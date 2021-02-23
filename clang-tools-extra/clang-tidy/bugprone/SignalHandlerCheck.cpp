//===--- SignalHandlerCheck.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SignalHandlerCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/CallGraph.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <iterator>
#include <queue>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

template <>
struct OptionEnumMapping<
    bugprone::SignalHandlerCheck::AsyncSafeFunctionSetType> {
  static llvm::ArrayRef<std::pair<
      bugprone::SignalHandlerCheck::AsyncSafeFunctionSetType, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<
        bugprone::SignalHandlerCheck::AsyncSafeFunctionSetType, StringRef>
        Mapping[] = {
            {bugprone::SignalHandlerCheck::AsyncSafeFunctionSetType::Minimal,
             "minimal"},
            {bugprone::SignalHandlerCheck::AsyncSafeFunctionSetType::POSIX,
             "POSIX"},
        };
    return makeArrayRef(Mapping);
  }
};

namespace bugprone {

static bool isSystemCall(const FunctionDecl *FD) {
  // Find a possible redeclaration in system header.
  // FIXME: Looking at the canonical declaration is not the most exact way
  // to do this.

  // Most common case will be inclusion directly from a header.
  // This works fine by using canonical declaration.
  // a.c
  // #include <sysheader.h>

  // Next most common case will be extern declaration.
  // Can't catch this with either approach.
  // b.c
  // extern void sysfunc(void);

  // Canonical declaration is the first found declaration, so this works.
  // c.c
  // #include <sysheader.h>
  // extern void sysfunc(void); // redecl won't matter

  // This does not work with canonical declaration.
  // Probably this is not a frequently used case but may happen (the first
  // declaration can be in a non-system header for example).
  // d.c
  // extern void sysfunc(void); // Canonical declaration, not in system header.
  // #include <sysheader.h>

  return FD->getASTContext().getSourceManager().isInSystemHeader(
      FD->getCanonicalDecl()->getLocation());
}

AST_MATCHER(FunctionDecl, isSystemCall) { return isSystemCall(&Node); }

SignalHandlerCheck::SignalHandlerCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AsyncSafeFunctionSet(
          Options.get("AsyncSafeFunctionSet", AsyncSafeFunctionSetType::POSIX)),
      ConformingFunctions(AsyncSafeFunctionSet ==
                                  AsyncSafeFunctionSetType::Minimal
                              ? MinimalConformingFunctions
                              : POSIXConformingFunctions) {}

void SignalHandlerCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AsyncSafeFunctionSet", AsyncSafeFunctionSet);
}

bool SignalHandlerCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  // FIXME: Make the checker useful on C++ code.
  if (LangOpts.CPlusPlus)
    return false;

  return true;
}

void SignalHandlerCheck::registerMatchers(MatchFinder *Finder) {
  auto SignalFunction = functionDecl(hasAnyName("::signal", "::std::signal"),
                                     parameterCountIs(2), isSystemCall());
  auto HandlerExpr =
      declRefExpr(hasDeclaration(functionDecl().bind("handler_decl")),
                  unless(isExpandedFromMacro("SIG_IGN")),
                  unless(isExpandedFromMacro("SIG_DFL")))
          .bind("handler_expr");
  Finder->addMatcher(
      callExpr(callee(SignalFunction), hasArgument(1, HandlerExpr))
          .bind("register_call"),
      this);
}

void SignalHandlerCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *SignalCall = Result.Nodes.getNodeAs<CallExpr>("register_call");
  const auto *HandlerDecl =
      Result.Nodes.getNodeAs<FunctionDecl>("handler_decl");
  const auto *HandlerExpr = Result.Nodes.getNodeAs<DeclRefExpr>("handler_expr");

  // Visit each function encountered in the callgraph only once.
  llvm::DenseSet<const FunctionDecl *> SeenFunctions;

  // The worklist of the callgraph visitation algorithm.
  std::deque<const CallExpr *> CalledFunctions;

  auto ProcessFunction = [&](const FunctionDecl *F, const Expr *CallOrRef) {
    // Ensure that canonical declaration is used.
    F = F->getCanonicalDecl();

    // Do not visit function if already encountered.
    if (!SeenFunctions.insert(F).second)
      return true;

    // Check if the call is allowed.
    // Non-system calls are not considered.
    if (isSystemCall(F)) {
      if (isSystemCallAllowed(F))
        return true;

      reportBug(F, CallOrRef, SignalCall, HandlerDecl);

      return false;
    }

    // Get the body of the encountered non-system call function.
    const FunctionDecl *FBody;
    if (!F->hasBody(FBody)) {
      reportBug(F, CallOrRef, SignalCall, HandlerDecl);
      return false;
    }

    // Collect all called functions.
    auto Matches = match(decl(forEachDescendant(callExpr().bind("call"))),
                         *FBody, FBody->getASTContext());
    for (const auto &Match : Matches) {
      const auto *CE = Match.getNodeAs<CallExpr>("call");
      if (isa<FunctionDecl>(CE->getCalleeDecl()))
        CalledFunctions.push_back(CE);
    }

    return true;
  };

  if (!ProcessFunction(HandlerDecl, HandlerExpr))
    return;

  // Visit the definition of every function referenced by the handler function.
  // Check for allowed function calls.
  while (!CalledFunctions.empty()) {
    const CallExpr *FunctionCall = CalledFunctions.front();
    CalledFunctions.pop_front();
    // At insertion we have already ensured that only function calls are there.
    const auto *F = cast<FunctionDecl>(FunctionCall->getCalleeDecl());

    if (!ProcessFunction(F, FunctionCall))
      break;
  }
}

bool SignalHandlerCheck::isSystemCallAllowed(const FunctionDecl *FD) const {
  const IdentifierInfo *II = FD->getIdentifier();
  // Unnamed functions are not explicitly allowed.
  if (!II)
    return false;

  // FIXME: Improve for C++ (check for namespace).
  if (ConformingFunctions.count(II->getName()))
    return true;

  return false;
}

void SignalHandlerCheck::reportBug(const FunctionDecl *CalledFunction,
                                   const Expr *CallOrRef,
                                   const CallExpr *SignalCall,
                                   const FunctionDecl *HandlerDecl) {
  diag(CallOrRef->getBeginLoc(),
       "%0 may not be asynchronous-safe; "
       "calling it from a signal handler may be dangerous")
      << CalledFunction;
  diag(SignalCall->getSourceRange().getBegin(),
       "signal handler registered here", DiagnosticIDs::Note);
  diag(HandlerDecl->getBeginLoc(), "handler function declared here",
       DiagnosticIDs::Note);
}

// This is the minimal set of safe functions.
// https://wiki.sei.cmu.edu/confluence/display/c/SIG30-C.+Call+only+asynchronous-safe+functions+within+signal+handlers
llvm::StringSet<> SignalHandlerCheck::MinimalConformingFunctions{
    "signal", "abort", "_Exit", "quick_exit"};

// The POSIX-defined set of safe functions.
// https://pubs.opengroup.org/onlinepubs/9699919799/functions/V2_chap02.html#tag_15_04_03
// 'quick_exit' is added to the set additionally because it looks like the
// mentioned POSIX specification was not updated after 'quick_exit' appeared
// in the C11 standard.
// Also, we want to keep the "minimal set" a subset of the "POSIX set".
llvm::StringSet<> SignalHandlerCheck::POSIXConformingFunctions{
    "_Exit",
    "_exit",
    "abort",
    "accept",
    "access",
    "aio_error",
    "aio_return",
    "aio_suspend",
    "alarm",
    "bind",
    "cfgetispeed",
    "cfgetospeed",
    "cfsetispeed",
    "cfsetospeed",
    "chdir",
    "chmod",
    "chown",
    "clock_gettime",
    "close",
    "connect",
    "creat",
    "dup",
    "dup2",
    "execl",
    "execle",
    "execv",
    "execve",
    "faccessat",
    "fchdir",
    "fchmod",
    "fchmodat",
    "fchown",
    "fchownat",
    "fcntl",
    "fdatasync",
    "fexecve",
    "ffs",
    "fork",
    "fstat",
    "fstatat",
    "fsync",
    "ftruncate",
    "futimens",
    "getegid",
    "geteuid",
    "getgid",
    "getgroups",
    "getpeername",
    "getpgrp",
    "getpid",
    "getppid",
    "getsockname",
    "getsockopt",
    "getuid",
    "htonl",
    "htons",
    "kill",
    "link",
    "linkat",
    "listen",
    "longjmp",
    "lseek",
    "lstat",
    "memccpy",
    "memchr",
    "memcmp",
    "memcpy",
    "memmove",
    "memset",
    "mkdir",
    "mkdirat",
    "mkfifo",
    "mkfifoat",
    "mknod",
    "mknodat",
    "ntohl",
    "ntohs",
    "open",
    "openat",
    "pause",
    "pipe",
    "poll",
    "posix_trace_event",
    "pselect",
    "pthread_kill",
    "pthread_self",
    "pthread_sigmask",
    "quick_exit",
    "raise",
    "read",
    "readlink",
    "readlinkat",
    "recv",
    "recvfrom",
    "recvmsg",
    "rename",
    "renameat",
    "rmdir",
    "select",
    "sem_post",
    "send",
    "sendmsg",
    "sendto",
    "setgid",
    "setpgid",
    "setsid",
    "setsockopt",
    "setuid",
    "shutdown",
    "sigaction",
    "sigaddset",
    "sigdelset",
    "sigemptyset",
    "sigfillset",
    "sigismember",
    "siglongjmp",
    "signal",
    "sigpause",
    "sigpending",
    "sigprocmask",
    "sigqueue",
    "sigset",
    "sigsuspend",
    "sleep",
    "sockatmark",
    "socket",
    "socketpair",
    "stat",
    "stpcpy",
    "stpncpy",
    "strcat",
    "strchr",
    "strcmp",
    "strcpy",
    "strcspn",
    "strlen",
    "strncat",
    "strncmp",
    "strncpy",
    "strnlen",
    "strpbrk",
    "strrchr",
    "strspn",
    "strstr",
    "strtok_r",
    "symlink",
    "symlinkat",
    "tcdrain",
    "tcflow",
    "tcflush",
    "tcgetattr",
    "tcgetpgrp",
    "tcsendbreak",
    "tcsetattr",
    "tcsetpgrp",
    "time",
    "timer_getoverrun",
    "timer_gettime",
    "timer_settime",
    "times",
    "umask",
    "uname",
    "unlink",
    "unlinkat",
    "utime",
    "utimensat",
    "utimes",
    "wait",
    "waitpid",
    "wcpcpy",
    "wcpncpy",
    "wcscat",
    "wcschr",
    "wcscmp",
    "wcscpy",
    "wcscspn",
    "wcslen",
    "wcsncat",
    "wcsncmp",
    "wcsncpy",
    "wcsnlen",
    "wcspbrk",
    "wcsrchr",
    "wcsspn",
    "wcsstr",
    "wcstok",
    "wmemchr",
    "wmemcmp",
    "wmemcpy",
    "wmemmove",
    "wmemset",
    "write"};

} // namespace bugprone
} // namespace tidy
} // namespace clang
