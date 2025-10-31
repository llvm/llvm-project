//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SignalHandlerCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"

// This is the minimal set of safe functions.
// https://wiki.sei.cmu.edu/confluence/display/c/SIG30-C.+Call+only+asynchronous-safe+functions+within+signal+handlers
constexpr llvm::StringLiteral MinimalConformingFunctions[] = {
    "signal", "abort", "_Exit", "quick_exit"};

// The POSIX-defined set of safe functions.
// https://pubs.opengroup.org/onlinepubs/9699919799/functions/V2_chap02.html#tag_15_04_03
// 'quick_exit' is added to the set additionally because it looks like the
// mentioned POSIX specification was not updated after 'quick_exit' appeared
// in the C11 standard.
// Also, we want to keep the "minimal set" a subset of the "POSIX set".
// The list is repeated in bugprone-signal-handler.rst and should be kept up to
// date.
// clang-format off
constexpr llvm::StringLiteral POSIXConformingFunctions[] = {
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
    "write"
};
// clang-format on

using namespace clang::ast_matchers;

namespace clang::tidy {

template <>
struct OptionEnumMapping<
    bugprone::SignalHandlerCheck::AsyncSafeFunctionSetKind> {
  static llvm::ArrayRef<std::pair<
      bugprone::SignalHandlerCheck::AsyncSafeFunctionSetKind, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<
        bugprone::SignalHandlerCheck::AsyncSafeFunctionSetKind, StringRef>
        Mapping[] = {
            {bugprone::SignalHandlerCheck::AsyncSafeFunctionSetKind::Minimal,
             "minimal"},
            {bugprone::SignalHandlerCheck::AsyncSafeFunctionSetKind::POSIX,
             "POSIX"},
        };
    return {Mapping};
  }
};

namespace bugprone {

/// Returns if a function is declared inside a system header.
/// These functions are considered to be "standard" (system-provided) library
/// functions.
static bool isStandardFunction(const FunctionDecl *FD) {
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

/// Check if a statement is "C++-only".
/// This includes all statements that have a class name with "CXX" prefix
/// and every other statement that is declared in file ExprCXX.h.
static bool isCXXOnlyStmt(const Stmt *S) {
  StringRef Name = S->getStmtClassName();
  if (Name.starts_with("CXX"))
    return true;
  // Check for all other class names in ExprCXX.h that have no 'CXX' prefix.
  return isa<ArrayTypeTraitExpr, BuiltinBitCastExpr, CUDAKernelCallExpr,
             CoawaitExpr, CoreturnStmt, CoroutineBodyStmt, CoroutineSuspendExpr,
             CoyieldExpr, DependentCoawaitExpr, DependentScopeDeclRefExpr,
             ExprWithCleanups, ExpressionTraitExpr, FunctionParmPackExpr,
             LambdaExpr, MSDependentExistsStmt, MSPropertyRefExpr,
             MSPropertySubscriptExpr, MaterializeTemporaryExpr, OverloadExpr,
             PackExpansionExpr, SizeOfPackExpr, SubstNonTypeTemplateParmExpr,
             SubstNonTypeTemplateParmPackExpr, TypeTraitExpr,
             UserDefinedLiteral>(S);
}

/// Given a call graph node of a \p Caller function and a \p Callee that is
/// called from \p Caller, get a \c CallExpr of the corresponding function call.
/// It is unspecified which call is found if multiple calls exist, but the order
/// should be deterministic (depend only on the AST).
static Expr *findCallExpr(const CallGraphNode *Caller,
                          const CallGraphNode *Callee) {
  const auto *FoundCallee = llvm::find_if(
      Caller->callees(), [Callee](const CallGraphNode::CallRecord &Call) {
        return Call.Callee == Callee;
      });
  assert(FoundCallee != Caller->end() &&
         "Callee should be called from the caller function here.");
  return FoundCallee->CallExpr;
}

static SourceRange getSourceRangeOfStmt(const Stmt *S, ASTContext &Ctx) {
  ParentMapContext &PM = Ctx.getParentMapContext();
  DynTypedNode P = DynTypedNode::create(*S);
  while (P.getSourceRange().isInvalid()) {
    DynTypedNodeList PL = PM.getParents(P);
    if (PL.size() != 1)
      return {};
    P = PL[0];
  }
  return P.getSourceRange();
}

namespace {

AST_MATCHER(FunctionDecl, isStandard) { return isStandardFunction(&Node); }

} // namespace

SignalHandlerCheck::SignalHandlerCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AsyncSafeFunctionSet(Options.get("AsyncSafeFunctionSet",
                                       AsyncSafeFunctionSetKind::POSIX)) {
  if (AsyncSafeFunctionSet == AsyncSafeFunctionSetKind::Minimal)
    ConformingFunctions.insert_range(MinimalConformingFunctions);
  else
    ConformingFunctions.insert_range(POSIXConformingFunctions);
}

void SignalHandlerCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AsyncSafeFunctionSet", AsyncSafeFunctionSet);
}

bool SignalHandlerCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return !LangOpts.CPlusPlus17;
}

void SignalHandlerCheck::registerMatchers(MatchFinder *Finder) {
  auto SignalFunction = functionDecl(hasAnyName("::signal", "::std::signal"),
                                     parameterCountIs(2), isStandard());
  auto HandlerExpr =
      declRefExpr(hasDeclaration(functionDecl().bind("handler_decl")),
                  unless(isExpandedFromMacro("SIG_IGN")),
                  unless(isExpandedFromMacro("SIG_DFL")))
          .bind("handler_expr");
  auto HandlerLambda = cxxMemberCallExpr(
      on(expr(ignoringParenImpCasts(lambdaExpr().bind("handler_lambda")))));
  Finder->addMatcher(callExpr(callee(SignalFunction),
                              hasArgument(1, anyOf(HandlerExpr, HandlerLambda)))
                         .bind("register_call"),
                     this);
}

void SignalHandlerCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *HandlerLambda =
          Result.Nodes.getNodeAs<LambdaExpr>("handler_lambda")) {
    diag(HandlerLambda->getBeginLoc(),
         "lambda function is not allowed as signal handler (until C++17)")
        << HandlerLambda->getSourceRange();
    return;
  }

  const auto *HandlerDecl =
      Result.Nodes.getNodeAs<FunctionDecl>("handler_decl");
  const auto *HandlerExpr = Result.Nodes.getNodeAs<DeclRefExpr>("handler_expr");
  assert(Result.Nodes.getNodeAs<CallExpr>("register_call") && HandlerDecl &&
         HandlerExpr && "All of these should exist in a match here.");

  if (CG.size() <= 1) {
    // Call graph must be populated with the entire TU at the beginning.
    // (It is possible to add a single function but the functions called from it
    // are not analysed in this case.)
    CG.addToCallGraph(const_cast<TranslationUnitDecl *>(
        HandlerDecl->getTranslationUnitDecl()));
    assert(CG.size() > 1 &&
           "There should be at least one function added to call graph.");
  }

  if (!HandlerDecl->hasBody()) {
    // Check the handler function.
    // The warning is placed to the signal handler registration.
    // No need to display a call chain and no need for more checks.
    (void)checkFunction(HandlerDecl, HandlerExpr, {});
    return;
  }

  // FIXME: Update CallGraph::getNode to use canonical decl?
  CallGraphNode *HandlerNode = CG.getNode(HandlerDecl->getCanonicalDecl());
  assert(HandlerNode &&
         "Handler with body should be present in the call graph.");
  // Start from signal handler and visit every function call.
  auto Itr = llvm::df_begin(HandlerNode), ItrE = llvm::df_end(HandlerNode);
  while (Itr != ItrE) {
    const auto *CallF = dyn_cast<FunctionDecl>((*Itr)->getDecl());
    unsigned int PathL = Itr.getPathLength();
    if (CallF) {
      // A signal handler or a function transitively reachable from the signal
      // handler was found to be unsafe.
      // Generate notes for the whole call chain (including the signal handler
      // registration).
      const Expr *CallOrRef = (PathL > 1)
                                  ? findCallExpr(Itr.getPath(PathL - 2), *Itr)
                                  : HandlerExpr;
      auto ChainReporter = [this, &Itr, HandlerExpr](bool SkipPathEnd) {
        reportHandlerChain(Itr, HandlerExpr, SkipPathEnd);
      };
      // If problems were found in a function (`CallF`), skip the analysis of
      // functions that are called from it.
      if (checkFunction(CallF, CallOrRef, ChainReporter))
        Itr.skipChildren();
      else
        ++Itr;
    } else {
      ++Itr;
    }
  }
}

bool SignalHandlerCheck::checkFunction(
    const FunctionDecl *FD, const Expr *CallOrRef,
    std::function<void(bool)> ChainReporter) {
  bool FunctionIsCalled = isa<CallExpr>(CallOrRef);

  if (isStandardFunction(FD)) {
    if (!isStandardFunctionAsyncSafe(FD)) {
      diag(CallOrRef->getBeginLoc(), "standard function %0 may not be "
                                     "asynchronous-safe; "
                                     "%select{using it as|calling it from}1 "
                                     "a signal handler may be dangerous")
          << FD << FunctionIsCalled << CallOrRef->getSourceRange();
      if (ChainReporter)
        ChainReporter(/*SkipPathEnd=*/true);
      return true;
    }
    return false;
  }

  if (!FD->hasBody()) {
    diag(CallOrRef->getBeginLoc(), "cannot verify that external function %0 is "
                                   "asynchronous-safe; "
                                   "%select{using it as|calling it from}1 "
                                   "a signal handler may be dangerous")
        << FD << FunctionIsCalled << CallOrRef->getSourceRange();
    if (ChainReporter)
      ChainReporter(/*SkipPathEnd=*/true);
    return true;
  }

  if (getLangOpts().CPlusPlus)
    return checkFunctionCPP14(FD, CallOrRef, ChainReporter);

  return false;
}

bool SignalHandlerCheck::checkFunctionCPP14(
    const FunctionDecl *FD, const Expr *CallOrRef,
    std::function<void(bool)> ChainReporter) {
  if (!FD->isExternC()) {
    diag(CallOrRef->getBeginLoc(),
         "functions without C linkage are not allowed as signal "
         "handler (until C++17)");
    if (ChainReporter)
      ChainReporter(/*SkipPathEnd=*/true);
    return true;
  }

  const FunctionDecl *FBody = nullptr;
  const Stmt *BodyS = FD->getBody(FBody);
  if (!BodyS)
    return false;

  bool StmtProblemsFound = false;
  ASTContext &Ctx = FBody->getASTContext();
  auto Matches =
      match(decl(forEachDescendant(stmt().bind("stmt"))), *FBody, Ctx);
  for (const auto &Match : Matches) {
    const auto *FoundS = Match.getNodeAs<Stmt>("stmt");
    if (isCXXOnlyStmt(FoundS)) {
      SourceRange R = getSourceRangeOfStmt(FoundS, Ctx);
      if (R.isInvalid())
        continue;
      diag(R.getBegin(),
           "C++-only construct is not allowed in signal handler (until C++17)")
          << R;
      diag(R.getBegin(), "internally, the statement is parsed as a '%0'",
           DiagnosticIDs::Remark)
          << FoundS->getStmtClassName();
      if (ChainReporter)
        ChainReporter(/*SkipPathEnd=*/false);
      StmtProblemsFound = true;
    }
  }

  return StmtProblemsFound;
}

bool SignalHandlerCheck::isStandardFunctionAsyncSafe(
    const FunctionDecl *FD) const {
  assert(isStandardFunction(FD));

  const IdentifierInfo *II = FD->getIdentifier();
  // Unnamed functions are not explicitly allowed.
  // C++ std operators may be unsafe and not within the
  // "common subset of C and C++".
  if (!II)
    return false;

  if (!FD->isInStdNamespace() && !FD->isGlobal())
    return false;

  if (ConformingFunctions.contains(II->getName()))
    return true;

  return false;
}

void SignalHandlerCheck::reportHandlerChain(
    const llvm::df_iterator<clang::CallGraphNode *> &Itr,
    const DeclRefExpr *HandlerRef, bool SkipPathEnd) {
  int CallLevel = Itr.getPathLength() - 2;
  assert(CallLevel >= -1 && "Empty iterator?");

  const CallGraphNode *Caller = Itr.getPath(CallLevel + 1), *Callee = nullptr;
  while (CallLevel >= 0) {
    Callee = Caller;
    Caller = Itr.getPath(CallLevel);
    const Expr *CE = findCallExpr(Caller, Callee);
    if (SkipPathEnd)
      SkipPathEnd = false;
    else
      diag(CE->getBeginLoc(), "function %0 called here from %1",
           DiagnosticIDs::Note)
          << cast<FunctionDecl>(Callee->getDecl())
          << cast<FunctionDecl>(Caller->getDecl());
    --CallLevel;
  }

  if (!SkipPathEnd)
    diag(HandlerRef->getBeginLoc(),
         "function %0 registered here as signal handler", DiagnosticIDs::Note)
        << cast<FunctionDecl>(Caller->getDecl())
        << HandlerRef->getSourceRange();
}

} // namespace bugprone
} // namespace clang::tidy
