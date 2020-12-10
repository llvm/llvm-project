//===--- MtUnsafeCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MtUnsafeCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

// Initial list was extracted from gcc documentation
static const clang::StringRef GlibcFunctions[] = {
    "::argp_error",
    "::argp_help",
    "::argp_parse",
    "::argp_state_help",
    "::argp_usage",
    "::asctime",
    "::clearenv",
    "::crypt",
    "::ctime",
    "::cuserid",
    "::drand48",
    "::ecvt",
    "::encrypt",
    "::endfsent",
    "::endgrent",
    "::endhostent",
    "::endnetent",
    "::endnetgrent",
    "::endprotoent",
    "::endpwent",
    "::endservent",
    "::endutent",
    "::endutxent",
    "::erand48",
    "::error_at_line",
    "::exit",
    "::fcloseall",
    "::fcvt",
    "::fgetgrent",
    "::fgetpwent",
    "::gammal",
    "::getchar_unlocked",
    "::getdate",
    "::getfsent",
    "::getfsfile",
    "::getfsspec",
    "::getgrent",
    "::getgrent_r",
    "::getgrgid",
    "::getgrnam",
    "::gethostbyaddr",
    "::gethostbyname",
    "::gethostbyname2",
    "::gethostent",
    "::getlogin",
    "::getmntent",
    "::getnetbyaddr",
    "::getnetbyname",
    "::getnetent",
    "::getnetgrent",
    "::getnetgrent_r",
    "::getopt",
    "::getopt_long",
    "::getopt_long_only",
    "::getpass",
    "::getprotobyname",
    "::getprotobynumber",
    "::getprotoent",
    "::getpwent",
    "::getpwent_r",
    "::getpwnam",
    "::getpwuid",
    "::getservbyname",
    "::getservbyport",
    "::getservent",
    "::getutent",
    "::getutent_r",
    "::getutid",
    "::getutid_r",
    "::getutline",
    "::getutline_r",
    "::getutxent",
    "::getutxid",
    "::getutxline",
    "::getwchar_unlocked",
    "::glob",
    "::glob64",
    "::gmtime",
    "::hcreate",
    "::hdestroy",
    "::hsearch",
    "::innetgr",
    "::jrand48",
    "::l64a",
    "::lcong48",
    "::lgammafNx",
    "::localeconv",
    "::localtime",
    "::login",
    "::login_tty",
    "::logout",
    "::logwtmp",
    "::lrand48",
    "::mallinfo",
    "::mallopt",
    "::mblen",
    "::mbrlen",
    "::mbrtowc",
    "::mbsnrtowcs",
    "::mbsrtowcs",
    "::mbtowc",
    "::mcheck",
    "::mprobe",
    "::mrand48",
    "::mtrace",
    "::muntrace",
    "::nrand48",
    "::__ppc_get_timebase_freq",
    "::ptsname",
    "::putchar_unlocked",
    "::putenv",
    "::pututline",
    "::pututxline",
    "::putwchar_unlocked",
    "::qecvt",
    "::qfcvt",
    "::register_printf_function",
    "::seed48",
    "::setenv",
    "::setfsent",
    "::setgrent",
    "::sethostent",
    "::sethostid",
    "::setkey",
    "::setlocale",
    "::setlogmask",
    "::setnetent",
    "::setnetgrent",
    "::setprotoent",
    "::setpwent",
    "::setservent",
    "::setutent",
    "::setutxent",
    "::siginterrupt",
    "::sigpause",
    "::sigprocmask",
    "::sigsuspend",
    "::sleep",
    "::srand48",
    "::strerror",
    "::strsignal",
    "::strtok",
    "::tcflow",
    "::tcsendbreak",
    "::tmpnam",
    "::ttyname",
    "::unsetenv",
    "::updwtmp",
    "::utmpname",
    "::utmpxname",
    "::valloc",
    "::vlimit",
    "::wcrtomb",
    "::wcsnrtombs",
    "::wcsrtombs",
    "::wctomb",
    "::wordexp",
};

static const clang::StringRef PosixFunctions[] = {
    "::asctime",
    "::basename",
    "::catgets",
    "::crypt",
    "::ctime",
    "::dbm_clearerr",
    "::dbm_close",
    "::dbm_delete",
    "::dbm_error",
    "::dbm_fetch",
    "::dbm_firstkey",
    "::dbm_nextkey",
    "::dbm_open",
    "::dbm_store",
    "::dirname",
    "::dlerror",
    "::drand48",
    "::encrypt",
    "::endgrent",
    "::endpwent",
    "::endutxent",
    "::ftw",
    "::getc_unlocked",
    "::getchar_unlocked",
    "::getdate",
    "::getenv",
    "::getgrent",
    "::getgrgid",
    "::getgrnam",
    "::gethostent",
    "::getlogin",
    "::getnetbyaddr",
    "::getnetbyname",
    "::getnetent",
    "::getopt",
    "::getprotobyname",
    "::getprotobynumber",
    "::getprotoent",
    "::getpwent",
    "::getpwnam",
    "::getpwuid",
    "::getservbyname",
    "::getservbyport",
    "::getservent",
    "::getutxent",
    "::getutxid",
    "::getutxline",
    "::gmtime",
    "::hcreate",
    "::hdestroy",
    "::hsearch",
    "::inet_ntoa",
    "::l64a",
    "::lgamma",
    "::lgammaf",
    "::lgammal",
    "::localeconv",
    "::localtime",
    "::lrand48",
    "::mrand48",
    "::nftw",
    "::nl_langinfo",
    "::ptsname",
    "::putc_unlocked",
    "::putchar_unlocked",
    "::putenv",
    "::pututxline",
    "::rand",
    "::readdir",
    "::setenv",
    "::setgrent",
    "::setkey",
    "::setpwent",
    "::setutxent",
    "::strerror",
    "::strsignal",
    "::strtok",
    "::system",
    "::ttyname",
    "::unsetenv",
    "::wcstombs",
    "::wctomb",
};

namespace clang {
namespace tidy {

template <> struct OptionEnumMapping<concurrency::MtUnsafeCheck::FunctionSet> {
  static llvm::ArrayRef<
      std::pair<concurrency::MtUnsafeCheck::FunctionSet, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<concurrency::MtUnsafeCheck::FunctionSet,
                               StringRef>
        Mapping[] = {{concurrency::MtUnsafeCheck::FunctionSet::Posix, "posix"},
                     {concurrency::MtUnsafeCheck::FunctionSet::Glibc, "glibc"},
                     {concurrency::MtUnsafeCheck::FunctionSet::Any, "any"}};
    return makeArrayRef(Mapping);
  }
};

namespace concurrency {

static ast_matchers::internal::Matcher<clang::NamedDecl>
hasAnyMtUnsafeNames(MtUnsafeCheck::FunctionSet libc) {
  switch (libc) {
  case MtUnsafeCheck::FunctionSet::Posix:
    return hasAnyName(PosixFunctions);
  case MtUnsafeCheck::FunctionSet::Glibc:
    return hasAnyName(GlibcFunctions);
  case MtUnsafeCheck::FunctionSet::Any:
    return anyOf(hasAnyName(PosixFunctions), hasAnyName(GlibcFunctions));
  }
  llvm_unreachable("invalid FunctionSet");
}

MtUnsafeCheck::MtUnsafeCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      FuncSet(Options.get("FunctionSet", MtUnsafeCheck::FunctionSet::Any)) {}

void MtUnsafeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "FunctionSet", FuncSet);
}

void MtUnsafeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasAnyMtUnsafeNames(FuncSet))))
          .bind("mt-unsafe"),
      this);
}

void MtUnsafeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("mt-unsafe");
  assert(Call && "Unhandled binding in the Matcher");

  diag(Call->getBeginLoc(), "function is not thread safe");
}

} // namespace concurrency
} // namespace tidy
} // namespace clang
