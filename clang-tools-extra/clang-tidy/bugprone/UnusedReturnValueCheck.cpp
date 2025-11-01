//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnusedReturnValueCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/OperatorKinds.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang::tidy::bugprone {

namespace {

// Matches functions that are instantiated from a class template member function
// matching InnerMatcher. Functions not instantiated from a class template
// member function are matched directly with InnerMatcher.
AST_MATCHER_P(FunctionDecl, isInstantiatedFrom, Matcher<FunctionDecl>,
              InnerMatcher) {
  FunctionDecl *InstantiatedFrom = Node.getInstantiatedFromMemberFunction();
  return InnerMatcher.matches(InstantiatedFrom ? *InstantiatedFrom : Node,
                              Finder, Builder);
}

constexpr std::initializer_list<OverloadedOperatorKind>
    AssignmentOverloadedOperatorKinds = {
        OO_Equal,      OO_PlusEqual,     OO_MinusEqual,          OO_StarEqual,
        OO_SlashEqual, OO_PercentEqual,  OO_CaretEqual,          OO_AmpEqual,
        OO_PipeEqual,  OO_LessLessEqual, OO_GreaterGreaterEqual, OO_PlusPlus,
        OO_MinusMinus};

AST_MATCHER(FunctionDecl, isAssignmentOverloadedOperator) {
  return llvm::is_contained(AssignmentOverloadedOperatorKinds,
                            Node.getOverloadedOperator());
}
} // namespace

UnusedReturnValueCheck::UnusedReturnValueCheck(llvm::StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CheckedFunctions(utils::options::parseStringList(
          Options.get("CheckedFunctions", "^::std::async$;"
                                          "^::std::launder$;"
                                          "^::std::remove$;"
                                          "^::std::remove_if$;"
                                          "^::std::unique$;"
                                          "^::std::unique_ptr::release$;"
                                          "^::std::basic_string::empty$;"
                                          "^::std::vector::empty$;"
                                          "^::std::back_inserter$;"
                                          "^::std::distance$;"
                                          "^::std::find$;"
                                          "^::std::find_if$;"
                                          "^::std::inserter$;"
                                          "^::std::lower_bound$;"
                                          "^::std::make_pair$;"
                                          "^::std::map::count$;"
                                          "^::std::map::find$;"
                                          "^::std::map::lower_bound$;"
                                          "^::std::multimap::equal_range$;"
                                          "^::std::multimap::upper_bound$;"
                                          "^::std::set::count$;"
                                          "^::std::set::find$;"
                                          "^::std::setfill$;"
                                          "^::std::setprecision$;"
                                          "^::std::setw$;"
                                          "^::std::upper_bound$;"
                                          "^::std::vector::at$;"
                                          // C standard library
                                          "^::bsearch$;"
                                          "^::ferror$;"
                                          "^::feof$;"
                                          "^::isalnum$;"
                                          "^::isalpha$;"
                                          "^::isblank$;"
                                          "^::iscntrl$;"
                                          "^::isdigit$;"
                                          "^::isgraph$;"
                                          "^::islower$;"
                                          "^::isprint$;"
                                          "^::ispunct$;"
                                          "^::isspace$;"
                                          "^::isupper$;"
                                          "^::iswalnum$;"
                                          "^::iswprint$;"
                                          "^::iswspace$;"
                                          "^::isxdigit$;"
                                          "^::memchr$;"
                                          "^::memcmp$;"
                                          "^::strcmp$;"
                                          "^::strcoll$;"
                                          "^::strncmp$;"
                                          "^::strpbrk$;"
                                          "^::strrchr$;"
                                          "^::strspn$;"
                                          "^::strstr$;"
                                          "^::wcscmp$;"
                                          // POSIX
                                          "^::access$;"
                                          "^::bind$;"
                                          "^::connect$;"
                                          "^::difftime$;"
                                          "^::dlsym$;"
                                          "^::fnmatch$;"
                                          "^::getaddrinfo$;"
                                          "^::getopt$;"
                                          "^::htonl$;"
                                          "^::htons$;"
                                          "^::iconv_open$;"
                                          "^::inet_addr$;"
                                          "^::isascii$;"
                                          "^::isatty$;"
                                          "^::mmap$;"
                                          "^::newlocale$;"
                                          "^::openat$;"
                                          "^::pathconf$;"
                                          "^::pthread_equal$;"
                                          "^::pthread_getspecific$;"
                                          "^::pthread_mutex_trylock$;"
                                          "^::readdir$;"
                                          "^::readlink$;"
                                          "^::recvmsg$;"
                                          "^::regexec$;"
                                          "^::scandir$;"
                                          "^::semget$;"
                                          "^::setjmp$;"
                                          "^::shm_open$;"
                                          "^::shmget$;"
                                          "^::sigismember$;"
                                          "^::strcasecmp$;"
                                          "^::strsignal$;"
                                          "^::ttyname$"))),
      CheckedReturnTypes(utils::options::parseStringList(
          Options.get("CheckedReturnTypes", "^::std::error_code$;"
                                            "^::std::error_condition$;"
                                            "^::std::errc$;"
                                            "^::std::expected$;"
                                            "^::boost::system::error_code$"))),
      AllowCastToVoid(Options.get("AllowCastToVoid", false)) {}

UnusedReturnValueCheck::UnusedReturnValueCheck(
    llvm::StringRef Name, ClangTidyContext *Context,
    std::vector<StringRef> CheckedFunctions)
    : UnusedReturnValueCheck(Name, Context, std::move(CheckedFunctions), {},
                             false) {}

UnusedReturnValueCheck::UnusedReturnValueCheck(
    llvm::StringRef Name, ClangTidyContext *Context,
    std::vector<StringRef> CheckedFunctions,
    std::vector<StringRef> CheckedReturnTypes, bool AllowCastToVoid)
    : ClangTidyCheck(Name, Context),
      CheckedFunctions(std::move(CheckedFunctions)),
      CheckedReturnTypes(std::move(CheckedReturnTypes)),
      AllowCastToVoid(AllowCastToVoid) {}

void UnusedReturnValueCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckedFunctions",
                utils::options::serializeStringList(CheckedFunctions));
  Options.store(Opts, "CheckedReturnTypes",
                utils::options::serializeStringList(CheckedReturnTypes));
  Options.store(Opts, "AllowCastToVoid", AllowCastToVoid);
}

void UnusedReturnValueCheck::registerMatchers(MatchFinder *Finder) {
  auto MatchedDirectCallExpr =
      expr(callExpr(callee(functionDecl(
                        // Don't match copy or move assignment operator.
                        unless(isAssignmentOverloadedOperator()),
                        // Don't match void overloads of checked functions.
                        unless(returns(voidType())),
                        anyOf(isInstantiatedFrom(matchers::matchesAnyListedName(
                                  CheckedFunctions)),
                              returns(hasCanonicalType(hasDeclaration(
                                  namedDecl(matchers::matchesAnyListedName(
                                      CheckedReturnTypes)))))))))
               .bind("match"));

  auto CheckCastToVoid =
      AllowCastToVoid ? castExpr(unless(hasCastKind(CK_ToVoid))) : castExpr();
  auto MatchedCallExpr = expr(
      anyOf(MatchedDirectCallExpr,
            explicitCastExpr(unless(cxxFunctionalCastExpr()), CheckCastToVoid,
                             hasSourceExpression(MatchedDirectCallExpr))));

  auto UnusedInCompoundStmt =
      compoundStmt(forEach(MatchedCallExpr),
                   // The checker can't currently differentiate between the
                   // return statement and other statements inside GNU statement
                   // expressions, so disable the checker inside them to avoid
                   // false positives.
                   unless(hasParent(stmtExpr())));
  auto UnusedInIfStmt =
      ifStmt(eachOf(hasThen(MatchedCallExpr), hasElse(MatchedCallExpr)));
  auto UnusedInWhileStmt = whileStmt(hasBody(MatchedCallExpr));
  auto UnusedInDoStmt = doStmt(hasBody(MatchedCallExpr));
  auto UnusedInForStmt =
      forStmt(eachOf(hasLoopInit(MatchedCallExpr),
                     hasIncrement(MatchedCallExpr), hasBody(MatchedCallExpr)));
  auto UnusedInRangeForStmt = cxxForRangeStmt(hasBody(MatchedCallExpr));
  auto UnusedInCaseStmt = switchCase(forEach(MatchedCallExpr));

  Finder->addMatcher(
      stmt(anyOf(UnusedInCompoundStmt, UnusedInIfStmt, UnusedInWhileStmt,
                 UnusedInDoStmt, UnusedInForStmt, UnusedInRangeForStmt,
                 UnusedInCaseStmt)),
      this);
}

void UnusedReturnValueCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Matched = Result.Nodes.getNodeAs<CallExpr>("match")) {
    diag(Matched->getBeginLoc(),
         "the value returned by this function should not be disregarded; "
         "neglecting it may lead to errors")
        << Matched->getSourceRange();

    if (!AllowCastToVoid)
      return;

    diag(Matched->getBeginLoc(),
         "cast the expression to void to silence this warning",
         DiagnosticIDs::Note);
  }
}

} // namespace clang::tidy::bugprone
