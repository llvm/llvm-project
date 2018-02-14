//===--- NoMallocCheck.cpp - clang-tidy------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NoMallocCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <algorithm>
#include <string>
#include <vector>

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

namespace {
Matcher<FunctionDecl> hasAnyListedName(const std::string &FunctionNames) {
  const std::vector<std::string> NameList =
      utils::options::parseStringList(FunctionNames);
  return hasAnyName(std::vector<StringRef>(NameList.begin(), NameList.end()));
}
} // namespace

void NoMallocCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Allocations", AllocList);
  Options.store(Opts, "Reallocations", ReallocList);
  Options.store(Opts, "Deallocations", DeallocList);
}

void NoMallocCheck::registerMatchers(MatchFinder *Finder) {
  // C-style memory management is only problematic in C++.
  if (!getLangOpts().CPlusPlus)
    return;

  // Registering malloc, will suggest RAII.
  Finder->addMatcher(callExpr(callee(functionDecl(hasAnyListedName(AllocList))))
                         .bind("allocation"),
                     this);

  // Registering realloc calls, suggest std::vector or std::string.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasAnyListedName(ReallocList))))
          .bind("realloc"),
      this);

  // Registering free calls, will suggest RAII instead.
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasAnyListedName(DeallocList))))
          .bind("free"),
      this);
}

void NoMallocCheck::check(const MatchFinder::MatchResult &Result) {
  const CallExpr *Call = nullptr;
  StringRef Recommendation;

  if ((Call = Result.Nodes.getNodeAs<CallExpr>("allocation")))
    Recommendation = "consider a container or a smart pointer";
  else if ((Call = Result.Nodes.getNodeAs<CallExpr>("realloc")))
    Recommendation = "consider std::vector or std::string";
  else if ((Call = Result.Nodes.getNodeAs<CallExpr>("free")))
    Recommendation = "use RAII";

  assert(Call && "Unhandled binding in the Matcher");

  diag(Call->getLocStart(), "do not manage memory manually; %0")
      << Recommendation << SourceRange(Call->getLocStart(), Call->getLocEnd());
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
