//===--- FunctionNamingCheck.cpp - clang-tidy -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionNamingCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/Support/Regex.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace google {
namespace objc {

namespace {

std::string validFunctionNameRegex(bool RequirePrefix) {
  // Allow the following name patterns for all functions:
  // • ABFoo (prefix + UpperCamelCase)
  // • ABURL (prefix + capitalized acronym/initialism)
  //
  // If no prefix is required, additionally allow the following name patterns:
  // • Foo (UpperCamelCase)
  // • URL (capitalized acronym/initialism)
  //
  // The function name following the prefix can contain standard and
  // non-standard capitalized character sequences including acronyms,
  // initialisms, and prefixes of symbols (e.g., UIColorFromNSString). For this
  // reason, the regex only verifies that the function name after the prefix
  // begins with a capital letter followed by an arbitrary sequence of
  // alphanumeric characters.
  //
  // If a prefix is required, the regex checks for a capital letter followed by
  // another capital letter or number that is part of the prefix and another
  // capital letter or number that begins the name following the prefix.
  std::string FunctionNameMatcher =
      std::string(RequirePrefix ? "[A-Z][A-Z0-9]+" : "") + "[A-Z][a-zA-Z0-9]*";
  return std::string("::(") + FunctionNameMatcher + ")$";
}

/// For now we will only fix functions of static storage class with names like
/// 'functionName' or 'function_name' and convert them to 'FunctionName'. For
/// other cases the user must determine an appropriate name on their own.
FixItHint generateFixItHint(const FunctionDecl *Decl) {
  // A fixit can be generated for functions of static storage class but
  // otherwise the check cannot determine the appropriate function name prefix
  // to use.
  if (Decl->getStorageClass() != SC_Static)
    return FixItHint();

  StringRef Name = Decl->getName();
  std::string NewName = Decl->getName().str();

  size_t Index = 0;
  bool AtWordBoundary = true;
  while (Index < NewName.size()) {
    char ch = NewName[Index];
    if (isalnum(ch)) {
      // Capitalize the first letter after every word boundary.
      if (AtWordBoundary) {
        NewName[Index] = toupper(NewName[Index]);
        AtWordBoundary = false;
      }

      // Advance the index after every alphanumeric character.
      Index++;
    } else {
      // Strip out any characters other than alphanumeric characters.
      NewName.erase(Index, 1);
      AtWordBoundary = true;
    }
  }

  // Generate a fixit hint if the new name is different.
  if (NewName != Name)
    return FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(SourceRange(Decl->getLocation())),
        llvm::StringRef(NewName));

  return FixItHint();
}

} // namespace

void FunctionNamingCheck::registerMatchers(MatchFinder *Finder) {
  // This check should only be applied to Objective-C sources.
  if (!getLangOpts().ObjC)
    return;

  // Enforce Objective-C function naming conventions on all functions except:
  // • Functions defined in system headers.
  // • C++ member functions.
  // • Namespaced functions.
  // • Implicitly defined functions.
  // • The main function.
  Finder->addMatcher(
      functionDecl(
          unless(anyOf(isExpansionInSystemHeader(), cxxMethodDecl(),
                       hasAncestor(namespaceDecl()), isMain(), isImplicit(),
                       matchesName(validFunctionNameRegex(true)),
                       allOf(isStaticStorageClass(),
                             matchesName(validFunctionNameRegex(false))))))
          .bind("function"),
      this);
}

void FunctionNamingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("function");

  bool IsGlobal = MatchedDecl->getStorageClass() != SC_Static;
  diag(MatchedDecl->getLocation(),
       "%select{static function|function in global namespace}1 named %0 must "
       "%select{be in|have an appropriate prefix followed by}1 Pascal case as "
       "required by Google Objective-C style guide")
      << MatchedDecl << IsGlobal << generateFixItHint(MatchedDecl);
}

} // namespace objc
} // namespace google
} // namespace tidy
} // namespace clang
