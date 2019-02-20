//===--- PropertyDeclarationCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PropertyDeclarationCheck.h"
#include <algorithm>
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Regex.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace objc {

namespace {

// For StandardProperty the naming style is 'lowerCamelCase'.
// For CategoryProperty especially in categories of system class,
// to avoid naming conflict, the suggested naming style is
// 'abc_lowerCamelCase' (adding lowercase prefix followed by '_').
// Regardless of the style, all acronyms and initialisms should be capitalized.
enum NamingStyle {
  StandardProperty = 1,
  CategoryProperty = 2,
};

/// For now we will only fix 'CamelCase' or 'abc_CamelCase' property to
/// 'camelCase' or 'abc_camelCase'. For other cases the users need to
/// come up with a proper name by their own.
/// FIXME: provide fix for snake_case to snakeCase
FixItHint generateFixItHint(const ObjCPropertyDecl *Decl, NamingStyle Style) {
  auto Name = Decl->getName();
  auto NewName = Decl->getName().str();
  size_t Index = 0;
  if (Style == CategoryProperty) {
    Index = Name.find_first_of('_') + 1;
    NewName.replace(0, Index - 1, Name.substr(0, Index - 1).lower());
  }
  if (Index < Name.size()) {
    NewName[Index] = tolower(NewName[Index]);
    if (NewName != Name) {
      return FixItHint::CreateReplacement(
          CharSourceRange::getTokenRange(SourceRange(Decl->getLocation())),
          llvm::StringRef(NewName));
    }
  }
  return FixItHint();
}

std::string validPropertyNameRegex(bool UsedInMatcher) {
  // Allow any of these names:
  // foo
  // fooBar
  // url
  // urlString
  // ID
  // IDs
  // URL
  // URLString
  // bundleID
  // CIColor
  //
  // Disallow names of this form:
  // LongString
  //
  // aRbITRaRyCapS is allowed to avoid generating false positives for names
  // like isVitaminBSupplement, CProgrammingLanguage, and isBeforeM.
  std::string StartMatcher = UsedInMatcher ? "::" : "^";
  return StartMatcher + "([a-z]|[A-Z][A-Z0-9])[a-z0-9A-Z]*$";
}

bool hasCategoryPropertyPrefix(llvm::StringRef PropertyName) {
  auto RegexExp = llvm::Regex("^[a-zA-Z]+_[a-zA-Z0-9][a-zA-Z0-9_]+$");
  return RegexExp.match(PropertyName);
}

bool prefixedPropertyNameValid(llvm::StringRef PropertyName) {
  size_t Start = PropertyName.find_first_of('_');
  assert(Start != llvm::StringRef::npos && Start + 1 < PropertyName.size());
  auto Prefix = PropertyName.substr(0, Start);
  if (Prefix.lower() != Prefix) {
    return false;
  }
  auto RegexExp =
      llvm::Regex(llvm::StringRef(validPropertyNameRegex(false)));
  return RegexExp.match(PropertyName.substr(Start + 1));
}
}  // namespace

PropertyDeclarationCheck::PropertyDeclarationCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SpecialAcronyms(
          utils::options::parseStringList(Options.get("Acronyms", ""))),
      IncludeDefaultAcronyms(Options.get("IncludeDefaultAcronyms", true)),
      EscapedAcronyms() {}

void PropertyDeclarationCheck::registerMatchers(MatchFinder *Finder) {
  // this check should only be applied to ObjC sources.
  if (!getLangOpts().ObjC) return;

  Finder->addMatcher(
      objcPropertyDecl(
          // the property name should be in Lower Camel Case like
          // 'lowerCamelCase'
          unless(matchesName(validPropertyNameRegex(true))))
          .bind("property"),
      this);
}

void PropertyDeclarationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<ObjCPropertyDecl>("property");
  assert(MatchedDecl->getName().size() > 0);
  auto *DeclContext = MatchedDecl->getDeclContext();
  auto *CategoryDecl = llvm::dyn_cast<ObjCCategoryDecl>(DeclContext);

  if (CategoryDecl != nullptr &&
      hasCategoryPropertyPrefix(MatchedDecl->getName())) {
    if (!prefixedPropertyNameValid(MatchedDecl->getName()) ||
        CategoryDecl->IsClassExtension()) {
      NamingStyle Style = CategoryDecl->IsClassExtension() ? StandardProperty
                                                           : CategoryProperty;
      diag(MatchedDecl->getLocation(),
           "property name '%0' not using lowerCamelCase style or not prefixed "
           "in a category, according to the Apple Coding Guidelines")
          << MatchedDecl->getName() << generateFixItHint(MatchedDecl, Style);
    }
    return;
  }
  diag(MatchedDecl->getLocation(),
       "property name '%0' not using lowerCamelCase style or not prefixed in "
       "a category, according to the Apple Coding Guidelines")
      << MatchedDecl->getName()
      << generateFixItHint(MatchedDecl, StandardProperty);
}

void PropertyDeclarationCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Acronyms",
                utils::options::serializeStringList(SpecialAcronyms));
  Options.store(Opts, "IncludeDefaultAcronyms", IncludeDefaultAcronyms);
}

}  // namespace objc
}  // namespace tidy
}  // namespace clang
