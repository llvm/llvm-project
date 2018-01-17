//===--- PropertyDeclarationCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PropertyDeclarationCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Regex.h"
#include <algorithm>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace objc {

namespace {
/// The acronyms are from
/// https://developer.apple.com/library/content/documentation/Cocoa/Conceptual/CodingGuidelines/Articles/APIAbbreviations.html#//apple_ref/doc/uid/20001285-BCIHCGAE
constexpr char DefaultSpecialAcronyms[] =
    "ASCII;"
    "PDF;"
    "XML;"
    "HTML;"
    "URL;"
    "RTF;"
    "HTTP;"
    "TIFF;"
    "JPG;"
    "PNG;"
    "GIF;"
    "LZW;"
    "ROM;"
    "RGB;"
    "CMYK;"
    "MIDI;"
    "FTP;"
    "ID";

/// For now we will only fix 'CamelCase' property to
/// 'camelCase'. For other cases the users need to
/// come up with a proper name by their own.
/// FIXME: provide fix for snake_case to snakeCase
FixItHint generateFixItHint(const ObjCPropertyDecl *Decl) {
  if (isupper(Decl->getName()[0])) {
    auto NewName = Decl->getName().str();
    NewName[0] = tolower(NewName[0]);
    return FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(SourceRange(Decl->getLocation())),
        llvm::StringRef(NewName));
  }
  return FixItHint();
}

std::string validPropertyNameRegex(const std::vector<std::string> &Acronyms) {
  std::vector<std::string> EscapedAcronyms;
  EscapedAcronyms.reserve(Acronyms.size());
  // In case someone defines a custom prefix which includes a regex
  // special character, escape all the prefixes.
  std::transform(Acronyms.begin(), Acronyms.end(),
                 std::back_inserter(EscapedAcronyms), [](const std::string& s) {
                   return llvm::Regex::escape(s); });
  // Allow any of these names:
  // foo
  // fooBar
  // url
  // urlString
  // URL
  // URLString
  // bundleID
  return std::string("::((") +
      llvm::join(EscapedAcronyms.begin(), EscapedAcronyms.end(), "|") +
      ")[A-Z]?)?[a-z]+[a-z0-9]*([A-Z][a-z0-9]+)*" + "(" +
      llvm::join(EscapedAcronyms.begin(), EscapedAcronyms.end(), "|") + ")?$";
}
}  // namespace

PropertyDeclarationCheck::PropertyDeclarationCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SpecialAcronyms(utils::options::parseStringList(
          Options.get("Acronyms", DefaultSpecialAcronyms))) {}

void PropertyDeclarationCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      objcPropertyDecl(
          // the property name should be in Lower Camel Case like
          // 'lowerCamelCase'
          unless(matchesName(validPropertyNameRegex(SpecialAcronyms))))
          .bind("property"),
      this);
}

void PropertyDeclarationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<ObjCPropertyDecl>("property");
  assert(MatchedDecl->getName().size() > 0);
  diag(MatchedDecl->getLocation(),
       "property name '%0' should use lowerCamelCase style, according to "
       "the Apple Coding Guidelines")
      << MatchedDecl->getName() << generateFixItHint(MatchedDecl);
}

void PropertyDeclarationCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Acronyms",
                utils::options::serializeStringList(SpecialAcronyms));
}

}  // namespace objc
}  // namespace tidy
}  // namespace clang
