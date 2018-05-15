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
enum NamingStyle {
  StandardProperty = 1,
  CategoryProperty = 2,
};

/// The acronyms are from
/// https://developer.apple.com/library/content/documentation/Cocoa/Conceptual/CodingGuidelines/Articles/APIAbbreviations.html#//apple_ref/doc/uid/20001285-BCIHCGAE
///
/// Keep this list sorted.
constexpr llvm::StringLiteral DefaultSpecialAcronyms[] = {
    "[2-9]G",
    "ACL",
    "API",
    "AR",
    "ARGB",
    "ASCII",
    "BGRA",
    "CA",
    "CF",
    "CG",
    "CI",
    "CV",
    "CMYK",
    "DNS",
    "FPS",
    "FTP",
    "GIF",
    "GL",
    "GPS",
    "GUID",
    "HD",
    "HDR",
    "HTML",
    "HTTP",
    "HTTPS",
    "HUD",
    "ID",
    "JPG",
    "JS",
    "LAN",
    "LZW",
    "MDNS",
    "MIDI",
    "NS",
    "OS",
    "PDF",
    "PIN",
    "PNG",
    "POI",
    "PSTN",
    "PTR",
    "QA",
    "QOS",
    "RGB",
    "RGBA",
    "RGBX",
    "ROM",
    "RPC",
    "RTF",
    "RTL",
    "SC",
    "SDK",
    "SSO",
    "TCP",
    "TIFF",
    "TTS",
    "UI",
    "URI",
    "URL",
    "UUID",
    "VC",
    "VOIP",
    "VPN",
    "VR",
    "W",
    "WAN",
    "X",
    "XML",
    "Y",
    "Z",
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

std::string AcronymsGroupRegex(llvm::ArrayRef<std::string> EscapedAcronyms) {
  return "(" +
         llvm::join(EscapedAcronyms.begin(), EscapedAcronyms.end(), "s?|") +
         "s?)";
}

std::string validPropertyNameRegex(llvm::ArrayRef<std::string> EscapedAcronyms,
                                   bool UsedInMatcher) {
  // Allow any of these names:
  // foo
  // fooBar
  // url
  // urlString
  // URL
  // URLString
  // bundleID
  std::string StartMatcher = UsedInMatcher ? "::" : "^";
  std::string AcronymsMatcher = AcronymsGroupRegex(EscapedAcronyms);
  return StartMatcher + "(" + AcronymsMatcher + "[A-Z]?)?[a-z]+[a-z0-9]*(" +
         AcronymsMatcher + "|([A-Z][a-z0-9]+))*$";
}

bool hasCategoryPropertyPrefix(llvm::StringRef PropertyName) {
  auto RegexExp = llvm::Regex("^[a-zA-Z]+_[a-zA-Z0-9][a-zA-Z0-9_]+$");
  return RegexExp.match(PropertyName);
}

bool prefixedPropertyNameValid(llvm::StringRef PropertyName,
                               llvm::ArrayRef<std::string> Acronyms) {
  size_t Start = PropertyName.find_first_of('_');
  assert(Start != llvm::StringRef::npos && Start + 1 < PropertyName.size());
  auto Prefix = PropertyName.substr(0, Start);
  if (Prefix.lower() != Prefix) {
    return false;
  }
  auto RegexExp =
      llvm::Regex(llvm::StringRef(validPropertyNameRegex(Acronyms, false)));
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
  if (!getLangOpts().ObjC1 && !getLangOpts().ObjC2) {
    return;
  }
  if (IncludeDefaultAcronyms) {
    EscapedAcronyms.reserve(llvm::array_lengthof(DefaultSpecialAcronyms) +
                            SpecialAcronyms.size());
    // No need to regex-escape the default acronyms.
    EscapedAcronyms.insert(EscapedAcronyms.end(),
                           std::begin(DefaultSpecialAcronyms),
                           std::end(DefaultSpecialAcronyms));
  } else {
    EscapedAcronyms.reserve(SpecialAcronyms.size());
  }
  // In case someone defines a prefix which includes a regex
  // special character, regex-escape all the user-defined prefixes.
  std::transform(SpecialAcronyms.begin(), SpecialAcronyms.end(),
                 std::back_inserter(EscapedAcronyms),
                 [](const std::string &s) { return llvm::Regex::escape(s); });
  Finder->addMatcher(
      objcPropertyDecl(
          // the property name should be in Lower Camel Case like
          // 'lowerCamelCase'
          unless(matchesName(validPropertyNameRegex(EscapedAcronyms, true))))
          .bind("property"),
      this);
}

void PropertyDeclarationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<ObjCPropertyDecl>("property");
  assert(MatchedDecl->getName().size() > 0);
  auto *DeclContext = MatchedDecl->getDeclContext();
  auto *CategoryDecl = llvm::dyn_cast<ObjCCategoryDecl>(DeclContext);

  auto AcronymsRegex =
      llvm::Regex("^" + AcronymsGroupRegex(EscapedAcronyms) + "$");
  if (AcronymsRegex.match(MatchedDecl->getName())) {
    return;
  }
  if (CategoryDecl != nullptr &&
      hasCategoryPropertyPrefix(MatchedDecl->getName())) {
    if (!prefixedPropertyNameValid(MatchedDecl->getName(), EscapedAcronyms) ||
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
