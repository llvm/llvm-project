//===--- Attributes.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AttributeCommonInfo interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Attributes.h"
#include "clang/Basic/AttrSubjectMatchRules.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/ParsedAttrInfo.h"
#include "clang/Basic/SimpleTypoCorrection.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/ADT/StringSwitch.h"

using namespace clang;

static StringRef canonicalizeScopeName(StringRef Name) {
  // Normalize the scope name, but only for gnu and clang attributes.
  if (Name == "__gnu__")
    return "gnu";

  if (Name == "_Clang")
    return "clang";

  return Name;
}

static StringRef canonicalizeAttrName(StringRef Name) {
  // Normalize the attribute name, __foo__ becomes foo.
  if (Name.size() >= 4 && Name.starts_with("__") && Name.ends_with("__"))
    return Name.substr(2, Name.size() - 4);

  return Name;
}

static int hasAttributeImpl(AttributeCommonInfo::Syntax Syntax, StringRef Name,
                            StringRef ScopeName, const TargetInfo &Target,
                            const LangOptions &LangOpts) {
#include "clang/Basic/AttrHasAttributeImpl.inc"
  return 0;
}

int clang::hasAttribute(AttributeCommonInfo::Syntax Syntax, StringRef ScopeName,
                        StringRef Name, const TargetInfo &Target,
                        const LangOptions &LangOpts, bool CheckPlugins) {
  ScopeName = canonicalizeScopeName(ScopeName);
  Name = canonicalizeAttrName(Name);

  // As a special case, look for the omp::sequence and omp::directive
  // attributes. We support those, but not through the typical attribute
  // machinery that goes through TableGen. We support this in all OpenMP modes
  // so long as double square brackets are enabled.
  //
  // Other OpenMP attributes (e.g. [[omp::assume]]) are handled via the
  // regular attribute parsing machinery.
  if (LangOpts.OpenMP && ScopeName == "omp" &&
      (Name == "directive" || Name == "sequence"))
    return 1;

  int res = hasAttributeImpl(Syntax, Name, ScopeName, Target, LangOpts);
  if (res)
    return res;

  if (CheckPlugins) {
    // Check if any plugin provides this attribute.
    for (auto &Ptr : getAttributePluginInstances())
      if (Ptr->hasSpelling(Syntax, Name))
        return 1;
  }

  return 0;
}

int clang::hasAttribute(AttributeCommonInfo::Syntax Syntax,
                        const IdentifierInfo *Scope, const IdentifierInfo *Attr,
                        const TargetInfo &Target, const LangOptions &LangOpts,
                        bool CheckPlugins) {
  return hasAttribute(Syntax, Scope ? Scope->getName() : "", Attr->getName(),
                      Target, LangOpts, CheckPlugins);
}

int clang::hasAttribute(AttributeCommonInfo::Syntax Syntax,
                        const IdentifierInfo *Scope, const IdentifierInfo *Attr,
                        const TargetInfo &Target, const LangOptions &LangOpts) {
  return hasAttribute(Syntax, Scope, Attr, Target, LangOpts,
                      /*CheckPlugins=*/true);
}

const char *attr::getSubjectMatchRuleSpelling(attr::SubjectMatchRule Rule) {
  switch (Rule) {
#define ATTR_MATCH_RULE(NAME, SPELLING, IsAbstract)                            \
  case attr::NAME:                                                             \
    return SPELLING;
#include "clang/Basic/AttrSubMatchRulesList.inc"
  }
  llvm_unreachable("Invalid subject match rule");
}

static StringRef
normalizeAttrScopeName(StringRef ScopeName,
                       AttributeCommonInfo::Syntax SyntaxUsed) {
  if (SyntaxUsed == AttributeCommonInfo::AS_CXX11 ||
      SyntaxUsed == AttributeCommonInfo::AS_C23)
    return canonicalizeScopeName(ScopeName);

  return ScopeName;
}

static StringRef
normalizeAttrScopeName(const IdentifierInfo *ScopeName,
                       AttributeCommonInfo::Syntax SyntaxUsed) {
  if (ScopeName)
    return normalizeAttrScopeName(ScopeName->getName(), SyntaxUsed);
  return "";
}

static StringRef normalizeAttrName(StringRef AttrName,
                                   StringRef NormalizedScopeName,
                                   AttributeCommonInfo::Syntax SyntaxUsed) {
  // Normalize the attribute name, __foo__ becomes foo. This is only allowable
  // for GNU attributes, and attributes using the double square bracket syntax.
  bool ShouldNormalize =
      SyntaxUsed == AttributeCommonInfo::AS_GNU ||
      ((SyntaxUsed == AttributeCommonInfo::AS_CXX11 ||
        SyntaxUsed == AttributeCommonInfo::AS_C23) &&
       (NormalizedScopeName.empty() || NormalizedScopeName == "gnu" ||
        NormalizedScopeName == "clang"));

  if (ShouldNormalize)
    return canonicalizeAttrName(AttrName);

  return AttrName;
}

StringRef AttributeCommonInfo::getNormalizedScopeName() const {
  return normalizeAttrScopeName(getScopeName(), getSyntax());
}

StringRef
AttributeCommonInfo::getNormalizedAttrName(StringRef ScopeName) const {
  return normalizeAttrName(getAttrName()->getName(), ScopeName, getSyntax());
}

bool AttributeCommonInfo::isGNUScope() const {
  return AttrScope.isValid() && (AttrScope.getName()->isStr("gnu") ||
                                 AttrScope.getName()->isStr("__gnu__"));
}

bool AttributeCommonInfo::isClangScope() const {
  return AttrScope.isValid() && (AttrScope.getName()->isStr("clang") ||
                                 AttrScope.getName()->isStr("_Clang"));
}

#include "clang/Sema/AttrParsedAttrKinds.inc"

static SmallString<64> normalizeName(StringRef AttrName, StringRef ScopeName,
                                     AttributeCommonInfo::Syntax SyntaxUsed) {
  std::string StrAttrName = SyntaxUsed == AttributeCommonInfo::AS_HLSLAnnotation
                                ? AttrName.lower()
                                : AttrName.str();
  SmallString<64> FullName = ScopeName;
  if (!ScopeName.empty()) {
    assert(SyntaxUsed == AttributeCommonInfo::AS_CXX11 ||
           SyntaxUsed == AttributeCommonInfo::AS_C23);
    FullName += "::";
  }
  FullName += StrAttrName;
  return FullName;
}

static SmallString<64> normalizeName(const IdentifierInfo *Name,
                                     const IdentifierInfo *Scope,
                                     AttributeCommonInfo::Syntax SyntaxUsed) {
  StringRef ScopeName = normalizeAttrScopeName(Scope, SyntaxUsed);
  StringRef AttrName =
      normalizeAttrName(Name->getName(), ScopeName, SyntaxUsed);
  return normalizeName(AttrName, ScopeName, SyntaxUsed);
}

AttributeCommonInfo::Kind
AttributeCommonInfo::getParsedKind(const IdentifierInfo *Name,
                                   const IdentifierInfo *ScopeName,
                                   Syntax SyntaxUsed) {
  return ::getAttrKind(normalizeName(Name, ScopeName, SyntaxUsed), SyntaxUsed);
}

AttributeCommonInfo::AttrArgsInfo
AttributeCommonInfo::getCXX11AttrArgsInfo(const IdentifierInfo *Name) {
  StringRef AttrName = normalizeAttrName(
      Name->getName(), /*NormalizedScopeName*/ "", Syntax::AS_CXX11);
#define CXX11_ATTR_ARGS_INFO
  return llvm::StringSwitch<AttributeCommonInfo::AttrArgsInfo>(AttrName)
#include "clang/Basic/CXX11AttributeInfo.inc"
      .Default(AttributeCommonInfo::AttrArgsInfo::None);
#undef CXX11_ATTR_ARGS_INFO
}

std::string AttributeCommonInfo::getNormalizedFullName() const {
  return static_cast<std::string>(
      normalizeName(getAttrName(), getScopeName(), getSyntax()));
}

std::string
AttributeCommonInfo::getNormalizedFullName(StringRef ScopeName,
                                           StringRef AttrName) const {
  return static_cast<std::string>(
      normalizeName(AttrName, ScopeName, getSyntax()));
}

SourceRange AttributeCommonInfo::getNormalizedRange() const {
  return hasScope() ? SourceRange(AttrScope.getNameLoc(), AttrRange.getEnd())
                    : AttrRange;
}

static AttributeCommonInfo::Scope
getScopeFromNormalizedScopeName(StringRef ScopeName) {
  return llvm::StringSwitch<AttributeCommonInfo::Scope>(ScopeName)
      .Case("", AttributeCommonInfo::Scope::NONE)
      .Case("clang", AttributeCommonInfo::Scope::CLANG)
      .Case("gnu", AttributeCommonInfo::Scope::GNU)
      .Case("gsl", AttributeCommonInfo::Scope::GSL)
      .Case("hlsl", AttributeCommonInfo::Scope::HLSL)
      .Case("vk", AttributeCommonInfo::Scope::VK)
      .Case("msvc", AttributeCommonInfo::Scope::MSVC)
      .Case("omp", AttributeCommonInfo::Scope::OMP)
      .Case("riscv", AttributeCommonInfo::Scope::RISCV);
}

unsigned AttributeCommonInfo::calculateAttributeSpellingListIndex() const {
  // Both variables will be used in tablegen generated
  // attribute spell list index matching code.
  auto Syntax = static_cast<AttributeCommonInfo::Syntax>(getSyntax());
  StringRef ScopeName = normalizeAttrScopeName(getScopeName(), Syntax);
  StringRef Name =
      normalizeAttrName(getAttrName()->getName(), ScopeName, Syntax);
  AttributeCommonInfo::Scope ComputedScope =
      getScopeFromNormalizedScopeName(ScopeName);

#include "clang/Sema/AttrSpellingListIndex.inc"
}

#define ATTR_NAME(NAME) NAME,
static constexpr const char *AttrSpellingList[] = {
#include "clang/Basic/AttributeSpellingList.inc"
};

#define ATTR_SCOPE_NAME(SCOPE_NAME) SCOPE_NAME,
static constexpr const char *AttrScopeSpellingList[] = {
#include "clang/Basic/AttributeSpellingList.inc"
};

std::optional<StringRef>
AttributeCommonInfo::tryGetCorrectedScopeName(StringRef ScopeName) const {
  if (ScopeName.size() > 0 &&
      !llvm::is_contained(AttrScopeSpellingList, ScopeName)) {
    SimpleTypoCorrection STC(ScopeName);
    for (const auto &Scope : AttrScopeSpellingList)
      STC.add(Scope);

    if (auto CorrectedScopeName = STC.getCorrection())
      return CorrectedScopeName;
  }
  return std::nullopt;
}

std::optional<StringRef> AttributeCommonInfo::tryGetCorrectedAttrName(
    StringRef ScopeName, StringRef AttrName, const TargetInfo &Target,
    const LangOptions &LangOpts) const {
  if (!llvm::is_contained(AttrSpellingList, AttrName)) {
    SimpleTypoCorrection STC(AttrName);
    for (const auto &Attr : AttrSpellingList)
      STC.add(Attr);

    if (auto CorrectedAttrName = STC.getCorrection()) {
      if (hasAttribute(getSyntax(), ScopeName, *CorrectedAttrName, Target,
                       LangOpts,
                       /*CheckPlugins=*/true))
        return CorrectedAttrName;
    }
  }
  return std::nullopt;
}
