#include "clang/Basic/Attributes.h"
#include "clang/Basic/AttrSubjectMatchRules.h"
#include "clang/Basic/AttributeCommonInfo.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringSwitch.h"
using namespace clang;

int clang::hasAttribute(AttributeCommonInfo::Syntax Syntax,
                        const IdentifierInfo *Scope, const IdentifierInfo *Attr,
                        const TargetInfo &Target, const LangOptions &LangOpts) {
  StringRef Name = Attr->getName();
  // Normalize the attribute name, __foo__ becomes foo.
  if (Name.size() >= 4 && Name.startswith("__") && Name.endswith("__"))
    Name = Name.substr(2, Name.size() - 4);

  // Normalize the scope name, but only for gnu and clang attributes.
  StringRef ScopeName = Scope ? Scope->getName() : "";
  if (ScopeName == "__gnu__")
    ScopeName = "gnu";
  else if (ScopeName == "_Clang")
    ScopeName = "clang";

  // As a special case, look for the omp::sequence and omp::directive
  // attributes. We support those, but not through the typical attribute
  // machinery that goes through TableGen. We support this in all OpenMP modes
  // so long as double square brackets are enabled.
  if (LangOpts.OpenMP && LangOpts.DoubleSquareBracketAttributes &&
      ScopeName == "omp")
    return (Name == "directive" || Name == "sequence") ? 1 : 0;

#include "clang/Basic/AttrHasAttributeImpl.inc"

  return 0;
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
normalizeAttrScopeName(const IdentifierInfo *Scope,
                       AttributeCommonInfo::Syntax SyntaxUsed) {
  if (!Scope)
    return "";

  // Normalize the "__gnu__" scope name to be "gnu" and the "_Clang" scope name
  // to be "clang".
  StringRef ScopeName = Scope->getName();
  if (SyntaxUsed == AttributeCommonInfo::AS_CXX11 ||
      SyntaxUsed == AttributeCommonInfo::AS_C2x) {
    if (ScopeName == "__gnu__")
      ScopeName = "gnu";
    else if (ScopeName == "_Clang")
      ScopeName = "clang";
  }
  return ScopeName;
}

static StringRef normalizeAttrName(const IdentifierInfo *Name,
                                   StringRef NormalizedScopeName,
                                   AttributeCommonInfo::Syntax SyntaxUsed) {
  // Normalize the attribute name, __foo__ becomes foo. This is only allowable
  // for GNU attributes, and attributes using the double square bracket syntax.
  bool ShouldNormalize =
      SyntaxUsed == AttributeCommonInfo::AS_GNU ||
      ((SyntaxUsed == AttributeCommonInfo::AS_CXX11 ||
        SyntaxUsed == AttributeCommonInfo::AS_C2x) &&
       (NormalizedScopeName.empty() || NormalizedScopeName == "gnu" ||
        NormalizedScopeName == "clang"));
  StringRef AttrName = Name->getName();
  if (ShouldNormalize && AttrName.size() >= 4 && AttrName.startswith("__") &&
      AttrName.endswith("__"))
    AttrName = AttrName.slice(2, AttrName.size() - 2);

  return AttrName;
}

bool AttributeCommonInfo::isGNUScope() const {
  return ScopeName && (ScopeName->isStr("gnu") || ScopeName->isStr("__gnu__"));
}

bool AttributeCommonInfo::isClangScope() const {
  return ScopeName && (ScopeName->isStr("clang") || ScopeName->isStr("_Clang"));
}

#include "clang/Sema/AttrParsedAttrKinds.inc"

static SmallString<64> normalizeName(const IdentifierInfo *Name,
                                     const IdentifierInfo *Scope,
                                     AttributeCommonInfo::Syntax SyntaxUsed) {
  StringRef ScopeName = normalizeAttrScopeName(Scope, SyntaxUsed);
  StringRef AttrName = normalizeAttrName(Name, ScopeName, SyntaxUsed);

  SmallString<64> FullName = ScopeName;
  if (!ScopeName.empty()) {
    assert(SyntaxUsed == AttributeCommonInfo::AS_CXX11 ||
           SyntaxUsed == AttributeCommonInfo::AS_C2x);
    FullName += "::";
  }
  FullName += AttrName;

  return FullName;
}

AttributeCommonInfo::Kind
AttributeCommonInfo::getParsedKind(const IdentifierInfo *Name,
                                   const IdentifierInfo *ScopeName,
                                   Syntax SyntaxUsed) {
  return ::getAttrKind(normalizeName(Name, ScopeName, SyntaxUsed), SyntaxUsed);
}

std::string AttributeCommonInfo::getNormalizedFullName() const {
  return static_cast<std::string>(
      normalizeName(getAttrName(), getScopeName(), getSyntax()));
}

unsigned AttributeCommonInfo::calculateAttributeSpellingListIndex() const {
  // Both variables will be used in tablegen generated
  // attribute spell list index matching code.
  auto Syntax = static_cast<AttributeCommonInfo::Syntax>(getSyntax());
  StringRef Scope = normalizeAttrScopeName(getScopeName(), Syntax);
  StringRef Name = normalizeAttrName(getAttrName(), Scope, Syntax);

#include "clang/Sema/AttrSpellingListIndex.inc"
}
