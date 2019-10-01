#include "clang/Basic/Attributes.h"
#include "clang/Basic/AttrSubjectMatchRules.h"
#include "clang/Basic/AttributeCommonInfo.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringSwitch.h"
using namespace clang;

int clang::hasAttribute(AttrSyntax Syntax, const IdentifierInfo *Scope,
                        const IdentifierInfo *Attr, const TargetInfo &Target,
                        const LangOptions &LangOpts) {
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
normalizeAttrScopeName(StringRef ScopeName,
                       AttributeCommonInfo::Syntax SyntaxUsed) {
  // Normalize the "__gnu__" scope name to be "gnu" and the "_Clang" scope name
  // to be "clang".
  if (SyntaxUsed == AttributeCommonInfo::AS_CXX11 ||
      SyntaxUsed == AttributeCommonInfo::AS_C2x) {
    if (ScopeName == "__gnu__")
      ScopeName = "gnu";
    else if (ScopeName == "_Clang")
      ScopeName = "clang";
  }
  return ScopeName;
}

static StringRef normalizeAttrName(StringRef AttrName,
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
  if (ShouldNormalize && AttrName.size() >= 4 && AttrName.startswith("__") &&
      AttrName.endswith("__"))
    AttrName = AttrName.slice(2, AttrName.size() - 2);

  return AttrName;
}

bool AttributeCommonInfo::isGNUScope() const {
  return ScopeName && (ScopeName->isStr("gnu") || ScopeName->isStr("__gnu__"));
}

#include "clang/Sema/AttrParsedAttrKinds.inc"

AttributeCommonInfo::Kind
AttributeCommonInfo::getParsedKind(const IdentifierInfo *Name,
                                   const IdentifierInfo *ScopeName,
                                   Syntax SyntaxUsed) {
  StringRef AttrName = Name->getName();

  SmallString<64> FullName;
  if (ScopeName)
    FullName += normalizeAttrScopeName(ScopeName->getName(), SyntaxUsed);

  AttrName = normalizeAttrName(AttrName, FullName, SyntaxUsed);

  // Ensure that in the case of C++11 attributes, we look for '::foo' if it is
  // unscoped.
  if (ScopeName || SyntaxUsed == AS_CXX11 || SyntaxUsed == AS_C2x)
    FullName += "::";
  FullName += AttrName;

  return ::getAttrKind(FullName, SyntaxUsed);
}

unsigned AttributeCommonInfo::calculateAttributeSpellingListIndex() const {
  // Both variables will be used in tablegen generated
  // attribute spell list index matching code.
  auto Syntax = static_cast<AttributeCommonInfo::Syntax>(getSyntax());
  StringRef Scope =
      getScopeName() ? normalizeAttrScopeName(getScopeName()->getName(), Syntax)
                     : "";
  StringRef Name = normalizeAttrName(getAttrName()->getName(), Scope, Syntax);

#include "clang/Sema/AttrSpellingListIndex.inc"
}
