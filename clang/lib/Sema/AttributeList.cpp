//===--- AttributeList.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AttributeList class implementation
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/AttributeList.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/AttrSubjectMatchRules.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;

IdentifierLoc *IdentifierLoc::create(ASTContext &Ctx, SourceLocation Loc,
                                     IdentifierInfo *Ident) {
  IdentifierLoc *Result = new (Ctx) IdentifierLoc;
  Result->Loc = Loc;
  Result->Ident = Ident;
  return Result;
}

size_t AttributeList::allocated_size() const {
  if (IsAvailability) return AttributeFactory::AvailabilityAllocSize;
  else if (IsTypeTagForDatatype)
    return AttributeFactory::TypeTagForDatatypeAllocSize;
  else if (IsProperty)
    return AttributeFactory::PropertyAllocSize;
  return (sizeof(AttributeList) + NumArgs * sizeof(ArgsUnion));
}

AttributeFactory::AttributeFactory() {
  // Go ahead and configure all the inline capacity.  This is just a memset.
  FreeLists.resize(InlineFreeListsCapacity);
}
AttributeFactory::~AttributeFactory() {}

static size_t getFreeListIndexForSize(size_t size) {
  assert(size >= sizeof(AttributeList));
  assert((size % sizeof(void*)) == 0);
  return ((size - sizeof(AttributeList)) / sizeof(void*));
}

void *AttributeFactory::allocate(size_t size) {
  // Check for a previously reclaimed attribute.
  size_t index = getFreeListIndexForSize(size);
  if (index < FreeLists.size()) {
    if (AttributeList *attr = FreeLists[index]) {
      FreeLists[index] = attr->NextInPool;
      return attr;
    }
  }

  // Otherwise, allocate something new.
  return Alloc.Allocate(size, alignof(AttributeFactory));
}

void AttributeFactory::reclaimPool(AttributeList *cur) {
  assert(cur && "reclaiming empty pool!");
  do {
    // Read this here, because we're going to overwrite NextInPool
    // when we toss 'cur' into the appropriate queue.
    AttributeList *next = cur->NextInPool;

    size_t size = cur->allocated_size();
    size_t freeListIndex = getFreeListIndexForSize(size);

    // Expand FreeLists to the appropriate size, if required.
    if (freeListIndex >= FreeLists.size())
      FreeLists.resize(freeListIndex+1);

    // Add 'cur' to the appropriate free-list.
    cur->NextInPool = FreeLists[freeListIndex];
    FreeLists[freeListIndex] = cur;
    
    cur = next;
  } while (cur);
}

void AttributePool::takePool(AttributeList *pool) {
  assert(pool);

  // Fast path:  this pool is empty.
  if (!Head) {
    Head = pool;
    return;
  }

  // Reverse the pool onto the current head.  This optimizes for the
  // pattern of pulling a lot of pools into a single pool.
  do {
    AttributeList *next = pool->NextInPool;
    pool->NextInPool = Head;
    Head = pool;
    pool = next;
  } while (pool);
}

#include "clang/Sema/AttrParsedAttrKinds.inc"

static StringRef normalizeAttrName(StringRef AttrName, StringRef ScopeName,
                                   AttributeList::Syntax SyntaxUsed) {
  // Normalize the attribute name, __foo__ becomes foo. This is only allowable
  // for GNU attributes.
  bool IsGNU = SyntaxUsed == AttributeList::AS_GNU ||
               ((SyntaxUsed == AttributeList::AS_CXX11 ||
                SyntaxUsed == AttributeList::AS_C2x) && ScopeName == "gnu");
  if (IsGNU && AttrName.size() >= 4 && AttrName.startswith("__") &&
      AttrName.endswith("__"))
    AttrName = AttrName.slice(2, AttrName.size() - 2);

  return AttrName;
}

AttributeList::Kind AttributeList::getKind(const IdentifierInfo *Name,
                                           const IdentifierInfo *ScopeName,
                                           Syntax SyntaxUsed) {
  StringRef AttrName = Name->getName();

  SmallString<64> FullName;
  if (ScopeName)
    FullName += ScopeName->getName();

  AttrName = normalizeAttrName(AttrName, FullName, SyntaxUsed);

  // Ensure that in the case of C++11 attributes, we look for '::foo' if it is
  // unscoped.
  if (ScopeName || SyntaxUsed == AS_CXX11 || SyntaxUsed == AS_C2x)
    FullName += "::";
  FullName += AttrName;

  return ::getAttrKind(FullName, SyntaxUsed);
}

unsigned AttributeList::getAttributeSpellingListIndex() const {
  // Both variables will be used in tablegen generated
  // attribute spell list index matching code.
  StringRef Scope = ScopeName ? ScopeName->getName() : "";
  StringRef Name = normalizeAttrName(AttrName->getName(), Scope,
                                     (AttributeList::Syntax)SyntaxUsed);

#include "clang/Sema/AttrSpellingListIndex.inc"

}

struct ParsedAttrInfo {
  unsigned NumArgs : 4;
  unsigned OptArgs : 4;
  unsigned HasCustomParsing : 1;
  unsigned IsTargetSpecific : 1;
  unsigned IsType : 1;
  unsigned IsStmt : 1;
  unsigned IsKnownToGCC : 1;
  unsigned IsSupportedByPragmaAttribute : 1;

  bool (*DiagAppertainsToDecl)(Sema &S, const AttributeList &Attr,
                               const Decl *);
  bool (*DiagLangOpts)(Sema &S, const AttributeList &Attr);
  bool (*ExistsInTarget)(const TargetInfo &Target);
  unsigned (*SpellingIndexToSemanticSpelling)(const AttributeList &Attr);
  void (*GetPragmaAttributeMatchRules)(
      llvm::SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &Rules,
      const LangOptions &LangOpts);
};

namespace {
  #include "clang/Sema/AttrParsedAttrImpl.inc"
}

static const ParsedAttrInfo &getInfo(const AttributeList &A) {
  return AttrInfoMap[A.getKind()];
}

unsigned AttributeList::getMinArgs() const {
  return getInfo(*this).NumArgs;
}

unsigned AttributeList::getMaxArgs() const {
  return getMinArgs() + getInfo(*this).OptArgs;
}

bool AttributeList::hasCustomParsing() const {
  return getInfo(*this).HasCustomParsing;
}

bool AttributeList::diagnoseAppertainsTo(Sema &S, const Decl *D) const {
  return getInfo(*this).DiagAppertainsToDecl(S, *this, D);
}

bool AttributeList::appliesToDecl(const Decl *D,
                                  attr::SubjectMatchRule MatchRule) const {
  return checkAttributeMatchRuleAppliesTo(D, MatchRule);
}

void AttributeList::getMatchRules(
    const LangOptions &LangOpts,
    SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules)
    const {
  return getInfo(*this).GetPragmaAttributeMatchRules(MatchRules, LangOpts);
}

bool AttributeList::diagnoseLangOpts(Sema &S) const {
  return getInfo(*this).DiagLangOpts(S, *this);
}

bool AttributeList::isTargetSpecificAttr() const {
  return getInfo(*this).IsTargetSpecific;
}

bool AttributeList::isTypeAttr() const {
  return getInfo(*this).IsType;
}

bool AttributeList::isStmtAttr() const {
  return getInfo(*this).IsStmt;
}

bool AttributeList::existsInTarget(const TargetInfo &Target) const {
  return getInfo(*this).ExistsInTarget(Target);
}

bool AttributeList::isKnownToGCC() const {
  return getInfo(*this).IsKnownToGCC;
}

bool AttributeList::isSupportedByPragmaAttribute() const {
  return getInfo(*this).IsSupportedByPragmaAttribute;
}

unsigned AttributeList::getSemanticSpelling() const {
  return getInfo(*this).SpellingIndexToSemanticSpelling(*this);
}

bool AttributeList::hasVariadicArg() const {
  // If the attribute has the maximum number of optional arguments, we will
  // claim that as being variadic. If we someday get an attribute that
  // legitimately bumps up against that maximum, we can use another bit to track
  // whether it's truly variadic or not.
  return getInfo(*this).OptArgs == 15;
}
