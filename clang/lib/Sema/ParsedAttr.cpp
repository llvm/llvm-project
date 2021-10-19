//======- ParsedAttr.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ParsedAttr class implementation
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/ParsedAttr.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/AttrSubjectMatchRules.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ManagedStatic.h"
#include <cassert>
#include <cstddef>
#include <utility>

using namespace clang;

LLVM_INSTANTIATE_REGISTRY(ParsedAttrInfoRegistry)

IdentifierLoc *IdentifierLoc::create(ASTContext &Ctx, SourceLocation Loc,
                                     IdentifierInfo *Ident) {
  IdentifierLoc *Result = new (Ctx) IdentifierLoc;
  Result->Loc = Loc;
  Result->Ident = Ident;
  return Result;
}

size_t ParsedAttr::allocated_size() const {
  if (IsAvailability) return AttributeFactory::AvailabilityAllocSize;
  else if (IsTypeTagForDatatype)
    return AttributeFactory::TypeTagForDatatypeAllocSize;
  else if (IsProperty)
    return AttributeFactory::PropertyAllocSize;
  else if (HasParsedType)
    return totalSizeToAlloc<ArgsUnion, detail::AvailabilityData,
                            detail::TypeTagForDatatypeData, ParsedType,
                            detail::PropertyData>(0, 0, 0, 1, 0);
  return totalSizeToAlloc<ArgsUnion, detail::AvailabilityData,
                          detail::TypeTagForDatatypeData, ParsedType,
                          detail::PropertyData>(NumArgs, 0, 0, 0, 0);
}

AttributeFactory::AttributeFactory() {
  // Go ahead and configure all the inline capacity.  This is just a memset.
  FreeLists.resize(InlineFreeListsCapacity);
}
AttributeFactory::~AttributeFactory() = default;

static size_t getFreeListIndexForSize(size_t size) {
  assert(size >= sizeof(ParsedAttr));
  assert((size % sizeof(void*)) == 0);
  return ((size - sizeof(ParsedAttr)) / sizeof(void *));
}

void *AttributeFactory::allocate(size_t size) {
  // Check for a previously reclaimed attribute.
  size_t index = getFreeListIndexForSize(size);
  if (index < FreeLists.size() && !FreeLists[index].empty()) {
    ParsedAttr *attr = FreeLists[index].back();
    FreeLists[index].pop_back();
    return attr;
  }

  // Otherwise, allocate something new.
  return Alloc.Allocate(size, alignof(AttributeFactory));
}

void AttributeFactory::deallocate(ParsedAttr *Attr) {
  size_t size = Attr->allocated_size();
  size_t freeListIndex = getFreeListIndexForSize(size);

  // Expand FreeLists to the appropriate size, if required.
  if (freeListIndex >= FreeLists.size())
    FreeLists.resize(freeListIndex + 1);

#ifndef NDEBUG
  // In debug mode, zero out the attribute to help find memory overwriting.
  memset(Attr, 0, size);
#endif

  // Add 'Attr' to the appropriate free-list.
  FreeLists[freeListIndex].push_back(Attr);
}

void AttributeFactory::reclaimPool(AttributePool &cur) {
  for (ParsedAttr *AL : cur.Attrs)
    deallocate(AL);
}

void AttributePool::takePool(AttributePool &pool) {
  Attrs.insert(Attrs.end(), pool.Attrs.begin(), pool.Attrs.end());
  pool.Attrs.clear();
}

namespace {

#include "clang/Sema/AttrParsedAttrImpl.inc"

} // namespace

const ParsedAttrInfo &ParsedAttrInfo::get(const AttributeCommonInfo &A) {
  // If we have a ParsedAttrInfo for this ParsedAttr then return that.
  if ((size_t)A.getParsedKind() < llvm::array_lengthof(AttrInfoMap))
    return *AttrInfoMap[A.getParsedKind()];

  // If this is an ignored attribute then return an appropriate ParsedAttrInfo.
  static const ParsedAttrInfo IgnoredParsedAttrInfo(
      AttributeCommonInfo::IgnoredAttribute);
  if (A.getParsedKind() == AttributeCommonInfo::IgnoredAttribute)
    return IgnoredParsedAttrInfo;

  // Otherwise this may be an attribute defined by a plugin. First instantiate
  // all plugin attributes if we haven't already done so.
  static llvm::ManagedStatic<std::list<std::unique_ptr<ParsedAttrInfo>>>
      PluginAttrInstances;
  if (PluginAttrInstances->empty())
    for (auto It : ParsedAttrInfoRegistry::entries())
      PluginAttrInstances->emplace_back(It.instantiate());

  // Search for a ParsedAttrInfo whose name and syntax match.
  std::string FullName = A.getNormalizedFullName();
  AttributeCommonInfo::Syntax SyntaxUsed = A.getSyntax();
  if (SyntaxUsed == AttributeCommonInfo::AS_ContextSensitiveKeyword)
    SyntaxUsed = AttributeCommonInfo::AS_Keyword;

  for (auto &Ptr : *PluginAttrInstances)
    for (auto &S : Ptr->Spellings)
      if (S.Syntax == SyntaxUsed && S.NormalizedFullName == FullName)
        return *Ptr;

  // If we failed to find a match then return a default ParsedAttrInfo.
  static const ParsedAttrInfo DefaultParsedAttrInfo(
      AttributeCommonInfo::UnknownAttribute);
  return DefaultParsedAttrInfo;
}

ArrayRef<const ParsedAttrInfo *> ParsedAttrInfo::getAllBuiltin() {
  return llvm::makeArrayRef(AttrInfoMap);
}

unsigned ParsedAttr::getMinArgs() const { return getInfo().NumArgs; }

unsigned ParsedAttr::getMaxArgs() const {
  return getMinArgs() + getInfo().OptArgs;
}

bool ParsedAttr::hasCustomParsing() const {
  return getInfo().HasCustomParsing;
}

bool ParsedAttr::diagnoseAppertainsTo(Sema &S, const Decl *D) const {
  return getInfo().diagAppertainsToDecl(S, *this, D);
}

bool ParsedAttr::diagnoseAppertainsTo(Sema &S, const Stmt *St) const {
  return getInfo().diagAppertainsToStmt(S, *this, St);
}

bool ParsedAttr::diagnoseMutualExclusion(Sema &S, const Decl *D) const {
  return getInfo().diagMutualExclusion(S, *this, D);
}

bool ParsedAttr::appliesToDecl(const Decl *D,
                               attr::SubjectMatchRule MatchRule) const {
  return checkAttributeMatchRuleAppliesTo(D, MatchRule);
}

void ParsedAttr::getMatchRules(
    const LangOptions &LangOpts,
    SmallVectorImpl<std::pair<attr::SubjectMatchRule, bool>> &MatchRules)
    const {
  return getInfo().getPragmaAttributeMatchRules(MatchRules, LangOpts);
}

bool ParsedAttr::diagnoseLangOpts(Sema &S) const {
  if (getInfo().acceptsLangOpts(S.getLangOpts()))
    return true;
  S.Diag(getLoc(), diag::warn_attribute_ignored) << *this;
  return false;
}

bool ParsedAttr::isTargetSpecificAttr() const {
  return getInfo().IsTargetSpecific;
}

bool ParsedAttr::isTypeAttr() const { return getInfo().IsType; }

bool ParsedAttr::isStmtAttr() const { return getInfo().IsStmt; }

bool ParsedAttr::existsInTarget(const TargetInfo &Target) const {
  return getInfo().existsInTarget(Target);
}

bool ParsedAttr::isKnownToGCC() const { return getInfo().IsKnownToGCC; }

bool ParsedAttr::isSupportedByPragmaAttribute() const {
  return getInfo().IsSupportedByPragmaAttribute;
}

unsigned ParsedAttr::getSemanticSpelling() const {
  return getInfo().spellingIndexToSemanticSpelling(*this);
}

bool ParsedAttr::hasVariadicArg() const {
  // If the attribute has the maximum number of optional arguments, we will
  // claim that as being variadic. If we someday get an attribute that
  // legitimately bumps up against that maximum, we can use another bit to track
  // whether it's truly variadic or not.
  return getInfo().OptArgs == 15;
}

static unsigned getNumAttributeArgs(const ParsedAttr &AL) {
  // FIXME: Include the type in the argument list.
  return AL.getNumArgs() + AL.hasParsedType();
}

template <typename Compare>
static bool checkAttributeNumArgsImpl(Sema &S, const ParsedAttr &AL,
                                      unsigned Num, unsigned Diag,
                                      Compare Comp) {
  if (Comp(getNumAttributeArgs(AL), Num)) {
    S.Diag(AL.getLoc(), Diag) << AL << Num;
    return false;
  }
  return true;
}

bool ParsedAttr::checkExactlyNumArgs(Sema &S, unsigned Num) const {
  return checkAttributeNumArgsImpl(S, *this, Num,
                                   diag::err_attribute_wrong_number_arguments,
                                   std::not_equal_to<unsigned>());
}
bool ParsedAttr::checkAtLeastNumArgs(Sema &S, unsigned Num) const {
  return checkAttributeNumArgsImpl(S, *this, Num,
                                   diag::err_attribute_too_few_arguments,
                                   std::less<unsigned>());
}
bool ParsedAttr::checkAtMostNumArgs(Sema &S, unsigned Num) const {
  return checkAttributeNumArgsImpl(S, *this, Num,
                                   diag::err_attribute_too_many_arguments,
                                   std::greater<unsigned>());
}
