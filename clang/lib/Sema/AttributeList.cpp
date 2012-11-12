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
#include "clang/AST/Expr.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;

size_t AttributeList::allocated_size() const {
  if (IsAvailability) return AttributeFactory::AvailabilityAllocSize;
  else if (IsTypeTagForDatatype)
    return AttributeFactory::TypeTagForDatatypeAllocSize;
  return (sizeof(AttributeList) + NumArgs * sizeof(Expr*));
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
  return Alloc.Allocate(size, llvm::AlignOf<AttributeFactory>::Alignment);
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

AttributeList *
AttributePool::createIntegerAttribute(ASTContext &C, IdentifierInfo *Name,
                                      SourceLocation TokLoc, int Arg) {
  Expr *IArg = IntegerLiteral::Create(C, llvm::APInt(32, (uint64_t) Arg),
                                      C.IntTy, TokLoc);
  return create(Name, TokLoc, 0, TokLoc, 0, TokLoc, &IArg, 1,
                AttributeList::AS_GNU);
}

#include "clang/Sema/AttrParsedAttrKinds.inc"

AttributeList::Kind AttributeList::getKind(const IdentifierInfo *Name,
                                           const IdentifierInfo *ScopeName,
                                           Syntax SyntaxUsed) {
  StringRef AttrName = Name->getName();

  // Normalize the attribute name, __foo__ becomes foo.
  if (AttrName.startswith("__") && AttrName.endswith("__") &&
      AttrName.size() >= 4)
    AttrName = AttrName.substr(2, AttrName.size() - 4);

  SmallString<64> Buf;
  if (ScopeName)
    Buf += ScopeName->getName();
  // Ensure that in the case of C++11 attributes, we look for '::foo' if it is
  // unscoped.
  if (ScopeName || SyntaxUsed == AS_CXX11)
    Buf += "::";
  Buf += AttrName;

  return ::getAttrKind(Buf);
}
