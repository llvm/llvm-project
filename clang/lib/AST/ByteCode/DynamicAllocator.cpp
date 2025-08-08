//==-------- DynamicAllocator.cpp - Dynamic allocations ----------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DynamicAllocator.h"
#include "InterpBlock.h"
#include "InterpState.h"

using namespace clang;
using namespace clang::interp;

// FIXME: There is a peculiar problem with the way we track pointers
// to blocks and the way we allocate dynamic memory.
//
// When we have code like this:
// while (true) {
//   char *buffer = new char[1024];
//   delete[] buffer;
// }
//
// We have a local variable 'buffer' pointing to the heap allocated memory.
// When deallocating the memory via delete[], that local variable still
// points to the memory, which means we will create a DeadBlock for it and move
// it over to that block, essentially duplicating the allocation. Moving
// the data is also slow.
//
// However, when we actually try to access the allocation after it has been
// freed, we need the block to still exist (alive or dead) so we can tell
// that it's a dynamic allocation.

DynamicAllocator::~DynamicAllocator() { cleanup(); }

void DynamicAllocator::cleanup() {
  // Invoke destructors of all the blocks and as a last restort,
  // reset all the pointers pointing to them to null pointees.
  // This should never show up in diagnostics, but it's necessary
  // for us to not cause use-after-free problems.
  for (auto &Iter : AllocationSites) {
    auto &AllocSite = Iter.second;
    for (auto &Alloc : AllocSite.Allocations) {
      Block *B = reinterpret_cast<Block *>(Alloc.Memory.get());
      B->invokeDtor();
      if (B->hasPointers()) {
        while (B->Pointers) {
          Pointer *Next = B->Pointers->asBlockPointer().Next;
          B->Pointers->PointeeStorage.BS.Pointee = nullptr;
          B->Pointers = Next;
        }
        B->Pointers = nullptr;
      }
    }
  }

  AllocationSites.clear();
}

Block *DynamicAllocator::allocate(const Expr *Source, PrimType T,
                                  size_t NumElements, unsigned EvalID,
                                  Form AllocForm) {
  // Create a new descriptor for an array of the specified size and
  // element type.
  const Descriptor *D = allocateDescriptor(
      Source, T, Descriptor::InlineDescMD, NumElements, /*IsConst=*/false,
      /*IsTemporary=*/false, /*IsMutable=*/false);

  return allocate(D, EvalID, AllocForm);
}

Block *DynamicAllocator::allocate(const Descriptor *ElementDesc,
                                  size_t NumElements, unsigned EvalID,
                                  Form AllocForm) {
  assert(ElementDesc->getMetadataSize() == 0);
  // Create a new descriptor for an array of the specified size and
  // element type.
  // FIXME: Pass proper element type.
  const Descriptor *D = allocateDescriptor(
      ElementDesc->asExpr(), nullptr, ElementDesc, Descriptor::InlineDescMD,
      NumElements,
      /*IsConst=*/false, /*IsTemporary=*/false, /*IsMutable=*/false);
  return allocate(D, EvalID, AllocForm);
}

Block *DynamicAllocator::allocate(const Descriptor *D, unsigned EvalID,
                                  Form AllocForm) {
  assert(D);
  assert(D->asExpr());

  auto Memory =
      std::make_unique<std::byte[]>(sizeof(Block) + D->getAllocSize());
  auto *B = new (Memory.get()) Block(EvalID, D, /*isStatic=*/false);
  B->invokeCtor();

  assert(D->getMetadataSize() == sizeof(InlineDescriptor));
  InlineDescriptor *ID = reinterpret_cast<InlineDescriptor *>(B->rawData());
  ID->Desc = D;
  ID->IsActive = true;
  ID->Offset = sizeof(InlineDescriptor);
  ID->IsBase = false;
  ID->IsFieldMutable = false;
  ID->IsConst = false;
  ID->IsInitialized = false;
  ID->IsVolatile = false;

  if (D->isCompositeArray())
    ID->LifeState = Lifetime::Started;
  else
    ID->LifeState =
        AllocForm == Form::Operator ? Lifetime::Ended : Lifetime::Started;

  B->IsDynamic = true;

  if (auto It = AllocationSites.find(D->asExpr()); It != AllocationSites.end())
    It->second.Allocations.emplace_back(std::move(Memory));
  else
    AllocationSites.insert(
        {D->asExpr(), AllocationSite(std::move(Memory), AllocForm)});
  return B;
}

bool DynamicAllocator::deallocate(const Expr *Source,
                                  const Block *BlockToDelete, InterpState &S) {
  auto It = AllocationSites.find(Source);
  if (It == AllocationSites.end())
    return false;

  auto &Site = It->second;
  assert(!Site.empty());

  // Find the Block to delete.
  auto AllocIt = llvm::find_if(Site.Allocations, [&](const Allocation &A) {
    const Block *B = reinterpret_cast<const Block *>(A.Memory.get());
    return BlockToDelete == B;
  });

  assert(AllocIt != Site.Allocations.end());

  Block *B = reinterpret_cast<Block *>(AllocIt->Memory.get());
  B->invokeDtor();

  S.deallocate(B);
  Site.Allocations.erase(AllocIt);

  if (Site.empty())
    AllocationSites.erase(It);

  return true;
}
