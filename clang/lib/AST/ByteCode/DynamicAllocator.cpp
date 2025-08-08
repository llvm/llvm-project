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

DynamicAllocator::~DynamicAllocator() { cleanup(); }

void DynamicAllocator::cleanup() {
  // Invoke destructors of all the blocks and as a last restort,
  // reset all the pointers pointing to them to null pointees.
  // This should never show up in diagnostics, but it's necessary
  // for us to not cause use-after-free problems.
  for (auto &Iter : AllocationSites) {
    auto &AllocSite = Iter.second;
    for (auto &Alloc : AllocSite.Allocations) {
      Block *B = Alloc.block();
      assert(!B->IsDead);
      assert(B->isInitialized());
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

  // Garbage collection. Remove all dead allocations that don't have pointers to
  // them anymore.
  llvm::erase_if(DeadAllocations, [](Allocation &Alloc) -> bool {
    return !Alloc.block()->hasPointers();
  });

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
    return BlockToDelete == A.block();
  });

  assert(AllocIt != Site.Allocations.end());

  Block *B = AllocIt->block();
  assert(B->isInitialized());
  assert(!B->IsDead);
  B->invokeDtor();

  // Almost all our dynamic allocations have a pointer pointing to them
  // when we deallocate them, since otherwise we can't call delete() at all.
  // This means that we would usually need to create DeadBlocks for all of them.
  // To work around that, we instead mark them as dead without moving the data
  // over to a DeadBlock and simply keep the block in a separate DeadAllocations
  // list.
  if (B->hasPointers()) {
    B->IsDead = true;
    DeadAllocations.push_back(std::move(*AllocIt));
    Site.Allocations.erase(AllocIt);

    if (Site.size() == 0)
      AllocationSites.erase(It);
    return true;
  }

  // Get rid of the allocation altogether.
  Site.Allocations.erase(AllocIt);
  if (Site.empty())
    AllocationSites.erase(It);

  return true;
}
