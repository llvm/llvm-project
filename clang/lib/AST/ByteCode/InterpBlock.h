//===-- InterpBlock.h - Allocated blocks for the interpreter -*- C++ ----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the classes describing allocated blocks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_BLOCK_H
#define LLVM_CLANG_AST_INTERP_BLOCK_H

#include "Descriptor.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace interp {
class Block;
class DeadBlock;
class InterpState;
class Pointer;
enum PrimType : uint8_t;

/// A memory block, either on the stack or in the heap.
///
/// The storage described by the block is immediately followed by
/// optional metadata, which is followed by the actual data.
///
/// Block*        rawData()                  data()
/// │               │                         │
/// │               │                         │
/// ▼               ▼                         ▼
/// ┌───────────────┬─────────────────────────┬─────────────────┐
/// │ Block         │ Metadata                │ Data            │
/// │ sizeof(Block) │ Desc->getMetadataSize() │ Desc->getSize() │
/// └───────────────┴─────────────────────────┴─────────────────┘
///
/// Desc->getAllocSize() describes the size after the Block, i.e.
/// the data size and the metadata size.
///
class Block final {
private:
  static constexpr uint8_t ExternFlag = 1 << 0;
  static constexpr uint8_t DeadFlag = 1 << 1;
  static constexpr uint8_t WeakFlag = 1 << 2;
  static constexpr uint8_t DummyFlag = 1 << 3;

public:
  /// Creates a new block.
  Block(unsigned EvalID, UnsignedOrNone DeclID, const Descriptor *Desc,
        bool IsStatic = false, bool IsExtern = false, bool IsWeak = false,
        bool IsDummy = false)
      : Desc(Desc), DeclID(DeclID), EvalID(EvalID), IsStatic(IsStatic) {
    assert(Desc);
    AccessFlags |= (ExternFlag * IsExtern);
    AccessFlags |= (WeakFlag * IsWeak);
    AccessFlags |= (DummyFlag * IsDummy);
  }

  Block(unsigned EvalID, const Descriptor *Desc, bool IsStatic = false,
        bool IsExtern = false, bool IsWeak = false, bool IsDummy = false)
      : Desc(Desc), EvalID(EvalID), IsStatic(IsStatic) {
    assert(Desc);
    AccessFlags |= (ExternFlag * IsExtern);
    AccessFlags |= (WeakFlag * IsWeak);
    AccessFlags |= (DummyFlag * IsDummy);
  }

  /// Returns the block's descriptor.
  const Descriptor *getDescriptor() const { return Desc; }
  /// Checks if the block has any live pointers.
  bool hasPointers() const { return Pointers; }
  /// Checks if the block is extern.
  bool isExtern() const { return AccessFlags & ExternFlag; }
  /// Checks if the block has static storage duration.
  bool isStatic() const { return IsStatic; }
  /// Checks if the block is temporary.
  bool isTemporary() const { return Desc->IsTemporary; }
  bool isWeak() const { return AccessFlags & WeakFlag; }
  bool isDynamic() const { return (DynAllocId != std::nullopt); }
  bool isDummy() const { return AccessFlags & DummyFlag; }
  bool isDead() const { return AccessFlags & DeadFlag; }
  /// Returns the size of the block.
  unsigned getSize() const { return Desc->getAllocSize(); }
  /// Returns the declaration ID.
  UnsignedOrNone getDeclID() const { return DeclID; }
  /// Returns whether the data of this block has been initialized via
  /// invoking the Ctor func.
  bool isInitialized() const { return IsInitialized; }
  /// The Evaluation ID this block was created in.
  unsigned getEvalID() const { return EvalID; }

  /// Returns a pointer to the stored data.
  /// You are allowed to read Desc->getSize() bytes from this address.
  std::byte *data() {
    // rawData might contain metadata as well.
    size_t DataOffset = Desc->getMetadataSize();
    return rawData() + DataOffset;
  }
  const std::byte *data() const {
    // rawData might contain metadata as well.
    size_t DataOffset = Desc->getMetadataSize();
    return rawData() + DataOffset;
  }

  /// Returns a pointer to the raw data, including metadata.
  /// You are allowed to read Desc->getAllocSize() bytes from this address.
  std::byte *rawData() {
    return reinterpret_cast<std::byte *>(this) + sizeof(Block);
  }
  const std::byte *rawData() const {
    return reinterpret_cast<const std::byte *>(this) + sizeof(Block);
  }

  template <typename T> T deref() const {
    return *reinterpret_cast<const T *>(data());
  }

  /// Invokes the constructor.
  void invokeCtor() {
    assert(!IsInitialized);
    std::memset(rawData(), 0, Desc->getAllocSize());
    if (Desc->CtorFn) {
      Desc->CtorFn(this, data(), Desc->IsConst, Desc->IsMutable,
                   Desc->IsVolatile,
                   /*isActive=*/true, /*InUnion=*/false, Desc);
    }
    IsInitialized = true;
  }

  /// Invokes the Destructor.
  void invokeDtor() {
    assert(IsInitialized);
    if (Desc->DtorFn)
      Desc->DtorFn(this, data(), Desc);
    IsInitialized = false;
  }

  void dump() const { dump(llvm::errs()); }
  void dump(llvm::raw_ostream &OS) const;

  bool isAccessible() const { return AccessFlags == 0; }

private:
  friend class Pointer;
  friend class DeadBlock;
  friend class InterpState;
  friend class DynamicAllocator;

  Block(unsigned EvalID, const Descriptor *Desc, bool IsExtern, bool IsStatic,
        bool IsWeak, bool IsDummy, bool IsDead)
      : Desc(Desc), EvalID(EvalID), IsStatic(IsStatic) {
    assert(Desc);
    AccessFlags |= (ExternFlag * IsExtern);
    AccessFlags |= (DeadFlag * IsDead);
    AccessFlags |= (WeakFlag * IsWeak);
    AccessFlags |= (DummyFlag * IsDummy);
  }

  /// To be called by DynamicAllocator.
  void setDynAllocId(unsigned ID) { DynAllocId = ID; }

  /// Deletes a dead block at the end of its lifetime.
  void cleanup();

  /// Pointer chain management.
  void addPointer(Pointer *P);
  void removePointer(Pointer *P);
  void replacePointer(Pointer *Old, Pointer *New);
#ifndef NDEBUG
  bool hasPointer(const Pointer *P) const;
#endif

  /// Pointer to the stack slot descriptor.
  const Descriptor *Desc;
  /// Start of the chain of pointers.
  Pointer *Pointers = nullptr;
  /// Unique identifier of the declaration.
  UnsignedOrNone DeclID = std::nullopt;
  const unsigned EvalID = ~0u;
  /// Flag indicating if the block has static storage duration.
  bool IsStatic = false;
  /// Flag indicating if the block contents have been initialized
  /// via invokeCtor.
  bool IsInitialized = false;
  /// Allocation ID for this dynamic allocation, if it is one.
  UnsignedOrNone DynAllocId = std::nullopt;
  /// AccessFlags containing IsExtern, IsDead, IsWeak, and IsDummy bits.
  uint8_t AccessFlags = 0;
};

/// Descriptor for a dead block.
///
/// Dead blocks are chained in a double-linked list to deallocate them
/// whenever pointers become dead.
class DeadBlock final {
public:
  /// Copies the block.
  DeadBlock(DeadBlock *&Root, Block *Blk);

  /// Returns a pointer to the stored data.
  std::byte *data() { return B.data(); }
  std::byte *rawData() { return B.rawData(); }

private:
  friend class Block;
  friend class InterpState;

  void free();

  /// Root pointer of the list.
  DeadBlock *&Root;
  /// Previous block in the list.
  DeadBlock *Prev;
  /// Next block in the list.
  DeadBlock *Next;

  /// Actual block storing data and tracking pointers.
  Block B;
};

} // namespace interp
} // namespace clang

#endif
