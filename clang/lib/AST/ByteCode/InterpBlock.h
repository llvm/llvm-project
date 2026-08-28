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
/// Block*        rawData()           data()
/// │               │                  │
/// │               │                  │
/// ▼               ▼                  ▼
/// ┌───────────────┬──────────────────┬─────────────────┐
/// │ Block         │ Metadata         │ Data            │
/// │ sizeof(Block) │ MDSize           │ Desc->getSize() │
/// └───────────────┴──────────────────┴─────────────────┘
///
/// getSize() returns MDSize + Desc->getAllocSize().
///
class Block final {
private:
  static constexpr uint8_t ExternFlag = 1 << 0;
  static constexpr uint8_t DeadFlag = 1 << 1;
  static constexpr uint8_t WeakFlag = 1 << 2;
  static constexpr uint8_t DummyFlag = 1 << 3;

public:
  static constexpr uint8_t InlineDescMD = sizeof(InlineDescriptor);
  static constexpr uint8_t GlobalMD = sizeof(GlobalInlineDescriptor);

  /// Creates a new block.
  Block(unsigned EvalID, UnsignedOrNone DeclID, const Descriptor *Desc,
        unsigned MDSize = 0, bool IsStatic = false, bool IsExtern = false,
        bool IsWeak = false, bool IsDummy = false)
      : Desc(Desc), DeclID(DeclID), EvalID(EvalID), MDSize(MDSize),
        IsStatic(IsStatic) {
    assert(Desc);
    AccessFlags |= (ExternFlag * IsExtern);
    AccessFlags |= (WeakFlag * IsWeak);
    AccessFlags |= (DummyFlag * IsDummy);
  }

  Block(unsigned EvalID, const Descriptor *Desc, unsigned MDSize = 0,
        bool IsStatic = false, bool IsExtern = false, bool IsWeak = false,
        bool IsDummy = false)
      : Desc(Desc), EvalID(EvalID), MDSize(MDSize), IsStatic(IsStatic) {
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
  /// Returns the size of the block, including metadata.
  unsigned getSize() const { return Desc->getAllocSize() + MDSize; }
  /// Returns the size of the metadata.
  unsigned getMetadataSize() const { return MDSize; }
  /// Returns the declaration ID.
  UnsignedOrNone getDeclID() const { return DeclID; }
  /// Returns whether the data of this block has been initialized via
  /// invoking the Ctor func.
  bool isInitialized() const { return IsInitialized; }
  /// The Evaluation ID this block was created in.
  unsigned getEvalID() const { return EvalID; }
  /// Move all pointers from this block to \param B.
  void movePointersTo(Block *B);
  /// Make all pointers that currently point to this block point to nullptr.
  void removePointers();

  /// Returns a pointer to the stored data.
  /// You are allowed to read Desc->getSize() bytes from this address.
  std::byte *data() { return rawData() + MDSize; }
  const std::byte *data() const { return rawData() + MDSize; }

  /// Returns a pointer to the raw data, including metadata.
  /// You are allowed to read Desc->getAllocSize() bytes from this address.
  std::byte *rawData() {
    return reinterpret_cast<std::byte *>(this) + sizeof(Block);
  }
  const std::byte *rawData() const {
    return reinterpret_cast<const std::byte *>(this) + sizeof(Block);
  }

  template <typename T> const T &deref() const {
    return *reinterpret_cast<const T *>(data());
  }
  template <typename T> T &deref() { return *reinterpret_cast<T *>(data()); }

  template <typename T> T &getBlockDesc() {
    assert(sizeof(T) == MDSize);
    return *reinterpret_cast<T *>(rawData());
  }
  template <typename T> const T &getBlockDesc() const {
    return const_cast<Block *>(this)->getBlockDesc<T>();
  }

  /// Invokes the constructor.
  void invokeCtor() {
    assert(!IsInitialized);
    std::memset(rawData(), 0, getSize());
    invokeCtorNoMemset();
  }
  /// The same, but won't memset() the memory first to zero.
  void invokeCtorNoMemset() {
    assert(!IsInitialized);
    if (Desc->CtorFn)
      Desc->CtorFn(this, data(), Desc->IsConst, Desc->IsMutable,
                   Desc->IsVolatile,
                   /*isActive=*/true, /*InUnion=*/false, Desc);

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
  friend class Program;

  Block(unsigned EvalID, const Descriptor *Desc, unsigned MDSize, bool IsExtern,
        bool IsStatic, bool IsWeak, bool IsDummy, bool IsDead)
      : Desc(Desc), EvalID(EvalID), MDSize(MDSize), IsStatic(IsStatic) {
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
  /// Allocation ID for this dynamic allocation, if it is one.
  UnsignedOrNone DynAllocId = std::nullopt;
  /// AccessFlags containing IsExtern, IsDead, IsWeak, and IsDummy bits.
  uint8_t AccessFlags = 0;
  /// Size of the metadata.
  const uint8_t MDSize = 0;
  /// Flag indicating if the block has static storage duration.
  bool IsStatic = false;
  /// Flag indicating if the block contents have been initialized
  /// via invokeCtor.
  bool IsInitialized = false;
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
