//===-------- Block.h - Allocated blocks for the interpreter -*- C++ ----*-===//
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

struct BlockFlags {
  static constexpr uint8_t Extern = 1u << 0u;
  static constexpr uint8_t Dead = 1u << 1u;
  static constexpr uint8_t Weak = 1u << 2u;
  static constexpr uint8_t ConstexprUnknown = 1u << 3u;

  uint8_t V = 0;
  constexpr BlockFlags(uint8_t V = 0) : V(V) {}
  static constexpr BlockFlags empty() { return BlockFlags(); }

  constexpr BlockFlags withExtern(bool B = true) const {
    return BlockFlags(V | (B * Extern));
  }
  constexpr BlockFlags withDead(bool B = true) const {
    return BlockFlags(V | (B * Dead));
  }
  constexpr BlockFlags withWeak(bool B = true) const {
    return BlockFlags(V | (B * Weak));
  }
  constexpr BlockFlags withConstexprUnknown(bool B = true) const {
    return BlockFlags(V | (B * ConstexprUnknown));
  }
};

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
/// getSize() returns Desc->getAllocSize().
/// getAllocSize() returns getMetadataSize() + Desc->getAllocSize().
///
class Block final {
public:
  enum BlockKind : uint8_t { None, Local, Global };

  Block(unsigned EvalID, UnsignedOrNone DeclID, const Descriptor *Desc,
        BlockKind K, BlockFlags Flags = BlockFlags::empty())
      : Desc(Desc), DescSize(Desc->getAllocSize()), DeclID(DeclID),
        EvalID(EvalID), Flags(Flags), Kind(K), MDSize(getMetadataSize(K)) {}

  Block(unsigned EvalID, const Descriptor *Desc, BlockKind K,
        BlockFlags Flags = BlockFlags::empty())
      : Desc(Desc), DescSize(Desc->getAllocSize()), EvalID(EvalID),
        Flags(Flags), Kind(K), MDSize(getMetadataSize(K)) {}

  /// Number of bytes needed to allocate the block.
  static unsigned allocSize(const Descriptor *D, BlockKind K) {
    return sizeof(Block) + getMetadataSize(K) + D->getAllocSize();
  }

  static constexpr unsigned getMetadataSize(BlockKind K) {
    switch (K) {
    case None:
      return 0;
    case Global:
      return sizeof(GlobalInlineDescriptor);
    case Local:
      return sizeof(InlineDescriptor);
    }
    llvm_unreachable("Unhandled Block kind");
  }

  bool isGlobal() const { return Kind == Global; }
  bool isLocal() const { return Kind == Local; }

  /// Returns the block's descriptor.
  const Descriptor *getDescriptor() const { return Desc; }
  /// Checks if the block has any live pointers.
  bool hasPointers() const { return Pointers; }
  /// Checks if the block is extern.
  bool isExtern() const { return Flags.V & BlockFlags::Extern; }
  /// Checks if the block is temporary.
  bool isTemporary() const { return Desc->IsTemporary; }
  bool isWeak() const { return Flags.V & BlockFlags::Weak; }
  bool isDynamic() const {
    bool Result = (DynAllocId != std::nullopt);
    assert((Result == Desc->isDynAlloc()) &&
           "Inconsistent block/descriptor dynamic alloc state");
    return Result;
  }
  bool isDead() const { return Flags.V & BlockFlags::Dead; }
  bool isConstexprUnknown() const {
    return Flags.V & BlockFlags::ConstexprUnknown;
  }
  bool isAccessible() const { return Flags.V == 0; }
  /// Returns the size of the block, excluding metadata.
  unsigned getSize() const { return DescSize; }
  /// Returns the size of the block, including metadata.
  unsigned getAllocSize() const { return DescSize + MDSize; }
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
    assert(sizeof(T) == getMetadataSize(Kind));

    if constexpr (std::is_same_v<T, GlobalInlineDescriptor>)
      assert(Kind == Global);
    else if constexpr (std::is_same_v<T, InlineDescriptor>)
      assert(Kind == Local);
    else
      assert(Kind == None);

    return *reinterpret_cast<T *>(rawData());
  }
  template <typename T> const T &getBlockDesc() const {
    return const_cast<Block *>(this)->getBlockDesc<T>();
  }

  /// Invokes the constructor.
  void invokeCtor() {
    assert(!IsInitialized);
    std::memset(rawData(), 0, getAllocSize());
    invokeCtorNoMemset();
  }
  /// The same, but won't memset() the memory first to zero.
  void invokeCtorNoMemset() {
    assert(!IsInitialized);
    if (Desc->CtorFn)
      Desc->CtorFn(data(), Desc->IsConst, Desc->IsMutable, Desc->IsVolatile,
                   /*isActive=*/true, /*InUnion=*/false, Desc);

    IsInitialized = true;
  }

  /// Invokes the Destructor.
  void invokeDtor() {
    assert(IsInitialized);
    if (Desc->DtorFn)
      Desc->DtorFn(data(), Desc);
    IsInitialized = false;
  }

  void dump() const { dump(llvm::errs()); }
  void dump(llvm::raw_ostream &OS) const;

private:
  friend class Pointer;
  friend class DeadBlock;
  friend class InterpState;
  friend class DynamicAllocator;
  friend class Program;

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

  /// For DynamicAllocator.
  void markDead() { Flags = Flags.withDead(); }

  /// Pointer to the stack slot descriptor.
  const Descriptor *const Desc;
  /// Start of the chain of pointers.
  Pointer *Pointers = nullptr;
  /// Size of the data(), excluding block-level metadata.
  const unsigned DescSize;
  /// Unique identifier of the declaration.
  const UnsignedOrNone DeclID = std::nullopt;
  const unsigned EvalID = ~0u;
  /// Allocation ID for this dynamic allocation, if it is one.
  UnsignedOrNone DynAllocId = std::nullopt;
  BlockFlags Flags;
  /// Type of this block.
  const BlockKind Kind;
  /// Size of block-level metadata.
  const uint8_t MDSize;
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
