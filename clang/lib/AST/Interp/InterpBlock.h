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
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ComparisonCategories.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace interp {
class Block;
class DeadBlock;
class InterpState;
class Pointer;
enum PrimType : unsigned;

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
public:
  // Creates a new block.
  Block(const std::optional<unsigned> &DeclID, Descriptor *Desc,
        bool IsStatic = false, bool IsExtern = false)
      : DeclID(DeclID), IsStatic(IsStatic), IsExtern(IsExtern), Desc(Desc) {}

  Block(Descriptor *Desc, bool IsStatic = false, bool IsExtern = false)
      : DeclID((unsigned)-1), IsStatic(IsStatic), IsExtern(IsExtern),
        Desc(Desc) {}

  /// Returns the block's descriptor.
  Descriptor *getDescriptor() const { return Desc; }
  /// Checks if the block has any live pointers.
  bool hasPointers() const { return Pointers; }
  /// Checks if the block is extern.
  bool isExtern() const { return IsExtern; }
  /// Checks if the block has static storage duration.
  bool isStatic() const { return IsStatic; }
  /// Checks if the block is temporary.
  bool isTemporary() const { return Desc->IsTemporary; }
  /// Returns the size of the block.
  InterpSize getSize() const { return Desc->getAllocSize(); }
  /// Returns the declaration ID.
  std::optional<unsigned> getDeclID() const { return DeclID; }

  /// Returns a pointer to the stored data.
  /// You are allowed to read Desc->getSize() bytes from this address.
  char *data() {
    // rawData might contain metadata as well.
    size_t DataOffset = Desc->getMetadataSize();
    return rawData() + DataOffset;
  }
  const char *data() const {
    // rawData might contain metadata as well.
    size_t DataOffset = Desc->getMetadataSize();
    return rawData() + DataOffset;
  }

  /// Returns a pointer to the raw data, including metadata.
  /// You are allowed to read Desc->getAllocSize() bytes from this address.
  char *rawData() { return reinterpret_cast<char *>(this) + sizeof(Block); }
  const char *rawData() const {
    return reinterpret_cast<const char *>(this) + sizeof(Block);
  }

  /// Returns a view over the data.
  template <typename T>
  T &deref() { return *reinterpret_cast<T *>(data()); }

  /// Invokes the constructor.
  void invokeCtor() {
    std::memset(rawData(), 0, Desc->getAllocSize());
    if (Desc->CtorFn)
      Desc->CtorFn(this, data(), Desc->IsConst, Desc->IsMutable,
                   /*isActive=*/true, Desc);
  }

  // Invokes the Destructor.
  void invokeDtor() {
    if (Desc->DtorFn)
      Desc->DtorFn(this, data(), Desc);
  }

protected:
  friend class Pointer;
  friend class DeadBlock;
  friend class InterpState;

  Block(Descriptor *Desc, bool IsExtern, bool IsStatic, bool IsDead)
    : IsStatic(IsStatic), IsExtern(IsExtern), IsDead(true), Desc(Desc) {}

  // Deletes a dead block at the end of its lifetime.
  void cleanup();

  // Pointer chain management.
  void addPointer(Pointer *P);
  void removePointer(Pointer *P);
  void movePointer(Pointer *From, Pointer *To);

  /// Start of the chain of pointers.
  Pointer *Pointers = nullptr;
  /// Unique identifier of the declaration.
  std::optional<unsigned> DeclID;
  /// Flag indicating if the block has static storage duration.
  bool IsStatic = false;
  /// Flag indicating if the block is an extern.
  bool IsExtern = false;
  /// Flag indicating if the pointer is dead.
  bool IsDead = false;
  /// Pointer to the stack slot descriptor.
  Descriptor *Desc;
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
  char *data() { return B.data(); }

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
