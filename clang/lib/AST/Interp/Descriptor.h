//===--- Descriptor.h - Types for the constexpr VM --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines descriptors which characterise allocations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_DESCRIPTOR_H
#define LLVM_CLANG_AST_INTERP_DESCRIPTOR_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"

namespace clang {
namespace interp {
class Block;
class Record;
struct Descriptor;
enum PrimType : unsigned;

using DeclTy = llvm::PointerUnion<const Decl *, const Expr *>;

/// Invoked whenever a block is created. The constructor method fills in the
/// inline descriptors of all fields and array elements. It also initializes
/// all the fields which contain non-trivial types.
using BlockCtorFn = void (*)(Block *Storage, char *FieldPtr, bool IsConst,
                             bool IsMutable, bool IsActive,
                             Descriptor *FieldDesc);

/// Invoked when a block is destroyed. Invokes the destructors of all
/// non-trivial nested fields of arrays and records.
using BlockDtorFn = void (*)(Block *Storage, char *FieldPtr,
                             Descriptor *FieldDesc);

/// Invoked when a block with pointers referencing it goes out of scope. Such
/// blocks are persisted: the move function copies all inline descriptors and
/// non-trivial fields, as existing pointers might need to reference those
/// descriptors. Data is not copied since it cannot be legally read.
using BlockMoveFn = void (*)(Block *Storage, char *SrcFieldPtr,
                             char *DstFieldPtr, Descriptor *FieldDesc);

/// Object size as used by the interpreter.
using InterpSize = unsigned;

/// Describes a memory block created by an allocation site.
struct Descriptor {
private:
  /// Original declaration, used to emit the error message.
  const DeclTy Source;
  /// Size of an element, in host bytes.
  const InterpSize ElemSize;
  /// Size of the storage, in host bytes.
  const InterpSize Size;
  /// Size of the allocation (storage + metadata), in host bytes.
  const InterpSize AllocSize;

  /// Value to denote arrays of unknown size.
  static constexpr unsigned UnknownSizeMark = (unsigned)-1;

public:
  /// Token to denote structures of unknown size.
  struct UnknownSize {};

  /// Pointer to the record, if block contains records.
  Record *const ElemRecord = nullptr;
  /// Descriptor of the array element.
  Descriptor *const ElemDesc = nullptr;
  /// Flag indicating if the block is mutable.
  const bool IsConst = false;
  /// Flag indicating if a field is mutable.
  const bool IsMutable = false;
  /// Flag indicating if the block is a temporary.
  const bool IsTemporary = false;
  /// Flag indicating if the block is an array.
  const bool IsArray = false;

  /// Storage management methods.
  const BlockCtorFn CtorFn = nullptr;
  const BlockDtorFn DtorFn = nullptr;
  const BlockMoveFn MoveFn = nullptr;

  /// Allocates a descriptor for a primitive.
  Descriptor(const DeclTy &D, PrimType Type, bool IsConst, bool IsTemporary,
             bool IsMutable);

  /// Allocates a descriptor for an array of primitives.
  Descriptor(const DeclTy &D, PrimType Type, size_t NumElems, bool IsConst,
             bool IsTemporary, bool IsMutable);

  /// Allocates a descriptor for an array of primitives of unknown size.
  Descriptor(const DeclTy &D, PrimType Type, bool IsTemporary, UnknownSize);

  /// Allocates a descriptor for an array of composites.
  Descriptor(const DeclTy &D, Descriptor *Elem, unsigned NumElems, bool IsConst,
             bool IsTemporary, bool IsMutable);

  /// Allocates a descriptor for an array of composites of unknown size.
  Descriptor(const DeclTy &D, Descriptor *Elem, bool IsTemporary, UnknownSize);

  /// Allocates a descriptor for a record.
  Descriptor(const DeclTy &D, Record *R, bool IsConst, bool IsTemporary,
             bool IsMutable);

  QualType getType() const;
  SourceLocation getLocation() const;

  const Decl *asDecl() const { return Source.dyn_cast<const Decl *>(); }
  const Expr *asExpr() const { return Source.dyn_cast<const Expr *>(); }

  const ValueDecl *asValueDecl() const {
    return dyn_cast_or_null<ValueDecl>(asDecl());
  }

  const FieldDecl *asFieldDecl() const {
    return dyn_cast_or_null<FieldDecl>(asDecl());
  }

  const RecordDecl *asRecordDecl() const {
    return dyn_cast_or_null<RecordDecl>(asDecl());
  }

  /// Returns the size of the object without metadata.
  unsigned getSize() const {
    assert(!isUnknownSizeArray() && "Array of unknown size");
    return Size;
  }

  /// Returns the allocated size, including metadata.
  unsigned getAllocSize() const { return AllocSize; }
  /// returns the size of an element when the structure is viewed as an array.
  unsigned getElemSize()  const { return ElemSize; }

  /// Returns the number of elements stored in the block.
  unsigned getNumElems() const {
    return Size == UnknownSizeMark ? 0 : (getSize() / getElemSize());
  }

  /// Checks if the descriptor is of an array of primitives.
  bool isPrimitiveArray() const { return IsArray && !ElemDesc; }
  /// Checks if the descriptor is of an array of zero size.
  bool isZeroSizeArray() const { return Size == 0; }
  /// Checks if the descriptor is of an array of unknown size.
  bool isUnknownSizeArray() const { return Size == UnknownSizeMark; }

  /// Checks if the descriptor is of a primitive.
  bool isPrimitive() const { return !IsArray && !ElemRecord; }

  /// Checks if the descriptor is of an array.
  bool isArray() const { return IsArray; }
};

/// Inline descriptor embedded in structures and arrays.
///
/// Such descriptors precede all composite array elements and structure fields.
/// If the base of a pointer is not zero, the base points to the end of this
/// structure. The offset field is used to traverse the pointer chain up
/// to the root structure which allocated the object.
struct InlineDescriptor {
  /// Offset inside the structure/array.
  unsigned Offset;

  /// Flag indicating if the storage is constant or not.
  /// Relevant for primitive fields.
  unsigned IsConst : 1;
  /// For primitive fields, it indicates if the field was initialized.
  /// Primitive fields in static storage are always initialized.
  /// Arrays are always initialized, even though their elements might not be.
  /// Base classes are initialized after the constructor is invoked.
  unsigned IsInitialized : 1;
  /// Flag indicating if the field is an embedded base class.
  unsigned IsBase : 1;
  /// Flag indicating if the field is the active member of a union.
  unsigned IsActive : 1;
  /// Flag indicating if the field is mutable (if in a record).
  unsigned IsMutable : 1;

  Descriptor *Desc;
};

/// Bitfield tracking the initialisation status of elements of primitive arrays.
/// A pointer to this is embedded at the end of all primitive arrays.
/// If the map was not yet created and nothing was initialied, the pointer to
/// this structure is 0. If the object was fully initialized, the pointer is -1.
struct InitMap {
private:
  /// Type packing bits.
  using T = uint64_t;
  /// Bits stored in a single field.
  static constexpr uint64_t PER_FIELD = sizeof(T) * CHAR_BIT;

  /// Initializes the map with no fields set.
  InitMap(unsigned N);

  /// Returns a pointer to storage.
  T *data();

public:
  /// Initializes an element. Returns true when object if fully initialized.
  bool initialize(unsigned I);

  /// Checks if an element was initialized.
  bool isInitialized(unsigned I);

  /// Allocates a map holding N elements.
  static InitMap *allocate(unsigned N);

private:
  /// Number of fields initialized.
  unsigned UninitFields;
};

} // namespace interp
} // namespace clang

#endif
