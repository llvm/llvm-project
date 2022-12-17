//===--- Pointer.h - Types for the constexpr VM -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the classes responsible for pointer tracking.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_POINTER_H
#define LLVM_CLANG_AST_INTERP_POINTER_H

#include "Descriptor.h"
#include "InterpBlock.h"
#include "clang/AST/ComparisonCategories.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace interp {
class Block;
class DeadBlock;
class Pointer;
enum PrimType : unsigned;

/// A pointer to a memory block, live or dead.
///
/// This object can be allocated into interpreter stack frames. If pointing to
/// a live block, it is a link in the chain of pointers pointing to the block.
class Pointer {
private:
  static constexpr unsigned PastEndMark = (unsigned)-1;
  static constexpr unsigned RootPtrMark = (unsigned)-1;

public:
  Pointer() {}
  Pointer(Block *B);
  Pointer(const Pointer &P);
  Pointer(Pointer &&P);
  ~Pointer();

  void operator=(const Pointer &P);
  void operator=(Pointer &&P);

  /// Converts the pointer to an APValue.
  APValue toAPValue() const;

  /// Offsets a pointer inside an array.
  Pointer atIndex(unsigned Idx) const {
    if (Base == RootPtrMark)
      return Pointer(Pointee, RootPtrMark, getDeclDesc()->getSize());
    unsigned Off = Idx * elemSize();
    if (getFieldDesc()->ElemDesc)
      Off += sizeof(InlineDescriptor);
    else
      Off += sizeof(InitMap *);
    return Pointer(Pointee, Base, Base + Off);
  }

  /// Creates a pointer to a field.
  Pointer atField(unsigned Off) const {
    unsigned Field = Offset + Off;
    return Pointer(Pointee, Field, Field);
  }

  /// Restricts the scope of an array element pointer.
  Pointer narrow() const {
    // Null pointers cannot be narrowed.
    if (isZero() || isUnknownSizeArray())
      return *this;

    // Pointer to an array of base types - enter block.
    if (Base == RootPtrMark)
      return Pointer(Pointee, 0, Offset == 0 ? Offset : PastEndMark);

    // Pointer is one past end - magic offset marks that.
    if (isOnePastEnd())
      return Pointer(Pointee, Base, PastEndMark);

    // Primitive arrays are a bit special since they do not have inline
    // descriptors. If Offset != Base, then the pointer already points to
    // an element and there is nothing to do. Otherwise, the pointer is
    // adjusted to the first element of the array.
    if (inPrimitiveArray()) {
      if (Offset != Base)
        return *this;
      return Pointer(Pointee, Base, Offset + sizeof(InitMap *));
    }

    // Pointer is to a field or array element - enter it.
    if (Offset != Base)
      return Pointer(Pointee, Offset, Offset);

    // Enter the first element of an array.
    if (!getFieldDesc()->isArray())
      return *this;

    const unsigned NewBase = Base + sizeof(InlineDescriptor);
    return Pointer(Pointee, NewBase, NewBase);
  }

  /// Expands a pointer to the containing array, undoing narrowing.
  Pointer expand() const {
    if (isElementPastEnd()) {
      // Revert to an outer one-past-end pointer.
      unsigned Adjust;
      if (inPrimitiveArray())
        Adjust = sizeof(InitMap *);
      else
        Adjust = sizeof(InlineDescriptor);
      return Pointer(Pointee, Base, Base + getSize() + Adjust);
    }

    // Do not step out of array elements.
    if (Base != Offset)
      return *this;

    // If at base, point to an array of base types.
    if (Base == 0)
      return Pointer(Pointee, RootPtrMark, 0);

    // Step into the containing array, if inside one.
    unsigned Next = Base - getInlineDesc()->Offset;
    Descriptor *Desc = Next == 0 ? getDeclDesc() : getDescriptor(Next)->Desc;
    if (!Desc->IsArray)
      return *this;
    return Pointer(Pointee, Next, Offset);
  }

  /// Checks if the pointer is null.
  bool isZero() const { return Pointee == nullptr; }
  /// Checks if the pointer is live.
  bool isLive() const { return Pointee && !Pointee->IsDead; }
  /// Checks if the item is a field in an object.
  bool isField() const { return Base != 0 && Base != RootPtrMark; }

  /// Accessor for information about the declaration site.
  Descriptor *getDeclDesc() const { return Pointee->Desc; }
  SourceLocation getDeclLoc() const { return getDeclDesc()->getLocation(); }

  /// Returns a pointer to the object of which this pointer is a field.
  Pointer getBase() const {
    if (Base == RootPtrMark) {
      assert(Offset == PastEndMark && "cannot get base of a block");
      return Pointer(Pointee, Base, 0);
    }
    assert(Offset == Base && "not an inner field");
    unsigned NewBase = Base - getInlineDesc()->Offset;
    return Pointer(Pointee, NewBase, NewBase);
  }
  /// Returns the parent array.
  Pointer getArray() const {
    if (Base == RootPtrMark) {
      assert(Offset != 0 && Offset != PastEndMark && "not an array element");
      return Pointer(Pointee, Base, 0);
    }
    assert(Offset != Base && "not an array element");
    return Pointer(Pointee, Base, Base);
  }

  /// Accessors for information about the innermost field.
  Descriptor *getFieldDesc() const {
    if (Base == 0 || Base == RootPtrMark)
      return getDeclDesc();
    return getInlineDesc()->Desc;
  }

  /// Returns the type of the innermost field.
  QualType getType() const { return getFieldDesc()->getType(); }

  /// Returns the element size of the innermost field.
  size_t elemSize() const {
    if (Base == RootPtrMark)
      return getDeclDesc()->getSize();
    return getFieldDesc()->getElemSize();
  }
  /// Returns the total size of the innermost field.
  size_t getSize() const { return getFieldDesc()->getSize(); }

  /// Returns the offset into an array.
  unsigned getOffset() const {
    assert(Offset != PastEndMark && "invalid offset");
    if (Base == RootPtrMark)
      return Offset;

    unsigned Adjust = 0;
    if (Offset != Base) {
      if (getFieldDesc()->ElemDesc)
        Adjust = sizeof(InlineDescriptor);
      else
        Adjust = sizeof(InitMap *);
    }
    return Offset - Base - Adjust;
  }

  /// Checks if the innermost field is an array.
  bool inArray() const { return getFieldDesc()->IsArray; }
  /// Checks if the structure is a primitive array.
  bool inPrimitiveArray() const { return getFieldDesc()->isPrimitiveArray(); }
  /// Checks if the structure is an array of unknown size.
  bool isUnknownSizeArray() const {
    return getFieldDesc()->isUnknownSizeArray();
  }
  /// Checks if the pointer points to an array.
  bool isArrayElement() const { return Base != Offset; }
  /// Pointer points directly to a block.
  bool isRoot() const {
    return (Base == 0 || Base == RootPtrMark) && Offset == 0;
  }

  /// Returns the record descriptor of a class.
  Record *getRecord() const { return getFieldDesc()->ElemRecord; }
  /// Returns the field information.
  const FieldDecl *getField() const { return getFieldDesc()->asFieldDecl(); }

  /// Checks if the object is a union.
  bool isUnion() const;

  /// Checks if the storage is extern.
  bool isExtern() const { return Pointee->isExtern(); }
  /// Checks if the storage is static.
  bool isStatic() const { return Pointee->isStatic(); }
  /// Checks if the storage is temporary.
  bool isTemporary() const { return Pointee->isTemporary(); }
  /// Checks if the storage is a static temporary.
  bool isStaticTemporary() const { return isStatic() && isTemporary(); }

  /// Checks if the field is mutable.
  bool isMutable() const { return Base != 0 && getInlineDesc()->IsMutable; }
  /// Checks if an object was initialized.
  bool isInitialized() const;
  /// Checks if the object is active.
  bool isActive() const { return Base == 0 || getInlineDesc()->IsActive; }
  /// Checks if a structure is a base class.
  bool isBaseClass() const { return isField() && getInlineDesc()->IsBase; }

  /// Checks if an object or a subfield is mutable.
  bool isConst() const {
    return Base == 0 ? getDeclDesc()->IsConst : getInlineDesc()->IsConst;
  }

  /// Returns the declaration ID.
  std::optional<unsigned> getDeclID() const { return Pointee->getDeclID(); }

  /// Returns the byte offset from the start.
  unsigned getByteOffset() const {
    return Offset;
  }

  /// Returns the number of elements.
  unsigned getNumElems() const { return getSize() / elemSize(); }

  /// Returns the index into an array.
  int64_t getIndex() const {
    if (isElementPastEnd())
      return 1;
    if (auto ElemSize = elemSize())
      return getOffset() / ElemSize;
    return 0;
  }

  /// Checks if the index is one past end.
  bool isOnePastEnd() const {
    return isElementPastEnd() || getSize() == getOffset();
  }

  /// Checks if the pointer is an out-of-bounds element pointer.
  bool isElementPastEnd() const { return Offset == PastEndMark; }

  /// Dereferences the pointer, if it's live.
  template <typename T> T &deref() const {
    assert(isLive() && "Invalid pointer");
    return *reinterpret_cast<T *>(Pointee->data() + Offset);
  }

  /// Dereferences a primitive element.
  template <typename T> T &elem(unsigned I) const {
    return reinterpret_cast<T *>(Pointee->data())[I];
  }

  /// Initializes a field.
  void initialize() const;
  /// Activats a field.
  void activate() const;
  /// Deactivates an entire strurcutre.
  void deactivate() const;

  /// Checks if two pointers are comparable.
  static bool hasSameBase(const Pointer &A, const Pointer &B);
  /// Checks if two pointers can be subtracted.
  static bool hasSameArray(const Pointer &A, const Pointer &B);

  /// Prints the pointer.
  void print(llvm::raw_ostream &OS) const {
    OS << "{" << Base << ", " << Offset << ", ";
    if (Pointee)
      OS << Pointee->getSize();
    else
      OS << "nullptr";
    OS << "}";
  }

private:
  friend class Block;
  friend class DeadBlock;

  Pointer(Block *Pointee, unsigned Base, unsigned Offset);

  /// Returns the embedded descriptor preceding a field.
  InlineDescriptor *getInlineDesc() const { return getDescriptor(Base); }

  /// Returns a descriptor at a given offset.
  InlineDescriptor *getDescriptor(unsigned Offset) const {
    assert(Offset != 0 && "Not a nested pointer");
    return reinterpret_cast<InlineDescriptor *>(Pointee->data() + Offset) - 1;
  }

  /// Returns a reference to the pointer which stores the initialization map.
  InitMap *&getInitMap() const {
    return *reinterpret_cast<InitMap **>(Pointee->data() + Base);
  }

  /// The block the pointer is pointing to.
  Block *Pointee = nullptr;
  /// Start of the current subfield.
  unsigned Base = 0;
  /// Offset into the block.
  unsigned Offset = 0;

  /// Previous link in the pointer chain.
  Pointer *Prev = nullptr;
  /// Next link in the pointer chain.
  Pointer *Next = nullptr;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Pointer &P) {
  P.print(OS);
  return OS;
}

} // namespace interp
} // namespace clang

#endif
