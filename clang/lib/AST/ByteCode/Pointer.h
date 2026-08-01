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
#include "Function.h"
#include "InitMap.h"
#include "InterpBlock.h"
#include "clang/AST/ComparisonCategories.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace interp {
class Block;
class DeadBlock;
class Pointer;
class Context;

class Pointer;
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Pointer &P);

struct PtrView {
  static constexpr unsigned PastEndMark = ~0u;

  Block *Pointee;
  unsigned Base;
  uint64_t Offset;

  bool isZero() const { return !Pointee; }
  bool isLive() const { return Pointee && !Pointee->isDead(); }
  bool isDummy() const { return Pointee && Pointee->isDummy(); }
  bool isActive() const { return isRoot() || getInlineDesc()->IsActive; }
  bool isArrayRoot() const { return inArray() && Offset == Base; }
  bool isElementPastEnd() const { return Offset == PastEndMark; }
  bool isZeroSizeArray() const { return getFieldDesc()->isZeroSizeArray(); }
  bool isMutable() const {
    return !isRoot() && getInlineDesc()->IsFieldMutable;
  }
  bool inUnion() const { return getInlineDesc()->InUnion; };
  bool inArray() const { return getFieldDesc()->IsArray; }
  bool inPrimitiveArray() const { return getFieldDesc()->isPrimitiveArray(); }
  const Block *block() const { return Pointee; }

  unsigned getEvalID() { return Pointee->getEvalID(); }

  bool isRoot() const {
    return Base == Pointee->getDescriptor()->getMetadataSize();
  }

  bool isConst() const {
    return isRoot() ? getDeclDesc()->IsConst : getInlineDesc()->IsConst;
  }

  InlineDescriptor *getInlineDesc() const {
    assert(Base != sizeof(GlobalInlineDescriptor));
    assert(Base <= Pointee->getSize());
    assert(Base >= sizeof(InlineDescriptor));
    return getDescriptor(Base);
  }

  InlineDescriptor *getDescriptor(unsigned Offset) const {
    assert(Offset != 0 && "Not a nested pointer");
    return reinterpret_cast<InlineDescriptor *>(Pointee->rawData() + Offset) -
           1;
  }

  const Descriptor *getFieldDesc() const {
    if (isRoot())
      return Pointee->getDescriptor();
    return getInlineDesc()->Desc;
  }

  const Descriptor *getDeclDesc() const { return Pointee->getDescriptor(); }

  size_t elemSize() const { return getFieldDesc()->getElemSize(); }

  [[nodiscard]] PtrView narrow() const {
    // Null pointers cannot be narrowed.
    if (isZero() || isUnknownSizeArray())
      return *this;

    if (inArray()) {
      // Pointer is one past end - magic offset marks that.
      if (isOnePastEnd())
        return PtrView{Pointee, Base, PastEndMark};

      if (Offset != Base) {
        // If we're pointing to a primitive array element, there's nothing to
        // do.
        if (inPrimitiveArray())
          return *this;
        // Pointer is to a composite array element - enter it.
        return PtrView{Pointee, static_cast<unsigned>(Offset), Offset};
      }
    }
    // Otherwise, we're pointing to a non-array element or
    // are already narrowed to a composite array element. Nothing to do.
    return *this;
  }

  [[nodiscard]] PtrView expand() const {
    if (isElementPastEnd()) {
      // Revert to an outer one-past-end pointer.
      unsigned Adjust;
      if (inPrimitiveArray())
        Adjust = sizeof(InitMapPtr);
      else
        Adjust = sizeof(InlineDescriptor);
      return PtrView{Pointee, Base, Base + getSize() + Adjust};
    }

    // Do not step out of array elements.
    if (Base != Offset)
      return *this;

    if (isRoot())
      return PtrView{Pointee, Base, Base};

    // Step into the containing array, if inside one.
    unsigned Next = Base - getInlineDesc()->Offset;
    const Descriptor *Desc =
        (Next == Pointee->getDescriptor()->getMetadataSize())
            ? getDeclDesc()
            : getDescriptor(Next)->Desc;
    if (!Desc->IsArray)
      return *this;
    return PtrView{Pointee, Next, Offset};
  }

  [[nodiscard]] PtrView stripBaseCasts() const {
    PtrView V = *this;
    while (V.isBaseClass())
      V = V.getBase();
    return V;
  }

  [[nodiscard]] PtrView getArray() const {
    assert(Offset != Base && "not an array element");
    return PtrView{Pointee, Base, Base};
  }

  const Record *getRecord() const { return getFieldDesc()->ElemRecord; }
  const Record *getElemRecord() const {
    const Descriptor *ElemDesc = getFieldDesc()->ElemDesc;
    return ElemDesc ? ElemDesc->ElemRecord : nullptr;
  }
  const FieldDecl *getField() const { return getFieldDesc()->asFieldDecl(); }

  bool isField() const {
    return !isZero() && !isRoot() && getFieldDesc()->asDecl();
  }

  bool isBaseClass() const { return isField() && getInlineDesc()->IsBase; }
  bool isVirtualBaseClass() const {
    return isField() && getInlineDesc()->IsVirtualBase;
  }
  bool isUnknownSizeArray() const {
    return getFieldDesc()->isUnknownSizeArray();
  }

  bool isPastEnd() const { return Offset > Pointee->getSize(); }

  unsigned getOffset() const {
    assert(Offset != PastEndMark);

    unsigned Adjust = 0;
    if (Offset != Base) {
      if (getFieldDesc()->ElemDesc)
        Adjust = sizeof(InlineDescriptor);
      else
        Adjust = sizeof(InitMapPtr);
    }
    return Offset - Base - Adjust;
  }
  size_t getSize() const { return getFieldDesc()->getSize(); }

  bool isOnePastEnd() const {
    if (!Pointee)
      return false;

    if (isUnknownSizeArray())
      return false;
    return isPastEnd() || (getSize() == getOffset());
  }

  PtrView atIndex(unsigned Idx) const {
    unsigned Off = Idx * elemSize();
    if (getFieldDesc()->ElemDesc)
      Off += sizeof(InlineDescriptor);
    else
      Off += sizeof(InitMapPtr);
    return PtrView{Pointee, Base, Base + Off};
  }

  int64_t getIndex() const {
    if (isZero())
      return 0;
    // narrow()ed element in a composite array.
    if (Base > sizeof(InlineDescriptor) && Base == Offset)
      return 0;

    if (auto ElemSize = elemSize())
      return getOffset() / ElemSize;
    return 0;
  }

  unsigned getNumElems() const { return getSize() / elemSize(); }

  bool isArrayElement() const {
    if (inArray() && Base != Offset)
      return true;

    // Might be a narrow()'ed element in a composite array.
    // Check the inline descriptor.
    if (Base >= sizeof(InlineDescriptor) && getInlineDesc()->IsArrayElement)
      return true;

    return false;
  }

  template <typename T> T &deref() const {
    assert(isLive() && "Invalid pointer");
    assert(Pointee);

    if (isArrayRoot())
      return *reinterpret_cast<T *>(Pointee->rawData() + Base +
                                    sizeof(InitMapPtr));

    return *reinterpret_cast<T *>(Pointee->rawData() + Offset);
  }

  template <typename T> T &elem(unsigned I) const {
    assert(isLive() && "Invalid pointer");
    assert(Pointee);
    assert(getFieldDesc()->isPrimitiveArray());
    assert(I < getFieldDesc()->getNumElems());

    unsigned ElemByteOffset = I * getFieldDesc()->getElemSize();
    unsigned ReadOffset = Base + sizeof(InitMapPtr) + ElemByteOffset;
    assert(ReadOffset + sizeof(T) <= Pointee->getDescriptor()->getAllocSize());

    return *reinterpret_cast<T *>(Pointee->rawData() + ReadOffset);
  }

  [[nodiscard]] PtrView getBase() const {
    unsigned NewBase = Base - getInlineDesc()->Offset;
    return PtrView{Pointee, NewBase, NewBase};
  }

  [[nodiscard]] PtrView atField(unsigned Offset) const {
    unsigned F = this->Offset + Offset;
    return PtrView{Pointee, F, F};
  }

  QualType getType() const {
    if (isRoot() && Base == Offset) {
      // If this pointer points to the root of a declaration, try to consult
      // the ValueDecl directly, since that has a type with more information,
      // e.g. the correct ElaboratedTypeKeyword.
      if (const ValueDecl *VD = getDeclDesc()->asValueDecl())
        return VD->getType();
      return getDeclDesc()->getType();
    }
    if (inPrimitiveArray() && Offset != Base) {
      // Unfortunately, complex and vector types are not array types in clang,
      // but they are for us.
      if (const auto *AT = getFieldDesc()->getType()->getAsArrayTypeUnsafe())
        return AT->getElementType();
      if (const auto *CT = getFieldDesc()->getType()->getAs<ComplexType>())
        return CT->getElementType();
      if (const auto *CT = getFieldDesc()->getType()->getAs<VectorType>())
        return CT->getElementType();
    }

    return getFieldDesc()->getType();
  }

  bool isInitialized() const {
    if (isRoot() && Base == sizeof(GlobalInlineDescriptor) && Offset == Base) {
      const auto &GD = Pointee->getBlockDesc<GlobalInlineDescriptor>();
      return GD.InitState == GlobalInitState::Initialized;
    }

    assert(Pointee && "Cannot check if null pointer was initialized");
    const Descriptor *Desc = getFieldDesc();
    assert(Desc);
    if (Desc->isPrimitiveArray())
      return isElementInitialized(getIndex());

    if (Base == 0)
      return true;
    // Field has its bit in an inline descriptor.
    return getInlineDesc()->IsInitialized;
  }

  void initializeElement(unsigned Index) const;
  bool allElementsInitialized() const;
  bool isElementInitialized(unsigned Index) const;
  InitMapPtr &getInitMap() const {
    return *reinterpret_cast<InitMapPtr *>(Pointee->rawData() + Base);
  }
  void initialize() const;
  void activate() const;

  void setLifeState(Lifetime L) const;
  Lifetime getLifetime() const;
  void startLifetime() const { setLifeState(Lifetime::Started); }
  void endLifetime() const { setLifeState(Lifetime::Ended); }

  bool operator==(const PtrView &Other) const {
    return Other.Pointee == Pointee && Base == Other.Base &&
           Offset == Other.Offset;
  }

  bool operator!=(const PtrView &Other) const { return !(Other == *this); }
};

struct BlockPointer {
  /// The block the pointer is pointing to.
  Block *Pointee;
  /// Start of the current subfield.
  unsigned Base;
  /// Previous link in the pointer chain.
  Pointer *Prev;
  /// Next link in the pointer chain.
  Pointer *Next;
};

struct IntPointer {
  const Type *Ty;
  uint64_t Value;

  std::optional<IntPointer> atOffset(const Context &Ctx, unsigned Offset) const;
  IntPointer baseCast(const Context &Ctx, unsigned BaseOffset) const;

  QualType getPointeeType() const {
    if (!Ty)
      return QualType();

    QualType QT(Ty, 0);
    if (QT->isPointerOrReferenceType())
      QT = QT->getPointeeType();
    else if (QT->isArrayType())
      QT = QT->getAsArrayTypeUnsafe()->getElementType();

    return QT.IgnoreParens();
  }
};

struct FunctionPointer {
  const Function *Func;
};

struct TypeidPointer {
  const Type *TypePtr;
  const Type *TypeInfoType;
};

enum class Storage { Int, Block, Fn, Typeid };

/// A pointer to a memory block, live or dead.
///
/// This object can be allocated into interpreter stack frames. If pointing to
/// a live block, it is a link in the chain of pointers pointing to the block.
///
/// In the simplest form, a Pointer has a Block* (the pointee) and both Base
/// and Offset are 0, which means it will point to raw data.
///
/// The Base field is used to access metadata about the data. For primitive
/// arrays, the Base is followed by an InitMap. In a variety of cases, the
/// Base is preceded by an InlineDescriptor, which is used to track the
/// initialization state, among other things.
///
/// The Offset field is used to access the actual data. In other words, the
/// data the pointer decribes can be found at
/// Pointee->rawData() + Pointer.Offset.
///
/// \verbatim
/// Pointee                      Offset
/// │                              │
/// │                              │
/// ▼                              ▼
/// ┌───────┬────────────┬─────────┬────────────────────────────┐
/// │ Block │ InlineDesc │ InitMap │ Actual Data                │
/// └───────┴────────────┴─────────┴────────────────────────────┘
///                      ▲
///                      │
///                      │
///                     Base
/// \endverbatim
class Pointer {
public:
  Pointer() : StorageKind(Storage::Int), Int{nullptr, 0} {}
  Pointer(IntPointer &&IntPtr)
      : StorageKind(Storage::Int), Int(std::move(IntPtr)) {}
  Pointer(Block *B);
  Pointer(Block *B, uint64_t BaseAndOffset);
  Pointer(const Pointer &P);
  Pointer(Pointer &&P);
  Pointer(uint64_t Address, const Type *Ty, uint64_t Offset = 0)
      : Offset(Offset), StorageKind(Storage::Int), Int{Ty, Address} {}
  Pointer(const Function *F, uint64_t Offset = 0)
      : Offset(Offset), StorageKind(Storage::Fn), Fn{F} {}
  Pointer(const Type *TypePtr, const Type *TypeInfoType, uint64_t Offset = 0)
      : Offset(Offset), StorageKind(Storage::Typeid) {
    Typeid.TypePtr = TypePtr;
    Typeid.TypeInfoType = TypeInfoType;
  }

  Pointer(Block *Pointee, unsigned Base, uint64_t Offset);
  explicit Pointer(PtrView V) : Pointer(V.Pointee, V.Base, V.Offset) {}
  ~Pointer();

  Pointer &operator=(const Pointer &P);
  Pointer &operator=(Pointer &&P);

  /// Equality operators are just for tests.
  bool operator==(const Pointer &P) const {
    if (P.StorageKind != StorageKind)
      return false;
    if (isIntegralPointer())
      return P.Int.Value == Int.Value && P.Int.Ty == Int.Ty &&
             P.Offset == Offset;

    if (isFunctionPointer())
      return P.Fn.Func == Fn.Func && P.Offset == Offset;

    return P.view() == view();
  }

  bool operator!=(const Pointer &P) const { return !(P == *this); }

  /// Converts the pointer to an APValue.
  APValue toAPValue(const ASTContext &ASTCtx) const;

  /// Converts the pointer to a string usable in diagnostics.
  std::string toDiagnosticString(const ASTContext &Ctx) const;

  uint64_t getIntegerRepresentation() const {
    if (isIntegralPointer())
      return Int.Value + (Offset * elemSize());
    if (isFunctionPointer())
      return reinterpret_cast<uint64_t>(Fn.Func) + Offset;
    return reinterpret_cast<uint64_t>(BS.Pointee) + Offset;
  }

  PtrView view() const {
    assert(isBlockPointer());
    return PtrView{BS.Pointee, BS.Base, Offset};
  }

  /// Converts the pointer to an APValue that is an rvalue.
  std::optional<APValue> toRValue(const Context &Ctx,
                                  QualType ResultType) const;

  /// Offsets a pointer inside an array.
  [[nodiscard]] Pointer atIndex(uint64_t Idx) const {
    if (isIntegralPointer())
      return Pointer(Int.Value, Int.Ty, Idx);
    if (isFunctionPointer())
      return Pointer(Fn.Func, Idx);

    return Pointer(view().atIndex(Idx));
  }

  /// Creates a pointer to a field.
  [[nodiscard]] Pointer atField(unsigned Off) const {
    return Pointer(view().atField(Off));
  }

  /// Subtract the given offset from the current Base and Offset
  /// of the pointer.
  [[nodiscard]] Pointer atFieldSub(unsigned Off) const {
    assert(Offset >= Off);
    unsigned O = Offset - Off;
    return Pointer(BS.Pointee, O, O);
  }

  /// Restricts the scope of an array element pointer.
  [[nodiscard]] Pointer narrow() const {
    if (!isBlockPointer())
      return *this;
    return Pointer(view().narrow());
  }

  /// Expands a pointer to the containing array, undoing narrowing.
  [[nodiscard]] Pointer expand() const {
    if (!isBlockPointer())
      return *this;
    return Pointer(view().expand());
  }

  /// Checks if the pointer is null.
  bool isZero() const {
    switch (StorageKind) {
    case Storage::Int:
      return Int.Value == 0 && Offset == 0;
    case Storage::Block:
      return BS.Pointee == nullptr;
    case Storage::Fn:
      return !Fn.Func;
    case Storage::Typeid:
      return false;
    }
    llvm_unreachable("Unknown clang::interp::Storage enum");
  }
  /// Checks if the pointer is live.
  bool isLive() const {
    if (!isBlockPointer())
      return true;
    return view().isLive();
  }
  /// Checks if the item is a field in an object.
  bool isField() const {
    if (!isBlockPointer())
      return false;

    return view().isField();
  }

  /// Accessor for information about the declaration site.
  const Descriptor *getDeclDesc() const {
    if (!isBlockPointer())
      return nullptr;

    assert(isBlockPointer());
    assert(BS.Pointee);
    return BS.Pointee->Desc;
  }
  SourceLocation getDeclLoc() const { return getDeclDesc()->getLocation(); }

  /// Returns the expression or declaration the pointer has been created for.
  DeclTy getSource() const {
    if (isBlockPointer())
      return getDeclDesc()->getSource();
    if (isFunctionPointer()) {
      const Function *F = Fn.Func;
      return F ? F->getDecl() : DeclTy();
    }
    llvm_unreachable("Unsupported pointer type in getSource()");
    return DeclTy();
  }

  /// Returns a pointer to the object of which this pointer is a field.
  [[nodiscard]] Pointer getBase() const { return Pointer(view().getBase()); }
  /// Returns the parent array.
  [[nodiscard]] Pointer getArray() const { return Pointer(view().getArray()); }

  /// Accessors for information about the innermost field.
  const Descriptor *getFieldDesc() const {
    if (isIntegralPointer())
      return nullptr;

    if (isRoot())
      return getDeclDesc();
    return getInlineDesc()->Desc;
  }

  /// Returns the type of the innermost field.
  QualType getType() const {
    switch (StorageKind) {
    case Storage::Int:
      return Int.getPointeeType();
    case Storage::Block:
      return view().getType();
    case Storage::Fn:
      return Fn.Func->getDecl()->getType();
    case Storage::Typeid:
      return QualType(Typeid.TypeInfoType, 0);
    }
    llvm_unreachable("Unhandled StorageKind");
  }

  const VarDecl *getRootVarDecl() const;

  [[nodiscard]] Pointer getDeclPtr() const { return Pointer(BS.Pointee); }

  /// Returns the element size of the innermost field.
  size_t elemSize() const {
    if (isIntegralPointer()) {
      // FIXME: Remove this and handle int ptrs specially?
      return 1;
    }

    return view().elemSize();
  }
  /// Returns the total size of the innermost field.
  size_t getSize() const {
    assert(isBlockPointer());
    return getFieldDesc()->getSize();
  }

  /// Returns the offset into an array.
  unsigned getOffset() const {
    assert(Offset != PtrView::PastEndMark && "invalid offset");
    return view().getOffset();
  }

  /// Whether this array refers to an array, but not
  /// to the first element.
  bool isArrayRoot() const { return view().isArrayRoot(); }

  /// Checks if the innermost field is an array.
  bool inArray() const {
    if (isBlockPointer())
      return view().inArray();
    return false;
  }
  bool inUnion() const {
    if (isBlockPointer() && BS.Base >= sizeof(InlineDescriptor))
      return view().inUnion();
    return false;
  };

  /// Checks if the structure is a primitive array.
  bool inPrimitiveArray() const {
    if (isBlockPointer())
      return view().inPrimitiveArray();
    return false;
  }
  /// Checks if the structure is an array of unknown size.
  bool isUnknownSizeArray() const {
    if (!isBlockPointer())
      return false;
    return getFieldDesc()->isUnknownSizeArray();
  }
  /// Checks if the pointer points to an array.
  bool isArrayElement() const {
    if (!isBlockPointer())
      return false;

    return view().isArrayElement();
  }
  /// Pointer points directly to a block.
  bool isRoot() const {
    if (isZero() || !isBlockPointer())
      return true;
    return view().isRoot();
  }
  /// If this pointer has an InlineDescriptor we can use to initialize.
  bool canBeInitialized() const {
    if (!isBlockPointer())
      return false;

    return BS.Pointee && BS.Base > 0;
  }

  [[nodiscard]] const BlockPointer &asBlockPointer() const {
    assert(isBlockPointer());
    return BS;
  }
  [[nodiscard]] const IntPointer &asIntPointer() const {
    assert(isIntegralPointer());
    return Int;
  }
  [[nodiscard]] const FunctionPointer &asFunctionPointer() const {
    assert(isFunctionPointer());
    return Fn;
  }
  [[nodiscard]] const TypeidPointer &asTypeidPointer() const {
    assert(isTypeidPointer());
    return Typeid;
  }

  bool isBlockPointer() const { return StorageKind == Storage::Block; }
  bool isIntegralPointer() const { return StorageKind == Storage::Int; }
  bool isFunctionPointer() const { return StorageKind == Storage::Fn; }
  bool isTypeidPointer() const { return StorageKind == Storage::Typeid; }

  /// Returns the record descriptor of a class.
  const Record *getRecord() const {
    if (!isBlockPointer())
      return nullptr;
    return view().getRecord();
  }
  /// Returns the element record type, if this is a non-primive array.
  const Record *getElemRecord() const { return view().getElemRecord(); }
  /// Returns the field information.
  const FieldDecl *getField() const {
    if (const Descriptor *FD = getFieldDesc())
      return FD->asFieldDecl();
    return nullptr;
  }

  /// Checks if the storage is extern.
  bool isExtern() const {
    if (isBlockPointer())
      return BS.Pointee && BS.Pointee->isExtern();
    return false;
  }
  /// Checks if the storage is static.
  bool isStatic() const {
    if (!isBlockPointer())
      return true;
    assert(BS.Pointee);
    return BS.Pointee->isStatic();
  }
  /// Checks if the storage is temporary.
  bool isTemporary() const {
    if (isBlockPointer()) {
      assert(BS.Pointee);
      return BS.Pointee->isTemporary();
    }
    return false;
  }
  /// Checks if the storage has been dynamically allocated.
  bool isDynamic() const {
    if (isBlockPointer()) {
      assert(BS.Pointee);
      return BS.Pointee->isDynamic();
    }
    return false;
  }
  /// Checks if the storage is a static temporary.
  bool isStaticTemporary() const { return isStatic() && isTemporary(); }

  /// Checks if the field is mutable.
  bool isMutable() const {
    if (!isBlockPointer())
      return false;
    return view().isMutable();
  }

  bool isWeak() const {
    if (isFunctionPointer()) {
      if (!Fn.Func || !Fn.Func->getDecl())
        return false;

      return Fn.Func->getDecl()->isWeak();
    }
    if (!isBlockPointer())
      return false;

    assert(isBlockPointer());
    return BS.Pointee->isWeak();
  }
  /// Checks if the object is active.
  bool isActive() const {
    if (!isBlockPointer())
      return true;
    return view().isActive();
  }
  /// Checks if a structure is a base class.
  bool isBaseClass() const { return view().isBaseClass(); }
  bool isVirtualBaseClass() const { return view().isVirtualBaseClass(); }

  /// Checks if the pointer points to a dummy value.
  bool isDummy() const {
    if (!isBlockPointer())
      return false;
    return view().isDummy();
  }

  /// Checks if an object or a subfield is mutable.
  bool isConst() const {
    if (isIntegralPointer())
      return true;
    return view().isConst();
  }
  bool isConstInMutable() const {
    if (!isBlockPointer())
      return false;
    return isRoot() ? false : getInlineDesc()->IsConstInMutable;
  }

  /// Checks if an object or a subfield is volatile.
  bool isVolatile() const {
    if (!isBlockPointer())
      return false;
    return isRoot() ? getDeclDesc()->IsVolatile : getInlineDesc()->IsVolatile;
  }

  /// Returns the declaration ID.
  UnsignedOrNone getDeclID() const {
    if (isBlockPointer()) {
      assert(BS.Pointee);
      return BS.Pointee->getDeclID();
    }
    return std::nullopt;
  }

  /// Returns the byte offset from the start.
  uint64_t getByteOffset() const {
    if (isIntegralPointer())
      return Int.Value + Offset;
    if (isTypeidPointer())
      return reinterpret_cast<uintptr_t>(Typeid.TypePtr) + Offset;
    if (isOnePastEnd())
      return PtrView::PastEndMark;
    return Offset;
  }

  /// Returns the number of elements.
  unsigned getNumElems() const {
    if (!isBlockPointer())
      return ~0u;
    return view().getNumElems();
  }

  const Block *block() const { return BS.Pointee; }

  /// If backed by actual data (i.e. a block pointer), return
  /// an address to that data.
  const std::byte *getRawAddress() const {
    assert(isBlockPointer());
    return BS.Pointee->rawData() + Offset;
  }

  /// Returns the index into an array.
  int64_t getIndex() const {
    if (!isBlockPointer())
      return getIntegerRepresentation();

    return view().getIndex();
  }

  /// Checks if the index is one past end.
  bool isOnePastEnd() const {
    if (!isBlockPointer())
      return false;

    if (!BS.Pointee)
      return false;

    if (isUnknownSizeArray())
      return false;

    return isPastEnd() || (getSize() == getOffset());
  }

  /// Checks if the pointer points past the end of the object.
  bool isPastEnd() const {
    if (isIntegralPointer())
      return false;

    return !isZero() && Offset > BS.Pointee->getSize();
  }

  /// Checks if the pointer is an out-of-bounds element pointer.
  bool isElementPastEnd() const { return Offset == PtrView::PastEndMark; }

  /// Checks if the pointer is pointing to a zero-size array.
  bool isZeroSizeArray() const {
    if (isFunctionPointer())
      return false;
    if (const auto *Desc = getFieldDesc())
      return Desc->isZeroSizeArray();
    return false;
  }

  /// Checks whether the pointer can be dereferenced to the given PrimType.
  bool canDeref(PrimType T) const {
    if (const Descriptor *FieldDesc = getFieldDesc()) {
      return (FieldDesc->isPrimitive() || FieldDesc->isPrimitiveArray()) &&
             FieldDesc->getPrimType() == T;
    }
    return false;
  }

  /// Dereferences the pointer, if it's live.
  template <typename T> T &deref() const {
    assert(isLive() && "Invalid pointer");
    assert(isBlockPointer());
    assert(BS.Pointee);
    assert(isDereferencable());
    assert(Offset + sizeof(T) <= BS.Pointee->getDescriptor()->getAllocSize());

    return view().deref<T>();
  }

  /// Dereferences the element at index \p I.
  /// This is equivalent to atIndex(I).deref<T>().
  template <typename T> T &elem(unsigned I) const {
    assert(isLive() && "Invalid pointer");
    assert(isBlockPointer());
    assert(BS.Pointee);
    assert(isDereferencable());
    assert(getFieldDesc()->isPrimitiveArray());
    assert(I < getFieldDesc()->getNumElems());

    return view().elem<T>(I);
  }

  bool isConstexprUnknown() const {
    if (!isBlockPointer())
      return false;
    return getDeclDesc()->IsConstexprUnknown;
  }

  /// Whether this block can be read from at all. This is only true for
  /// block pointers that point to a valid location inside that block.
  bool isDereferencable() const {
    if (!isBlockPointer())
      return false;
    if (isDummy())
      return false;
    if (isConstexprUnknown())
      return false;
    if (isPastEnd())
      return false;

    return true;
  }

  /// Initializes a field.
  void initialize() const {
    if (!isBlockPointer())
      return;
    view().initialize();
  }
  /// Initialized the given element of a primitive array.
  void initializeElement(unsigned Index) const {
    view().initializeElement(Index);
  }
  /// Initialize all elements of a primitive array at once. This can be
  /// used in situations where we *know* we have initialized *all* elements
  /// of a primtive array.
  void initializeAllElements() const;
  /// Checks if an object was initialized.
  bool isInitialized() const;
  /// Like isInitialized(), but for primitive arrays.
  bool isElementInitialized(unsigned Index) const {
    if (!isBlockPointer())
      return true;

    return view().isElementInitialized(Index);
  }
  bool allElementsInitialized() const {
    assert(getFieldDesc()->isPrimitiveArray());
    assert(isArrayRoot());
    return view().allElementsInitialized();
  }
  bool allElementsAlive() const;
  bool isElementAlive(unsigned Index) const;

  /// Activates a field.
  void activate() const { view().activate(); }
  /// Deactivates an entire strurcutre.
  void deactivate() const {
    // TODO: this only appears in constructors, so nothing to deactivate.
  }

  Lifetime getLifetime() const {
    if (!isBlockPointer())
      return Lifetime::Started;
    return view().getLifetime();
  }

  /// Start the lifetime of this pointer. This works for pointer with an
  /// InlineDescriptor as well as primitive array elements. Pointers are usually
  /// alive by default, unless the underlying object has been allocated with
  /// std::allocator. This function is used by std::construct_at.
  void startLifetime() const { setLifeState(Lifetime::Started); }
  /// Ends the lifetime of the pointer. This works for pointer with an
  /// InlineDescriptor as well as primitive array elements. This function is
  /// used by std::destroy_at.
  void endLifetime() const { setLifeState(Lifetime::Ended); }

  void setLifeState(Lifetime L) const {
    if (!isBlockPointer())
      return;
    view().setLifeState(L);
  };

  /// Strip base casts from this Pointer.
  /// The result is either a root pointer or something
  /// that isn't a base class anymore.
  [[nodiscard]] Pointer stripBaseCasts() const {
    return Pointer(view().stripBaseCasts());
  }

  /// Compare two pointers.
  ComparisonCategoryResult compare(const Pointer &Other) const {
    if (!hasSameBase(*this, Other))
      return ComparisonCategoryResult::Unordered;

    if (Offset < Other.Offset)
      return ComparisonCategoryResult::Less;
    if (Offset > Other.Offset)
      return ComparisonCategoryResult::Greater;

    return ComparisonCategoryResult::Equal;
  }

  /// Checks if two pointers are comparable.
  static bool hasSameBase(const Pointer &A, const Pointer &B);
  /// Checks if two pointers can be subtracted.
  static bool elemsOfSameArray(const Pointer &A, const Pointer &B);
  /// Checks if both given pointers point to the same block.
  static bool pointToSameBlock(const Pointer &A, const Pointer &B);

  static std::optional<std::pair<PtrView, PtrView>>
  computeSplitPoint(const Pointer &A, const Pointer &B);

  /// Whether this points to a block that's been created for a "literal lvalue",
  /// i.e. a non-MaterializeTemporaryExpr Expr.
  bool pointsToLiteral() const;
  bool pointsToStringLiteral() const;
  /// Whether this points to a block created for an AddrLabelExpr.
  bool pointsToLabel() const;
  /// Returns the AddrLabelExpr the Pointer points to, if any.
  const AddrLabelExpr *getPointedToLabel() const {
    if (const Descriptor *Desc = getDeclDesc())
      return dyn_cast_if_present<AddrLabelExpr>(Desc->asExpr());
    return nullptr;
  }

  /// Prints the pointer.
  void print(llvm::raw_ostream &OS) const;

  /// Compute an integer that can be used to compare this pointer to
  /// another one. This is usually NOT the same as the pointer offset
  /// regarding the AST record layout.
  std::optional<size_t>
  computeOffsetForComparison(const ASTContext &ASTCtx) const;
  /// Compute the pointer offset as given by the ASTRecordLayout.
  /// Returns the result in bytes.
  std::optional<size_t> computeLayoutOffset(const ASTContext &ASTCtx) const;

private:
  friend class Block;
  friend class DeadBlock;
  friend class MemberPointer;
  friend class InterpState;
  friend class DynamicAllocator;
  friend class Program;

  /// Returns the embedded descriptor preceding a field.
  InlineDescriptor *getInlineDesc() const {
    assert(isBlockPointer());
    assert(BS.Base != sizeof(GlobalInlineDescriptor));
    assert(BS.Base <= BS.Pointee->getSize());
    assert(BS.Base >= sizeof(InlineDescriptor));
    return getDescriptor(BS.Base);
  }

  /// Returns a descriptor at a given offset.
  InlineDescriptor *getDescriptor(unsigned Offset) const {
    assert(Offset != 0 && "Not a nested pointer");
    assert(isBlockPointer());
    assert(!isZero());
    return view().getDescriptor(Offset);
  }

  /// Returns a reference to the InitMapPtr which stores the initialization map.
  InitMapPtr &getInitMap() const {
    assert(isBlockPointer());
    assert(!isZero());
    return view().getInitMap();
  }

  /// Offset into the storage.
  uint64_t Offset = 0;

  Storage StorageKind = Storage::Int;
  union {
    IntPointer Int;
    BlockPointer BS;
    FunctionPointer Fn;
    TypeidPointer Typeid;
  };
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Pointer &P) {
  P.print(OS);
  OS << ' ';
  if (P.isZero())
    return OS;

  if (const Descriptor *D = P.getFieldDesc())
    D->dump(OS);
  if (P.isArrayElement()) {
    if (P.isOnePastEnd())
      OS << " one-past-the-end";
    else {
      OS << ' ';
      std::string Indices;
      llvm::raw_string_ostream SS(Indices);
      Pointer K = P;
      while (K.isArrayElement()) {
        SS << ']' << K.expand().getIndex() << '[';
        K = K.expand().getArray();
      }
      std::reverse(Indices.begin(), Indices.end());
      OS << Indices;
    }
  } else if (P.isBlockPointer() && P.isArrayRoot())
    OS << " arrayroot";

  if (P.isBlockPointer() && P.block() && P.block()->isDummy())
    OS << " dummy";
  if (!P.isLive())
    OS << " dead";
  if (P.isBlockPointer() && P.isBaseClass())
    OS << " base-class";
  return OS;
}

} // namespace interp
} // namespace clang

#endif
