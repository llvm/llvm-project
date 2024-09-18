//===-- Optimizer/Dialect/FIRType.h -- FIR types ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_FIRTYPE_H
#define FORTRAN_OPTIMIZER_DIALECT_FIRTYPE_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Type.h"

namespace fir {
class FIROpsDialect;
class KindMapping;
using KindTy = unsigned;

namespace detail {
struct RecordTypeStorage;
} // namespace detail

} // namespace fir

//===----------------------------------------------------------------------===//
// BaseBoxType
//===----------------------------------------------------------------------===//

namespace fir {

/// This class provides a shared interface for box and class types.
class BaseBoxType : public mlir::Type {
public:
  using mlir::Type::Type;

  /// Box attributes.
  enum class Attribute { None, Allocatable, Pointer };

  /// Returns the element type of this box type.
  mlir::Type getEleTy() const;

  /// Unwrap element type from fir.heap, fir.ptr and fir.array.
  mlir::Type unwrapInnerType() const;

  /// Is this the box for an assumed rank?
  bool isAssumedRank() const;

  /// Return the same type, except for the shape, that is taken the shape
  /// of shapeMold.
  BaseBoxType getBoxTypeWithNewShape(mlir::Type shapeMold) const;
  BaseBoxType getBoxTypeWithNewShape(int rank) const;

  /// Return the same type, except for the attribute (fir.heap/fir.ptr).
  BaseBoxType getBoxTypeWithNewAttr(Attribute attr) const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(mlir::Type type);
};

} // namespace fir

#define GET_TYPEDEF_CLASSES
#include "flang/Optimizer/Dialect/FIROpsTypes.h.inc"

namespace llvm {
class raw_ostream;
class StringRef;
template <typename>
class ArrayRef;
class hash_code;
} // namespace llvm

namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
class ComplexType;
class FloatType;
class ValueRange;
} // namespace mlir

namespace fir {
namespace detail {
struct RecordTypeStorage;
} // namespace detail

// These isa_ routines follow the precedent of llvm::isa_or_null<>

/// Is `t` any of the FIR dialect types?
bool isa_fir_type(mlir::Type t);

/// Is `t` any of the Standard dialect types?
bool isa_std_type(mlir::Type t);

/// Is `t` any of the FIR dialect or Standard dialect types?
bool isa_fir_or_std_type(mlir::Type t);

/// Is `t` a FIR dialect type that implies a memory (de)reference?
inline bool isa_ref_type(mlir::Type t) {
  return mlir::isa<fir::ReferenceType, fir::PointerType, fir::HeapType,
                   fir::LLVMPointerType>(t);
}

/// Is `t` a boxed type?
inline bool isa_box_type(mlir::Type t) {
  return mlir::isa<fir::BaseBoxType, fir::BoxCharType, fir::BoxProcType>(t);
}

/// Is `t` a type that is always trivially pass-by-reference? Specifically, this
/// is testing if `t` is a ReferenceType or any box type. Compare this to
/// conformsWithPassByRef(), which includes pointers and allocatables.
inline bool isa_passbyref_type(mlir::Type t) {
  return mlir::isa<fir::ReferenceType, mlir::FunctionType>(t) ||
         isa_box_type(t);
}

/// Is `t` a type that can conform to be pass-by-reference? Depending on the
/// context, these types may simply demote to pass-by-reference or a reference
/// to them may have to be passed instead. Functions are always referent.
inline bool conformsWithPassByRef(mlir::Type t) {
  return isa_ref_type(t) || isa_box_type(t) || mlir::isa<mlir::FunctionType>(t);
}

/// Is `t` a derived (record) type?
inline bool isa_derived(mlir::Type t) { return mlir::isa<fir::RecordType>(t); }

/// Is `t` type(c_ptr) or type(c_funptr)?
inline bool isa_builtin_cptr_type(mlir::Type t) {
  if (auto recTy = mlir::dyn_cast_or_null<fir::RecordType>(t))
    return recTy.getName().ends_with("T__builtin_c_ptr") ||
           recTy.getName().ends_with("T__builtin_c_funptr");
  return false;
}

/// Is `t` type(c_devptr)?
inline bool isa_builtin_cdevptr_type(mlir::Type t) {
  if (auto recTy = mlir::dyn_cast_or_null<fir::RecordType>(t))
    return recTy.getName().ends_with("T__builtin_c_devptr");
  return false;
}

/// Is `t` a FIR dialect aggregate type?
inline bool isa_aggregate(mlir::Type t) {
  return mlir::isa<SequenceType, mlir::TupleType>(t) || fir::isa_derived(t);
}

/// Extract the `Type` pointed to from a FIR memory reference type. If `t` is
/// not a memory reference type, then returns a null `Type`.
mlir::Type dyn_cast_ptrEleTy(mlir::Type t);

/// Extract the `Type` pointed to from a FIR memory reference or box type. If
/// `t` is not a memory reference or box type, then returns a null `Type`.
mlir::Type dyn_cast_ptrOrBoxEleTy(mlir::Type t);

/// Is `t` a FIR Real or MLIR Float type?
inline bool isa_real(mlir::Type t) {
  return mlir::isa<fir::RealType, mlir::FloatType>(t);
}

/// Is `t` an integral type?
inline bool isa_integer(mlir::Type t) {
  return mlir::isa<mlir::IndexType, mlir::IntegerType, fir::IntegerType>(t);
}

/// Is `t` a vector type?
inline bool isa_vector(mlir::Type t) {
  return mlir::isa<mlir::VectorType, fir::VectorType>(t);
}

mlir::Type parseFirType(FIROpsDialect *, mlir::DialectAsmParser &parser);

void printFirType(FIROpsDialect *, mlir::Type ty, mlir::DialectAsmPrinter &p);

/// Guarantee `type` is a scalar integral type (standard Integer, standard
/// Index, or FIR Int). Aborts execution if condition is false.
void verifyIntegralType(mlir::Type type);

/// Is `t` a FIR or MLIR Complex type?
inline bool isa_complex(mlir::Type t) {
  return mlir::isa<fir::ComplexType, mlir::ComplexType>(t);
}

/// Is `t` a CHARACTER type? Does not check the length.
inline bool isa_char(mlir::Type t) { return mlir::isa<fir::CharacterType>(t); }

/// Is `t` a trivial intrinsic type? CHARACTER is <em>excluded</em> because it
/// is a dependent type.
inline bool isa_trivial(mlir::Type t) {
  return isa_integer(t) || isa_real(t) || isa_complex(t) || isa_vector(t) ||
         mlir::isa<fir::LogicalType>(t);
}

/// Is `t` a CHARACTER type with a LEN other than 1?
inline bool isa_char_string(mlir::Type t) {
  if (auto ct = mlir::dyn_cast_or_null<fir::CharacterType>(t))
    return ct.getLen() != fir::CharacterType::singleton();
  return false;
}

/// Is `t` a box type for which it is not possible to deduce the box size?
/// It is not possible to deduce the size of a box that describes an entity
/// of unknown rank.
/// Unknown type are always considered to have the size of derived type box
/// (since they may hold one), and are not considered to be unknown size.
bool isa_unknown_size_box(mlir::Type t);

/// Returns true iff `t` is a fir.char type and has an unknown length.
inline bool characterWithDynamicLen(mlir::Type t) {
  if (auto charTy = mlir::dyn_cast<fir::CharacterType>(t))
    return charTy.hasDynamicLen();
  return false;
}

/// Returns true iff `seqTy` has either an unknown shape or a non-constant shape
/// (where rank > 0).
inline bool sequenceWithNonConstantShape(fir::SequenceType seqTy) {
  return seqTy.hasUnknownShape() || seqTy.hasDynamicExtents();
}

/// Returns true iff the type `t` does not have a constant size.
bool hasDynamicSize(mlir::Type t);

inline unsigned getRankOfShapeType(mlir::Type t) {
  if (auto shTy = mlir::dyn_cast<fir::ShapeType>(t))
    return shTy.getRank();
  if (auto shTy = mlir::dyn_cast<fir::ShapeShiftType>(t))
    return shTy.getRank();
  if (auto shTy = mlir::dyn_cast<fir::ShiftType>(t))
    return shTy.getRank();
  return 0;
}

/// Get the memory reference type of the data pointer from the box type,
inline mlir::Type boxMemRefType(fir::BaseBoxType t) {
  auto eleTy = t.getEleTy();
  if (!mlir::isa<fir::PointerType, fir::HeapType>(eleTy))
    eleTy = fir::ReferenceType::get(t);
  return eleTy;
}

/// If `t` is a SequenceType return its element type, otherwise return `t`.
inline mlir::Type unwrapSequenceType(mlir::Type t) {
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(t))
    return seqTy.getEleTy();
  return t;
}

/// Return the nested sequence type if any.
mlir::Type extractSequenceType(mlir::Type ty);

inline mlir::Type unwrapRefType(mlir::Type t) {
  if (auto eleTy = dyn_cast_ptrEleTy(t))
    return eleTy;
  return t;
}

/// If `t` conforms with a pass-by-reference type (box, ref, ptr, etc.) then
/// return the element type of `t`. Otherwise, return `t`.
inline mlir::Type unwrapPassByRefType(mlir::Type t) {
  if (auto eleTy = dyn_cast_ptrOrBoxEleTy(t))
    return eleTy;
  return t;
}

/// Unwrap either a sequence or a boxed sequence type, returning the element
/// type of the sequence type.
/// e.g.,
///   !fir.array<...xT>  ->  T
///   !fir.box<!fir.ptr<!fir.array<...xT>>>  ->  T
/// otherwise
///   T -> T
mlir::Type unwrapSeqOrBoxedSeqType(mlir::Type ty);

/// Unwrap all referential and sequential outer types (if any). Returns the
/// element type. This is useful for determining the element type of any object
/// memory reference, whether it is a single instance or a series of instances.
mlir::Type unwrapAllRefAndSeqType(mlir::Type ty);

/// Unwrap all pointer and box types and return the element type if it is a
/// sequence type, otherwise return null.
inline fir::SequenceType unwrapUntilSeqType(mlir::Type t) {
  while (true) {
    if (!t)
      return {};
    if (auto ty = dyn_cast_ptrOrBoxEleTy(t)) {
      t = ty;
      continue;
    }
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(t))
      return seqTy;
    return {};
  }
}

/// Unwrap the referential and sequential outer types (if any). Returns the
/// the element if type is fir::RecordType
inline fir::RecordType unwrapIfDerived(fir::BaseBoxType boxTy) {
  return mlir::dyn_cast<fir::RecordType>(
      fir::unwrapSequenceType(fir::unwrapRefType(boxTy.getEleTy())));
}

/// Return true iff `boxTy` wraps a fir::RecordType with length parameters
inline bool isDerivedTypeWithLenParams(fir::BaseBoxType boxTy) {
  auto recTy = unwrapIfDerived(boxTy);
  return recTy && recTy.getNumLenParams() > 0;
}

/// Return true iff `boxTy` wraps a fir::RecordType
inline bool isDerivedType(fir::BaseBoxType boxTy) {
  return static_cast<bool>(unwrapIfDerived(boxTy));
}

#ifndef NDEBUG
// !fir.ptr<X> and !fir.heap<X> where X is !fir.ptr, !fir.heap, or !fir.ref
// is undefined and disallowed.
inline bool singleIndirectionLevel(mlir::Type ty) {
  return !fir::isa_ref_type(ty);
}
#endif

/// Return true iff `ty` is the type of a POINTER entity or value.
/// `isa_ref_type()` can be used to distinguish.
bool isPointerType(mlir::Type ty);

/// Return true iff `ty` is the type of an ALLOCATABLE entity or value.
bool isAllocatableType(mlir::Type ty);

/// Return true iff `ty` is !fir.box<none>.
bool isBoxNone(mlir::Type ty);

/// Return true iff `ty` is the type of a boxed record type.
/// e.g. !fir.box<!fir.type<derived>>
bool isBoxedRecordType(mlir::Type ty);

/// Return true iff `ty` is a type that contains descriptor information.
bool isTypeWithDescriptor(mlir::Type ty);

/// Return true iff `ty` is a scalar boxed record type.
/// e.g. !fir.box<!fir.type<derived>>
///      !fir.box<!fir.heap<!fir.type<derived>>>
///      !fir.class<!fir.type<derived>>
bool isScalarBoxedRecordType(mlir::Type ty);

/// Return the nested RecordType if one if found. Return ty otherwise.
mlir::Type getDerivedType(mlir::Type ty);

/// Return true iff `ty` is the type of an polymorphic entity or
/// value.
bool isPolymorphicType(mlir::Type ty);

/// Return true iff `ty` is the type of an unlimited polymorphic entity or
/// value.
bool isUnlimitedPolymorphicType(mlir::Type ty);

/// Return true iff `ty` is the type of an assumed type. In FIR,
/// assumed types are of the form `[fir.ref|ptr|heap]fir.box<[fir.array]none>`,
/// or `fir.ref|ptr|heap<[fir.array]none>`.
bool isAssumedType(mlir::Type ty);

/// Return true iff `ty` is the type of an assumed shape array.
bool isAssumedShape(mlir::Type ty);

/// Return true iff `ty` is the type of an allocatable array.
bool isAllocatableOrPointerArray(mlir::Type ty);

/// Return true iff `boxTy` wraps a record type or an unlimited polymorphic
/// entity. Polymorphic entities with intrinsic type spec do not have addendum
inline bool boxHasAddendum(fir::BaseBoxType boxTy) {
  return static_cast<bool>(unwrapIfDerived(boxTy)) ||
         fir::isUnlimitedPolymorphicType(boxTy);
}

/// Get the rank from a !fir.box type.
unsigned getBoxRank(mlir::Type boxTy);

/// Return the inner type of the given type.
mlir::Type unwrapInnerType(mlir::Type ty);

/// Return true iff `ty` is a RecordType with members that are allocatable.
bool isRecordWithAllocatableMember(mlir::Type ty);

/// Return true iff `ty` is a scalar/array of RecordType
/// with members that are descriptors.
bool isRecordWithDescriptorMember(mlir::Type ty);

/// Return true iff `ty` is a RecordType with type parameters.
inline bool isRecordWithTypeParameters(mlir::Type ty) {
  if (auto recTy = mlir::dyn_cast_or_null<fir::RecordType>(ty))
    return recTy.isDependentType();
  return false;
}

/// Is this tuple type holding a character function and its result length?
bool isCharacterProcedureTuple(mlir::Type type, bool acceptRawFunc = true);

/// Apply the components specified by `path` to `rootTy` to determine the type
/// of the resulting component element. `rootTy` should be an aggregate type.
/// Returns null on error.
mlir::Type applyPathToType(mlir::Type rootTy, mlir::ValueRange path);

/// Does this function type has a result that requires binding the result value
/// with a storage in a fir.save_result operation in order to use the result?
bool hasAbstractResult(mlir::FunctionType ty);

/// Convert llvm::Type::TypeID to mlir::Type
mlir::Type fromRealTypeID(mlir::MLIRContext *context, llvm::Type::TypeID typeID,
                          fir::KindTy kind);

int getTypeCode(mlir::Type ty, const KindMapping &kindMap);

inline bool BaseBoxType::classof(mlir::Type type) {
  return mlir::isa<fir::BoxType, fir::ClassType>(type);
}

/// Return true iff `ty` is none or fir.array<none>.
inline bool isNoneOrSeqNone(mlir::Type type) {
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(type))
    return mlir::isa<mlir::NoneType>(seqTy.getEleTy());
  return mlir::isa<mlir::NoneType>(type);
}

/// Return a fir.box<T> or fir.class<T> if the type is polymorphic. If the type
/// is polymorphic and assumed shape return fir.box<T>.
inline mlir::Type wrapInClassOrBoxType(mlir::Type eleTy,
                                       bool isPolymorphic = false,
                                       bool isAssumedType = false) {
  if (isPolymorphic && !isAssumedType)
    return fir::ClassType::get(eleTy);
  return fir::BoxType::get(eleTy);
}

/// Return the elementType where intrinsic types are replaced with none for
/// unlimited polymorphic entities.
///
/// i32 -> none
/// !fir.array<2xf32> -> !fir.array<2xnone>
/// !fir.heap<!fir.array<2xf32>> -> !fir.heap<!fir.array<2xnone>>
inline mlir::Type updateTypeForUnlimitedPolymorphic(mlir::Type ty) {
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
    return fir::SequenceType::get(
        seqTy.getShape(), updateTypeForUnlimitedPolymorphic(seqTy.getEleTy()));
  if (auto heapTy = mlir::dyn_cast<fir::HeapType>(ty))
    return fir::HeapType::get(
        updateTypeForUnlimitedPolymorphic(heapTy.getEleTy()));
  if (auto pointerTy = mlir::dyn_cast<fir::PointerType>(ty))
    return fir::PointerType::get(
        updateTypeForUnlimitedPolymorphic(pointerTy.getEleTy()));
  if (!mlir::isa<mlir::NoneType, fir::RecordType>(ty))
    return mlir::NoneType::get(ty.getContext());
  return ty;
}

/// Replace the element type of \p type by \p newElementType, preserving
/// all other layers of the type (fir.ref/ptr/heap/array/box/class).
/// If \p turnBoxIntoClass and the input is a fir.box, it will be turned into
/// a fir.class in the result.
mlir::Type changeElementType(mlir::Type type, mlir::Type newElementType,
                             bool turnBoxIntoClass);

/// Is `t` an address to fir.box or class type?
inline bool isBoxAddress(mlir::Type t) {
  return fir::isa_ref_type(t) &&
         mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(t));
}

/// Is `t` a fir.box or class address or value type?
inline bool isBoxAddressOrValue(mlir::Type t) {
  return mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(t));
}

/// Is this a fir.boxproc address type?
inline bool isBoxProcAddressType(mlir::Type t) {
  t = fir::dyn_cast_ptrEleTy(t);
  return t && mlir::isa<fir::BoxProcType>(t);
}

/// Return a string representation of `ty`.
///
/// fir.array<10x10xf32> -> prefix_10x10xf32
/// fir.ref<i32> -> prefix_ref_i32
std::string getTypeAsString(mlir::Type ty, const KindMapping &kindMap,
                            llvm::StringRef prefix = "");

/// Return the size and alignment of FIR types.
/// TODO: consider moving this to a DataLayoutTypeInterface implementation
/// for FIR types. It should first be ensured that it is OK to open the gate of
/// target dependent type size inquiries in lowering. It would also not be
/// straightforward given the need for a kind map that would need to be
/// converted in terms of mlir::DataLayoutEntryKey.

/// This variant terminates the compilation if an unsupported type is passed.
std::pair<std::uint64_t, unsigned short>
getTypeSizeAndAlignmentOrCrash(mlir::Location loc, mlir::Type ty,
                               const mlir::DataLayout &dl,
                               const fir::KindMapping &kindMap);

/// This variant returns std::nullopt if an unsupported type is passed.
std::optional<std::pair<uint64_t, unsigned short>>
getTypeSizeAndAlignment(mlir::Location loc, mlir::Type ty,
                        const mlir::DataLayout &dl,
                        const fir::KindMapping &kindMap);
} // namespace fir

#endif // FORTRAN_OPTIMIZER_DIALECT_FIRTYPE_H
