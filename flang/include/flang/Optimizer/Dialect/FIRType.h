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
#include "llvm/ADT/SmallVector.h"

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

class FIROpsDialect;

using KindTy = unsigned;

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
  return t.isa<ReferenceType>() || t.isa<PointerType>() || t.isa<HeapType>() ||
         t.isa<fir::LLVMPointerType>();
}

/// Is `t` a boxed type?
inline bool isa_box_type(mlir::Type t) {
  return t.isa<BoxType>() || t.isa<BoxCharType>() || t.isa<BoxProcType>();
}

/// Is `t` a type that is always trivially pass-by-reference? Specifically, this
/// is testing if `t` is a ReferenceType or any box type. Compare this to
/// conformsWithPassByRef(), which includes pointers and allocatables.
inline bool isa_passbyref_type(mlir::Type t) {
  return t.isa<ReferenceType>() || isa_box_type(t) ||
         t.isa<mlir::FunctionType>();
}

/// Is `t` a type that can conform to be pass-by-reference? Depending on the
/// context, these types may simply demote to pass-by-reference or a reference
/// to them may have to be passed instead.
inline bool conformsWithPassByRef(mlir::Type t) {
  return isa_ref_type(t) || isa_box_type(t);
}

/// Is `t` a derived (record) type?
inline bool isa_derived(mlir::Type t) { return t.isa<fir::RecordType>(); }

/// Is `t` a FIR dialect aggregate type?
inline bool isa_aggregate(mlir::Type t) {
  return t.isa<SequenceType>() || fir::isa_derived(t) ||
         t.isa<mlir::TupleType>();
}

/// Extract the `Type` pointed to from a FIR memory reference type. If `t` is
/// not a memory reference type, then returns a null `Type`.
mlir::Type dyn_cast_ptrEleTy(mlir::Type t);

/// Extract the `Type` pointed to from a FIR memory reference or box type. If
/// `t` is not a memory reference or box type, then returns a null `Type`.
mlir::Type dyn_cast_ptrOrBoxEleTy(mlir::Type t);

/// Is `t` a FIR Real or MLIR Float type?
inline bool isa_real(mlir::Type t) {
  return t.isa<fir::RealType>() || t.isa<mlir::FloatType>();
}

/// Is `t` an integral type?
inline bool isa_integer(mlir::Type t) {
  return t.isa<mlir::IndexType>() || t.isa<mlir::IntegerType>() ||
         t.isa<fir::IntegerType>();
}

mlir::Type parseFirType(FIROpsDialect *, mlir::DialectAsmParser &parser);

void printFirType(FIROpsDialect *, mlir::Type ty, mlir::DialectAsmPrinter &p);

/// Guarantee `type` is a scalar integral type (standard Integer, standard
/// Index, or FIR Int). Aborts execution if condition is false.
void verifyIntegralType(mlir::Type type);

/// Is `t` a FIR or MLIR Complex type?
inline bool isa_complex(mlir::Type t) {
  return t.isa<fir::ComplexType>() || t.isa<mlir::ComplexType>();
}

/// Is `t` a CHARACTER type? Does not check the length.
inline bool isa_char(mlir::Type t) { return t.isa<fir::CharacterType>(); }

/// Is `t` a trivial intrinsic type? CHARACTER is <em>excluded</em> because it
/// is a dependent type.
inline bool isa_trivial(mlir::Type t) {
  return isa_integer(t) || isa_real(t) || isa_complex(t) ||
         t.isa<fir::LogicalType>();
}

/// Is `t` a CHARACTER type with a LEN other than 1?
inline bool isa_char_string(mlir::Type t) {
  if (auto ct = t.dyn_cast_or_null<fir::CharacterType>())
    return ct.getLen() != fir::CharacterType::singleton();
  return false;
}

/// Is `t` a box type for which it is not possible to deduce the box size?
/// It is not possible to deduce the size of a box that describes an entity
/// of unknown rank or type.
bool isa_unknown_size_box(mlir::Type t);

/// Returns true iff `t` is a fir.char type and has an unknown length.
inline bool characterWithDynamicLen(mlir::Type t) {
  if (auto charTy = t.dyn_cast<fir::CharacterType>())
    return charTy.hasDynamicLen();
  return false;
}

/// Returns true iff `seqTy` has either an unknown shape or a non-constant shape
/// (where rank > 0).
inline bool sequenceWithNonConstantShape(fir::SequenceType seqTy) {
  return seqTy.hasUnknownShape() || !seqTy.hasConstantShape();
}

/// Returns true iff the type `t` does not have a constant size.
bool hasDynamicSize(mlir::Type t);

/// If `t` is a SequenceType return its element type, otherwise return `t`.
inline mlir::Type unwrapSequenceType(mlir::Type t) {
  if (auto seqTy = t.dyn_cast<fir::SequenceType>())
    return seqTy.getEleTy();
  return t;
}

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

#ifndef NDEBUG
// !fir.ptr<X> and !fir.heap<X> where X is !fir.ptr, !fir.heap, or !fir.ref
// is undefined and disallowed.
inline bool singleIndirectionLevel(mlir::Type ty) {
  return !fir::isa_ref_type(ty);
}
#endif

/// Return true iff `ty` is the type of an ALLOCATABLE entity or value.
bool isAllocatableType(mlir::Type ty);

/// Return true iff `ty` is a RecordType with members that are allocatable.
bool isRecordWithAllocatableMember(mlir::Type ty);

/// Return true iff `ty` is a RecordType with type parameters.
inline bool isRecordWithTypeParameters(mlir::Type ty) {
  if (auto recTy = ty.dyn_cast_or_null<fir::RecordType>())
    return recTy.getNumLenParams() != 0;
  return false;
}

/// Is this tuple type holding a character function and its result length ?
bool isCharacterProcedureTuple(mlir::Type type, bool acceptRawFunc = true);

/// Apply the components specified by `path` to `rootTy` to determine the type
/// of the resulting component element. `rootTy` should be an aggregate type.
/// Returns null on error.
mlir::Type applyPathToType(mlir::Type rootTy, mlir::ValueRange path);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_DIALECT_FIRTYPE_H
