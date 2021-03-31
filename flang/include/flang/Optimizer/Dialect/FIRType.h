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
bool isa_ref_type(mlir::Type t);

/// Is `t` a type that is always trivially pass-by-reference? Specifically, this
/// is testing if `t` is a ReferenceType or any box type. Compare this to
/// conformsWithPassByRef(), which includes pointers and allocatables.
bool isa_passbyref_type(mlir::Type t);

/// Is `t` a boxed type?
bool isa_box_type(mlir::Type t);

/// Is `t` a type that can conform to be pass-by-reference? Depending on the
/// context, these types may simply demote to pass-by-reference or a reference
/// to them may have to be passed instead.
inline bool conformsWithPassByRef(mlir::Type t) {
  return isa_ref_type(t) || isa_box_type(t);
}

/// Is `t` a FIR dialect aggregate type?
bool isa_aggregate(mlir::Type t);

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

inline bool isa_char_string(mlir::Type t) {
  if (auto ct = t.dyn_cast_or_null<fir::CharacterType>())
    return ct.getLen() != fir::CharacterType::singleton();
  return false;
}

/// Is `t` a box type for which it is not possible to deduce the box size.
/// It is not possible to deduce the size of a box that describes an entity
/// of unknown rank or type.
bool isa_unknown_size_box(mlir::Type t);

#ifndef NDEBUG
// !fir.ptr<X> and !fir.heap<X> where X is !fir.ptr, !fir.heap, or !fir.ref
// is undefined and disallowed.
inline bool singleIndirectionLevel(mlir::Type ty) {
  return !fir::isa_ref_type(ty);
}
#endif

} // namespace fir

#endif // FORTRAN_OPTIMIZER_DIALECT_FIRTYPE_H
