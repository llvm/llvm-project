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

#ifndef OPTIMIZER_DIALECT_FIRTYPE_H
#define OPTIMIZER_DIALECT_FIRTYPE_H

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
struct BoxProcTypeStorage;
struct ComplexTypeStorage;
struct HeapTypeStorage;
struct IntegerTypeStorage;
struct LenTypeStorage;
struct LogicalTypeStorage;
struct PointerTypeStorage;
struct RealTypeStorage;
struct RecordTypeStorage;
struct ReferenceTypeStorage;
struct SequenceTypeStorage;
struct SliceTypeStorage;
struct TypeDescTypeStorage;
struct VectorTypeStorage;
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

// Intrinsic types

/// Model of a Fortran COMPLEX intrinsic type, including the KIND type
/// parameter. COMPLEX is a floating point type with a real and imaginary
/// member.
class ComplexType : public mlir::Type::TypeBase<fir::ComplexType, mlir::Type,
                                                detail::ComplexTypeStorage> {
public:
  using Base::Base;
  static fir::ComplexType get(mlir::MLIRContext *ctxt, KindTy kind);

  /// Get the corresponding fir.real<k> type.
  mlir::Type getElementType() const;

  KindTy getFKind() const;
};

/// Model of a Fortran INTEGER intrinsic type, including the KIND type
/// parameter.
class IntegerType : public mlir::Type::TypeBase<fir::IntegerType, mlir::Type,
                                                detail::IntegerTypeStorage> {
public:
  using Base::Base;
  static fir::IntegerType get(mlir::MLIRContext *ctxt, KindTy kind);
  KindTy getFKind() const;
};

/// Model of a Fortran LOGICAL intrinsic type, including the KIND type
/// parameter.
class LogicalType : public mlir::Type::TypeBase<LogicalType, mlir::Type,
                                                detail::LogicalTypeStorage> {
public:
  using Base::Base;
  static LogicalType get(mlir::MLIRContext *ctxt, KindTy kind);
  KindTy getFKind() const;
};

/// Model of a Fortran REAL (and DOUBLE PRECISION) intrinsic type, including the
/// KIND type parameter.
class RealType : public mlir::Type::TypeBase<RealType, mlir::Type,
                                             detail::RealTypeStorage> {
public:
  using Base::Base;
  static RealType get(mlir::MLIRContext *ctxt, KindTy kind);
  KindTy getFKind() const;
};

// FIR support types

/// The type of a pair that describes a PROCEDURE reference. Pointers to
/// internal procedures must carry an additional reference to the host's
/// variables that are referenced.
class BoxProcType : public mlir::Type::TypeBase<BoxProcType, mlir::Type,
                                                detail::BoxProcTypeStorage> {
public:
  using Base::Base;
  static BoxProcType get(mlir::Type eleTy);
  mlir::Type getEleTy() const;

  static mlir::LogicalResult verifyConstructionInvariants(mlir::Location,
                                                          mlir::Type eleTy);
};

/// Type of a vector that represents an array slice operation on an array.
/// Fortran slices are triples of lower bound, upper bound, and stride. The rank
/// of a SliceType must be at least 1.
class SliceType : public mlir::Type::TypeBase<SliceType, mlir::Type,
                                              detail::SliceTypeStorage> {
public:
  using Base::Base;
  static SliceType get(mlir::MLIRContext *ctx, unsigned rank);
  unsigned getRank() const;
};

/// The type of a heap pointer. Fortran entities with the ALLOCATABLE attribute
/// may be allocated on the heap at runtime. These pointers are explicitly
/// distinguished to disallow the composition of multiple levels of
/// indirection. For example, an ALLOCATABLE POINTER is invalid.
class HeapType : public mlir::Type::TypeBase<HeapType, mlir::Type,
                                             detail::HeapTypeStorage> {
public:
  using Base::Base;
  static HeapType get(mlir::Type elementType);

  mlir::Type getEleTy() const;

  static mlir::LogicalResult verifyConstructionInvariants(mlir::Location,
                                                          mlir::Type eleTy);
};

/// The type of a LEN parameter name. Implementations may defer the layout of a
/// Fortran derived type until runtime. This implies that the runtime must be
/// able to determine the offset of LEN type parameters related to an entity.
class LenType
    : public mlir::Type::TypeBase<LenType, mlir::Type, detail::LenTypeStorage> {
public:
  using Base::Base;
  static LenType get(mlir::MLIRContext *ctxt);
};

/// The type of entities with the POINTER attribute.  These pointers are
/// explicitly distinguished to disallow the composition of multiple levels of
/// indirection. For example, an ALLOCATABLE POINTER is invalid.
class PointerType : public mlir::Type::TypeBase<PointerType, mlir::Type,
                                                detail::PointerTypeStorage> {
public:
  using Base::Base;
  static PointerType get(mlir::Type elementType);

  mlir::Type getEleTy() const;

  static mlir::LogicalResult verifyConstructionInvariants(mlir::Location,
                                                          mlir::Type eleTy);
};

/// The type of a reference to an entity in memory.
class ReferenceType
    : public mlir::Type::TypeBase<ReferenceType, mlir::Type,
                                  detail::ReferenceTypeStorage> {
public:
  using Base::Base;
  static ReferenceType get(mlir::Type elementType);

  mlir::Type getEleTy() const;

  static mlir::LogicalResult verifyConstructionInvariants(mlir::Location,
                                                          mlir::Type eleTy);
};

/// A sequence type is a multi-dimensional array of values. The sequence type
/// may have an unknown number of dimensions or the extent of dimensions may be
/// unknown. A sequence type models a Fortran array entity, giving it a type in
/// FIR. A sequence type is assumed to be stored in a column-major order, which
/// differs from LLVM IR and other dialects of MLIR.
class SequenceType : public mlir::Type::TypeBase<SequenceType, mlir::Type,
                                                 detail::SequenceTypeStorage> {
public:
  using Base::Base;
  using Extent = int64_t;
  using Shape = llvm::SmallVector<Extent, 8>;

  /// Return a sequence type with the specified shape and element type
  static SequenceType get(const Shape &shape, mlir::Type elementType,
                          mlir::AffineMapAttr map = {});

  /// The element type of this sequence
  mlir::Type getEleTy() const;

  /// The shape of the sequence. If the sequence has an unknown shape, the shape
  /// returned will be empty.
  Shape getShape() const;

  mlir::AffineMapAttr getLayoutMap() const;

  /// The number of dimensions of the sequence
  unsigned getDimension() const { return getShape().size(); }

  /// Number of rows of constant extent
  unsigned getConstantRows() const;

  /// Is the shape of the sequence constant?
  bool hasConstantShape() const { return getConstantRows() == getDimension(); }

  /// Does the sequence have unknown shape? (`array<* x T>`)
  bool hasUnknownShape() const { return getShape().empty(); }

  /// Is the interior of the sequence constant? Check if the array is
  /// one of constant shape (`array<C...xCxT>`), unknown shape
  /// (`array<*xT>`), or rows with shape and ending with column(s) of
  /// unknown extent (`array<C...xCx?...x?xT>`).
  bool hasConstantInterior() const;

  /// The value `-1` represents an unknown extent for a dimension
  static constexpr Extent getUnknownExtent() { return -1; }

  static mlir::LogicalResult
  verifyConstructionInvariants(mlir::Location loc, const Shape &shape,
                               mlir::Type eleTy, mlir::AffineMapAttr map);
};

bool operator==(const SequenceType::Shape &, const SequenceType::Shape &);
llvm::hash_code hash_value(const SequenceType::Extent &);
llvm::hash_code hash_value(const SequenceType::Shape &);

/// The type of a type descriptor object. The runtime may generate type
/// descriptor objects to determine the type of an entity at runtime, etc.
class TypeDescType : public mlir::Type::TypeBase<TypeDescType, mlir::Type,
                                                 detail::TypeDescTypeStorage> {
public:
  using Base::Base;
  static TypeDescType get(mlir::Type ofType);
  mlir::Type getOfTy() const;

  static mlir::LogicalResult verifyConstructionInvariants(mlir::Location,
                                                          mlir::Type ofType);
};

// Derived types

/// Model of Fortran's derived type, TYPE. The name of the TYPE includes any
/// KIND type parameters. The record includes runtime slots for LEN type
/// parameters and for data components.
class RecordType : public mlir::Type::TypeBase<RecordType, mlir::Type,
                                               detail::RecordTypeStorage> {
public:
  using Base::Base;
  using TypePair = std::pair<std::string, mlir::Type>;
  using TypeList = std::vector<TypePair>;

  llvm::StringRef getName();
  TypeList getTypeList();
  TypeList getLenParamList();

  mlir::Type getType(llvm::StringRef ident);
  mlir::Type getType(unsigned index) {
    assert(index < getNumFields());
    return getTypeList()[index].second;
  }
  unsigned getNumFields() { return getTypeList().size(); }
  unsigned getNumLenParams() { return getLenParamList().size(); }

  static RecordType get(mlir::MLIRContext *ctxt, llvm::StringRef name);
  void finalize(llvm::ArrayRef<TypePair> lenPList,
                llvm::ArrayRef<TypePair> typeList);

  detail::RecordTypeStorage const *uniqueKey() const;

  static mlir::LogicalResult verifyConstructionInvariants(mlir::Location,
                                                          llvm::StringRef name);
};

/// Is `t` a FIR Real or MLIR Float type?
inline bool isa_real(mlir::Type t) {
  return t.isa<fir::RealType>() || t.isa<mlir::FloatType>();
}

/// Is `t` an integral type?
inline bool isa_integer(mlir::Type t) {
  return t.isa<mlir::IndexType>() || t.isa<mlir::IntegerType>() ||
         t.isa<fir::IntegerType>();
}

/// Replacement for the builtin vector type.
/// The FIR vector type is always rank one. It's size is always a constant.
/// A vector's element type must be real or integer.
class VectorType : public mlir::Type::TypeBase<fir::VectorType, mlir::Type,
                                               detail::VectorTypeStorage> {
public:
  using Base::Base;

  static fir::VectorType get(uint64_t len, mlir::Type eleTy);
  mlir::Type getEleTy() const;
  uint64_t getLen() const;

  static mlir::LogicalResult
  verifyConstructionInvariants(mlir::Location, uint64_t len, mlir::Type eleTy);
  static bool isValidElementType(mlir::Type t) {
    return isa_real(t) || isa_integer(t);
  }
};

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

} // namespace fir

#endif // OPTIMIZER_DIALECT_FIRTYPE_H
