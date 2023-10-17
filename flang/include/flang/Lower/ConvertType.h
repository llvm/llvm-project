//===-- Lower/ConvertType.h -- lowering of types ----------------*- C++ -*-===//
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
///
/// Conversion of front-end TYPE, KIND, ATTRIBUTE (TKA) information to FIR/MLIR.
/// This is meant to be the single point of truth (SPOT) for all type
/// conversions when lowering to FIR.  This implements all lowering of parse
/// tree TKA to the FIR type system. If one is converting front-end types and
/// not using one of the routines provided here, it's being done wrong.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERT_TYPE_H
#define FORTRAN_LOWER_CONVERT_TYPE_H

#include "flang/Common/Fortran.h"
#include "flang/Evaluate/type.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
class Location;
class MLIRContext;
class Type;
} // namespace mlir

namespace Fortran {
namespace common {
template <typename>
class Reference;
} // namespace common

namespace evaluate {
template <typename>
class Expr;
template <typename>
class FunctionRef;
struct SomeType;
} // namespace evaluate

namespace semantics {
class Symbol;
class DerivedTypeSpec;
class DerivedTypeDetails;
class Scope;
} // namespace semantics

namespace lower {
class AbstractConverter;
namespace pft {
struct Variable;
}

using SomeExpr = evaluate::Expr<evaluate::SomeType>;
using SymbolRef = common::Reference<const semantics::Symbol>;

// Type for compile time constant length type parameters.
using LenParameterTy = std::int64_t;

/// Get a FIR type based on a category and kind.
mlir::Type getFIRType(mlir::MLIRContext *ctxt, common::TypeCategory tc,
                      int kind, llvm::ArrayRef<LenParameterTy>);

/// Get a FIR type for a derived type
mlir::Type
translateDerivedTypeToFIRType(Fortran::lower::AbstractConverter &,
                              const Fortran::semantics::DerivedTypeSpec &);

/// Translate a SomeExpr to an mlir::Type.
mlir::Type translateSomeExprToFIRType(Fortran::lower::AbstractConverter &,
                                      const SomeExpr &expr);

/// Translate a Fortran::semantics::Symbol to an mlir::Type.
mlir::Type translateSymbolToFIRType(Fortran::lower::AbstractConverter &,
                                    const SymbolRef symbol);

/// Translate a Fortran::lower::pft::Variable to an mlir::Type.
mlir::Type translateVariableToFIRType(Fortran::lower::AbstractConverter &,
                                      const pft::Variable &variable);

/// Translate a REAL of KIND to the mlir::Type.
mlir::Type convertReal(mlir::MLIRContext *ctxt, int KIND);

bool isDerivedTypeWithLenParameters(const semantics::Symbol &);

template <typename T>
class TypeBuilder {
public:
  static mlir::Type genType(Fortran::lower::AbstractConverter &,
                            const Fortran::evaluate::FunctionRef<T> &);
};
using namespace evaluate;
FOR_EACH_SPECIFIC_TYPE(extern template class TypeBuilder, )

/// A helper class to reverse iterate through the component names of a derived
/// type, including the parent component and the component of the parents. This
/// is useful to deal with StructureConstructor lowering.
class ComponentReverseIterator {
public:
  ComponentReverseIterator(const Fortran::semantics::DerivedTypeSpec &derived) {
    setCurrentType(derived);
  }
  /// Does the current type has a component with \name (does not look-up the
  /// components of the parent if any). If there is a match, the iterator
  /// is advanced to the search result.
  bool lookup(const Fortran::parser::CharBlock &name) {
    componentIt = std::find(componentIt, componentItEnd, name);
    return componentIt != componentItEnd;
  };

  /// Advance iterator to the last components of the current type parent.
  const Fortran::semantics::DerivedTypeSpec &advanceToParentType();

private:
  void setCurrentType(const Fortran::semantics::DerivedTypeSpec &derived);
  const Fortran::semantics::DerivedTypeSpec *currentParentType = nullptr;
  const Fortran::semantics::DerivedTypeDetails *currentTypeDetails = nullptr;
  using name_iterator =
      std::list<Fortran::parser::CharBlock>::const_reverse_iterator;
  name_iterator componentIt{};
  name_iterator componentItEnd{};
};
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CONVERT_TYPE_H
