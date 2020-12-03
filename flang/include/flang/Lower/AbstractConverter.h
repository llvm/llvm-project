//===-- Lower/AbstractConverter.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_ABSTRACTCONVERTER_H
#define FORTRAN_LOWER_ABSTRACTCONVERTER_H

#include "flang/Common/Fortran.h"
#include "mlir/IR/BuiltinOps.h"

namespace Fortran {
namespace common {
template <typename>
class Reference;
}
namespace evaluate {
struct DataRef;
template <typename>
class Expr;
class FoldingContext;
struct SomeType;
} // namespace evaluate

namespace parser {
class CharBlock;
}
namespace semantics {
class Symbol;
}

namespace lower {
namespace pft {
struct Variable;
}

using SomeExpr = Fortran::evaluate::Expr<Fortran::evaluate::SomeType>;
using SymbolRef = Fortran::common::Reference<const Fortran::semantics::Symbol>;
class FirOpBuilder;

//===----------------------------------------------------------------------===//
// AbstractConverter interface
//===----------------------------------------------------------------------===//

/// The abstract interface for converter implementations to lower Fortran
/// front-end fragments such as expressions, types, etc. to the FIR dialect of
/// MLIR.
class AbstractConverter {
public:
  //===--------------------------------------------------------------------===//
  // Symbols
  //===--------------------------------------------------------------------===//

  /// Get the mlir instance of a symbol.
  virtual mlir::Value getSymbolAddress(SymbolRef sym) = 0;

  //===--------------------------------------------------------------------===//
  // Expressions
  //===--------------------------------------------------------------------===//

  /// Generate the address of the location holding the expression, someExpr
  virtual mlir::Value genExprAddr(const SomeExpr &,
                                  mlir::Location *loc = nullptr) = 0;
  /// Generate the address of the location holding the expression, someExpr
  mlir::Value genExprAddr(const SomeExpr *someExpr, mlir::Location loc) {
    return genExprAddr(*someExpr, &loc);
  }

  /// Generate the computations of the expression to produce a value
  virtual mlir::Value genExprValue(const SomeExpr &,
                                   mlir::Location *loc = nullptr) = 0;
  /// Generate the computations of the expression, someExpr, to produce a value
  mlir::Value genExprValue(const SomeExpr *someExpr, mlir::Location loc) {
    return genExprValue(*someExpr, &loc);
  }

  /// Get FoldingContext that is required for some expression
  /// analysis.
  virtual Fortran::evaluate::FoldingContext &getFoldingContext() = 0;

  //===--------------------------------------------------------------------===//
  // Types
  //===--------------------------------------------------------------------===//

  /// Generate the type of a DataRef
  virtual mlir::Type genType(const Fortran::evaluate::DataRef &) = 0;
  /// Generate the type of an Expr
  virtual mlir::Type genType(const SomeExpr &) = 0;
  /// Generate the type of a Symbol
  virtual mlir::Type genType(SymbolRef) = 0;
  /// Generate the type from a category
  virtual mlir::Type genType(Fortran::common::TypeCategory tc) = 0;
  /// Generate the type from a category and kind
  virtual mlir::Type genType(Fortran::common::TypeCategory tc, int kind) = 0;
  /// Generate the type from a Variable
  virtual mlir::Type genType(const pft::Variable &) = 0;

  //===--------------------------------------------------------------------===//
  // Locations
  //===--------------------------------------------------------------------===//

  /// Get the converter's current location
  virtual mlir::Location getCurrentLocation() = 0;
  /// Generate a dummy location
  virtual mlir::Location genLocation() = 0;
  /// Generate the location as converted from a CharBlock
  virtual mlir::Location genLocation(const Fortran::parser::CharBlock &) = 0;

  //===--------------------------------------------------------------------===//
  // FIR/MLIR
  //===--------------------------------------------------------------------===//

  /// Get the OpBuilder
  virtual Fortran::lower::FirOpBuilder &getFirOpBuilder() = 0;
  /// Get the ModuleOp
  virtual mlir::ModuleOp &getModuleOp() = 0;
  /// Get the MLIRContext
  virtual mlir::MLIRContext &getMLIRContext() = 0;
  /// Unique a symbol
  virtual std::string mangleName(const Fortran::semantics::Symbol &) = 0;
  /// Unique a compiler generated identifier. A short prefix should be provided
  /// to hint at the origin of the identifier.
  virtual std::string uniqueCGIdent(llvm::StringRef prefix,
                                    llvm::StringRef name) = 0;

  virtual ~AbstractConverter() = default;
};

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_ABSTRACTCONVERTER_H
