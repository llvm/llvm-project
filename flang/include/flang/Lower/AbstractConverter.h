//===-- Lower/AbstractConverter.h -------------------------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_ABSTRACTCONVERTER_H
#define FORTRAN_LOWER_ABSTRACTCONVERTER_H

#include "flang/Common/Fortran.h"
#include "flang/Lower/LoweringOptions.h"
#include "flang/Lower/PFTDefs.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Semantics/symbol.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"

namespace fir {
class KindMapping;
class FirOpBuilder;
} // namespace fir

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
class DerivedTypeSpec;
} // namespace semantics

namespace lower {
class SymMap;
namespace pft {
struct Variable;
}

using SomeExpr = Fortran::evaluate::Expr<Fortran::evaluate::SomeType>;
using SymbolRef = Fortran::common::Reference<const Fortran::semantics::Symbol>;
class StatementContext;

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

  virtual fir::ExtendedValue
  getSymbolExtendedValue(const Fortran::semantics::Symbol &sym,
                         Fortran::lower::SymMap *symMap = nullptr) = 0;

  /// Get the binding of an implied do variable by name.
  virtual mlir::Value impliedDoBinding(llvm::StringRef name) = 0;

  /// Copy the binding of src to target symbol.
  virtual void copySymbolBinding(SymbolRef src, SymbolRef target) = 0;

  /// Binds the symbol to an fir extended value. The symbol binding will be
  /// added or replaced at the inner-most level of the local symbol map.
  virtual void bindSymbol(SymbolRef sym, const fir::ExtendedValue &exval) = 0;

  /// Get the label set associated with a symbol.
  virtual bool lookupLabelSet(SymbolRef sym, pft::LabelSet &labelSet) = 0;

  /// Get the code defined by a label
  virtual pft::Evaluation *lookupLabel(pft::Label label) = 0;

  /// For a given symbol which is host-associated, create a clone using
  /// parameters from the host-associated symbol.
  virtual bool
  createHostAssociateVarClone(const Fortran::semantics::Symbol &sym) = 0;

  virtual void
  createHostAssociateVarCloneDealloc(const Fortran::semantics::Symbol &sym) = 0;

  virtual void copyHostAssociateVar(
      const Fortran::semantics::Symbol &sym,
      mlir::OpBuilder::InsertPoint *copyAssignIP = nullptr) = 0;

  /// Collect the set of symbols with \p flag in \p eval
  /// region if \p collectSymbols is true. Likewise, collect the
  /// set of the host symbols with \p flag of the associated symbols in \p eval
  /// region if collectHostAssociatedSymbols is true.
  virtual void collectSymbolSet(
      pft::Evaluation &eval,
      llvm::SetVector<const Fortran::semantics::Symbol *> &symbolSet,
      Fortran::semantics::Symbol::Flag flag, bool collectSymbols = true,
      bool collectHostAssociatedSymbols = false) = 0;

  /// For the given literal constant \p expression, returns a unique name
  /// that can be used to create a global object to represent this
  /// literal constant. It will return the same name for equivalent
  /// literal constant expressions. \p eleTy specifies the data type
  /// of the constant elements. For array constants it specifies
  /// the array's element type.
  virtual llvm::StringRef
  getUniqueLitName(mlir::Location loc,
                   std::unique_ptr<Fortran::lower::SomeExpr> expression,
                   mlir::Type eleTy) = 0;

  //===--------------------------------------------------------------------===//
  // Expressions
  //===--------------------------------------------------------------------===//

  /// Generate the address of the location holding the expression, \p expr.
  /// If \p expr is a Designator that is not compile time contiguous, the
  /// address returned is the one of a contiguous temporary storage holding the
  /// expression value. The clean-up for this temporary is added to \p context.
  virtual fir::ExtendedValue genExprAddr(const SomeExpr &expr,
                                         StatementContext &context,
                                         mlir::Location *locPtr = nullptr) = 0;

  /// Generate the address of the location holding the expression, \p expr.
  fir::ExtendedValue genExprAddr(mlir::Location loc, const SomeExpr *expr,
                                 StatementContext &stmtCtx) {
    return genExprAddr(*expr, stmtCtx, &loc);
  }
  fir::ExtendedValue genExprAddr(mlir::Location loc, const SomeExpr &expr,
                                 StatementContext &stmtCtx) {
    return genExprAddr(expr, stmtCtx, &loc);
  }

  /// Generate the computations of the expression to produce a value.
  virtual fir::ExtendedValue genExprValue(const SomeExpr &expr,
                                          StatementContext &context,
                                          mlir::Location *locPtr = nullptr) = 0;

  /// Generate the computations of the expression, \p expr, to produce a value.
  fir::ExtendedValue genExprValue(mlir::Location loc, const SomeExpr *expr,
                                  StatementContext &stmtCtx) {
    return genExprValue(*expr, stmtCtx, &loc);
  }
  fir::ExtendedValue genExprValue(mlir::Location loc, const SomeExpr &expr,
                                  StatementContext &stmtCtx) {
    return genExprValue(expr, stmtCtx, &loc);
  }

  /// Generate or get a fir.box describing the expression. If SomeExpr is
  /// a Designator, the fir.box describes an entity over the Designator base
  /// storage without making a temporary.
  virtual fir::ExtendedValue genExprBox(mlir::Location loc,
                                        const SomeExpr &expr,
                                        StatementContext &stmtCtx) = 0;

  /// Generate the address of the box describing the variable designated
  /// by the expression. The expression must be an allocatable or pointer
  /// designator.
  virtual fir::MutableBoxValue genExprMutableBox(mlir::Location loc,
                                                 const SomeExpr &expr) = 0;

  /// Get FoldingContext that is required for some expression
  /// analysis.
  virtual Fortran::evaluate::FoldingContext &getFoldingContext() = 0;

  /// Host associated variables are grouped as a tuple. This returns that value,
  /// which is itself a reference. Use bindTuple() to set this value.
  virtual mlir::Value hostAssocTupleValue() = 0;

  /// Record a binding for the ssa-value of the host assoications tuple for this
  /// function.
  virtual void bindHostAssocTuple(mlir::Value val) = 0;

  //===--------------------------------------------------------------------===//
  // Types
  //===--------------------------------------------------------------------===//

  /// Generate the type of an Expr
  virtual mlir::Type genType(const SomeExpr &) = 0;
  /// Generate the type of a Symbol
  virtual mlir::Type genType(SymbolRef) = 0;
  /// Generate the type from a category
  virtual mlir::Type genType(Fortran::common::TypeCategory tc) = 0;
  /// Generate the type from a category and kind and length parameters.
  virtual mlir::Type
  genType(Fortran::common::TypeCategory tc, int kind,
          llvm::ArrayRef<std::int64_t> lenParameters = std::nullopt) = 0;
  /// Generate the type from a DerivedTypeSpec.
  virtual mlir::Type genType(const Fortran::semantics::DerivedTypeSpec &) = 0;
  /// Generate the type from a Variable
  virtual mlir::Type genType(const pft::Variable &) = 0;

  /// Register a runtime derived type information object symbol to ensure its
  /// object will be generated as a global.
  virtual void registerRuntimeTypeInfo(mlir::Location loc,
                                       SymbolRef typeInfoSym) = 0;

  virtual void registerDispatchTableInfo(
      mlir::Location loc,
      const Fortran::semantics::DerivedTypeSpec *typeSpec) = 0;

  //===--------------------------------------------------------------------===//
  // Locations
  //===--------------------------------------------------------------------===//

  /// Get the converter's current location
  virtual mlir::Location getCurrentLocation() = 0;
  /// Generate a dummy location
  virtual mlir::Location genUnknownLocation() = 0;
  /// Generate the location as converted from a CharBlock
  virtual mlir::Location genLocation(const Fortran::parser::CharBlock &) = 0;

  /// Get the converter's current scope
  virtual const Fortran::semantics::Scope &getCurrentScope() = 0;

  //===--------------------------------------------------------------------===//
  // FIR/MLIR
  //===--------------------------------------------------------------------===//

  /// Get the OpBuilder
  virtual fir::FirOpBuilder &getFirOpBuilder() = 0;
  /// Get the ModuleOp
  virtual mlir::ModuleOp &getModuleOp() = 0;
  /// Get the MLIRContext
  virtual mlir::MLIRContext &getMLIRContext() = 0;
  /// Unique a symbol (add a containing scope specific prefix)
  virtual std::string mangleName(const Fortran::semantics::Symbol &) = 0;
  /// Unique a derived type (add a containing scope specific prefix)
  virtual std::string
  mangleName(const Fortran::semantics::DerivedTypeSpec &) = 0;
  /// Unique a compiler generated name (add a containing scope specific prefix)
  virtual std::string mangleName(std::string &) = 0;
  /// Get the KindMap.
  virtual const fir::KindMapping &getKindMap() = 0;

  virtual Fortran::lower::StatementContext &getFctCtx() = 0;

  AbstractConverter(const Fortran::lower::LoweringOptions &loweringOptions)
      : loweringOptions(loweringOptions) {}
  virtual ~AbstractConverter() = default;

  //===--------------------------------------------------------------------===//
  // Miscellaneous
  //===--------------------------------------------------------------------===//

  /// Return options controlling lowering behavior.
  const Fortran::lower::LoweringOptions &getLoweringOptions() const {
    return loweringOptions;
  }

private:
  /// Options controlling lowering behavior.
  const Fortran::lower::LoweringOptions &loweringOptions;
};

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_ABSTRACTCONVERTER_H
