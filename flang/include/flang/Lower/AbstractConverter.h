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

#include "flang/Lower/LoweringOptions.h"
#include "flang/Lower/PFTDefs.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Semantics/symbol.h"
#include "flang/Support/Fortran.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class SymbolTable;
}

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
class Scope;
class DerivedTypeSpec;
} // namespace semantics

namespace lower {
class SymMap;
struct SymbolBox;
namespace pft {
struct Variable;
struct FunctionLikeUnit;
} // namespace pft

using SomeExpr = Fortran::evaluate::Expr<Fortran::evaluate::SomeType>;
using SymbolRef = Fortran::common::Reference<const Fortran::semantics::Symbol>;
using TypeConstructionStack =
    llvm::DenseMap<const Fortran::semantics::Scope *, mlir::Type>;
class StatementContext;

using ExprToValueMap = llvm::DenseMap<const SomeExpr *, mlir::Value>;

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
  symBoxToExtendedValue(const Fortran::lower::SymbolBox &symBox) = 0;

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

  /// Override lowering of expression with pre-lowered values.
  /// Associate mlir::Value to evaluate::Expr. All subsequent call to
  /// genExprXXX() will replace any occurrence of an overridden
  /// expression in the expression tree by the pre-lowered values.
  virtual void overrideExprValues(const ExprToValueMap *) = 0;
  void resetExprOverrides() { overrideExprValues(nullptr); }
  virtual const ExprToValueMap *getExprOverrides() = 0;

  /// Get the label set associated with a symbol.
  virtual bool lookupLabelSet(SymbolRef sym, pft::LabelSet &labelSet) = 0;

  /// Get the code defined by a label
  virtual pft::Evaluation *lookupLabel(pft::Label label) = 0;

  /// For a given symbol which is host-associated, create a clone using
  /// parameters from the host-associated symbol.
  /// The clone is default initialized if its type has any default
  /// initialization unless `skipDefaultInit` is set.
  virtual bool
  createHostAssociateVarClone(const Fortran::semantics::Symbol &sym,
                              bool skipDefaultInit) = 0;

  virtual void
  createHostAssociateVarCloneDealloc(const Fortran::semantics::Symbol &sym) = 0;

  /// For a host-associated symbol (a symbol associated with another symbol from
  /// an enclosing scope), either:
  ///
  /// * if \p hostIsSource == true: copy \p sym's value *from* its corresponding
  /// host symbol,
  ///
  /// * if \p hostIsSource == false: copy \p sym's value *to* its corresponding
  /// host symbol.
  virtual void
  copyHostAssociateVar(const Fortran::semantics::Symbol &sym,
                       mlir::OpBuilder::InsertPoint *copyAssignIP = nullptr,
                       bool hostIsSource = true) = 0;

  virtual void copyVar(mlir::Location loc, mlir::Value dst, mlir::Value src,
                       fir::FortranVariableFlagsEnum attrs) = 0;

  /// For a given symbol, check if it is present in the inner-most
  /// level of the symbol map.
  virtual bool
  isPresentShallowLookup(const Fortran::semantics::Symbol &sym) = 0;

  /// Collect the set of symbols with \p flag in \p eval
  /// region if \p collectSymbols is true. Otherwise, collect the
  /// set of the host symbols with \p flag of the associated symbols in \p eval
  /// region if collectHostAssociatedSymbols is true. This allows gathering
  /// host association details of symbols particularly in nested directives
  /// irrespective of \p flag \p, and can be useful where host
  /// association details are needed in flag-agnostic manner.
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

  /// Returns fir.dummy_scope operation's result value to be used
  /// as dummy_scope operand of hlfir.declare operations for the dummy
  /// arguments of this function.
  virtual mlir::Value dummyArgsScopeValue() const = 0;

  /// Returns true if the given symbol is a dummy argument of this function.
  /// Note that it returns false for all the symbols after all the variables
  /// are instantiated for this function, i.e. it can only be used reliably
  /// during the instatiation of the variables.
  virtual bool
  isRegisteredDummySymbol(Fortran::semantics::SymbolRef symRef) const = 0;

  /// Returns the FunctionLikeUnit being lowered, if any.
  virtual const Fortran::lower::pft::FunctionLikeUnit *
  getCurrentFunctionUnit() const = 0;

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
  virtual void
  registerTypeInfo(mlir::Location loc, SymbolRef typeInfoSym,
                   const Fortran::semantics::DerivedTypeSpec &typeSpec,
                   fir::RecordType type) = 0;

  /// Get stack of derived type in construction. This is an internal entry point
  /// for the type conversion utility to allow lowering recursive derived types.
  virtual TypeConstructionStack &getTypeConstructionStack() = 0;

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
  virtual mlir::ModuleOp getModuleOp() = 0;
  /// Get the MLIRContext
  virtual mlir::MLIRContext &getMLIRContext() = 0;
  /// Unique a symbol (add a containing scope specific prefix)
  virtual std::string mangleName(const Fortran::semantics::Symbol &) = 0;
  /// Unique a derived type (add a containing scope specific prefix)
  virtual std::string
  mangleName(const Fortran::semantics::DerivedTypeSpec &) = 0;
  /// Unique a compiler generated name (add a containing scope specific prefix)
  virtual std::string mangleName(std::string &) = 0;
  /// Unique a compiler generated name (add a provided scope specific prefix)
  virtual std::string mangleName(std::string &, const semantics::Scope &) = 0;
  /// Return the field name for a derived type component inside a fir.record
  /// type.
  virtual std::string
  getRecordTypeFieldName(const Fortran::semantics::Symbol &component) = 0;

  /// Get the KindMap.
  virtual const fir::KindMapping &getKindMap() = 0;

  virtual Fortran::lower::StatementContext &getFctCtx() = 0;

  AbstractConverter(const Fortran::lower::LoweringOptions &loweringOptions)
      : loweringOptions(loweringOptions) {}
  virtual ~AbstractConverter() = default;

  //===--------------------------------------------------------------------===//
  // Miscellaneous
  //===--------------------------------------------------------------------===//

  /// Generate IR for Evaluation \p eval.
  virtual void genEval(pft::Evaluation &eval,
                       bool unstructuredContext = true) = 0;

  /// Return options controlling lowering behavior.
  const Fortran::lower::LoweringOptions &getLoweringOptions() const {
    return loweringOptions;
  }

  /// Find the symbol in one level up of symbol map such as for host-association
  /// in OpenMP code or return null.
  virtual Fortran::lower::SymbolBox
  lookupOneLevelUpSymbol(const Fortran::semantics::Symbol &sym) = 0;

  /// Return the mlir::SymbolTable associated to the ModuleOp.
  /// Look-ups are faster using it than using module.lookup<>,
  /// but the module op should be queried in case of failure
  /// because this symbol table is not guaranteed to contain
  /// all the symbols from the ModuleOp (the symbol table should
  /// always be provided to the builder helper creating globals and
  /// functions in order to be in sync).
  virtual mlir::SymbolTable *getMLIRSymbolTable() = 0;

private:
  /// Options controlling lowering behavior.
  const Fortran::lower::LoweringOptions &loweringOptions;
};

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_ABSTRACTCONVERTER_H
