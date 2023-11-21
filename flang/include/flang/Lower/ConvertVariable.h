//===- Lower/ConvertVariable.h -- lowering of variables to FIR --*- C++ -*-===//
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
/// Instantiation of pft::Variable in FIR/MLIR.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERT_VARIABLE_H
#define FORTRAN_LOWER_CONVERT_VARIABLE_H

#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace fir {
class ExtendedValue;
class FirOpBuilder;
class GlobalOp;
class FortranVariableFlagsAttr;
} // namespace fir

namespace Fortran ::lower {
class AbstractConverter;
class CallerInterface;
class StatementContext;
class SymMap;
namespace pft {
struct Variable;
}

/// AggregateStoreMap is used to keep track of instantiated aggregate stores
/// when lowering a scope containing equivalences (aliases). It must only be
/// owned by the code lowering a scope and provided to instantiateVariable.
using AggregateStoreKey =
    std::tuple<const Fortran::semantics::Scope *, std::size_t>;
using AggregateStoreMap = llvm::DenseMap<AggregateStoreKey, mlir::Value>;

/// Instantiate variable \p var and add it to \p symMap.
/// The AbstractConverter builder must be set.
/// The AbstractConverter own symbol mapping is not used during the
/// instantiation and can be different form \p symMap.
void instantiateVariable(AbstractConverter &, const pft::Variable &var,
                         SymMap &symMap, AggregateStoreMap &storeMap);

/// Create a fir::GlobalOp given a module variable definition. This is intended
/// to be used when lowering a module definition, not when lowering variables
/// used from a module. For used variables instantiateVariable must directly be
/// called.
void defineModuleVariable(AbstractConverter &, const pft::Variable &var);

/// Create fir::GlobalOp for all common blocks, including their initial values
/// if they have one. This should be called before lowering any scopes so that
/// common block globals are available when a common appear in a scope.
void defineCommonBlocks(
    AbstractConverter &,
    const std::vector<std::pair<semantics::SymbolRef, std::size_t>>
        &commonBlocks);

/// Lower a symbol attributes given an optional storage \p and add it to the
/// provided symbol map. If \preAlloc is not provided, a temporary storage will
/// be allocated. This is a low level function that should only be used if
/// instantiateVariable cannot be called.
void mapSymbolAttributes(AbstractConverter &, const pft::Variable &, SymMap &,
                         StatementContext &, mlir::Value preAlloc = {});
void mapSymbolAttributes(AbstractConverter &, const semantics::SymbolRef &,
                         SymMap &, StatementContext &,
                         mlir::Value preAlloc = {});

/// Instantiate the variables that appear in the specification expressions
/// of the result of a function call. The instantiated variables are added
/// to \p symMap.
void mapCallInterfaceSymbols(AbstractConverter &,
                             const Fortran::lower::CallerInterface &caller,
                             SymMap &symMap);

// TODO: consider saving the initial expression symbol dependence analysis in
// in the PFT variable and dealing with the dependent symbols instantiation in
// the fir::GlobalOp body at the fir::GlobalOp creation point rather than by
// having genExtAddrInInitializer and genInitialDataTarget custom entry points
// here to deal with this while lowering the initial expression value.

/// Create initial-data-target fir.box in a global initializer region.
/// This handles the local instantiation of the target variable.
mlir::Value genInitialDataTarget(Fortran::lower::AbstractConverter &,
                                 mlir::Location, mlir::Type boxType,
                                 const SomeExpr &initialTarget,
                                 bool couldBeInEquivalence = false);

/// Call \p genInit to generate code inside \p global initializer region.
void createGlobalInitialization(
    fir::FirOpBuilder &builder, fir::GlobalOp global,
    std::function<void(fir::FirOpBuilder &)> genInit);

/// Generate address \p addr inside an initializer.
fir::ExtendedValue
genExtAddrInInitializer(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc, const SomeExpr &addr);

/// Create global variable from a compiler generated object symbol that
/// describes a derived type for the runtime.
void createRuntimeTypeInfoGlobal(Fortran::lower::AbstractConverter &converter,
                                 mlir::Location loc,
                                 const Fortran::semantics::Symbol &typeInfoSym);

/// Translate the Fortran attributes of \p sym into the FIR variable attribute
/// representation.
fir::FortranVariableFlagsAttr
translateSymbolAttributes(mlir::MLIRContext *mlirContext,
                          const Fortran::semantics::Symbol &sym,
                          fir::FortranVariableFlagsEnum extraFlags =
                              fir::FortranVariableFlagsEnum::None);

/// Map a symbol to a given fir::ExtendedValue. This will generate an
/// hlfir.declare when lowering to HLFIR and map the hlfir.declare result to the
/// symbol.
void genDeclareSymbol(Fortran::lower::AbstractConverter &converter,
                      Fortran::lower::SymMap &symMap,
                      const Fortran::semantics::Symbol &sym,
                      const fir::ExtendedValue &exv,
                      fir::FortranVariableFlagsEnum extraFlags =
                          fir::FortranVariableFlagsEnum::None,
                      bool force = false);

/// For the given Cray pointee symbol return the corresponding
/// Cray pointer symbol. Assert if the pointer symbol cannot be found.
Fortran::semantics::SymbolRef getCrayPointer(Fortran::semantics::SymbolRef sym);

} // namespace Fortran::lower
#endif // FORTRAN_LOWER_CONVERT_VARIABLE_H
