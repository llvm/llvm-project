//===-- ConvertVariable.cpp -- bridge to lower to MLIR --------------------===//
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

#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-variable"

//===----------------------------------------------------------------===//
// Local variables instantiation (not for alias)
//===----------------------------------------------------------------===//

/// Create a stack slot for a local variable. Precondition: the insertion
/// point of the builder must be in the entry block, which is currently being
/// constructed.
static mlir::Value createNewLocal(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc,
                                  const Fortran::lower::pft::Variable &var,
                                  mlir::Value preAlloc,
                                  llvm::ArrayRef<mlir::Value> shape = {},
                                  llvm::ArrayRef<mlir::Value> lenParams = {}) {
  if (preAlloc)
    return preAlloc;
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  std::string nm = Fortran::lower::mangle::mangleName(var.getSymbol());
  mlir::Type ty = converter.genType(var);
  const Fortran::semantics::Symbol &ultimateSymbol =
      var.getSymbol().GetUltimate();
  llvm::StringRef symNm = toStringRef(ultimateSymbol.name());
  bool isTarg = var.isTarget();
  // Let the builder do all the heavy lifting.
  return builder.allocateLocal(loc, ty, nm, symNm, shape, lenParams, isTarg);
}

/// Instantiate a local variable. Precondition: Each variable will be visited
/// such that if its properties depend on other variables, the variables upon
/// which its properties depend will already have been visited.
static void instantiateLocal(Fortran::lower::AbstractConverter &converter,
                             const Fortran::lower::pft::Variable &var,
                             Fortran::lower::SymMap &symMap) {
  assert(!var.isAlias());
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  const bool isDummy = Fortran::semantics::IsDummy(sym);
  const bool isResult = Fortran::semantics::IsFunctionResult(sym);
  if (symMap.lookupSymbol(sym))
    return;

  const mlir::Location loc = converter.genLocation(sym.name());
  if (isDummy) {
    // This is an argument.
    if (!symMap.lookupSymbol(sym))
      mlir::emitError(loc, "symbol \"")
          << toStringRef(sym.name()) << "\" must already be in map";
    return;
  } else if (isResult) {
    // Some Fortran results may be passed by argument (e.g. derived
    // types)
    if (symMap.lookupSymbol(sym))
      return;
  }
  // Otherwise, it's a local variable or function result.
  mlir::Value local = createNewLocal(converter, loc, var, {});
  symMap.addSymbol(sym, local);
}

void Fortran::lower::instantiateVariable(AbstractConverter &converter,
                                         const pft::Variable &var,
                                         SymMap &symMap) {
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  const mlir::Location loc = converter.genLocation(sym.name());
  if (var.isAggregateStore()) {
    TODO(loc, "instantiateVariable AggregateStore");
  } else if (Fortran::semantics::FindCommonBlockContaining(
                 var.getSymbol().GetUltimate())) {
    TODO(loc, "instantiateVariable Common");
  } else if (var.isAlias()) {
    TODO(loc, "instantiateVariable Alias");
  } else if (var.isGlobal()) {
    TODO(loc, "instantiateVariable Global");
  } else {
    instantiateLocal(converter, var, symMap);
  }
}
