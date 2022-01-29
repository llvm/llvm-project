//===-- CallInterface.cpp -- Procedure call interface ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/CallInterface.h"
#include "flang/Evaluate/fold.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"

//===----------------------------------------------------------------------===//
// BIND(C) mangling helpers
//===----------------------------------------------------------------------===//

// Return the binding label (from BIND(C...)) or the mangled name of a symbol.
static std::string getMangledName(const Fortran::semantics::Symbol &symbol) {
  const std::string *bindName = symbol.GetBindName();
  return bindName ? *bindName : Fortran::lower::mangle::mangleName(symbol);
}

//===----------------------------------------------------------------------===//
// Callee side interface implementation
//===----------------------------------------------------------------------===//

std::string Fortran::lower::CalleeInterface::getMangledName() const {
  if (funit.isMainProgram())
    return fir::NameUniquer::doProgramEntry().str();
  return ::getMangledName(funit.getSubprogramSymbol());
}

const Fortran::semantics::Symbol *
Fortran::lower::CalleeInterface::getProcedureSymbol() const {
  if (funit.isMainProgram())
    return nullptr;
  return &funit.getSubprogramSymbol();
}

mlir::Location Fortran::lower::CalleeInterface::getCalleeLocation() const {
  // FIXME: do NOT use unknown for the anonymous PROGRAM case. We probably
  // should just stash the location in the funit regardless.
  return converter.genLocation(funit.getStartingSourceLoc());
}

mlir::FuncOp Fortran::lower::CalleeInterface::addEntryBlockAndMapArguments() {
  // On the callee side, directly map the mlir::value argument of
  // the function block to the Fortran symbols.
  func.addEntryBlock();
  return func;
}

//===----------------------------------------------------------------------===//
// CallInterface implementation: this part is common to both callee and caller
// sides.
//===----------------------------------------------------------------------===//

static void addSymbolAttribute(mlir::FuncOp func,
                               const Fortran::semantics::Symbol &sym,
                               mlir::MLIRContext &mlirContext) {
  // Only add this on bind(C) functions for which the symbol is not reflected in
  // the current context.
  if (!Fortran::semantics::IsBindCProcedure(sym))
    return;
  std::string name =
      Fortran::lower::mangle::mangleName(sym, /*keepExternalInScope=*/true);
  func->setAttr(fir::getSymbolAttrName(),
                mlir::StringAttr::get(&mlirContext, name));
}

/// Declare drives the different actions to be performed while analyzing the
/// signature and building/finding the mlir::FuncOp.
template <typename T>
void Fortran::lower::CallInterface<T>::declare() {
  // Create / get funcOp for direct calls. For indirect calls (only meaningful
  // on the caller side), no funcOp has to be created here. The mlir::Value
  // holding the indirection is used when creating the fir::CallOp.
  if (!side().isIndirectCall()) {
    std::string name = side().getMangledName();
    mlir::ModuleOp module = converter.getModuleOp();
    func = fir::FirOpBuilder::getNamedFunction(module, name);
    if (!func) {
      mlir::Location loc = side().getCalleeLocation();
      mlir::FunctionType ty = genFunctionType();
      func = fir::FirOpBuilder::createFunction(loc, module, name, ty);
      if (const Fortran::semantics::Symbol *sym = side().getProcedureSymbol())
        addSymbolAttribute(func, *sym, converter.getMLIRContext());
    }
  }
}

template <typename T>
mlir::FunctionType Fortran::lower::CallInterface<T>::genFunctionType() {
  return mlir::FunctionType::get(&converter.getMLIRContext(), {}, {});
}

template class Fortran::lower::CallInterface<Fortran::lower::CalleeInterface>;
