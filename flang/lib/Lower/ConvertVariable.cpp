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

void Fortran::lower::mapCallInterfaceSymbols(
    AbstractConverter &converter, const Fortran::lower::CallerInterface &caller,
    SymMap &symMap) {
  const Fortran::semantics::Symbol &result = caller.getResultSymbol();
  for (Fortran::lower::pft::Variable var :
       Fortran::lower::pft::buildFuncResultDependencyList(result)) {
    if (var.isAggregateStore()) {
      instantiateVariable(converter, var, symMap);
    } else {
      const Fortran::semantics::Symbol &sym = var.getSymbol();
      const auto *hostDetails =
          sym.detailsIf<Fortran::semantics::HostAssocDetails>();
      if (hostDetails && !var.isModuleVariable()) {
        // The callee is an internal procedure `A` whose result properties
        // depend on host variables. The caller may be the host, or another
        // internal procedure `B` contained in the same host.  In the first
        // case, the host symbol is obviously mapped, in the second case, it
        // must also be mapped because
        // HostAssociations::internalProcedureBindings that was called when
        // lowering `B` will have mapped all host symbols of captured variables
        // to the tuple argument containing the composite of all host associated
        // variables, whether or not the host symbol is actually referred to in
        // `B`. Hence it is possible to simply lookup the variable associated to
        // the host symbol without having to go back to the tuple argument.
        Fortran::lower::SymbolBox hostValue =
            symMap.lookupSymbol(hostDetails->symbol());
        assert(hostValue && "callee host symbol must be mapped on caller side");
        symMap.addSymbol(sym, hostValue.toExtendedValue());
        // The SymbolBox associated to the host symbols is complete, skip
        // instantiateVariable that would try to allocate a new storage.
        continue;
      }
      if (Fortran::semantics::IsDummy(sym) && sym.owner() == result.owner()) {
        // Get the argument for the dummy argument symbols of the current call.
        symMap.addSymbol(sym, caller.getArgumentValue(sym));
        // All the properties of the dummy variable may not come from the actual
        // argument, let instantiateVariable handle this.
      }
      // If this is neither a host associated or dummy symbol, it must be a
      // module or common block variable to satisfy specification expression
      // requirements in 10.1.11, instantiateVariable will get its address and
      // properties.
      instantiateVariable(converter, var, symMap);
    }
  }
}
