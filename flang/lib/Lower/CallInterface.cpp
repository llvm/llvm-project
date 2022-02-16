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
#include "flang/Lower/Todo.h"
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

bool Fortran::lower::CalleeInterface::hasAlternateReturns() const {
  return !funit.isMainProgram() &&
         Fortran::semantics::HasAlternateReturns(funit.getSubprogramSymbol());
}

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

Fortran::evaluate::characteristics::Procedure
Fortran::lower::CalleeInterface::characterize() const {
  Fortran::evaluate::FoldingContext &foldingContext =
      converter.getFoldingContext();
  std::optional<Fortran::evaluate::characteristics::Procedure> characteristic =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          funit.getSubprogramSymbol(), foldingContext);
  assert(characteristic && "Fail to get characteristic from symbol");
  return *characteristic;
}

bool Fortran::lower::CalleeInterface::isMainProgram() const {
  return funit.isMainProgram();
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
  if (!side().isMainProgram()) {
    characteristic.emplace(side().characterize());
    bool isImplicit = characteristic->CanBeCalledViaImplicitInterface();
    determineInterface(isImplicit, *characteristic);
  }
  // No input/output for main program

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

//===----------------------------------------------------------------------===//
// CallInterface implementation: this part is common to both caller and caller
// sides.
//===----------------------------------------------------------------------===//

/// This is the actual part that defines the FIR interface based on the
/// characteristic. It directly mutates the CallInterface members.
template <typename T>
class Fortran::lower::CallInterfaceImpl {
  using CallInterface = Fortran::lower::CallInterface<T>;
  using FirPlaceHolder = typename CallInterface::FirPlaceHolder;
  using Property = typename CallInterface::Property;
  using TypeAndShape = Fortran::evaluate::characteristics::TypeAndShape;

public:
  CallInterfaceImpl(CallInterface &i)
      : interface(i), mlirContext{i.converter.getMLIRContext()} {}

  void buildImplicitInterface(
      const Fortran::evaluate::characteristics::Procedure &procedure) {
    // Handle result
    if (const std::optional<Fortran::evaluate::characteristics::FunctionResult>
            &result = procedure.functionResult)
      handleImplicitResult(*result);
    else if (interface.side().hasAlternateReturns())
      addFirResult(mlir::IndexType::get(&mlirContext),
                   FirPlaceHolder::resultEntityPosition, Property::Value);
  }

  void buildExplicitInterface(
      const Fortran::evaluate::characteristics::Procedure &procedure) {
    // Handle result
    if (const std::optional<Fortran::evaluate::characteristics::FunctionResult>
            &result = procedure.functionResult) {
      if (result->CanBeReturnedViaImplicitInterface())
        handleImplicitResult(*result);
      else
        handleExplicitResult(*result);
    } else if (interface.side().hasAlternateReturns()) {
      addFirResult(mlir::IndexType::get(&mlirContext),
                   FirPlaceHolder::resultEntityPosition, Property::Value);
    }
  }

private:
  void handleImplicitResult(
      const Fortran::evaluate::characteristics::FunctionResult &result) {
    if (result.IsProcedurePointer())
      TODO(interface.converter.getCurrentLocation(),
           "procedure pointer result not yet handled");
    const Fortran::evaluate::characteristics::TypeAndShape *typeAndShape =
        result.GetTypeAndShape();
    assert(typeAndShape && "expect type for non proc pointer result");
    Fortran::evaluate::DynamicType dynamicType = typeAndShape->type();
    if (dynamicType.category() == Fortran::common::TypeCategory::Character) {
      TODO(interface.converter.getCurrentLocation(),
           "implicit result character type");
    } else if (dynamicType.category() ==
               Fortran::common::TypeCategory::Derived) {
      TODO(interface.converter.getCurrentLocation(),
           "implicit result derived type");
    } else {
      // All result other than characters/derived are simply returned by value
      // in implicit interfaces
      mlir::Type mlirType =
          getConverter().genType(dynamicType.category(), dynamicType.kind());
      addFirResult(mlirType, FirPlaceHolder::resultEntityPosition,
                   Property::Value);
    }
  }

  void handleExplicitResult(
      const Fortran::evaluate::characteristics::FunctionResult &result) {
    using Attr = Fortran::evaluate::characteristics::FunctionResult::Attr;

    if (result.IsProcedurePointer())
      TODO(interface.converter.getCurrentLocation(),
           "procedure pointer results");
    const Fortran::evaluate::characteristics::TypeAndShape *typeAndShape =
        result.GetTypeAndShape();
    assert(typeAndShape && "expect type for non proc pointer result");
    Fortran::evaluate::DynamicType dynamicType = typeAndShape->type();
    if (dynamicType.category() == Fortran::common::TypeCategory::Character) {
      TODO(interface.converter.getCurrentLocation(),
           "implicit result character type");
    } else if (dynamicType.category() ==
               Fortran::common::TypeCategory::Derived) {
      TODO(interface.converter.getCurrentLocation(),
           "implicit result derived type");
    }
    mlir::Type mlirType =
        getConverter().genType(dynamicType.category(), dynamicType.kind());
    fir::SequenceType::Shape bounds = getBounds(typeAndShape->shape());
    if (!bounds.empty())
      mlirType = fir::SequenceType::get(bounds, mlirType);
    if (result.attrs.test(Attr::Allocatable))
      mlirType = fir::BoxType::get(fir::HeapType::get(mlirType));
    if (result.attrs.test(Attr::Pointer))
      mlirType = fir::BoxType::get(fir::PointerType::get(mlirType));

    addFirResult(mlirType, FirPlaceHolder::resultEntityPosition,
                 Property::Value);
  }

  fir::SequenceType::Shape getBounds(const Fortran::evaluate::Shape &shape) {
    fir::SequenceType::Shape bounds;
    for (Fortran::evaluate::MaybeExtentExpr extentExpr : shape) {
      fir::SequenceType::Extent extent = fir::SequenceType::getUnknownExtent();
      if (std::optional<std::int64_t> constantExtent =
              toInt64(std::move(extentExpr)))
        extent = *constantExtent;
      bounds.push_back(extent);
    }
    return bounds;
  }

  template <typename A>
  std::optional<std::int64_t> toInt64(A &&expr) {
    return Fortran::evaluate::ToInt64(Fortran::evaluate::Fold(
        getConverter().getFoldingContext(), std::move(expr)));
  }

  void addFirResult(mlir::Type type, int entityPosition, Property p) {
    interface.outputs.emplace_back(FirPlaceHolder{type, entityPosition, p});
  }

  Fortran::lower::AbstractConverter &getConverter() {
    return interface.converter;
  }
  CallInterface &interface;
  mlir::MLIRContext &mlirContext;
};

template <typename T>
void Fortran::lower::CallInterface<T>::determineInterface(
    bool isImplicit,
    const Fortran::evaluate::characteristics::Procedure &procedure) {
  CallInterfaceImpl<T> impl(*this);
  if (isImplicit)
    impl.buildImplicitInterface(procedure);
  else
    impl.buildExplicitInterface(procedure);
}

template <typename T>
mlir::FunctionType Fortran::lower::CallInterface<T>::genFunctionType() {
  llvm::SmallVector<mlir::Type> returnTys;
  for (const FirPlaceHolder &placeHolder : outputs)
    returnTys.emplace_back(placeHolder.type);
  return mlir::FunctionType::get(&converter.getMLIRContext(), {}, returnTys);
}

template class Fortran::lower::CallInterface<Fortran::lower::CalleeInterface>;
