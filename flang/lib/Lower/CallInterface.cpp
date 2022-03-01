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
// Caller side interface implementation
//===----------------------------------------------------------------------===//

bool Fortran::lower::CallerInterface::hasAlternateReturns() const {
  return procRef.hasAlternateReturns();
}

std::string Fortran::lower::CallerInterface::getMangledName() const {
  const Fortran::evaluate::ProcedureDesignator &proc = procRef.proc();
  if (const Fortran::semantics::Symbol *symbol = proc.GetSymbol())
    return ::getMangledName(symbol->GetUltimate());
  assert(proc.GetSpecificIntrinsic() &&
         "expected intrinsic procedure in designator");
  return proc.GetName();
}

const Fortran::semantics::Symbol *
Fortran::lower::CallerInterface::getProcedureSymbol() const {
  return procRef.proc().GetSymbol();
}

bool Fortran::lower::CallerInterface::isIndirectCall() const {
  if (const Fortran::semantics::Symbol *symbol = procRef.proc().GetSymbol())
    return Fortran::semantics::IsPointer(*symbol) ||
           Fortran::semantics::IsDummy(*symbol);
  return false;
}

const Fortran::semantics::Symbol *
Fortran::lower::CallerInterface::getIfIndirectCallSymbol() const {
  if (const Fortran::semantics::Symbol *symbol = procRef.proc().GetSymbol())
    if (Fortran::semantics::IsPointer(*symbol) ||
        Fortran::semantics::IsDummy(*symbol))
      return symbol;
  return nullptr;
}

mlir::Location Fortran::lower::CallerInterface::getCalleeLocation() const {
  const Fortran::evaluate::ProcedureDesignator &proc = procRef.proc();
  // FIXME: If the callee is defined in the same file but after the current
  // unit we cannot get its location here and the funcOp is created at the
  // wrong location (i.e, the caller location).
  if (const Fortran::semantics::Symbol *symbol = proc.GetSymbol())
    return converter.genLocation(symbol->name());
  // Use current location for intrinsics.
  return converter.getCurrentLocation();
}

// Get dummy argument characteristic for a procedure with implicit interface
// from the actual argument characteristic. The actual argument may not be a F77
// entity. The attribute must be dropped and the shape, if any, must be made
// explicit.
static Fortran::evaluate::characteristics::DummyDataObject
asImplicitArg(Fortran::evaluate::characteristics::DummyDataObject &&dummy) {
  Fortran::evaluate::Shape shape =
      dummy.type.attrs().none() ? dummy.type.shape()
                                : Fortran::evaluate::Shape(dummy.type.Rank());
  return Fortran::evaluate::characteristics::DummyDataObject(
      Fortran::evaluate::characteristics::TypeAndShape(dummy.type.type(),
                                                       std::move(shape)));
}

static Fortran::evaluate::characteristics::DummyArgument
asImplicitArg(Fortran::evaluate::characteristics::DummyArgument &&dummy) {
  return std::visit(
      Fortran::common::visitors{
          [&](Fortran::evaluate::characteristics::DummyDataObject &obj) {
            return Fortran::evaluate::characteristics::DummyArgument(
                std::move(dummy.name), asImplicitArg(std::move(obj)));
          },
          [&](Fortran::evaluate::characteristics::DummyProcedure &proc) {
            return Fortran::evaluate::characteristics::DummyArgument(
                std::move(dummy.name), std::move(proc));
          },
          [](Fortran::evaluate::characteristics::AlternateReturn &x) {
            return Fortran::evaluate::characteristics::DummyArgument(
                std::move(x));
          }},
      dummy.u);
}

Fortran::evaluate::characteristics::Procedure
Fortran::lower::CallerInterface::characterize() const {
  Fortran::evaluate::FoldingContext &foldingContext =
      converter.getFoldingContext();
  std::optional<Fortran::evaluate::characteristics::Procedure> characteristic =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          procRef.proc(), foldingContext);
  assert(characteristic && "Failed to get characteristic from procRef");
  // The characteristic may not contain the argument characteristic if the
  // ProcedureDesignator has no interface.
  if (!characteristic->HasExplicitInterface()) {
    for (const std::optional<Fortran::evaluate::ActualArgument> &arg :
         procRef.arguments()) {
      if (arg.value().isAlternateReturn()) {
        characteristic->dummyArguments.emplace_back(
            Fortran::evaluate::characteristics::AlternateReturn{});
      } else {
        // Argument cannot be optional with implicit interface
        const Fortran::lower::SomeExpr *expr = arg.value().UnwrapExpr();
        assert(
            expr &&
            "argument in call with implicit interface cannot be assumed type");
        std::optional<Fortran::evaluate::characteristics::DummyArgument>
            argCharacteristic =
                Fortran::evaluate::characteristics::DummyArgument::FromActual(
                    "actual", *expr, foldingContext);
        assert(argCharacteristic &&
               "failed to characterize argument in implicit call");
        characteristic->dummyArguments.emplace_back(
            asImplicitArg(std::move(*argCharacteristic)));
      }
    }
  }
  return *characteristic;
}

void Fortran::lower::CallerInterface::placeInput(
    const PassedEntity &passedEntity, mlir::Value arg) {
  assert(static_cast<int>(actualInputs.size()) > passedEntity.firArgument &&
         passedEntity.firArgument >= 0 &&
         passedEntity.passBy != CallInterface::PassEntityBy::AddressAndLength &&
         "bad arg position");
  actualInputs[passedEntity.firArgument] = arg;
}

void Fortran::lower::CallerInterface::placeAddressAndLengthInput(
    const PassedEntity &passedEntity, mlir::Value addr, mlir::Value len) {
  assert(static_cast<int>(actualInputs.size()) > passedEntity.firArgument &&
         static_cast<int>(actualInputs.size()) > passedEntity.firLength &&
         passedEntity.firArgument >= 0 && passedEntity.firLength >= 0 &&
         passedEntity.passBy == CallInterface::PassEntityBy::AddressAndLength &&
         "bad arg position");
  actualInputs[passedEntity.firArgument] = addr;
  actualInputs[passedEntity.firLength] = len;
}

bool Fortran::lower::CallerInterface::verifyActualInputs() const {
  if (getNumFIRArguments() != actualInputs.size())
    return false;
  for (mlir::Value arg : actualInputs) {
    if (!arg)
      return false;
  }
  return true;
}

void Fortran::lower::CallerInterface::walkResultLengths(
    ExprVisitor visitor) const {
  assert(characteristic && "characteristic was not computed");
  const Fortran::evaluate::characteristics::FunctionResult &result =
      characteristic->functionResult.value();
  const Fortran::evaluate::characteristics::TypeAndShape *typeAndShape =
      result.GetTypeAndShape();
  assert(typeAndShape && "no result type");
  Fortran::evaluate::DynamicType dynamicType = typeAndShape->type();
  // Visit result length specification expressions that are explicit.
  if (dynamicType.category() == Fortran::common::TypeCategory::Character) {
    if (std::optional<Fortran::evaluate::ExtentExpr> length =
            dynamicType.GetCharLength())
      visitor(toEvExpr(*length));
  } else if (dynamicType.category() == common::TypeCategory::Derived) {
    const Fortran::semantics::DerivedTypeSpec &derivedTypeSpec =
        dynamicType.GetDerivedTypeSpec();
    if (Fortran::semantics::CountLenParameters(derivedTypeSpec) > 0)
      TODO(converter.getCurrentLocation(),
           "function result with derived type length parameters");
  }
}

// Compute extent expr from shapeSpec of an explicit shape.
// TODO: Allow evaluate shape analysis to work in a mode where it disregards
// the non-constant aspects when building the shape to avoid having this here.
static Fortran::evaluate::ExtentExpr
getExtentExpr(const Fortran::semantics::ShapeSpec &shapeSpec) {
  const auto &ubound = shapeSpec.ubound().GetExplicit();
  const auto &lbound = shapeSpec.lbound().GetExplicit();
  assert(lbound && ubound && "shape must be explicit");
  return Fortran::common::Clone(*ubound) - Fortran::common::Clone(*lbound) +
         Fortran::evaluate::ExtentExpr{1};
}

void Fortran::lower::CallerInterface::walkResultExtents(
    ExprVisitor visitor) const {
  // Walk directly the result symbol shape (the characteristic shape may contain
  // descriptor inquiries to it that would fail to lower on the caller side).
  const Fortran::semantics::Symbol *interfaceSymbol =
      procRef.proc().GetInterfaceSymbol();
  if (interfaceSymbol) {
    const Fortran::semantics::Symbol &result =
        interfaceSymbol->get<Fortran::semantics::SubprogramDetails>().result();
    if (const auto *objectDetails =
            result.detailsIf<Fortran::semantics::ObjectEntityDetails>())
      if (objectDetails->shape().IsExplicitShape())
        for (const Fortran::semantics::ShapeSpec &shapeSpec :
             objectDetails->shape())
          visitor(Fortran::evaluate::AsGenericExpr(getExtentExpr(shapeSpec)));
  } else {
    if (procRef.Rank() != 0)
      fir::emitFatalError(
          converter.getCurrentLocation(),
          "only scalar functions may not have an interface symbol");
  }
}

bool Fortran::lower::CallerInterface::mustMapInterfaceSymbols() const {
  assert(characteristic && "characteristic was not computed");
  const std::optional<Fortran::evaluate::characteristics::FunctionResult>
      &result = characteristic->functionResult;
  if (!result || result->CanBeReturnedViaImplicitInterface() ||
      !procRef.proc().GetInterfaceSymbol())
    return false;
  bool allResultSpecExprConstant = true;
  auto visitor = [&](const Fortran::lower::SomeExpr &e) {
    allResultSpecExprConstant &= Fortran::evaluate::IsConstantExpr(e);
  };
  walkResultLengths(visitor);
  walkResultExtents(visitor);
  return !allResultSpecExprConstant;
}

mlir::Value Fortran::lower::CallerInterface::getArgumentValue(
    const semantics::Symbol &sym) const {
  mlir::Location loc = converter.getCurrentLocation();
  const Fortran::semantics::Symbol *iface = procRef.proc().GetInterfaceSymbol();
  if (!iface)
    fir::emitFatalError(
        loc, "mapping actual and dummy arguments requires an interface");
  const std::vector<Fortran::semantics::Symbol *> &dummies =
      iface->get<semantics::SubprogramDetails>().dummyArgs();
  auto it = std::find(dummies.begin(), dummies.end(), &sym);
  if (it == dummies.end())
    fir::emitFatalError(loc, "symbol is not a dummy in this call");
  FirValue mlirArgIndex = passedArguments[it - dummies.begin()].firArgument;
  return actualInputs[mlirArgIndex];
}

mlir::Type Fortran::lower::CallerInterface::getResultStorageType() const {
  if (passedResult)
    return fir::dyn_cast_ptrEleTy(inputs[passedResult->firArgument].type);
  assert(saveResult && !outputs.empty());
  return outputs[0].type;
}

const Fortran::semantics::Symbol &
Fortran::lower::CallerInterface::getResultSymbol() const {
  mlir::Location loc = converter.getCurrentLocation();
  const Fortran::semantics::Symbol *iface = procRef.proc().GetInterfaceSymbol();
  if (!iface)
    fir::emitFatalError(
        loc, "mapping actual and dummy arguments requires an interface");
  return iface->get<semantics::SubprogramDetails>().result();
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
  mapPassedEntities();
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
      for (const auto &placeHolder : llvm::enumerate(inputs))
        if (!placeHolder.value().attributes.empty())
          func.setArgAttrs(placeHolder.index(), placeHolder.value().attributes);
    }
  }
}

/// Once the signature has been analyzed and the mlir::FuncOp was built/found,
/// map the fir inputs to Fortran entities (the symbols or expressions).
template <typename T>
void Fortran::lower::CallInterface<T>::mapPassedEntities() {
  // map back fir inputs to passed entities
  if constexpr (std::is_same_v<T, Fortran::lower::CalleeInterface>) {
    assert(inputs.size() == func.front().getArguments().size() &&
           "function previously created with different number of arguments");
    for (auto [fst, snd] : llvm::zip(inputs, func.front().getArguments()))
      mapBackInputToPassedEntity(fst, snd);
  } else {
    // On the caller side, map the index of the mlir argument position
    // to Fortran ActualArguments.
    int firPosition = 0;
    for (const FirPlaceHolder &placeHolder : inputs)
      mapBackInputToPassedEntity(placeHolder, firPosition++);
  }
}

template <typename T>
void Fortran::lower::CallInterface<T>::mapBackInputToPassedEntity(
    const FirPlaceHolder &placeHolder, FirValue firValue) {
  PassedEntity &passedEntity =
      placeHolder.passedEntityPosition == FirPlaceHolder::resultEntityPosition
          ? passedResult.value()
          : passedArguments[placeHolder.passedEntityPosition];
  if (placeHolder.property == Property::CharLength)
    passedEntity.firLength = firValue;
  else
    passedEntity.firArgument = firValue;
}

/// Helpers to access ActualArgument/Symbols
static const Fortran::evaluate::ActualArguments &
getEntityContainer(const Fortran::evaluate::ProcedureRef &proc) {
  return proc.arguments();
}

static const std::vector<Fortran::semantics::Symbol *> &
getEntityContainer(Fortran::lower::pft::FunctionLikeUnit &funit) {
  return funit.getSubprogramSymbol()
      .get<Fortran::semantics::SubprogramDetails>()
      .dummyArgs();
}

static const Fortran::evaluate::ActualArgument *getDataObjectEntity(
    const std::optional<Fortran::evaluate::ActualArgument> &arg) {
  if (arg)
    return &*arg;
  return nullptr;
}

static const Fortran::semantics::Symbol &
getDataObjectEntity(const Fortran::semantics::Symbol *arg) {
  assert(arg && "expect symbol for data object entity");
  return *arg;
}

static const Fortran::evaluate::ActualArgument *
getResultEntity(const Fortran::evaluate::ProcedureRef &) {
  return nullptr;
}

static const Fortran::semantics::Symbol &
getResultEntity(Fortran::lower::pft::FunctionLikeUnit &funit) {
  return funit.getSubprogramSymbol()
      .get<Fortran::semantics::SubprogramDetails>()
      .result();
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
  using PassEntityBy = typename CallInterface::PassEntityBy;
  using PassedEntity = typename CallInterface::PassedEntity;
  using FirValue = typename CallInterface::FirValue;
  using FortranEntity = typename CallInterface::FortranEntity;
  using FirPlaceHolder = typename CallInterface::FirPlaceHolder;
  using Property = typename CallInterface::Property;
  using TypeAndShape = Fortran::evaluate::characteristics::TypeAndShape;
  using DummyCharacteristics =
      Fortran::evaluate::characteristics::DummyArgument;

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
    // Handle arguments
    const auto &argumentEntities =
        getEntityContainer(interface.side().getCallDescription());
    for (auto pair : llvm::zip(procedure.dummyArguments, argumentEntities)) {
      const Fortran::evaluate::characteristics::DummyArgument
          &argCharacteristics = std::get<0>(pair);
      std::visit(
          Fortran::common::visitors{
              [&](const auto &dummy) {
                const auto &entity = getDataObjectEntity(std::get<1>(pair));
                handleImplicitDummy(&argCharacteristics, dummy, entity);
              },
              [&](const Fortran::evaluate::characteristics::AlternateReturn &) {
                // nothing to do
              },
          },
          argCharacteristics.u);
    }
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
    bool isBindC = procedure.IsBindC();
    // Handle arguments
    const auto &argumentEntities =
        getEntityContainer(interface.side().getCallDescription());
    for (auto pair : llvm::zip(procedure.dummyArguments, argumentEntities)) {
      const Fortran::evaluate::characteristics::DummyArgument
          &argCharacteristics = std::get<0>(pair);
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::evaluate::characteristics::DummyDataObject
                      &dummy) {
                const auto &entity = getDataObjectEntity(std::get<1>(pair));
                if (dummy.CanBePassedViaImplicitInterface())
                  handleImplicitDummy(&argCharacteristics, dummy, entity);
                else
                  handleExplicitDummy(&argCharacteristics, dummy, entity,
                                      isBindC);
              },
              [&](const Fortran::evaluate::characteristics::DummyProcedure
                      &dummy) {
                const auto &entity = getDataObjectEntity(std::get<1>(pair));
                handleImplicitDummy(&argCharacteristics, dummy, entity);
              },
              [&](const Fortran::evaluate::characteristics::AlternateReturn &) {
                // nothing to do
              },
          },
          argCharacteristics.u);
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
    // Character result allocated by caller and passed as hidden arguments
    if (dynamicType.category() == Fortran::common::TypeCategory::Character) {
      handleImplicitCharacterResult(dynamicType);
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

  void
  handleImplicitCharacterResult(const Fortran::evaluate::DynamicType &type) {
    int resultPosition = FirPlaceHolder::resultEntityPosition;
    setPassedResult(PassEntityBy::AddressAndLength,
                    getResultEntity(interface.side().getCallDescription()));
    mlir::Type lenTy = mlir::IndexType::get(&mlirContext);
    std::optional<std::int64_t> constantLen = type.knownLength();
    fir::CharacterType::LenType len =
        constantLen ? *constantLen : fir::CharacterType::unknownLen();
    mlir::Type charRefTy = fir::ReferenceType::get(
        fir::CharacterType::get(&mlirContext, type.kind(), len));
    mlir::Type boxCharTy = fir::BoxCharType::get(&mlirContext, type.kind());
    addFirOperand(charRefTy, resultPosition, Property::CharAddress);
    addFirOperand(lenTy, resultPosition, Property::CharLength);
    /// For now, also return it by boxchar
    addFirResult(boxCharTy, resultPosition, Property::BoxChar);
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
    mlir::Type mlirType = translateDynamicType(typeAndShape->type());
    fir::SequenceType::Shape bounds = getBounds(typeAndShape->shape());
    if (!bounds.empty())
      mlirType = fir::SequenceType::get(bounds, mlirType);
    if (result.attrs.test(Attr::Allocatable))
      mlirType = fir::BoxType::get(fir::HeapType::get(mlirType));
    if (result.attrs.test(Attr::Pointer))
      mlirType = fir::BoxType::get(fir::PointerType::get(mlirType));

    if (fir::isa_char(mlirType)) {
      // Character scalar results must be passed as arguments in lowering so
      // that an assumed length character function callee can access the result
      // length. A function with a result requiring an explicit interface does
      // not have to be compatible with assumed length function, but most
      // compilers supports it.
      handleImplicitCharacterResult(typeAndShape->type());
      return;
    }

    addFirResult(mlirType, FirPlaceHolder::resultEntityPosition,
                 Property::Value);
    // Explicit results require the caller to allocate the storage and save the
    // function result in the storage with a fir.save_result.
    setSaveResult();
  }

  fir::SequenceType::Shape getBounds(const Fortran::evaluate::Shape &shape) {
    fir::SequenceType::Shape bounds;
    for (const std::optional<Fortran::evaluate::ExtentExpr> &extent : shape) {
      fir::SequenceType::Extent bound = fir::SequenceType::getUnknownExtent();
      if (std::optional<std::int64_t> i = toInt64(extent))
        bound = *i;
      bounds.emplace_back(bound);
    }
    return bounds;
  }
  std::optional<std::int64_t>
  toInt64(std::optional<
          Fortran::evaluate::Expr<Fortran::evaluate::SubscriptInteger>>
              expr) {
    if (expr)
      return Fortran::evaluate::ToInt64(Fortran::evaluate::Fold(
          getConverter().getFoldingContext(), toEvExpr(*expr)));
    return std::nullopt;
  }

  /// Return a vector with an attribute with the name of the argument if this
  /// is a callee interface and the name is available. Otherwise, just return
  /// an empty vector.
  llvm::SmallVector<mlir::NamedAttribute>
  dummyNameAttr(const FortranEntity &entity) {
    if constexpr (std::is_same_v<FortranEntity,
                                 std::optional<Fortran::common::Reference<
                                     const Fortran::semantics::Symbol>>>) {
      if (entity.has_value()) {
        const Fortran::semantics::Symbol *argument = &*entity.value();
        // "fir.bindc_name" is used for arguments for the sake of consistency
        // with other attributes carrying surface syntax names in FIR.
        return {mlir::NamedAttribute(
            mlir::StringAttr::get(&mlirContext, "fir.bindc_name"),
            mlir::StringAttr::get(&mlirContext,
                                  toStringRef(argument->name())))};
      }
    }
    return {};
  }

  // Define when an explicit argument must be passed in a fir.box.
  bool dummyRequiresBox(
      const Fortran::evaluate::characteristics::DummyDataObject &obj) {
    using ShapeAttr = Fortran::evaluate::characteristics::TypeAndShape::Attr;
    using ShapeAttrs = Fortran::evaluate::characteristics::TypeAndShape::Attrs;
    constexpr ShapeAttrs shapeRequiringBox = {
        ShapeAttr::AssumedShape, ShapeAttr::DeferredShape,
        ShapeAttr::AssumedRank, ShapeAttr::Coarray};
    if ((obj.type.attrs() & shapeRequiringBox).any())
      // Need to pass shape/coshape info in fir.box.
      return true;
    if (obj.type.type().IsPolymorphic())
      // Need to pass dynamic type info in fir.box.
      return true;
    if (const Fortran::semantics::DerivedTypeSpec *derived =
            Fortran::evaluate::GetDerivedTypeSpec(obj.type.type()))
      // Need to pass type parameters in fir.box if any.
      return derived->parameters().empty();
    return false;
  }

  mlir::Type
  translateDynamicType(const Fortran::evaluate::DynamicType &dynamicType) {
    Fortran::common::TypeCategory cat = dynamicType.category();
    // DERIVED
    if (cat == Fortran::common::TypeCategory::Derived) {
      TODO(interface.converter.getCurrentLocation(),
           "[translateDynamicType] Derived");
    }
    // CHARACTER with compile time constant length.
    if (cat == Fortran::common::TypeCategory::Character)
      if (std::optional<std::int64_t> constantLen =
              toInt64(dynamicType.GetCharLength()))
        return getConverter().genType(cat, dynamicType.kind(), {*constantLen});
    // INTEGER, REAL, LOGICAL, COMPLEX, and CHARACTER with dynamic length.
    return getConverter().genType(cat, dynamicType.kind());
  }

  void handleExplicitDummy(
      const DummyCharacteristics *characteristics,
      const Fortran::evaluate::characteristics::DummyDataObject &obj,
      const FortranEntity &entity, bool isBindC) {
    using Attrs = Fortran::evaluate::characteristics::DummyDataObject::Attr;

    bool isValueAttr = false;
    [[maybe_unused]] mlir::Location loc =
        interface.converter.getCurrentLocation();
    llvm::SmallVector<mlir::NamedAttribute> attrs = dummyNameAttr(entity);
    auto addMLIRAttr = [&](llvm::StringRef attr) {
      attrs.emplace_back(mlir::StringAttr::get(&mlirContext, attr),
                         mlir::UnitAttr::get(&mlirContext));
    };
    if (obj.attrs.test(Attrs::Optional))
      addMLIRAttr(fir::getOptionalAttrName());
    if (obj.attrs.test(Attrs::Asynchronous))
      TODO(loc, "Asynchronous in procedure interface");
    if (obj.attrs.test(Attrs::Contiguous))
      addMLIRAttr(fir::getContiguousAttrName());
    if (obj.attrs.test(Attrs::Value))
      isValueAttr = true; // TODO: do we want an mlir::Attribute as well?
    if (obj.attrs.test(Attrs::Volatile))
      TODO(loc, "Volatile in procedure interface");
    if (obj.attrs.test(Attrs::Target))
      addMLIRAttr(fir::getTargetAttrName());

    // TODO: intents that require special care (e.g finalization)

    using ShapeAttr = Fortran::evaluate::characteristics::TypeAndShape::Attr;
    const Fortran::evaluate::characteristics::TypeAndShape::Attrs &shapeAttrs =
        obj.type.attrs();
    if (shapeAttrs.test(ShapeAttr::AssumedRank))
      TODO(loc, "Assumed Rank in procedure interface");
    if (shapeAttrs.test(ShapeAttr::Coarray))
      TODO(loc, "Coarray in procedure interface");

    // So far assume that if the argument cannot be passed by implicit interface
    // it must be by box. That may no be always true (e.g for simple optionals)

    Fortran::evaluate::DynamicType dynamicType = obj.type.type();
    mlir::Type type = translateDynamicType(dynamicType);
    fir::SequenceType::Shape bounds = getBounds(obj.type.shape());
    if (!bounds.empty())
      type = fir::SequenceType::get(bounds, type);
    if (obj.attrs.test(Attrs::Allocatable))
      type = fir::HeapType::get(type);
    if (obj.attrs.test(Attrs::Pointer))
      type = fir::PointerType::get(type);
    mlir::Type boxType = fir::BoxType::get(type);

    if (obj.attrs.test(Attrs::Allocatable) || obj.attrs.test(Attrs::Pointer)) {
      // Pass as fir.ref<fir.box>
      mlir::Type boxRefType = fir::ReferenceType::get(boxType);
      addFirOperand(boxRefType, nextPassedArgPosition(), Property::MutableBox,
                    attrs);
      addPassedArg(PassEntityBy::MutableBox, entity, characteristics);
    } else if (dummyRequiresBox(obj)) {
      // Pass as fir.box
      addFirOperand(boxType, nextPassedArgPosition(), Property::Box, attrs);
      addPassedArg(PassEntityBy::Box, entity, characteristics);
    } else if (dynamicType.category() ==
               Fortran::common::TypeCategory::Character) {
      // Pass as fir.box_char
      mlir::Type boxCharTy =
          fir::BoxCharType::get(&mlirContext, dynamicType.kind());
      addFirOperand(boxCharTy, nextPassedArgPosition(), Property::BoxChar,
                    attrs);
      addPassedArg(isValueAttr ? PassEntityBy::CharBoxValueAttribute
                               : PassEntityBy::BoxChar,
                   entity, characteristics);
    } else {
      // Pass as fir.ref unless it's by VALUE and BIND(C)
      mlir::Type passType = fir::ReferenceType::get(type);
      PassEntityBy passBy = PassEntityBy::BaseAddress;
      Property prop = Property::BaseAddress;
      if (isValueAttr) {
        if (isBindC) {
          passBy = PassEntityBy::Value;
          prop = Property::Value;
          passType = type;
        } else {
          passBy = PassEntityBy::BaseAddressValueAttribute;
        }
      }
      addFirOperand(passType, nextPassedArgPosition(), prop, attrs);
      addPassedArg(passBy, entity, characteristics);
    }
  }

  void handleImplicitDummy(
      const DummyCharacteristics *characteristics,
      const Fortran::evaluate::characteristics::DummyDataObject &obj,
      const FortranEntity &entity) {
    Fortran::evaluate::DynamicType dynamicType = obj.type.type();
    if (dynamicType.category() == Fortran::common::TypeCategory::Character) {
      mlir::Type boxCharTy =
          fir::BoxCharType::get(&mlirContext, dynamicType.kind());
      addFirOperand(boxCharTy, nextPassedArgPosition(), Property::BoxChar,
                    dummyNameAttr(entity));
      addPassedArg(PassEntityBy::BoxChar, entity, characteristics);
    } else {
      // non-PDT derived type allowed in implicit interface.
      Fortran::common::TypeCategory cat = dynamicType.category();
      mlir::Type type = getConverter().genType(cat, dynamicType.kind());
      fir::SequenceType::Shape bounds = getBounds(obj.type.shape());
      if (!bounds.empty())
        type = fir::SequenceType::get(bounds, type);
      mlir::Type refType = fir::ReferenceType::get(type);
      addFirOperand(refType, nextPassedArgPosition(), Property::BaseAddress,
                    dummyNameAttr(entity));
      addPassedArg(PassEntityBy::BaseAddress, entity, characteristics);
    }
  }

  void handleImplicitDummy(
      const DummyCharacteristics *characteristics,
      const Fortran::evaluate::characteristics::DummyProcedure &proc,
      const FortranEntity &entity) {
    TODO(interface.converter.getCurrentLocation(),
         "handleImlicitDummy DummyProcedure");
  }

  void
  addFirOperand(mlir::Type type, int entityPosition, Property p,
                llvm::ArrayRef<mlir::NamedAttribute> attributes = llvm::None) {
    interface.inputs.emplace_back(
        FirPlaceHolder{type, entityPosition, p, attributes});
  }
  void
  addFirResult(mlir::Type type, int entityPosition, Property p,
               llvm::ArrayRef<mlir::NamedAttribute> attributes = llvm::None) {
    interface.outputs.emplace_back(
        FirPlaceHolder{type, entityPosition, p, attributes});
  }
  void addPassedArg(PassEntityBy p, FortranEntity entity,
                    const DummyCharacteristics *characteristics) {
    interface.passedArguments.emplace_back(
        PassedEntity{p, entity, {}, {}, characteristics});
  }
  void setPassedResult(PassEntityBy p, FortranEntity entity) {
    interface.passedResult =
        PassedEntity{p, entity, emptyValue(), emptyValue()};
  }
  void setSaveResult() { interface.saveResult = true; }
  int nextPassedArgPosition() { return interface.passedArguments.size(); }

  static FirValue emptyValue() {
    if constexpr (std::is_same_v<Fortran::lower::CalleeInterface, T>) {
      return {};
    } else {
      return -1;
    }
  }

  Fortran::lower::AbstractConverter &getConverter() {
    return interface.converter;
  }
  CallInterface &interface;
  mlir::MLIRContext &mlirContext;
};

template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::isOptional() const {
  if (!characteristics)
    return false;
  return characteristics->IsOptional();
}
template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::mayBeModifiedByCall()
    const {
  if (!characteristics)
    return true;
  return characteristics->GetIntent() != Fortran::common::Intent::In;
}
template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::mayBeReadByCall() const {
  if (!characteristics)
    return true;
  return characteristics->GetIntent() != Fortran::common::Intent::Out;
}

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
  llvm::SmallVector<mlir::Type> inputTys;
  for (const FirPlaceHolder &placeHolder : outputs)
    returnTys.emplace_back(placeHolder.type);
  for (const FirPlaceHolder &placeHolder : inputs)
    inputTys.emplace_back(placeHolder.type);
  return mlir::FunctionType::get(&converter.getMLIRContext(), inputTys,
                                 returnTys);
}

template class Fortran::lower::CallInterface<Fortran::lower::CalleeInterface>;
template class Fortran::lower::CallInterface<Fortran::lower::CallerInterface>;
