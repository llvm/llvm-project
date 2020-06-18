//===-- CallInterface.cpp -- Procedure call interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/CallInterface.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/fold.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"

//===----------------------------------------------------------------------===//
// Caller side interface implementation
//===----------------------------------------------------------------------===//

bool Fortran::lower::CallerInterface::hasAlternateReturns() const {
  return procRef.HasAlternateReturns();
}

std::string Fortran::lower::CallerInterface::getMangledName() const {
  const auto &proc = procRef.proc();
  if (const auto *symbol = proc.GetSymbol())
    return converter.mangleName(*symbol);
  assert(proc.GetSpecificIntrinsic() &&
         "expected intrinsic procedure in designator");
  return proc.GetName();
}

bool Fortran::lower::CallerInterface::isIndirectCall() const {
  if (const auto *symbol = procRef.proc().GetSymbol())
    return Fortran::semantics::IsPointer(*symbol) ||
           Fortran::semantics::IsDummy(*symbol);
  return false;
}

const Fortran::semantics::Symbol *
Fortran::lower::CallerInterface::getIfIndirectCallSymbol() const {
  if (const auto *symbol = procRef.proc().GetSymbol())
    if (Fortran::semantics::IsPointer(*symbol) ||
        Fortran::semantics::IsDummy(*symbol))
      return symbol;
  return nullptr;
}

mlir::Location Fortran::lower::CallerInterface::getCalleeLocation() const {
  const auto &proc = procRef.proc();
  // FIXME: If the callee is defined in the same file but after the current
  // unit we do cannot get its location here and the funcOp is created at the
  // wrong location (i.e, the caller location).
  if (const auto *symbol = proc.GetSymbol())
    return converter.genLocation(symbol->name());
  // Unknown location for intrinsics.
  return converter.genLocation();
}

Fortran::evaluate::characteristics::Procedure
Fortran::lower::CallerInterface::characterize() const {
  auto &foldingContext = converter.getFoldingContext();
  auto characteristic =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          procRef.proc(), foldingContext.intrinsics());
  assert(characteristic && "Failed to get characteristic from procRef");
  // The characteristic may not contain the argument characteristic if no
  // the ProcedureDesignator has no interface.
  if (!characteristic->HasExplicitInterface()) {
    for (const auto &arg : procRef.arguments()) {
      // Argument cannot be optional with implicit interface
      const auto *expr = arg.value().UnwrapExpr();
      assert(expr &&
             "argument in call with implicit interface cannot be assumed type");
      auto argCharacteristic =
          Fortran::evaluate::characteristics::DummyArgument::FromActual(
              "actual", *expr, foldingContext);
      assert(argCharacteristic &&
             "failed to characterize argument in implicit call");
      characteristic->dummyArguments.emplace_back(
          std::move(*argCharacteristic));
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
  for (auto arg : actualInputs) {
    if (!arg)
      return false;
  }
  return true;
}

template <typename T>
static inline auto AsGenericExpr(T e) {
  return Fortran::evaluate::AsGenericExpr(Fortran::common::Clone(e));
}

mlir::Value Fortran::lower::CallerInterface::getResultLength() {
  // FIXME: technically, this is a specification expression,
  // so it should be evaluated on entry of the region we are
  // in, it can go wrong if the specification expression
  // uses a symbol that may have change.
  //
  // The characteristic has to be explicit for such
  // cases, so the allocation could also be handled on callee side, in such
  // case. For now, protect with an unreachable.
  assert(characteristic && "characteristic was not computed");
  const auto *typeAndShape =
      characteristic->functionResult.value().GetTypeAndShape();
  assert(typeAndShape && "no result type");
  auto expr = AsGenericExpr(typeAndShape->LEN().value());
  if (Fortran::evaluate::IsConstantExpr(expr))
    return converter.genExprValue(expr);
  llvm_unreachable(
      "non constant result length on caller side not yet safely handled");
}

//===----------------------------------------------------------------------===//
// Callee side interface implementation
//===----------------------------------------------------------------------===//

bool Fortran::lower::CalleeInterface::hasAlternateReturns() const {
  return !funit.isMainProgram() &&
         Fortran::semantics::HasAlternateReturns(funit.getSubprogramSymbol());
}

std::string Fortran::lower::CalleeInterface::getMangledName() const {
  return funit.isMainProgram()
             ? fir::NameUniquer::doProgramEntry().str()
             : converter.mangleName(funit.getSubprogramSymbol());
}

mlir::Location Fortran::lower::CalleeInterface::getCalleeLocation() const {
  // FIXME: do NOT use unknown for the anonymous PROGRAM case. We probably
  // should just stash the location in the funit regardless.
  return converter.genLocation(funit.getStartingSourceLoc());
}

Fortran::evaluate::characteristics::Procedure
Fortran::lower::CalleeInterface::characterize() const {
  auto &foldingContext = converter.getFoldingContext();
  auto characteristic =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          funit.getSubprogramSymbol(), foldingContext.intrinsics());
  assert(characteristic && "Fail to get characteristic from symbol");
  return *characteristic;
}

bool Fortran::lower::CalleeInterface::isMainProgram() const {
  return funit.isMainProgram();
}

//===----------------------------------------------------------------------===//
// CallInterface implementation: this part is common to both caller and caller
// sides.
//===----------------------------------------------------------------------===//

/// Init drives the different actions to be performed while building a
/// CallInterface, it does not decide anything about the interface.
template <typename T>
void Fortran::lower::CallInterface<T>::init() {
  if (!side().isMainProgram()) {
    characteristic =
        std::make_unique<Fortran::evaluate::characteristics::Procedure>(
            side().characterize());
    if (characteristic->CanBeCalledViaImplicitInterface())
      buildImplicitInterface(*characteristic);
    else
      buildExplicitInterface(*characteristic);
  }
  // No input/output for main program

  // Create / get funcOp for direct calls. For indirect calls (only meaningful
  // on the caller side), no funcOp has to be created here. The mlir::Value
  // holding the indirection is used when creating the fir::CallOp.
  if (!side().isIndirectCall()) {
    auto name = side().getMangledName();
    auto module = converter.getModuleOp();
    func = Fortran::lower::FirOpBuilder::getNamedFunction(module, name);
    if (!func) {
      mlir::Location loc = side().getCalleeLocation();
      mlir::FunctionType ty = genFunctionType();
      func =
          Fortran::lower::FirOpBuilder::createFunction(loc, module, name, ty);
    }
  }

  // map back fir inputs to passed entities
  if constexpr (std::is_same_v<T, Fortran::lower::CalleeInterface>) {
    // On the callee side, directly map the mlir::value argument of
    // the function block to the Fortran symbols.
    func.addEntryBlock();
    assert(inputs.size() == func.front().getArguments().size() &&
           "function previously created with different number of arguments");
    for (const auto &pair : llvm::zip(inputs, func.front().getArguments()))
      mapBackInputToPassedEntity(std::get<0>(pair), std::get<1>(pair));
  } else {
    // On the caller side, map the index of the mlir argument position
    // to Fortran ActualArguments.
    auto firPosition = 0;
    for (const auto &placeHolder : inputs)
      mapBackInputToPassedEntity(placeHolder, firPosition++);
  }
}

template <typename T>
void Fortran::lower::CallInterface<T>::mapBackInputToPassedEntity(
    const FirPlaceHolder &placeHolder, FirValue firValue) {
  auto &passedEntity =
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
  const auto &details =
      funit.getSubprogramSymbol().get<Fortran::semantics::SubprogramDetails>();
  return details.result();
}

/// This is the actual part that defines the FIR interface based on the
/// charcteristic. It directly mutates the CallInterface members.
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

public:
  CallInterfaceImpl(CallInterface &i)
      : interface{i}, mlirContext{i.converter.getMLIRContext()} {}

  void buildImplicitInterface(
      const Fortran::evaluate::characteristics::Procedure &procedure) {
    // Handle result
    auto resultPosition = FirPlaceHolder::resultEntityPosition;
    if (const auto &result = procedure.functionResult) {
      if (result->IsProcedurePointer()) // TODO
        llvm_unreachable("procedure pointer result not yet handled");
      const auto *typeAndShape = result->GetTypeAndShape();
      assert(typeAndShape && "expect type for non proc pointer result");
      auto dynamicType = typeAndShape->type();
      // Character result allocated by caller and passed has hidden arguments
      if (dynamicType.category() == Fortran::common::TypeCategory::Character) {
        handleImplicitCharacterResult(dynamicType);
      } else {
        // All result other than characters are simply returned by value in
        // implicit interfaces
        auto mlirType =
            getConverter().genType(dynamicType.category(), dynamicType.kind());
        addFirOutput(mlirType, resultPosition, Property::Value);
      }
    } else if (interface.side().hasAlternateReturns()) {
      addFirOutput(mlir::IndexType::get(&mlirContext), resultPosition,
                   Property::Value);
    }
    // Handle arguments
    const auto &argumentEntities =
        getEntityContainer(interface.side().getCallDescription());
    for (const auto &pair :
         llvm::zip(procedure.dummyArguments, argumentEntities)) {
      const auto &dummyCharacteristic = std::get<0>(pair);
      std::visit(
          Fortran::common::visitors{
              [&](const auto &dummy) {
                const auto &entity = getDataObjectEntity(std::get<1>(pair));
                handleImplicitDummy(dummy, entity);
              },
              [&](const Fortran::evaluate::characteristics::AlternateReturn &) {
                // nothing to do
              },
          },
          dummyCharacteristic.u);
    }
  }
  void buildExplicitInterface(
      const Fortran::evaluate::characteristics::Procedure &procedure) {
    // TODO
    llvm_unreachable("Explicit interface lowering TODO");
  }

private:
  void
  handleImplicitCharacterResult(const Fortran::evaluate::DynamicType &type) {
    auto resultPosition = FirPlaceHolder::resultEntityPosition;
    setPassedResult(PassEntityBy::AddressAndLength,
                    getResultEntity(interface.side().getCallDescription()));
    auto lenTy = mlir::IndexType::get(&mlirContext);
    auto charRefTy = fir::ReferenceType::get(
        fir::CharacterType::get(&mlirContext, type.kind()));
    auto boxCharTy = fir::BoxCharType::get(&mlirContext, type.kind());
    addFirInput(charRefTy, resultPosition, Property::CharAddress);
    addFirInput(lenTy, resultPosition, Property::CharLength);
    /// For now, still also return it by boxchar
    addFirOutput(boxCharTy, resultPosition, Property::BoxChar);
  }

  void handleImplicitDummy(
      const Fortran::evaluate::characteristics::DummyDataObject &obj,
      const FortranEntity &entity) {
    auto dynamicType = obj.type.type();
    if (dynamicType.category() == Fortran::common::TypeCategory::Character) {
      auto boxCharTy = fir::BoxCharType::get(&mlirContext, dynamicType.kind());
      addFirInput(boxCharTy, nextPassedArgPosition(), Property::BoxChar);
      addPassedArg(PassEntityBy::BoxChar, entity);
      // FIXME: non PDT derived type allowed here.
    } else {
      mlir::Type type =
          getConverter().genType(dynamicType.category(), dynamicType.kind());
      fir::SequenceType::Shape bounds = getBounds(obj.type.shape());
      if (!bounds.empty())
        type = fir::SequenceType::get(bounds, type);
      auto refType = fir::ReferenceType::get(type);

      addFirInput(refType, nextPassedArgPosition(), Property::BaseAddress);
      addPassedArg(PassEntityBy::BaseAddress, entity);
    }
  }

  void handleImplicitDummy(
      const Fortran::evaluate::characteristics::DummyProcedure &proc,
      const FortranEntity &entity) {
    if (proc.attrs.test(
            Fortran::evaluate::characteristics::DummyProcedure::Attr::Pointer))
      llvm_unreachable("TODO: procedure pointer arguments");
    // Otherwise, it is a dummy procedure

    // TODO: Get actual function type of the dummy procedure, at least when an
    // interface is given.
    // In general, that is a nice to have but we cannot guarantee to find the
    // function type that will match the one of the calls, we may not even know
    // how many arguments the dummy procedure accepts (e.g. if a procedure
    // pointer is only transiting through the current procedure without being
    // called), so a function type cast must always be inserted.
    auto funcType = mlir::FunctionType::get({}, {}, &mlirContext);
    addFirInput(funcType, nextPassedArgPosition(), Property::BaseAddress);
    addPassedArg(PassEntityBy::BaseAddress, entity);
  }

  fir::SequenceType::Shape getBounds(const Fortran::evaluate::Shape &shape) {
    fir::SequenceType::Shape bounds;
    for (const auto &extent : shape) {
      auto bound = fir::SequenceType::getUnknownExtent();
      if (extent)
        if (auto i = Fortran::evaluate::ToInt64(Fortran::evaluate::Fold(
                getConverter().getFoldingContext(), AsGenericExpr(*extent))))
          bound = *i;
      bounds.emplace_back(bound);
    }
    return bounds;
  }
  void addFirInput(mlir::Type type, int entityPosition, Property p) {
    interface.inputs.emplace_back(FirPlaceHolder{type, entityPosition, p});
  }
  void addFirOutput(mlir::Type type, int entityPosition, Property p) {
    interface.outputs.emplace_back(FirPlaceHolder{type, entityPosition, p});
  }
  void addPassedArg(PassEntityBy p, FortranEntity entity) {
    interface.passedArguments.emplace_back(
        PassedEntity{p, entity, emptyValue(), emptyValue()});
  }
  void setPassedResult(PassEntityBy p, FortranEntity entity) {
    interface.passedResult =
        PassedEntity{p, entity, emptyValue(), emptyValue()};
  }
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
void Fortran::lower::CallInterface<T>::buildImplicitInterface(
    const Fortran::evaluate::characteristics::Procedure &procedure) {
  CallInterfaceImpl<T> impl(*this);
  impl.buildImplicitInterface(procedure);
}

template <typename T>
void Fortran::lower::CallInterface<T>::buildExplicitInterface(
    const Fortran::evaluate::characteristics::Procedure &procedure) {
  CallInterfaceImpl<T> impl(*this);
  impl.buildExplicitInterface(procedure);
}

template <typename T>
mlir::FunctionType Fortran::lower::CallInterface<T>::genFunctionType() const {
  llvm::SmallVector<mlir::Type, 1> returnTys;
  llvm::SmallVector<mlir::Type, 4> inputTys;
  for (const auto &placeHolder : outputs)
    returnTys.emplace_back(placeHolder.type);
  for (const auto &placeHolder : inputs)
    inputTys.emplace_back(placeHolder.type);
  return mlir::FunctionType::get(inputTys, returnTys,
                                 &converter.getMLIRContext());
}

template <typename T>
llvm::SmallVector<mlir::Type, 1>
Fortran::lower::CallInterface<T>::getResultType() const {
  llvm::SmallVector<mlir::Type, 1> types;
  for (const auto &out : outputs)
    types.emplace_back(out.type);
  return types;
}

template class Fortran::lower::CallInterface<Fortran::lower::CalleeInterface>;
template class Fortran::lower::CallInterface<Fortran::lower::CallerInterface>;
