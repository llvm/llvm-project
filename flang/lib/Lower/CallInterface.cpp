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
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include <optional>

mlir::Type Fortran::lower::getUntypedBoxProcType(mlir::MLIRContext *context) {
  llvm::SmallVector<mlir::Type> resultTys;
  llvm::SmallVector<mlir::Type> inputTys;
  auto untypedFunc = mlir::FunctionType::get(context, inputTys, resultTys);
  return fir::BoxProcType::get(context, untypedFunc);
}

/// Return the type of a dummy procedure given its characteristic (if it has
/// one).
static mlir::Type getProcedureDesignatorType(
    const Fortran::evaluate::characteristics::Procedure *,
    Fortran::lower::AbstractConverter &converter) {
  // TODO: Get actual function type of the dummy procedure, at least when an
  // interface is given. The result type should be available even if the arity
  // and type of the arguments is not.
  // In general, that is a nice to have but we cannot guarantee to find the
  // function type that will match the one of the calls, we may not even know
  // how many arguments the dummy procedure accepts (e.g. if a procedure
  // pointer is only transiting through the current procedure without being
  // called), so a function type cast must always be inserted.
  return Fortran::lower::getUntypedBoxProcType(&converter.getMLIRContext());
}

//===----------------------------------------------------------------------===//
// Caller side interface implementation
//===----------------------------------------------------------------------===//

bool Fortran::lower::CallerInterface::hasAlternateReturns() const {
  return procRef.hasAlternateReturns();
}

std::string Fortran::lower::CallerInterface::getMangledName() const {
  const Fortran::evaluate::ProcedureDesignator &proc = procRef.proc();
  // Return the binding label (from BIND(C...)) or the mangled name of the
  // symbol.
  if (const Fortran::semantics::Symbol *symbol = proc.GetSymbol())
    return converter.mangleName(symbol->GetUltimate());
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

bool Fortran::lower::CallerInterface::requireDispatchCall() const {
  // calls with NOPASS attribute still have their component so check if it is
  // polymorphic.
  if (const Fortran::evaluate::Component *component =
          procRef.proc().GetComponent()) {
    if (Fortran::semantics::IsPolymorphic(component->GetFirstSymbol()))
      return true;
  }
  // calls with PASS attribute have the passed-object already set in its
  // arguments. Just check if their is one.
  std::optional<unsigned> passArg = getPassArgIndex();
  if (passArg)
    return true;
  return false;
}

std::optional<unsigned>
Fortran::lower::CallerInterface::getPassArgIndex() const {
  unsigned passArgIdx = 0;
  std::optional<unsigned> passArg;
  for (const auto &arg : getCallDescription().arguments()) {
    if (arg && arg->isPassedObject()) {
      passArg = passArgIdx;
      break;
    }
    ++passArgIdx;
  }
  if (!passArg)
    return passArg;
  // Take into account result inserted as arguments.
  if (std::optional<Fortran::lower::CallInterface<
          Fortran::lower::CallerInterface>::PassedEntity>
          resultArg = getPassedResult()) {
    if (resultArg->passBy == PassEntityBy::AddressAndLength)
      passArg = *passArg + 2;
    else if (resultArg->passBy == PassEntityBy::BaseAddress)
      passArg = *passArg + 1;
  }
  return passArg;
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

static bool isExternalDefinedInSameCompilationUnit(
    const Fortran::evaluate::ProcedureDesignator &proc) {
  if (const auto *symbol{proc.GetSymbol()})
    return symbol->has<Fortran::semantics::SubprogramDetails>() &&
           symbol->owner().IsGlobal();
  return false;
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
  // ProcedureDesignator has no interface, or may mismatch in case of implicit
  // interface.
  if (!characteristic->HasExplicitInterface() ||
      (converter.getLoweringOptions().getLowerToHighLevelFIR() &&
       isExternalDefinedInSameCompilationUnit(procRef.proc()) &&
       characteristic->CanBeCalledViaImplicitInterface())) {
    // In HLFIR lowering, calls to subprogram with implicit interfaces are
    // always prepared according to the actual arguments. This is to support
    // cases where the implicit interfaces are "abused" in old and not so old
    // Fortran code (e.g, passing REAL(8) to CHARACTER(8), passing object
    // pointers to procedure dummies, passing regular procedure dummies to
    // character procedure dummies, omitted arguments....).
    // In all those case, if the subprogram definition is in the same
    // compilation unit, the "characteristic" from Characterize will be the one
    // from the definition, in case of "abuses" (for which semantics raise a
    // warning), lowering will be placed in a difficult position if it is given
    // the dummy characteristic from the definition and an actual that has
    // seemingly nothing to do with it: it would need to battle to anticipate
    // and handle these mismatches (e.g., be able to prepare a fir.boxchar<>
    // from a fir.real<> and so one). This was the approach of the lowering to
    // FIR, and usually lead to compiler bug every time a new "abuse" was met in
    // the wild.
    // Instead, in HLFIR, the dummy characteristic is always computed from the
    // actual for subprogram with implicit interfaces, and in case of call site
    // vs fun.func MLIR function type signature mismatch, a function cast is
    // done before placing the call. This is a hammer that should cover all
    // cases and behave like existing compiler that "do not see" the definition
    // when placing the call.
    characteristic->dummyArguments.clear();
    for (const std::optional<Fortran::evaluate::ActualArgument> &arg :
         procRef.arguments()) {
      // "arg" may be null if this is a call with missing arguments compared
      // to the subprogram definition. Do not compute any characteristic
      // in this case.
      if (arg.has_value()) {
        if (arg.value().isAlternateReturn()) {
          characteristic->dummyArguments.emplace_back(
              Fortran::evaluate::characteristics::AlternateReturn{});
        } else {
          // Argument cannot be optional with implicit interface
          const Fortran::lower::SomeExpr *expr = arg.value().UnwrapExpr();
          assert(expr && "argument in call with implicit interface cannot be "
                         "assumed type");
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
  } else if (dynamicType.category() == common::TypeCategory::Derived &&
             !dynamicType.IsUnlimitedPolymorphic()) {
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
  const Fortran::semantics::SubprogramDetails *interfaceDetails =
      getInterfaceDetails();
  if (interfaceDetails) {
    const Fortran::semantics::Symbol &result = interfaceDetails->result();
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
      !getInterfaceDetails())
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
  const Fortran::semantics::SubprogramDetails *ifaceDetails =
      getInterfaceDetails();
  if (!ifaceDetails)
    fir::emitFatalError(
        loc, "mapping actual and dummy arguments requires an interface");
  const std::vector<Fortran::semantics::Symbol *> &dummies =
      ifaceDetails->dummyArgs();
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
  const Fortran::semantics::SubprogramDetails *ifaceDetails =
      getInterfaceDetails();
  if (!ifaceDetails)
    fir::emitFatalError(
        loc, "mapping actual and dummy arguments requires an interface");
  return ifaceDetails->result();
}

const Fortran::semantics::SubprogramDetails *
Fortran::lower::CallerInterface::getInterfaceDetails() const {
  if (const Fortran::semantics::Symbol *iface =
          procRef.proc().GetInterfaceSymbol())
    return iface->GetUltimate()
        .detailsIf<Fortran::semantics::SubprogramDetails>();
  return nullptr;
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
  return converter.mangleName(funit.getSubprogramSymbol());
}

const Fortran::semantics::Symbol *
Fortran::lower::CalleeInterface::getProcedureSymbol() const {
  if (funit.isMainProgram())
    return funit.getMainProgramSymbol();
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

mlir::func::FuncOp
Fortran::lower::CalleeInterface::addEntryBlockAndMapArguments() {
  // Check for bugs in the front end. The front end must not present multiple
  // definitions of the same procedure.
  if (!func.getBlocks().empty())
    fir::emitFatalError(func.getLoc(),
                        "cannot process subprogram that was already processed");

  // On the callee side, directly map the mlir::value argument of the function
  // block to the Fortran symbols.
  func.addEntryBlock();
  mapPassedEntities();
  return func;
}

bool Fortran::lower::CalleeInterface::hasHostAssociated() const {
  return funit.parentHasTupleHostAssoc();
}

mlir::Type Fortran::lower::CalleeInterface::getHostAssociatedTy() const {
  assert(hasHostAssociated());
  return funit.parentHostAssoc().getArgumentType(converter);
}

mlir::Value Fortran::lower::CalleeInterface::getHostAssociatedTuple() const {
  assert(hasHostAssociated() || !funit.getHostAssoc().empty());
  return converter.hostAssocTupleValue();
}

void Fortran::lower::CalleeInterface::setFuncAttrs(
    mlir::func::FuncOp func) const {
  if (funit.parentHasHostAssoc())
    func->setAttr(fir::getInternalProcedureAttrName(),
                  mlir::UnitAttr::get(func->getContext()));
}

//===----------------------------------------------------------------------===//
// CallInterface implementation: this part is common to both caller and callee.
//===----------------------------------------------------------------------===//

static void addSymbolAttribute(mlir::func::FuncOp func,
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
/// signature and building/finding the mlir::func::FuncOp.
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
      if (const Fortran::semantics::Symbol *sym = side().getProcedureSymbol()) {
        if (side().isMainProgram()) {
          func->setAttr(fir::getSymbolAttrName(),
                        mlir::StringAttr::get(&converter.getMLIRContext(),
                                              sym->name().ToString()));
        } else {
          addSymbolAttribute(func, *sym, converter.getMLIRContext());
        }
      }
      for (const auto &placeHolder : llvm::enumerate(inputs))
        if (!placeHolder.value().attributes.empty())
          func.setArgAttrs(placeHolder.index(), placeHolder.value().attributes);
      side().setFuncAttrs(func);
    }
  }
}

/// Once the signature has been analyzed and the mlir::func::FuncOp was
/// built/found, map the fir inputs to Fortran entities (the symbols or
/// expressions).
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

/// Bypass helpers to manipulate entities since they are not any symbol/actual
/// argument to associate. See SignatureBuilder below.
using FakeEntity = bool;
using FakeEntities = llvm::SmallVector<FakeEntity>;
static FakeEntities
getEntityContainer(const Fortran::evaluate::characteristics::Procedure &proc) {
  FakeEntities enities(proc.dummyArguments.size());
  return enities;
}
static const FakeEntity &getDataObjectEntity(const FakeEntity &e) { return e; }
static FakeEntity
getResultEntity(const Fortran::evaluate::characteristics::Procedure &proc) {
  return false;
}

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
      handleImplicitResult(*result, procedure.IsBindC());
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
    bool isBindC = procedure.IsBindC();
    // Handle result
    if (const std::optional<Fortran::evaluate::characteristics::FunctionResult>
            &result = procedure.functionResult) {
      if (result->CanBeReturnedViaImplicitInterface())
        handleImplicitResult(*result, isBindC);
      else
        handleExplicitResult(*result);
    } else if (interface.side().hasAlternateReturns()) {
      addFirResult(mlir::IndexType::get(&mlirContext),
                   FirPlaceHolder::resultEntityPosition, Property::Value);
    }
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
                if (!isBindC && dummy.CanBePassedViaImplicitInterface())
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

  void appendHostAssocTupleArg(mlir::Type tupTy) {
    mlir::MLIRContext *ctxt = tupTy.getContext();
    addFirOperand(tupTy, nextPassedArgPosition(), Property::BaseAddress,
                  {mlir::NamedAttribute{
                      mlir::StringAttr::get(ctxt, fir::getHostAssocAttrName()),
                      mlir::UnitAttr::get(ctxt)}});
    interface.passedArguments.emplace_back(
        PassedEntity{PassEntityBy::BaseAddress, std::nullopt,
                     interface.side().getHostAssociatedTuple(), emptyValue()});
  }

  static std::optional<Fortran::evaluate::DynamicType> getResultDynamicType(
      const Fortran::evaluate::characteristics::Procedure &procedure) {
    if (const std::optional<Fortran::evaluate::characteristics::FunctionResult>
            &result = procedure.functionResult)
      if (const auto *resultTypeAndShape = result->GetTypeAndShape())
        return resultTypeAndShape->type();
    return std::nullopt;
  }

  static bool mustPassLengthWithDummyProcedure(
      const Fortran::evaluate::characteristics::Procedure &procedure) {
    // When passing a character function designator `bar` as dummy procedure to
    // `foo` (e.g. `foo(bar)`), pass the result length of `bar` to `foo` so that
    // `bar` can be called inside `foo` even if its length is assumed there.
    // From an ABI perspective, the extra length argument must be handled
    // exactly as if passing a character object. Using an argument of
    // fir.boxchar type gives the expected behavior: after codegen, the
    // fir.boxchar lengths are added after all the arguments as extra value
    // arguments (the extra arguments order is the order of the fir.boxchar).

    // This ABI is compatible with ifort, nag, nvfortran, and xlf, but not
    // gfortran. Gfortran does not pass the length and is therefore unable to
    // handle later call to `bar` in `foo` where the length would be assumed. If
    // the result is an array, nag and ifort and xlf still pass the length, but
    // not nvfortran (and gfortran). It is not clear it is possible to call an
    // array function with assumed length (f18 forbides defining such
    // interfaces). Hence, passing the length is most likely useless, but stick
    // with ifort/nag/xlf interface here.
    if (std::optional<Fortran::evaluate::DynamicType> type =
            getResultDynamicType(procedure))
      return type->category() == Fortran::common::TypeCategory::Character;
    return false;
  }

private:
  void handleImplicitResult(
      const Fortran::evaluate::characteristics::FunctionResult &result,
      bool isBindC) {
    if (result.IsProcedurePointer())
      TODO(interface.converter.getCurrentLocation(),
           "procedure pointer result not yet handled");
    const Fortran::evaluate::characteristics::TypeAndShape *typeAndShape =
        result.GetTypeAndShape();
    assert(typeAndShape && "expect type for non proc pointer result");
    Fortran::evaluate::DynamicType dynamicType = typeAndShape->type();
    // Character result allocated by caller and passed as hidden arguments
    if (dynamicType.category() == Fortran::common::TypeCategory::Character) {
      if (isBindC) {
        mlir::Type mlirType = translateDynamicType(dynamicType);
        addFirResult(mlirType, FirPlaceHolder::resultEntityPosition,
                     Property::Value);
      } else {
        handleImplicitCharacterResult(dynamicType);
      }
    } else if (dynamicType.category() ==
               Fortran::common::TypeCategory::Derived) {
      if (!dynamicType.GetDerivedTypeSpec().IsVectorType()) {
        // Derived result need to be allocated by the caller and the result
        // value must be saved. Derived type in implicit interface cannot have
        // length parameters.
        setSaveResult();
      }
      mlir::Type mlirType = translateDynamicType(dynamicType);
      addFirResult(mlirType, FirPlaceHolder::resultEntityPosition,
                   Property::Value);
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
      mlir::Type type = translateDynamicType(dynamicType);
      fir::SequenceType::Shape bounds = getBounds(obj.type.shape());
      if (!bounds.empty())
        type = fir::SequenceType::get(bounds, type);
      mlir::Type refType = fir::ReferenceType::get(type);
      addFirOperand(refType, nextPassedArgPosition(), Property::BaseAddress,
                    dummyNameAttr(entity));
      addPassedArg(PassEntityBy::BaseAddress, entity, characteristics);
    }
  }

  // Define when an explicit argument must be passed in a fir.box.
  bool dummyRequiresBox(
      const Fortran::evaluate::characteristics::DummyDataObject &obj,
      bool isBindC) {
    using ShapeAttr = Fortran::evaluate::characteristics::TypeAndShape::Attr;
    using ShapeAttrs = Fortran::evaluate::characteristics::TypeAndShape::Attrs;
    constexpr ShapeAttrs shapeRequiringBox = {
        ShapeAttr::AssumedShape, ShapeAttr::DeferredShape,
        ShapeAttr::AssumedRank, ShapeAttr::Coarray};
    if ((obj.type.attrs() & shapeRequiringBox).any())
      // Need to pass shape/coshape info in fir.box.
      return true;
    if (obj.type.type().IsPolymorphic() && !obj.type.type().IsAssumedType())
      // Need to pass dynamic type info in fir.box.
      return true;
    if (const Fortran::semantics::DerivedTypeSpec *derived =
            Fortran::evaluate::GetDerivedTypeSpec(obj.type.type()))
      if (const Fortran::semantics::Scope *scope = derived->scope())
        // Need to pass length type parameters in fir.box if any.
        return scope->IsDerivedTypeWithLengthParameter();
    if (isBindC && obj.type.type().IsAssumedLengthCharacter())
      return true; // Fortran 2018 18.3.6 point 2 (5)
    return false;
  }

  mlir::Type
  translateDynamicType(const Fortran::evaluate::DynamicType &dynamicType) {
    Fortran::common::TypeCategory cat = dynamicType.category();
    // DERIVED
    if (cat == Fortran::common::TypeCategory::Derived) {
      // TODO is kept under experimental flag until feature is complete.
      if (dynamicType.IsPolymorphic() &&
          !getConverter().getLoweringOptions().getPolymorphicTypeImpl())
        TODO(interface.converter.getCurrentLocation(),
             "support for polymorphic types");

      if (dynamicType.IsUnlimitedPolymorphic())
        return mlir::NoneType::get(&mlirContext);
      return getConverter().genType(dynamicType.GetDerivedTypeSpec());
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
      TODO(loc, "ASYNCHRONOUS in procedure interface");
    if (obj.attrs.test(Attrs::Contiguous))
      addMLIRAttr(fir::getContiguousAttrName());
    if (obj.attrs.test(Attrs::Value))
      isValueAttr = true; // TODO: do we want an mlir::Attribute as well?
    if (obj.attrs.test(Attrs::Volatile))
      TODO(loc, "VOLATILE in procedure interface");
    if (obj.attrs.test(Attrs::Target))
      addMLIRAttr(fir::getTargetAttrName());

    // TODO: intents that require special care (e.g finalization)

    using ShapeAttr = Fortran::evaluate::characteristics::TypeAndShape::Attr;
    const Fortran::evaluate::characteristics::TypeAndShape::Attrs &shapeAttrs =
        obj.type.attrs();
    if (shapeAttrs.test(ShapeAttr::AssumedRank))
      TODO(loc, "assumed rank in procedure interface");
    if (shapeAttrs.test(ShapeAttr::Coarray))
      TODO(loc, "coarray: dummy argument coarray in procedure interface");

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
    mlir::Type boxType = fir::wrapInClassOrBoxType(
        type, obj.type.type().IsPolymorphic(), obj.type.type().IsAssumedType());

    if (obj.attrs.test(Attrs::Allocatable) || obj.attrs.test(Attrs::Pointer)) {
      // Pass as fir.ref<fir.box> or fir.ref<fir.class>
      mlir::Type boxRefType = fir::ReferenceType::get(boxType);
      addFirOperand(boxRefType, nextPassedArgPosition(), Property::MutableBox,
                    attrs);
      addPassedArg(PassEntityBy::MutableBox, entity, characteristics);
    } else if (dummyRequiresBox(obj, isBindC)) {
      // Pass as fir.box or fir.class
      if (isValueAttr)
        TODO(loc, "assumed shape dummy argument with VALUE attribute");
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
      // Pass as fir.ref unless it's by VALUE and BIND(C). Also pass-by-value
      // for numerical/logical scalar without OPTIONAL so that the behavior is
      // consistent with gfortran/nvfortran.
      // TODO: pass-by-value for derived type is not supported yet
      mlir::Type passType = fir::ReferenceType::get(type);
      PassEntityBy passBy = PassEntityBy::BaseAddress;
      Property prop = Property::BaseAddress;
      if (isValueAttr) {
        bool isBuiltinCptrType = fir::isa_builtin_cptr_type(type);
        if (isBindC || (!type.isa<fir::SequenceType>() &&
                        !obj.attrs.test(Attrs::Optional) &&
                        (dynamicType.category() !=
                             Fortran::common::TypeCategory::Derived ||
                         isBuiltinCptrType))) {
          passBy = PassEntityBy::Value;
          prop = Property::Value;
          if (isBuiltinCptrType) {
            auto recTy = type.dyn_cast<fir::RecordType>();
            mlir::Type fieldTy = recTy.getTypeList()[0].second;
            passType = fir::ReferenceType::get(fieldTy);
          } else {
            passType = type;
          }
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
      const Fortran::evaluate::characteristics::DummyProcedure &proc,
      const FortranEntity &entity) {
    if (proc.attrs.test(
            Fortran::evaluate::characteristics::DummyProcedure::Attr::Pointer))
      TODO(interface.converter.getCurrentLocation(),
           "procedure pointer arguments");
    // Otherwise, it is a dummy procedure.
    const Fortran::evaluate::characteristics::Procedure &procedure =
        proc.procedure.value();
    mlir::Type funcType =
        getProcedureDesignatorType(&procedure, interface.converter);
    std::optional<Fortran::evaluate::DynamicType> resultTy =
        getResultDynamicType(procedure);
    if (resultTy && mustPassLengthWithDummyProcedure(procedure)) {
      // The result length of dummy procedures that are character functions must
      // be passed so that the dummy procedure can be called if it has assumed
      // length on the callee side.
      mlir::Type tupleType =
          fir::factory::getCharacterProcedureTupleType(funcType);
      llvm::StringRef charProcAttr = fir::getCharacterProcedureDummyAttrName();
      addFirOperand(tupleType, nextPassedArgPosition(), Property::CharProcTuple,
                    {mlir::NamedAttribute{
                        mlir::StringAttr::get(&mlirContext, charProcAttr),
                        mlir::UnitAttr::get(&mlirContext)}});
      addPassedArg(PassEntityBy::CharProcTuple, entity, characteristics);
      return;
    }
    addFirOperand(funcType, nextPassedArgPosition(), Property::BaseAddress);
    addPassedArg(PassEntityBy::BaseAddress, entity, characteristics);
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
    const auto *resTypeAndShape{result.GetTypeAndShape()};
    bool resIsPolymorphic =
        resTypeAndShape && resTypeAndShape->type().IsPolymorphic();
    bool resIsAssumedType =
        resTypeAndShape && resTypeAndShape->type().IsAssumedType();
    if (!bounds.empty())
      mlirType = fir::SequenceType::get(bounds, mlirType);
    if (result.attrs.test(Attr::Allocatable))
      mlirType = fir::wrapInClassOrBoxType(fir::HeapType::get(mlirType),
                                           resIsPolymorphic, resIsAssumedType);
    if (result.attrs.test(Attr::Pointer))
      mlirType = fir::wrapInClassOrBoxType(fir::PointerType::get(mlirType),
                                           resIsPolymorphic, resIsAssumedType);

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
  void addFirOperand(
      mlir::Type type, int entityPosition, Property p,
      llvm::ArrayRef<mlir::NamedAttribute> attributes = std::nullopt) {
    interface.inputs.emplace_back(
        FirPlaceHolder{type, entityPosition, p, attributes});
  }
  void
  addFirResult(mlir::Type type, int entityPosition, Property p,
               llvm::ArrayRef<mlir::NamedAttribute> attributes = std::nullopt) {
    interface.outputs.emplace_back(
        FirPlaceHolder{type, entityPosition, p, attributes});
  }
  void addPassedArg(PassEntityBy p, FortranEntity entity,
                    const DummyCharacteristics *characteristics) {
    interface.passedArguments.emplace_back(
        PassedEntity{p, entity, emptyValue(), emptyValue(), characteristics});
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
  if (characteristics->GetIntent() == Fortran::common::Intent::In)
    return false;
  return !hasValueAttribute();
}
template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::mayBeReadByCall() const {
  if (!characteristics)
    return true;
  return characteristics->GetIntent() != Fortran::common::Intent::Out;
}

template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::testTKR(
    Fortran::common::IgnoreTKR flag) const {
  if (!characteristics)
    return false;
  const auto *dummy =
      std::get_if<Fortran::evaluate::characteristics::DummyDataObject>(
          &characteristics->u);
  if (!dummy)
    return false;
  return dummy->ignoreTKR.test(flag);
}

template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::isIntentOut() const {
  if (!characteristics)
    return true;
  return characteristics->GetIntent() == Fortran::common::Intent::Out;
}
template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::mustBeMadeContiguous()
    const {
  if (!characteristics)
    return true;
  const auto *dummy =
      std::get_if<Fortran::evaluate::characteristics::DummyDataObject>(
          &characteristics->u);
  if (!dummy)
    return false;
  const auto &shapeAttrs = dummy->type.attrs();
  using ShapeAttrs = Fortran::evaluate::characteristics::TypeAndShape::Attr;
  if (shapeAttrs.test(ShapeAttrs::AssumedRank) ||
      shapeAttrs.test(ShapeAttrs::AssumedShape))
    return dummy->attrs.test(
        Fortran::evaluate::characteristics::DummyDataObject::Attr::Contiguous);
  if (shapeAttrs.test(ShapeAttrs::DeferredShape))
    return false;
  // Explicit shape arrays are contiguous.
  return dummy->type.Rank() > 0;
}

template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::hasValueAttribute() const {
  if (!characteristics)
    return false;
  const auto *dummy =
      std::get_if<Fortran::evaluate::characteristics::DummyDataObject>(
          &characteristics->u);
  return dummy &&
         dummy->attrs.test(
             Fortran::evaluate::characteristics::DummyDataObject::Attr::Value);
}

template <typename T>
bool Fortran::lower::CallInterface<T>::PassedEntity::hasAllocatableAttribute()
    const {
  if (!characteristics)
    return false;
  const auto *dummy =
      std::get_if<Fortran::evaluate::characteristics::DummyDataObject>(
          &characteristics->u);
  using Attrs = Fortran::evaluate::characteristics::DummyDataObject::Attr;
  return dummy && dummy->attrs.test(Attrs::Allocatable);
}

template <typename T>
bool Fortran::lower::CallInterface<
    T>::PassedEntity::mayRequireIntentoutFinalization() const {
  // Conservatively assume that the finalization is needed.
  if (!characteristics)
    return true;

  // No INTENT(OUT) dummy arguments do not require finalization on entry.
  if (!isIntentOut())
    return false;

  const auto *dummy =
      std::get_if<Fortran::evaluate::characteristics::DummyDataObject>(
          &characteristics->u);
  if (!dummy)
    return true;

  // POINTER/ALLOCATABLE dummy arguments do not require finalization.
  using Attrs = Fortran::evaluate::characteristics::DummyDataObject::Attr;
  if (dummy->attrs.test(Attrs::Allocatable) ||
      dummy->attrs.test(Attrs::Pointer))
    return false;

  // Polymorphic and unlimited polymorphic INTENT(OUT) dummy arguments
  // may need finalization.
  const Fortran::evaluate::DynamicType &type = dummy->type.type();
  if (type.IsPolymorphic() || type.IsUnlimitedPolymorphic())
    return true;

  // INTENT(OUT) dummy arguments of derived types require finalization,
  // if their type has finalization.
  const Fortran::semantics::DerivedTypeSpec *derived =
      Fortran::evaluate::GetDerivedTypeSpec(type);
  if (!derived)
    return false;

  return Fortran::semantics::IsFinalizable(*derived);
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
  // We only expect the extra host asspciations argument from the callee side as
  // the definition of internal procedures will be present, and we'll always
  // have a FuncOp definition in the ModuleOp, when lowering.
  if constexpr (std::is_same_v<T, Fortran::lower::CalleeInterface>) {
    if (side().hasHostAssociated())
      impl.appendHostAssocTupleArg(side().getHostAssociatedTy());
  }
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

template <typename T>
llvm::SmallVector<mlir::Type>
Fortran::lower::CallInterface<T>::getResultType() const {
  llvm::SmallVector<mlir::Type> types;
  for (const FirPlaceHolder &out : outputs)
    types.emplace_back(out.type);
  return types;
}

template class Fortran::lower::CallInterface<Fortran::lower::CalleeInterface>;
template class Fortran::lower::CallInterface<Fortran::lower::CallerInterface>;

//===----------------------------------------------------------------------===//
// Function Type Translation
//===----------------------------------------------------------------------===//

/// Build signature from characteristics when there is no Fortran entity to
/// associate with the arguments (i.e, this is not a call site or a procedure
/// declaration. This is needed when dealing with function pointers/dummy
/// arguments.

class SignatureBuilder;
template <>
struct Fortran::lower::PassedEntityTypes<SignatureBuilder> {
  using FortranEntity = FakeEntity;
  using FirValue = int;
};

/// SignatureBuilder is a CRTP implementation of CallInterface intended to
/// help translating characteristics::Procedure to mlir::FunctionType using
/// the CallInterface translation.
class SignatureBuilder
    : public Fortran::lower::CallInterface<SignatureBuilder> {
public:
  SignatureBuilder(const Fortran::evaluate::characteristics::Procedure &p,
                   Fortran::lower::AbstractConverter &c, bool forceImplicit)
      : CallInterface{c}, proc{p} {
    bool isImplicit = forceImplicit || proc.CanBeCalledViaImplicitInterface();
    determineInterface(isImplicit, proc);
  }
  /// Does the procedure characteristics being translated have alternate
  /// returns ?
  bool hasAlternateReturns() const {
    for (const Fortran::evaluate::characteristics::DummyArgument &dummy :
         proc.dummyArguments)
      if (std::holds_alternative<
              Fortran::evaluate::characteristics::AlternateReturn>(dummy.u))
        return true;
    return false;
  };

  /// This is only here to fulfill CRTP dependencies and should not be called.
  std::string getMangledName() const {
    llvm_unreachable("trying to get name from SignatureBuilder");
  }

  /// This is only here to fulfill CRTP dependencies and should not be called.
  mlir::Location getCalleeLocation() const {
    llvm_unreachable("trying to get callee location from SignatureBuilder");
  }

  /// This is only here to fulfill CRTP dependencies and should not be called.
  const Fortran::semantics::Symbol *getProcedureSymbol() const {
    llvm_unreachable("trying to get callee symbol from SignatureBuilder");
  };

  Fortran::evaluate::characteristics::Procedure characterize() const {
    return proc;
  }
  /// SignatureBuilder cannot be used on main program.
  static constexpr bool isMainProgram() { return false; }

  /// Return the characteristics::Procedure that is being translated to
  /// mlir::FunctionType.
  const Fortran::evaluate::characteristics::Procedure &
  getCallDescription() const {
    return proc;
  }

  /// This is not the description of an indirect call.
  static constexpr bool isIndirectCall() { return false; }

  /// Return the translated signature.
  mlir::FunctionType getFunctionType() { return genFunctionType(); }

  // Copy of base implementation.
  static constexpr bool hasHostAssociated() { return false; }
  mlir::Type getHostAssociatedTy() const {
    llvm_unreachable("getting host associated type in SignatureBuilder");
  }

private:
  const Fortran::evaluate::characteristics::Procedure &proc;
};

mlir::FunctionType Fortran::lower::translateSignature(
    const Fortran::evaluate::ProcedureDesignator &proc,
    Fortran::lower::AbstractConverter &converter) {
  std::optional<Fortran::evaluate::characteristics::Procedure> characteristics =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          proc, converter.getFoldingContext());
  // Most unrestricted intrinsic characteristic has the Elemental attribute
  // which triggers CanBeCalledViaImplicitInterface to return false. However,
  // using implicit interface rules is just fine here.
  bool forceImplicit = proc.GetSpecificIntrinsic();
  return SignatureBuilder{characteristics.value(), converter, forceImplicit}
      .getFunctionType();
}

mlir::func::FuncOp Fortran::lower::getOrDeclareFunction(
    llvm::StringRef name, const Fortran::evaluate::ProcedureDesignator &proc,
    Fortran::lower::AbstractConverter &converter) {
  mlir::ModuleOp module = converter.getModuleOp();
  mlir::func::FuncOp func = fir::FirOpBuilder::getNamedFunction(module, name);
  if (func)
    return func;

  const Fortran::semantics::Symbol *symbol = proc.GetSymbol();
  assert(symbol && "non user function in getOrDeclareFunction");
  // getOrDeclareFunction is only used for functions not defined in the current
  // program unit, so use the location of the procedure designator symbol, which
  // is the first occurrence of the procedure in the program unit.
  mlir::Location loc = converter.genLocation(symbol->name());
  std::optional<Fortran::evaluate::characteristics::Procedure> characteristics =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          proc, converter.getFoldingContext());
  mlir::FunctionType ty = SignatureBuilder{characteristics.value(), converter,
                                           /*forceImplicit=*/false}
                              .getFunctionType();
  mlir::func::FuncOp newFunc =
      fir::FirOpBuilder::createFunction(loc, module, name, ty);
  addSymbolAttribute(newFunc, *symbol, converter.getMLIRContext());
  return newFunc;
}

// Is it required to pass a dummy procedure with \p characteristics as a tuple
// containing the function address and the result length ?
static bool mustPassLengthWithDummyProcedure(
    const std::optional<Fortran::evaluate::characteristics::Procedure>
        &characteristics) {
  return characteristics &&
         Fortran::lower::CallInterfaceImpl<SignatureBuilder>::
             mustPassLengthWithDummyProcedure(*characteristics);
}

bool Fortran::lower::mustPassLengthWithDummyProcedure(
    const Fortran::evaluate::ProcedureDesignator &procedure,
    Fortran::lower::AbstractConverter &converter) {
  std::optional<Fortran::evaluate::characteristics::Procedure> characteristics =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          procedure, converter.getFoldingContext());
  return ::mustPassLengthWithDummyProcedure(characteristics);
}

mlir::Type Fortran::lower::getDummyProcedureType(
    const Fortran::semantics::Symbol &dummyProc,
    Fortran::lower::AbstractConverter &converter) {
  std::optional<Fortran::evaluate::characteristics::Procedure> iface =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          dummyProc, converter.getFoldingContext());
  mlir::Type procType = getProcedureDesignatorType(
      iface.has_value() ? &*iface : nullptr, converter);
  if (::mustPassLengthWithDummyProcedure(iface))
    return fir::factory::getCharacterProcedureTupleType(procType);
  return procType;
}

bool Fortran::lower::isCPtrArgByValueType(mlir::Type ty) {
  return ty.isa<fir::ReferenceType>() &&
         fir::isa_integer(fir::unwrapRefType(ty));
}
