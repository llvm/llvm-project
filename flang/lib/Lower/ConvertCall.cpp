//===-- ConvertCall.cpp ---------------------------------------------------===//
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

#include "flang/Lower/ConvertCall.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/CustomIntrinsicCall.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "flang-lower-expr"

/// Helper to package a Value and its properties into an ExtendedValue.
static fir::ExtendedValue toExtendedValue(mlir::Location loc, mlir::Value base,
                                          llvm::ArrayRef<mlir::Value> extents,
                                          llvm::ArrayRef<mlir::Value> lengths) {
  mlir::Type type = base.getType();
  if (type.isa<fir::BaseBoxType>())
    return fir::BoxValue(base, /*lbounds=*/{}, lengths, extents);
  type = fir::unwrapRefType(type);
  if (type.isa<fir::BaseBoxType>())
    return fir::MutableBoxValue(base, lengths, /*mutableProperties*/ {});
  if (auto seqTy = type.dyn_cast<fir::SequenceType>()) {
    if (seqTy.getDimension() != extents.size())
      fir::emitFatalError(loc, "incorrect number of extents for array");
    if (seqTy.getEleTy().isa<fir::CharacterType>()) {
      if (lengths.empty())
        fir::emitFatalError(loc, "missing length for character");
      assert(lengths.size() == 1);
      return fir::CharArrayBoxValue(base, lengths[0], extents);
    }
    return fir::ArrayBoxValue(base, extents);
  }
  if (type.isa<fir::CharacterType>()) {
    if (lengths.empty())
      fir::emitFatalError(loc, "missing length for character");
    assert(lengths.size() == 1);
    return fir::CharBoxValue(base, lengths[0]);
  }
  return base;
}

/// Lower a type(C_PTR/C_FUNPTR) argument with VALUE attribute into a
/// reference. A C pointer can correspond to a Fortran dummy argument of type
/// C_PTR with the VALUE attribute. (see 18.3.6 note 3).
static mlir::Value
genRecordCPtrValueArg(Fortran::lower::AbstractConverter &converter,
                      mlir::Value rec, mlir::Type ty) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Location loc = converter.getCurrentLocation();
  mlir::Value cAddr = fir::factory::genCPtrOrCFunptrAddr(builder, loc, rec, ty);
  mlir::Value cVal = builder.create<fir::LoadOp>(loc, cAddr);
  return builder.createConvert(loc, cAddr.getType(), cVal);
}

// Find the argument that corresponds to the host associations.
// Verify some assumptions about how the signature was built here.
[[maybe_unused]] static unsigned findHostAssocTuplePos(mlir::func::FuncOp fn) {
  // Scan the argument list from last to first as the host associations are
  // appended for now.
  for (unsigned i = fn.getNumArguments(); i > 0; --i)
    if (fn.getArgAttr(i - 1, fir::getHostAssocAttrName())) {
      // Host assoc tuple must be last argument (for now).
      assert(i == fn.getNumArguments() && "tuple must be last");
      return i - 1;
    }
  llvm_unreachable("anyFuncArgsHaveAttr failed");
}

mlir::Value
Fortran::lower::argumentHostAssocs(Fortran::lower::AbstractConverter &converter,
                                   mlir::Value arg) {
  if (auto addr = mlir::dyn_cast_or_null<fir::AddrOfOp>(arg.getDefiningOp())) {
    auto &builder = converter.getFirOpBuilder();
    if (auto funcOp = builder.getNamedFunction(addr.getSymbol()))
      if (fir::anyFuncArgsHaveAttr(funcOp, fir::getHostAssocAttrName()))
        return converter.hostAssocTupleValue();
  }
  return {};
}

fir::ExtendedValue Fortran::lower::genCallOpAndResult(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
    Fortran::lower::CallerInterface &caller, mlir::FunctionType callSiteType,
    std::optional<mlir::Type> resultType) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  using PassBy = Fortran::lower::CallerInterface::PassEntityBy;
  // Handle cases where caller must allocate the result or a fir.box for it.
  bool mustPopSymMap = false;
  if (caller.mustMapInterfaceSymbols()) {
    symMap.pushScope();
    mustPopSymMap = true;
    Fortran::lower::mapCallInterfaceSymbols(converter, caller, symMap);
  }
  // If this is an indirect call, retrieve the function address. Also retrieve
  // the result length if this is a character function (note that this length
  // will be used only if there is no explicit length in the local interface).
  mlir::Value funcPointer;
  mlir::Value charFuncPointerLength;
  if (const Fortran::semantics::Symbol *sym =
          caller.getIfIndirectCallSymbol()) {
    funcPointer = symMap.lookupSymbol(*sym).getAddr();
    if (!funcPointer)
      fir::emitFatalError(loc, "failed to find indirect call symbol address");
    if (fir::isCharacterProcedureTuple(funcPointer.getType(),
                                       /*acceptRawFunc=*/false))
      std::tie(funcPointer, charFuncPointerLength) =
          fir::factory::extractCharacterProcedureTuple(builder, loc,
                                                       funcPointer);
  }

  mlir::IndexType idxTy = builder.getIndexType();
  auto lowerSpecExpr = [&](const auto &expr) -> mlir::Value {
    mlir::Value convertExpr = builder.createConvert(
        loc, idxTy, fir::getBase(converter.genExprValue(expr, stmtCtx)));
    return fir::factory::genMaxWithZero(builder, loc, convertExpr);
  };
  llvm::SmallVector<mlir::Value> resultLengths;
  auto allocatedResult = [&]() -> std::optional<fir::ExtendedValue> {
    llvm::SmallVector<mlir::Value> extents;
    llvm::SmallVector<mlir::Value> lengths;
    if (!caller.callerAllocateResult())
      return {};
    mlir::Type type = caller.getResultStorageType();
    if (type.isa<fir::SequenceType>())
      caller.walkResultExtents([&](const Fortran::lower::SomeExpr &e) {
        extents.emplace_back(lowerSpecExpr(e));
      });
    caller.walkResultLengths([&](const Fortran::lower::SomeExpr &e) {
      lengths.emplace_back(lowerSpecExpr(e));
    });

    // Result length parameters should not be provided to box storage
    // allocation and save_results, but they are still useful information to
    // keep in the ExtendedValue if non-deferred.
    if (!type.isa<fir::BoxType>()) {
      if (fir::isa_char(fir::unwrapSequenceType(type)) && lengths.empty()) {
        // Calling an assumed length function. This is only possible if this
        // is a call to a character dummy procedure.
        if (!charFuncPointerLength)
          fir::emitFatalError(loc, "failed to retrieve character function "
                                   "length while calling it");
        lengths.push_back(charFuncPointerLength);
      }
      resultLengths = lengths;
    }

    if (!extents.empty() || !lengths.empty()) {
      auto *bldr = &converter.getFirOpBuilder();
      auto stackSaveFn = fir::factory::getLlvmStackSave(builder);
      auto stackSaveSymbol = bldr->getSymbolRefAttr(stackSaveFn.getName());
      mlir::Value sp = bldr->create<fir::CallOp>(
                               loc, stackSaveFn.getFunctionType().getResults(),
                               stackSaveSymbol, mlir::ValueRange{})
                           .getResult(0);
      stmtCtx.attachCleanup([bldr, loc, sp]() {
        auto stackRestoreFn = fir::factory::getLlvmStackRestore(*bldr);
        auto stackRestoreSymbol =
            bldr->getSymbolRefAttr(stackRestoreFn.getName());
        bldr->create<fir::CallOp>(loc,
                                  stackRestoreFn.getFunctionType().getResults(),
                                  stackRestoreSymbol, mlir::ValueRange{sp});
      });
    }
    mlir::Value temp =
        builder.createTemporary(loc, type, ".result", extents, resultLengths);
    return toExtendedValue(loc, temp, extents, lengths);
  }();

  if (mustPopSymMap)
    symMap.popScope();

  // Place allocated result or prepare the fir.save_result arguments.
  mlir::Value arrayResultShape;
  if (allocatedResult) {
    if (std::optional<Fortran::lower::CallInterface<
            Fortran::lower::CallerInterface>::PassedEntity>
            resultArg = caller.getPassedResult()) {
      if (resultArg->passBy == PassBy::AddressAndLength)
        caller.placeAddressAndLengthInput(*resultArg,
                                          fir::getBase(*allocatedResult),
                                          fir::getLen(*allocatedResult));
      else if (resultArg->passBy == PassBy::BaseAddress)
        caller.placeInput(*resultArg, fir::getBase(*allocatedResult));
      else
        fir::emitFatalError(
            loc, "only expect character scalar result to be passed by ref");
    } else {
      assert(caller.mustSaveResult());
      arrayResultShape = allocatedResult->match(
          [&](const fir::CharArrayBoxValue &) {
            return builder.createShape(loc, *allocatedResult);
          },
          [&](const fir::ArrayBoxValue &) {
            return builder.createShape(loc, *allocatedResult);
          },
          [&](const auto &) { return mlir::Value{}; });
    }
  }

  // In older Fortran, procedure argument types are inferred. This may lead
  // different view of what the function signature is in different locations.
  // Casts are inserted as needed below to accommodate this.

  // The mlir::func::FuncOp type prevails, unless it has a different number of
  // arguments which can happen in legal program if it was passed as a dummy
  // procedure argument earlier with no further type information.
  mlir::SymbolRefAttr funcSymbolAttr;
  bool addHostAssociations = false;
  if (!funcPointer) {
    mlir::FunctionType funcOpType = caller.getFuncOp().getFunctionType();
    mlir::SymbolRefAttr symbolAttr =
        builder.getSymbolRefAttr(caller.getMangledName());
    if (callSiteType.getNumResults() == funcOpType.getNumResults() &&
        callSiteType.getNumInputs() + 1 == funcOpType.getNumInputs() &&
        fir::anyFuncArgsHaveAttr(caller.getFuncOp(),
                                 fir::getHostAssocAttrName())) {
      // The number of arguments is off by one, and we're lowering a function
      // with host associations. Modify call to include host associations
      // argument by appending the value at the end of the operands.
      assert(funcOpType.getInput(findHostAssocTuplePos(caller.getFuncOp())) ==
             converter.hostAssocTupleValue().getType());
      addHostAssociations = true;
    }
    if (!addHostAssociations &&
        (callSiteType.getNumResults() != funcOpType.getNumResults() ||
         callSiteType.getNumInputs() != funcOpType.getNumInputs())) {
      // Deal with argument number mismatch by making a function pointer so
      // that function type cast can be inserted. Do not emit a warning here
      // because this can happen in legal program if the function is not
      // defined here and it was first passed as an argument without any more
      // information.
      funcPointer = builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
    } else if (callSiteType.getResults() != funcOpType.getResults()) {
      // Implicit interface result type mismatch are not standard Fortran, but
      // some compilers are not complaining about it.  The front end is not
      // protecting lowering from this currently. Support this with a
      // discouraging warning.
      LLVM_DEBUG(mlir::emitWarning(
          loc, "a return type mismatch is not standard compliant and may "
               "lead to undefined behavior."));
      // Cast the actual function to the current caller implicit type because
      // that is the behavior we would get if we could not see the definition.
      funcPointer = builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
    } else {
      funcSymbolAttr = symbolAttr;
    }
  }

  mlir::FunctionType funcType =
      funcPointer ? callSiteType : caller.getFuncOp().getFunctionType();
  llvm::SmallVector<mlir::Value> operands;
  // First operand of indirect call is the function pointer. Cast it to
  // required function type for the call to handle procedures that have a
  // compatible interface in Fortran, but that have different signatures in
  // FIR.
  if (funcPointer) {
    operands.push_back(
        funcPointer.getType().isa<fir::BoxProcType>()
            ? builder.create<fir::BoxAddrOp>(loc, funcType, funcPointer)
            : builder.createConvert(loc, funcType, funcPointer));
  }

  // Deal with potential mismatches in arguments types. Passing an array to a
  // scalar argument should for instance be tolerated here.
  bool callingImplicitInterface = caller.canBeCalledViaImplicitInterface();
  for (auto [fst, snd] : llvm::zip(caller.getInputs(), funcType.getInputs())) {
    // When passing arguments to a procedure that can be called by implicit
    // interface, allow any character actual arguments to be passed to dummy
    // arguments of any type and vice versa.
    mlir::Value cast;
    auto *context = builder.getContext();
    if (snd.isa<fir::BoxProcType>() &&
        fst.getType().isa<mlir::FunctionType>()) {
      auto funcTy =
          mlir::FunctionType::get(context, std::nullopt, std::nullopt);
      auto boxProcTy = builder.getBoxProcType(funcTy);
      if (mlir::Value host = argumentHostAssocs(converter, fst)) {
        cast = builder.create<fir::EmboxProcOp>(
            loc, boxProcTy, llvm::ArrayRef<mlir::Value>{fst, host});
      } else {
        cast = builder.create<fir::EmboxProcOp>(loc, boxProcTy, fst);
      }
    } else {
      mlir::Type fromTy = fir::unwrapRefType(fst.getType());
      if (fir::isa_builtin_cptr_type(fromTy) &&
          Fortran::lower::isCPtrArgByValueType(snd)) {
        cast = genRecordCPtrValueArg(converter, fst, fromTy);
      } else if (fir::isa_derived(snd)) {
        // FIXME: This seems like a serious bug elsewhere in lowering. Paper
        // over the problem for now.
        TODO(loc, "derived type argument passed by value");
      } else {
        cast = builder.convertWithSemantics(loc, snd, fst,
                                            callingImplicitInterface);
      }
    }
    operands.push_back(cast);
  }

  // Add host associations as necessary.
  if (addHostAssociations)
    operands.push_back(converter.hostAssocTupleValue());

  mlir::Value callResult;
  unsigned callNumResults;
  if (caller.requireDispatchCall()) {
    // Procedure call requiring a dynamic dispatch. Call is created with
    // fir.dispatch.

    // Get the raw procedure name. The procedure name is not mangled in the
    // binding table.
    const auto &ultimateSymbol =
        caller.getCallDescription().proc().GetSymbol()->GetUltimate();
    auto procName = toStringRef(ultimateSymbol.name());

    fir::DispatchOp dispatch;
    if (std::optional<unsigned> passArg = caller.getPassArgIndex()) {
      // PASS, PASS(arg-name)
      dispatch = builder.create<fir::DispatchOp>(
          loc, funcType.getResults(), builder.getStringAttr(procName),
          operands[*passArg], operands, builder.getI32IntegerAttr(*passArg));
    } else {
      // NOPASS
      const Fortran::evaluate::Component *component =
          caller.getCallDescription().proc().GetComponent();
      assert(component && "expect component for type-bound procedure call.");
      fir::ExtendedValue pass =
          symMap.lookupSymbol(component->GetFirstSymbol()).toExtendedValue();
      mlir::Value passObject = fir::getBase(pass);
      if (fir::isa_ref_type(passObject.getType()))
        passObject = builder.create<fir::ConvertOp>(
            loc, passObject.getType().dyn_cast<fir::ReferenceType>().getEleTy(),
            passObject);
      dispatch = builder.create<fir::DispatchOp>(
          loc, funcType.getResults(), builder.getStringAttr(procName),
          passObject, operands, nullptr);
    }
    callResult = dispatch.getResult(0);
    callNumResults = dispatch.getNumResults();
  } else {
    // Standard procedure call with fir.call.
    auto call = builder.create<fir::CallOp>(loc, funcType.getResults(),
                                            funcSymbolAttr, operands);
    callResult = call.getResult(0);
    callNumResults = call.getNumResults();
  }

  if (caller.mustSaveResult()) {
    assert(allocatedResult.has_value());
    builder.create<fir::SaveResultOp>(loc, callResult,
                                      fir::getBase(*allocatedResult),
                                      arrayResultShape, resultLengths);
  }

  if (allocatedResult) {
    allocatedResult->match(
        [&](const fir::MutableBoxValue &box) {
          if (box.isAllocatable()) {
            // 9.7.3.2 point 4. Finalize allocatables.
            fir::FirOpBuilder *bldr = &converter.getFirOpBuilder();
            stmtCtx.attachCleanup([bldr, loc, box]() {
              fir::factory::genFinalization(*bldr, loc, box);
            });
          }
        },
        [](const auto &) {});
    return *allocatedResult;
  }

  if (!resultType)
    return mlir::Value{}; // subroutine call
  // For now, Fortran return values are implemented with a single MLIR
  // function return value.
  assert(callNumResults == 1 && "Expected exactly one result in FUNCTION call");
  (void)callNumResults;

  // Call a BIND(C) function that return a char.
  if (caller.characterize().IsBindC() &&
      funcType.getResults()[0].isa<fir::CharacterType>()) {
    fir::CharacterType charTy =
        funcType.getResults()[0].dyn_cast<fir::CharacterType>();
    mlir::Value len = builder.createIntegerConstant(
        loc, builder.getCharacterLengthType(), charTy.getLen());
    return fir::CharBoxValue{callResult, len};
  }

  return callResult;
}

static hlfir::EntityWithAttributes genStmtFunctionRef(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
    const Fortran::evaluate::ProcedureRef &procRef) {
  const Fortran::semantics::Symbol *symbol = procRef.proc().GetSymbol();
  assert(symbol && "expected symbol in ProcedureRef of statement functions");
  const auto &details = symbol->get<Fortran::semantics::SubprogramDetails>();
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  // Statement functions have their own scope, we just need to associate
  // the dummy symbols to argument expressions. There are no
  // optional/alternate return arguments. Statement functions cannot be
  // recursive (directly or indirectly) so it is safe to add dummy symbols to
  // the local map here.
  symMap.pushScope();
  llvm::SmallVector<hlfir::AssociateOp> exprAssociations;
  for (auto [arg, bind] : llvm::zip(details.dummyArgs(), procRef.arguments())) {
    assert(arg && "alternate return in statement function");
    assert(bind && "optional argument in statement function");
    const auto *expr = bind->UnwrapExpr();
    // TODO: assumed type in statement function, that surprisingly seems
    // allowed, probably because nobody thought of restricting this usage.
    // gfortran/ifort compiles this.
    assert(expr && "assumed type used as statement function argument");
    // As per Fortran 2018 C1580, statement function arguments can only be
    // scalars.
    // The only care is to use the dummy character explicit length if any
    // instead of the actual argument length (that can be bigger).
    hlfir::EntityWithAttributes loweredArg = Fortran::lower::convertExprToHLFIR(
        loc, converter, *expr, symMap, stmtCtx);
    fir::FortranVariableOpInterface variableIface = loweredArg.getIfVariable();
    if (!variableIface) {
      // So far only FortranVariableOpInterface can be mapped to symbols.
      // Create an hlfir.associate to create a variable from a potential
      // value argument.
      mlir::Type argType = converter.genType(*arg);
      auto associate = hlfir::genAssociateExpr(
          loc, builder, loweredArg, argType, toStringRef(arg->name()));
      exprAssociations.push_back(associate);
      variableIface = associate;
    }
    const Fortran::semantics::DeclTypeSpec *type = arg->GetType();
    if (type &&
        type->category() == Fortran::semantics::DeclTypeSpec::Character) {
      // Instantiate character as if it was a normal dummy argument so that the
      // statement function dummy character length is applied and dealt with
      // correctly.
      symMap.addSymbol(*arg, variableIface.getBase());
      Fortran::lower::mapSymbolAttributes(converter, *arg, symMap, stmtCtx);
    } else {
      // No need to create an extra hlfir.declare otherwise for
      // numerical and logical scalar dummies.
      symMap.addVariableDefinition(*arg, variableIface);
    }
  }

  // Explicitly map statement function host associated symbols to their
  // parent scope lowered symbol box.
  for (const Fortran::semantics::SymbolRef &sym :
       Fortran::evaluate::CollectSymbols(*details.stmtFunction()))
    if (const auto *details =
            sym->detailsIf<Fortran::semantics::HostAssocDetails>())
      converter.copySymbolBinding(details->symbol(), sym);

  hlfir::Entity result = Fortran::lower::convertExprToHLFIR(
      loc, converter, details.stmtFunction().value(), symMap, stmtCtx);
  symMap.popScope();
  // The result must not be a variable.
  result = hlfir::loadTrivialScalar(loc, builder, result);
  if (result.isVariable())
    result = hlfir::Entity{builder.create<hlfir::AsExprOp>(loc, result)};
  for (auto associate : exprAssociations)
    builder.create<hlfir::EndAssociateOp>(loc, associate);
  return hlfir::EntityWithAttributes{result};
}

namespace {
// Structure to hold the information about the call and the lowering context.
// This structure is intended to help threading the information
// through the various lowering calls without having to pass every
// required structure one by one.
struct CallContext {
  CallContext(const Fortran::evaluate::ProcedureRef &procRef,
              std::optional<mlir::Type> resultType, mlir::Location loc,
              Fortran::lower::AbstractConverter &converter,
              Fortran::lower::SymMap &symMap,
              Fortran::lower::StatementContext &stmtCtx)
      : procRef{procRef}, converter{converter}, symMap{symMap},
        stmtCtx{stmtCtx}, resultType{resultType}, loc{loc} {}

  fir::FirOpBuilder &getBuilder() { return converter.getFirOpBuilder(); }

  /// Is this a call to an elemental procedure with at least one array argument?
  bool isElementalProcWithArrayArgs() const {
    if (procRef.IsElemental())
      for (const std::optional<Fortran::evaluate::ActualArgument> &arg :
           procRef.arguments())
        if (arg && arg->Rank() != 0)
          return true;
    return false;
  }

  /// Is this a statement function reference?
  bool isStatementFunctionCall() const {
    if (const Fortran::semantics::Symbol *symbol = procRef.proc().GetSymbol())
      if (const auto *details =
              symbol->detailsIf<Fortran::semantics::SubprogramDetails>())
        return details->stmtFunction().has_value();
    return false;
  }

  const Fortran::evaluate::ProcedureRef &procRef;
  Fortran::lower::AbstractConverter &converter;
  Fortran::lower::SymMap &symMap;
  Fortran::lower::StatementContext &stmtCtx;
  std::optional<mlir::Type> resultType;
  mlir::Location loc;
};

/// This structure holds the initial lowered value of an actual argument that
/// was lowered regardless of the interface, and it holds whether or not it
/// may be absent at runtime and the dummy is optional.
struct PreparedActualArgument {
  hlfir::Entity actual;
  bool handleDynamicOptional;
};
} // namespace

/// Vector of pre-lowered actual arguments. nullopt if the actual is
/// "statically" absent (if it was not syntactically  provided).
using PreparedActualArguments =
    llvm::SmallVector<std::optional<PreparedActualArgument>>;

// Helper to transform a fir::ExtendedValue to an hlfir::EntityWithAttributes.
static hlfir::EntityWithAttributes
extendedValueToHlfirEntity(mlir::Location loc, fir::FirOpBuilder &builder,
                           const fir::ExtendedValue &exv,
                           llvm::StringRef name) {
  mlir::Value firBase = fir::getBase(exv);
  if (fir::isa_trivial(firBase.getType()))
    return hlfir::EntityWithAttributes{firBase};
  return hlfir::genDeclare(loc, builder, exv, name,
                           fir::FortranVariableFlagsAttr{});
}

/// Lower calls to user procedures with actual arguments that have been
/// pre-lowered but not yet prepared according to the interface.
/// This can be called for elemental procedures, but only with scalar
/// arguments: if there are array arguments, it must be provided with
/// the array argument elements value and will return the corresponding
/// scalar result value.
static std::optional<hlfir::EntityWithAttributes>
genUserCall(PreparedActualArguments &loweredActuals,
            Fortran::lower::CallerInterface &caller,
            mlir::FunctionType callSiteType, CallContext &callContext) {
  using PassBy = Fortran::lower::CallerInterface::PassEntityBy;
  mlir::Location loc = callContext.loc;
  fir::FirOpBuilder &builder = callContext.getBuilder();
  llvm::SmallVector<hlfir::AssociateOp> exprAssociations;
  for (auto [preparedActual, arg] :
       llvm::zip(loweredActuals, caller.getPassedArguments())) {
    mlir::Type argTy = callSiteType.getInput(arg.firArgument);
    if (!preparedActual) {
      // Optional dummy argument for which there is no actual argument.
      caller.placeInput(arg, builder.create<fir::AbsentOp>(loc, argTy));
      continue;
    }
    hlfir::Entity actual = preparedActual->actual;
    const auto *expr = arg.entity->UnwrapExpr();
    if (!expr)
      TODO(loc, "assumed type actual argument");

    if (preparedActual->handleDynamicOptional)
      TODO(loc, "passing optional arguments in HLFIR");

    const bool isSimplyContiguous =
        actual.isScalar() ||
        Fortran::evaluate::IsSimplyContiguous(
            *expr, callContext.converter.getFoldingContext());

    switch (arg.passBy) {
    case PassBy::Value: {
      // True pass-by-value semantics.
      auto value = hlfir::loadTrivialScalar(loc, builder, actual);
      if (!value.isValue())
        TODO(loc, "Passing CPTR an CFUNCTPTR VALUE in HLFIR");
      caller.placeInput(arg, builder.createConvert(loc, argTy, value));
    } break;
    case PassBy::BaseAddressValueAttribute: {
      // VALUE attribute or pass-by-reference to a copy semantics. (byval*)
      TODO(loc, "HLFIR PassBy::BaseAddressValueAttribute");
    } break;
    case PassBy::BaseAddress:
    case PassBy::BoxChar: {
      hlfir::Entity entity = actual;
      if (entity.isVariable()) {
        entity = hlfir::derefPointersAndAllocatables(loc, builder, entity);
        // Copy-in non contiguous variable
        if (!isSimplyContiguous)
          TODO(loc, "HLFIR copy-in/copy-out");
      } else {
        hlfir::AssociateOp associate = hlfir::genAssociateExpr(
            loc, builder, entity, argTy, "adapt.valuebyref");
        exprAssociations.push_back(associate);
        entity = hlfir::Entity{associate.getBase()};
      }
      mlir::Value addr =
          arg.passBy == PassBy::BaseAddress
              ? hlfir::genVariableRawAddress(loc, builder, entity)
              : hlfir::genVariableBoxChar(loc, builder, entity);
      caller.placeInput(arg, builder.createConvert(loc, argTy, addr));
    } break;
    case PassBy::CharBoxValueAttribute: {
      TODO(loc, "HLFIR PassBy::CharBoxValueAttribute");
    } break;
    case PassBy::AddressAndLength:
      // PassBy::AddressAndLength is only used for character results. Results
      // are not handled here.
      fir::emitFatalError(
          loc, "unexpected PassBy::AddressAndLength for actual arguments");
      break;
    case PassBy::CharProcTuple: {
      TODO(loc, "HLFIR PassBy::CharProcTuple");
    } break;
    case PassBy::Box: {
      TODO(loc, "HLFIR PassBy::Box");
    } break;
    case PassBy::MutableBox: {
      if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
              *expr)) {
        // If expr is NULL(), the mutableBox created must be a deallocated
        // pointer with the dummy argument characteristics (see table 16.5
        // in Fortran 2018 standard).
        // No length parameters are set for the created box because any non
        // deferred type parameters of the dummy will be evaluated on the
        // callee side, and it is illegal to use NULL without a MOLD if any
        // dummy length parameters are assumed.
        mlir::Type boxTy = fir::dyn_cast_ptrEleTy(argTy);
        assert(boxTy && boxTy.isa<fir::BoxType>() && "must be a fir.box type");
        mlir::Value boxStorage = builder.createTemporary(loc, boxTy);
        mlir::Value nullBox = fir::factory::createUnallocatedBox(
            builder, loc, boxTy, /*nonDeferredParams=*/{});
        builder.create<fir::StoreOp>(loc, nullBox, boxStorage);
        caller.placeInput(arg, boxStorage);
        continue;
      }
      if (fir::isPointerType(argTy) &&
          !Fortran::evaluate::IsObjectPointer(
              *expr, callContext.converter.getFoldingContext())) {
        // Passing a non POINTER actual argument to a POINTER dummy argument.
        // Create a pointer of the dummy argument type and assign the actual
        // argument to it.
        TODO(loc, "Associate POINTER dummy to TARGET argument in HLFIR");
        continue;
      }
      // Passing a POINTER to a POINTER, or an ALLOCATABLE to an ALLOCATABLE.
      assert(actual.isMutableBox() && "actual must be a mutable box");
      caller.placeInput(arg, actual);
      if (fir::isAllocatableType(argTy) && arg.isIntentOut() &&
          Fortran::semantics::IsBindCProcedure(
              *callContext.procRef.proc().GetSymbol())) {
        TODO(loc, "BIND(C) INTENT(OUT) allocatable deallocation in HLFIR");
      }
    } break;
    }
  }
  // Prepare lowered arguments according to the interface
  // and map the lowered values to the dummy
  // arguments.
  fir::ExtendedValue result = Fortran::lower::genCallOpAndResult(
      loc, callContext.converter, callContext.symMap, callContext.stmtCtx,
      caller, callSiteType, callContext.resultType);

  /// Clean-up associations and copy-in.
  for (auto associate : exprAssociations)
    builder.create<hlfir::EndAssociateOp>(loc, associate);
  if (!fir::getBase(result))
    return std::nullopt; // subroutine call.
  // TODO: "move" non pointer results into hlfir.expr.
  return extendedValueToHlfirEntity(loc, builder, result, ".tmp.func_result");
}

/// Lower calls to intrinsic procedures with actual arguments that have been
/// pre-lowered but have not yet been prepared according to the interface.
static hlfir::EntityWithAttributes genIntrinsicRefCore(
    PreparedActualArguments &loweredActuals,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    const Fortran::lower::IntrinsicArgumentLoweringRules *argLowering,
    CallContext &callContext) {
  llvm::SmallVector<fir::ExtendedValue> operands;
  auto &stmtCtx = callContext.stmtCtx;
  auto &converter = callContext.converter;
  mlir::Location loc = callContext.loc;
  for (auto arg : llvm::enumerate(loweredActuals)) {
    if (!arg.value()) {
      operands.emplace_back(Fortran::lower::getAbsentIntrinsicArgument());
      continue;
    }
    hlfir::Entity actual = arg.value()->actual;
    if (arg.value()->handleDynamicOptional)
      TODO(loc, "intrinsic dynamically optional arguments");
    if (!argLowering) {
      // No argument lowering instruction, lower by value.
      operands.emplace_back(
          Fortran::lower::convertToValue(loc, converter, actual, stmtCtx));
      continue;
    }
    // Ad-hoc argument lowering handling.
    Fortran::lower::ArgLoweringRule argRules =
        Fortran::lower::lowerIntrinsicArgumentAs(*argLowering, arg.index());
    switch (argRules.lowerAs) {
    case Fortran::lower::LowerIntrinsicArgAs::Value:
      operands.emplace_back(
          Fortran::lower::convertToValue(loc, converter, actual, stmtCtx));
      continue;
    case Fortran::lower::LowerIntrinsicArgAs::Addr:
      operands.emplace_back(
          Fortran::lower::convertToAddress(loc, converter, actual, stmtCtx));
      continue;
    case Fortran::lower::LowerIntrinsicArgAs::Box:
      operands.emplace_back(
          Fortran::lower::convertToBox(loc, converter, actual, stmtCtx));
      continue;
    case Fortran::lower::LowerIntrinsicArgAs::Inquired:
      TODO(loc, "as inquired arguments in HLFIR");
      continue;
    }
    llvm_unreachable("bad switch");
  }
  fir::FirOpBuilder &builder = callContext.getBuilder();
  // genIntrinsicCall needs the scalar type, even if this is a transformational
  // procedure returning an array.
  std::optional<mlir::Type> scalarResultType;
  if (callContext.resultType)
    scalarResultType = hlfir::getFortranElementType(*callContext.resultType);
  // Let the intrinsic library lower the intrinsic procedure call.
  auto [resultExv, mustBeFreed] = Fortran::lower::genIntrinsicCall(
      callContext.getBuilder(), loc, intrinsic.name, scalarResultType,
      operands);
  hlfir::EntityWithAttributes resultEntity = extendedValueToHlfirEntity(
      loc, builder, resultExv, ".tmp.intrinsic_result");
  // Move result into memory into an hlfir.expr since they are immutable from
  // that point, and the result storage is some temp.
  if (!fir::isa_trivial(resultEntity.getType())) {
    hlfir::AsExprOp asExpr;
    // Character/Derived MERGE lowering returns one of its argument address
    // (this is the only intrinsic implemented in that way so far). The
    // ownership of this address cannot be taken here since it may not be a
    // temp.
    if (intrinsic.name == "merge")
      asExpr = builder.create<hlfir::AsExprOp>(loc, resultEntity);
    else
      asExpr = builder.create<hlfir::AsExprOp>(
          loc, resultEntity, builder.createBool(loc, mustBeFreed));
    resultEntity = hlfir::EntityWithAttributes{asExpr.getResult()};
  }
  return resultEntity;
}

namespace {
template <typename ElementalCallBuilderImpl>
class ElementalCallBuilder {
public:
  std::optional<hlfir::EntityWithAttributes>
  genElementalCall(PreparedActualArguments &loweredActuals, bool isImpure,
                   CallContext &callContext) {
    mlir::Location loc = callContext.loc;
    fir::FirOpBuilder &builder = callContext.getBuilder();
    unsigned numArgs = loweredActuals.size();
    // Step 1: dereference pointers/allocatables and compute elemental shape.
    mlir::Value shape;
    // 10.1.4 p5. Impure elemental procedures must be called in element order.
    bool mustBeOrdered = isImpure;
    for (unsigned i = 0; i < numArgs; ++i) {
      auto &preparedActual = loweredActuals[i];
      if (preparedActual) {
        hlfir::Entity &actual = preparedActual->actual;
        // Elemental procedure dummy arguments cannot be pointer/allocatables
        // (C15100), so it is safe to dereference any pointer or allocatable
        // actual argument now instead of doing this inside the elemental
        // region.
        actual = hlfir::derefPointersAndAllocatables(loc, builder, actual);
        // Better to load scalars outside of the loop when possible.
        if (!preparedActual->handleDynamicOptional &&
            impl().canLoadActualArgumentBeforeLoop(i))
          actual = hlfir::loadTrivialScalar(loc, builder, actual);
        // TODO: merge shape instead of using the first one.
        if (!shape && actual.isArray()) {
          if (preparedActual->handleDynamicOptional)
            TODO(loc, "deal with optional with shapes in HLFIR elemental call");
          shape = hlfir::genShape(loc, builder, actual);
        }
        // 15.8.3 p1. Elemental procedure with intent(out)/intent(inout)
        // arguments must be called in element order.
        if (impl().argMayBeModifiedByCall(i))
          mustBeOrdered = true;
      }
    }
    assert(shape &&
           "elemental array calls must have at least one array arguments");
    if (mustBeOrdered)
      TODO(loc, "ordered elemental calls in HLFIR");
    // Push a new local scope so that any temps made inside the elemental
    // iterations are cleaned up inside the iterations.
    if (!callContext.resultType) {
      // Subroutine case. Generate call inside loop nest.
      auto [innerLoop, oneBasedIndices] =
          hlfir::genLoopNest(loc, builder, shape);
      auto insPt = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(innerLoop.getBody());
      callContext.stmtCtx.pushScope();
      for (auto &preparedActual : loweredActuals)
        if (preparedActual)
          preparedActual->actual = hlfir::getElementAt(
              loc, builder, preparedActual->actual, oneBasedIndices);
      impl().genElementalKernel(loweredActuals, callContext);
      callContext.stmtCtx.finalizeAndPop();
      builder.restoreInsertionPoint(insPt);
      return std::nullopt;
    }
    // Function case: generate call inside hlfir.elemental
    mlir::Type elementType =
        hlfir::getFortranElementType(*callContext.resultType);
    // Get result length parameters.
    llvm::SmallVector<mlir::Value> typeParams;
    if (elementType.isa<fir::CharacterType>() ||
        fir::isRecordWithTypeParameters(elementType)) {
      auto charType = elementType.dyn_cast<fir::CharacterType>();
      if (charType && charType.hasConstantLen())
        typeParams.push_back(builder.createIntegerConstant(
            loc, builder.getIndexType(), charType.getLen()));
      else if (charType)
        typeParams.push_back(impl().computeDynamicCharacterResultLength(
            loweredActuals, callContext));
      else
        TODO(
            loc,
            "compute elemental PDT function result length parameters in HLFIR");
    }
    auto genKernel = [&](mlir::Location l, fir::FirOpBuilder &b,
                         mlir::ValueRange oneBasedIndices) -> hlfir::Entity {
      callContext.stmtCtx.pushScope();
      for (auto &preparedActual : loweredActuals)
        if (preparedActual)
          preparedActual->actual = hlfir::getElementAt(
              l, b, preparedActual->actual, oneBasedIndices);
      auto res = *impl().genElementalKernel(loweredActuals, callContext);
      callContext.stmtCtx.finalizeAndPop();
      // Note that an hlfir.destroy is not emitted for the result since it
      // is still used by the hlfir.yield_element that also marks its last
      // use.
      return res;
    };
    mlir::Value elemental = hlfir::genElementalOp(loc, builder, elementType,
                                                  shape, typeParams, genKernel);
    fir::FirOpBuilder *bldr = &builder;
    callContext.stmtCtx.attachCleanup(
        [=]() { bldr->create<hlfir::DestroyOp>(loc, elemental); });
    return hlfir::EntityWithAttributes{elemental};
  }

private:
  ElementalCallBuilderImpl &impl() {
    return *static_cast<ElementalCallBuilderImpl *>(this);
  }
};

class ElementalUserCallBuilder
    : public ElementalCallBuilder<ElementalUserCallBuilder> {
public:
  ElementalUserCallBuilder(Fortran::lower::CallerInterface &caller,
                           mlir::FunctionType callSiteType)
      : caller{caller}, callSiteType{callSiteType} {}
  std::optional<hlfir::Entity>
  genElementalKernel(PreparedActualArguments &loweredActuals,
                     CallContext &callContext) {
    return genUserCall(loweredActuals, caller, callSiteType, callContext);
  }

  bool argMayBeModifiedByCall(unsigned argIdx) const {
    assert(argIdx < caller.getPassedArguments().size() && "bad argument index");
    return caller.getPassedArguments()[argIdx].mayBeModifiedByCall();
  }

  bool canLoadActualArgumentBeforeLoop(unsigned argIdx) const {
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;
    assert(argIdx < caller.getPassedArguments().size() && "bad argument index");
    // If the actual argument does not need to be passed via an address,
    // or will be passed in the address of a temporary copy, it can be loaded
    // before the elemental loop nest.
    const auto &arg = caller.getPassedArguments()[argIdx];
    return arg.passBy == PassBy::Value ||
           arg.passBy == PassBy::BaseAddressValueAttribute;
  }

  mlir::Value
  computeDynamicCharacterResultLength(PreparedActualArguments &loweredActuals,
                                      CallContext &callContext) {
    TODO(callContext.loc,
         "compute elemental function result length parameters in HLFIR");
  }

private:
  Fortran::lower::CallerInterface &caller;
  mlir::FunctionType callSiteType;
};

class ElementalIntrinsicCallBuilder
    : public ElementalCallBuilder<ElementalIntrinsicCallBuilder> {
public:
  ElementalIntrinsicCallBuilder(
      const Fortran::evaluate::SpecificIntrinsic &intrinsic,
      const Fortran::lower::IntrinsicArgumentLoweringRules *argLowering,
      bool isFunction)
      : intrinsic{intrinsic}, argLowering{argLowering}, isFunction{isFunction} {
  }
  std::optional<hlfir::Entity>
  genElementalKernel(PreparedActualArguments &loweredActuals,
                     CallContext &callContext) {
    return genIntrinsicRefCore(loweredActuals, intrinsic, argLowering,
                               callContext);
  }
  // Elemental intrinsic functions cannot modify their arguments.
  bool argMayBeModifiedByCall(int) const { return !isFunction; }
  bool canLoadActualArgumentBeforeLoop(int) const {
    // Elemental intrinsic functions never need the actual addresses
    // of their arguments.
    return isFunction;
  }

  mlir::Value
  computeDynamicCharacterResultLength(PreparedActualArguments &loweredActuals,
                                      CallContext &callContext) {
    if (intrinsic.name == "adjustr" || intrinsic.name == "adjustl" ||
        intrinsic.name == "merge")
      return hlfir::genCharLength(callContext.loc, callContext.getBuilder(),
                                  loweredActuals[0].value().actual);
    // Character MIN/MAX is the min/max of the arguments length that are
    // present.
    TODO(callContext.loc,
         "compute elemental character min/max function result length in HLFIR");
  }

private:
  const Fortran::evaluate::SpecificIntrinsic &intrinsic;
  const Fortran::lower::IntrinsicArgumentLoweringRules *argLowering;
  const bool isFunction;
};
} // namespace

/// Lower an intrinsic procedure reference.
static hlfir::EntityWithAttributes
genIntrinsicRef(const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                CallContext &callContext) {
  mlir::Location loc = callContext.loc;
  auto &converter = callContext.converter;
  if (Fortran::lower::intrinsicRequiresCustomOptionalHandling(
          callContext.procRef, intrinsic, converter))
    TODO(loc, "special cases of intrinsic with optional arguments");

  PreparedActualArguments loweredActuals;
  const Fortran::lower::IntrinsicArgumentLoweringRules *argLowering =
      Fortran::lower::getIntrinsicArgumentLowering(intrinsic.name);
  for (const auto &arg : llvm::enumerate(callContext.procRef.arguments())) {
    auto *expr =
        Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg.value());
    if (!expr) {
      // Absent optional.
      loweredActuals.push_back(std::nullopt);
      continue;
    }
    auto loweredActual = Fortran::lower::convertExprToHLFIR(
        loc, callContext.converter, *expr, callContext.symMap,
        callContext.stmtCtx);
    bool handleDynamicOptional = false;
    if (argLowering) {
      Fortran::lower::ArgLoweringRule argRules =
          Fortran::lower::lowerIntrinsicArgumentAs(*argLowering, arg.index());
      handleDynamicOptional = argRules.handleDynamicOptional &&
                              Fortran::evaluate::MayBePassedAsAbsentOptional(
                                  *expr, converter.getFoldingContext());
    }
    loweredActuals.push_back(
        PreparedActualArgument{loweredActual, handleDynamicOptional});
  }

  if (callContext.isElementalProcWithArrayArgs()) {
    // All intrinsic elemental functions are pure.
    const bool isFunction = callContext.resultType.has_value();
    return ElementalIntrinsicCallBuilder{intrinsic, argLowering, isFunction}
        .genElementalCall(loweredActuals, /*isImpure=*/!isFunction, callContext)
        .value();
  }
  hlfir::EntityWithAttributes result =
      genIntrinsicRefCore(loweredActuals, intrinsic, argLowering, callContext);
  if (result.getType().isa<hlfir::ExprType>()) {
    fir::FirOpBuilder *bldr = &callContext.getBuilder();
    callContext.stmtCtx.attachCleanup(
        [=]() { bldr->create<hlfir::DestroyOp>(loc, result); });
  }
  return result;
}

/// Main entry point to lower procedure references, regardless of what they are.
static std::optional<hlfir::EntityWithAttributes>
genProcedureRef(CallContext &callContext) {
  mlir::Location loc = callContext.loc;
  if (auto *intrinsic = callContext.procRef.proc().GetSpecificIntrinsic())
    return genIntrinsicRef(*intrinsic, callContext);

  if (callContext.isStatementFunctionCall())
    return genStmtFunctionRef(loc, callContext.converter, callContext.symMap,
                              callContext.stmtCtx, callContext.procRef);

  Fortran::lower::CallerInterface caller(callContext.procRef,
                                         callContext.converter);
  mlir::FunctionType callSiteType = caller.genFunctionType();

  PreparedActualArguments loweredActuals;
  // Lower the actual arguments
  for (const Fortran::lower::CallInterface<
           Fortran::lower::CallerInterface>::PassedEntity &arg :
       caller.getPassedArguments())
    if (const auto *actual = arg.entity) {
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(loc, "assumed type actual argument");

      const bool handleDynamicOptional =
          arg.isOptional() &&
          Fortran::evaluate::MayBePassedAsAbsentOptional(
              *expr, callContext.converter.getFoldingContext());
      auto loweredActual = Fortran::lower::convertExprToHLFIR(
          loc, callContext.converter, *expr, callContext.symMap,
          callContext.stmtCtx);
      loweredActuals.emplace_back(
          PreparedActualArgument{loweredActual, handleDynamicOptional});
    } else {
      // Optional dummy argument for which there is no actual argument.
      loweredActuals.emplace_back(std::nullopt);
    }
  if (callContext.isElementalProcWithArrayArgs()) {
    bool isImpure = false;
    if (const Fortran::semantics::Symbol *procSym =
            callContext.procRef.proc().GetSymbol())
      isImpure = !Fortran::semantics::IsPureProcedure(*procSym);
    return ElementalUserCallBuilder{caller, callSiteType}.genElementalCall(
        loweredActuals, isImpure, callContext);
  }
  return genUserCall(loweredActuals, caller, callSiteType, callContext);
}

std::optional<hlfir::EntityWithAttributes> Fortran::lower::convertCallToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const evaluate::ProcedureRef &procRef, std::optional<mlir::Type> resultType,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  CallContext callContext(procRef, resultType, loc, converter, symMap, stmtCtx);
  return genProcedureRef(callContext);
}
