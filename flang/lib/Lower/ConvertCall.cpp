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
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "flang-lower-expr"

static llvm::cl::opt<bool> useHlfirIntrinsicOps(
    "use-hlfir-intrinsic-ops", llvm::cl::init(true),
    llvm::cl::desc("Lower via HLFIR transformational intrinsic operations such as hlfir.sum"));

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

static bool mustCastFuncOpToCopeWithImplicitInterfaceMismatch(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    mlir::FunctionType callSiteType, mlir::FunctionType funcOpType) {
  // Deal with argument number mismatch by making a function pointer so
  // that function type cast can be inserted. Do not emit a warning here
  // because this can happen in legal program if the function is not
  // defined here and it was first passed as an argument without any more
  // information.
  if (callSiteType.getNumResults() != funcOpType.getNumResults() ||
      callSiteType.getNumInputs() != funcOpType.getNumInputs())
    return true;

  // Implicit interface result type mismatch are not standard Fortran, but
  // some compilers are not complaining about it.  The front end is not
  // protecting lowering from this currently. Support this with a
  // discouraging warning.
  // Cast the actual function to the current caller implicit type because
  // that is the behavior we would get if we could not see the definition.
  if (callSiteType.getResults() != funcOpType.getResults()) {
    LLVM_DEBUG(mlir::emitWarning(
        loc, "a return type mismatch is not standard compliant and may "
             "lead to undefined behavior."));
    return true;
  }

  // In HLFIR, there is little attempt to cope with implicit interface
  // mismatch on the arguments. The argument are always prepared according
  // to the implicit interface. Cast the actual function if any of the
  // argument mismatch cannot be dealt with a simple fir.convert.
  if (converter.getLoweringOptions().getLowerToHighLevelFIR())
    for (auto [actualType, dummyType] :
         llvm::zip(callSiteType.getInputs(), funcOpType.getInputs()))
      if (actualType != dummyType &&
          !fir::ConvertOp::canBeConverted(actualType, dummyType))
        return true;
  return false;
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
    funcPointer = fir::getBase(converter.getSymbolExtendedValue(*sym, &symMap));
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
    // When this is not a call to an internal procedure (where there is a
    // mismatch due to the extra argument, but the interface is otherwise
    // explicit and safe), handle interface mismatch due to F77 implicit
    // interface "abuse" with a function address cast if needed.
    if (!addHostAssociations &&
        mustCastFuncOpToCopeWithImplicitInterfaceMismatch(
            loc, converter, callSiteType, funcOpType))
      funcPointer = builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
    else
      funcSymbolAttr = symbolAttr;
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
      fir::ExtendedValue pass = converter.getSymbolExtendedValue(
          component->GetFirstSymbol(), &symMap);
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
    // 7.5.6.3 point 5. Derived-type finalization for nonpointer function.
    // Check if the derived-type is finalizable if it is a monorphic
    // derived-type.
    // For polymorphic and unlimited polymorphic enities call the runtime
    // in any cases.
    std::optional<Fortran::evaluate::DynamicType> retTy =
        caller.getCallDescription().proc().GetType();
    if (!fir::isPointerType(funcType.getResults()[0]) && retTy &&
        (retTy->category() == Fortran::common::TypeCategory::Derived ||
         retTy->IsPolymorphic() || retTy->IsUnlimitedPolymorphic())) {
      if (retTy->IsPolymorphic() || retTy->IsUnlimitedPolymorphic()) {
        auto *bldr = &converter.getFirOpBuilder();
        stmtCtx.attachCleanup([bldr, loc, allocatedResult]() {
          fir::runtime::genDerivedTypeDestroy(*bldr, loc,
                                              fir::getBase(*allocatedResult));
        });
      } else {
        const Fortran::semantics::DerivedTypeSpec &typeSpec =
            retTy->GetDerivedTypeSpec();
        if (Fortran::semantics::IsFinalizable(typeSpec)) {
          auto *bldr = &converter.getFirOpBuilder();
          stmtCtx.attachCleanup([bldr, loc, allocatedResult]() {
            mlir::Value box = bldr->createBox(loc, *allocatedResult);
            fir::runtime::genDerivedTypeDestroy(*bldr, loc, box);
          });
        }
      }
    }
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

  PreparedActualArgument(hlfir::Entity actual,
                         std::optional<mlir::Value> isPresent)
      : actual{actual}, isPresent{isPresent} {}
  void setElementalIndices(mlir::ValueRange &indices) {
    oneBasedElementalIndices = &indices;
  }
  hlfir::Entity getActual(mlir::Location loc,
                          fir::FirOpBuilder &builder) const {
    if (oneBasedElementalIndices)
      return hlfir::getElementAt(loc, builder, actual,
                                 *oneBasedElementalIndices);
    return actual;
  }
  hlfir::Entity getOriginalActual() const { return actual; }
  void setOriginalActual(hlfir::Entity newActual) { actual = newActual; }
  bool handleDynamicOptional() const { return isPresent.has_value(); }
  mlir::Value getIsPresent() const {
    assert(handleDynamicOptional() && "not a dynamic optional");
    return *isPresent;
  }

  void resetOptionalAspect() { isPresent = std::nullopt; }

private:
  hlfir::Entity actual;
  mlir::ValueRange *oneBasedElementalIndices{nullptr};
  // When the actual may be dynamically optional, "isPresent"
  // holds a boolean value indicating the presence of the
  // actual argument at runtime.
  std::optional<mlir::Value> isPresent;
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
  mlir::Type firBaseTy = firBase.getType();
  if (fir::isa_trivial(firBaseTy))
    return hlfir::EntityWithAttributes{firBase};
  if (auto charTy = firBase.getType().dyn_cast<fir::CharacterType>()) {
    // CHAR() intrinsic and BIND(C) procedures returning CHARACTER(1)
    // are lowered to a fir.char<kind,1> that is not in memory.
    // This tends to cause a lot of bugs because the rest of the
    // infrastructure is mostly tested with characters that are
    // in memory.
    // To avoid having to deal with this special case here and there,
    // place it in memory here. If this turns out to be suboptimal,
    // this could be fixed, but for now llvm opt -O1 is able to get
    // rid of the memory indirection in a = char(b), so there is
    // little incentive to increase the compiler complexity.
    hlfir::Entity storage{builder.createTemporary(loc, charTy)};
    builder.create<fir::StoreOp>(loc, firBase, storage);
    auto asExpr = builder.create<hlfir::AsExprOp>(
        loc, storage, /*mustFree=*/builder.createBool(loc, false));
    return hlfir::EntityWithAttributes{asExpr.getResult()};
  }
  return hlfir::genDeclare(loc, builder, exv, name,
                           fir::FortranVariableFlagsAttr{});
}
namespace {
/// Structure to hold the clean-up related to a dummy argument preparation
/// that may have to be done after a call (copy-out or temporary deallocation).
struct CallCleanUp {
  struct CopyIn {
    void genCleanUp(mlir::Location loc, fir::FirOpBuilder &builder) {
      builder.create<hlfir::CopyOutOp>(loc, copiedIn, wasCopied, copyBackVar);
    }
    mlir::Value copiedIn;
    mlir::Value wasCopied;
    // copyBackVar may be null if copy back is not needed.
    mlir::Value copyBackVar;
  };
  struct ExprAssociate {
    void genCleanUp(mlir::Location loc, fir::FirOpBuilder &builder) {
      builder.create<hlfir::EndAssociateOp>(loc, tempVar, mustFree);
    }
    mlir::Value tempVar;
    mlir::Value mustFree;
  };
  void genCleanUp(mlir::Location loc, fir::FirOpBuilder &builder) {
    std::visit([&](auto &c) { c.genCleanUp(loc, builder); }, cleanUp);
  }
  std::variant<CopyIn, ExprAssociate> cleanUp;
};

/// Structure representing a prepared dummy argument.
/// It holds the value to be passed in the call and any related
/// clean-ups to be done after the call.
struct PreparedDummyArgument {
  void setCopyInCleanUp(mlir::Value copiedIn, mlir::Value wasCopied,
                        mlir::Value copyBackVar) {
    assert(!maybeCleanUp.has_value() && "clean-up already set");
    maybeCleanUp =
        CallCleanUp{CallCleanUp::CopyIn{copiedIn, wasCopied, copyBackVar}};
  }
  void setExprAssociateCleanUp(mlir::Value tempVar, mlir::Value wasCopied) {
    assert(!maybeCleanUp.has_value() && "clean-up already set");
    maybeCleanUp = CallCleanUp{CallCleanUp::ExprAssociate{tempVar, wasCopied}};
  }

  mlir::Value dummy;
  std::optional<CallCleanUp> maybeCleanUp;
};

/// Structure to help conditionally preparing a dummy argument based
/// on the actual argument presence.
/// It helps "wrapping" the dummy and the clean-up information in
/// an if (present) {...}:
///
///  %conditionallyPrepared = fir.if (%present) {
///    fir.result %preparedDummy
///  } else {
///    fir.result %absent
///  }
///
struct ConditionallyPreparedDummy {
  /// Create ConditionallyPreparedDummy from a preparedDummy that must
  /// be wrapped in a fir.if.
  ConditionallyPreparedDummy(PreparedDummyArgument &preparedDummy) {
    thenResultValues.push_back(preparedDummy.dummy);
    if (preparedDummy.maybeCleanUp) {
      if (const auto *copyInCleanUp = std::get_if<CallCleanUp::CopyIn>(
              &preparedDummy.maybeCleanUp->cleanUp)) {
        thenResultValues.push_back(copyInCleanUp->copiedIn);
        thenResultValues.push_back(copyInCleanUp->wasCopied);
        if (copyInCleanUp->copyBackVar)
          thenResultValues.push_back(copyInCleanUp->copyBackVar);
      } else {
        const auto &exprAssociate = std::get<CallCleanUp::ExprAssociate>(
            preparedDummy.maybeCleanUp->cleanUp);
        thenResultValues.push_back(exprAssociate.tempVar);
        thenResultValues.push_back(exprAssociate.mustFree);
      }
    }
  }

  /// Get the result types of the wrapping fir.if that must be created.
  llvm::SmallVector<mlir::Type> getIfResulTypes() const {
    llvm::SmallVector<mlir::Type> types;
    for (mlir::Value res : thenResultValues)
      types.push_back(res.getType());
    return types;
  }

  /// Generate the "fir.result %preparedDummy" in the then branch of the
  /// wrapping fir.if.
  void genThenResult(mlir::Location loc, fir::FirOpBuilder &builder) const {
    builder.create<fir::ResultOp>(loc, thenResultValues);
  }

  /// Generate the "fir.result %absent" in the else branch of the
  /// wrapping fir.if.
  void genElseResult(mlir::Location loc, fir::FirOpBuilder &builder) const {
    llvm::SmallVector<mlir::Value> elseResultValues;
    mlir::Type i1Type = builder.getI1Type();
    for (mlir::Value res : thenResultValues) {
      mlir::Type type = res.getType();
      if (type == i1Type)
        elseResultValues.push_back(builder.createBool(loc, false));
      else
        elseResultValues.push_back(builder.create<fir::AbsentOp>(loc, type));
    }
    builder.create<fir::ResultOp>(loc, elseResultValues);
  }

  /// Once the fir.if has been created, get the resulting %conditionallyPrepared
  /// dummy argument.
  PreparedDummyArgument
  getPreparedDummy(fir::IfOp ifOp,
                   const PreparedDummyArgument &unconditionalDummy) {
    PreparedDummyArgument preparedDummy;
    preparedDummy.dummy = ifOp.getResults()[0];
    if (unconditionalDummy.maybeCleanUp) {
      if (const auto *copyInCleanUp = std::get_if<CallCleanUp::CopyIn>(
              &unconditionalDummy.maybeCleanUp->cleanUp)) {
        mlir::Value copyBackVar;
        if (copyInCleanUp->copyBackVar)
          copyBackVar = ifOp.getResults().back();
        preparedDummy.setCopyInCleanUp(ifOp.getResults()[1],
                                       ifOp.getResults()[2], copyBackVar);
      } else {
        preparedDummy.setExprAssociateCleanUp(ifOp.getResults()[1],
                                              ifOp.getResults()[2]);
      }
    }
    return preparedDummy;
  }

  llvm::SmallVector<mlir::Value> thenResultValues;
};
} // namespace

/// Fix-up the fact that it is supported to pass a character procedure
/// designator to a non character procedure dummy procedure and vice-versa, even
/// in case of explicit interface. Uglier cases where an object is passed as
/// procedure designator or vice versa are handled only for implicit interfaces
/// (refused by semantics with explicit interface), and handled with a funcOp
/// cast like other implicit interface mismatches.
static hlfir::Entity fixProcedureDummyMismatch(mlir::Location loc,
                                               fir::FirOpBuilder &builder,
                                               hlfir::Entity actual,
                                               mlir::Type dummyType) {
  if (actual.getType().isa<fir::BoxProcType>() &&
      fir::isCharacterProcedureTuple(dummyType)) {
    mlir::Value length =
        builder.create<fir::UndefOp>(loc, builder.getCharacterLengthType());
    mlir::Value tuple = fir::factory::createCharacterProcedureTuple(
        builder, loc, dummyType, actual, length);
    return hlfir::Entity{tuple};
  }
  assert(fir::isCharacterProcedureTuple(actual.getType()) &&
         dummyType.isa<fir::BoxProcType>() &&
         "unsupported dummy procedure mismatch with the actual argument");
  mlir::Value boxProc = fir::factory::extractCharacterProcedureTuple(
                            builder, loc, actual, /*openBoxProc=*/false)
                            .first;
  return hlfir::Entity{boxProc};
}

/// When dummy is not ALLOCATABLE, POINTER and is not passed in register,
/// prepare the actual argument according to the interface. Do as needed:
/// - address element if this is an array argument in an elemental call.
/// - set dynamic type to the dummy type if the dummy is not polymorphic.
/// - copy-in into contiguous variable if the dummy must be contiguous
/// - copy into a temporary if the dummy has the VALUE attribute.
/// - package the prepared dummy as required (fir.box, fir.class,
///   fir.box_char...).
/// This function should only be called with an actual that is present.
/// The optional aspects must be handled by this function user.
static PreparedDummyArgument preparePresentUserCallActualArgument(
    mlir::Location loc, fir::FirOpBuilder &builder,
    const PreparedActualArgument &preparedActual, mlir::Type dummyType,
    const Fortran::lower::CallerInterface::PassedEntity &arg,
    const Fortran::lower::SomeExpr &expr,
    Fortran::evaluate::FoldingContext &foldingContext) {

  // Step 1: get the actual argument, which includes addressing the
  // element if this is an array in an elemental call.
  hlfir::Entity actual = preparedActual.getActual(loc, builder);

  // Do nothing if this is a procedure argument. It is already a
  // fir.boxproc/fir.tuple<fir.boxproc, len> as it should.
  if (actual.isProcedure()) {
    if (actual.getType() != dummyType)
      actual = fixProcedureDummyMismatch(loc, builder, actual, dummyType);
    return PreparedDummyArgument{actual, std::nullopt};
  }

  const bool passingPolymorphicToNonPolymorphic =
      actual.isPolymorphic() && !fir::isPolymorphicType(dummyType);

  // When passing a CLASS(T) to TYPE(T), only the "T" part must be
  // passed. Unless the entity is a scalar passed by raw address, a
  // new descriptor must be made using the dummy argument type as
  // dynamic type. This must be done before any copy/copy-in because the
  // dynamic type matters to determine the contiguity.
  const bool mustSetDynamicTypeToDummyType =
      passingPolymorphicToNonPolymorphic &&
      (actual.isArray() || dummyType.isa<fir::BaseBoxType>());

  // The simple contiguity of the actual is "lost" when passing a polymorphic
  // to a non polymorphic entity because the dummy dynamic type matters for
  // the contiguity.
  const bool mustDoCopyInOut =
      actual.isArray() && arg.mustBeMadeContiguous() &&
      (passingPolymorphicToNonPolymorphic ||
       !Fortran::evaluate::IsSimplyContiguous(expr, foldingContext));

  // Step 2: prepare the storage for the dummy arguments, ensuring that it
  // matches the dummy requirements (e.g., must be contiguous or must be
  // a temporary).
  PreparedDummyArgument preparedDummy;
  hlfir::Entity entity =
      hlfir::derefPointersAndAllocatables(loc, builder, actual);
  if (entity.isVariable()) {
    if (mustSetDynamicTypeToDummyType) {
      // Note: this is important to do this before any copy-in or copy so
      // that the dummy is contiguous according to the dummy type.
      mlir::Type boxType =
          fir::BoxType::get(hlfir::getFortranElementOrSequenceType(dummyType));
      entity = hlfir::Entity{builder.create<fir::ReboxOp>(
          loc, boxType, entity, /*shape=*/mlir::Value{},
          /*slice=*/mlir::Value{})};
    }
    if (arg.hasValueAttribute()) {
      // Make a copy in a temporary.
      auto copy = builder.create<hlfir::AsExprOp>(loc, entity);
      hlfir::AssociateOp associate = hlfir::genAssociateExpr(
          loc, builder, hlfir::Entity{copy}, dummyType, "adapt.valuebyref");
      entity = hlfir::Entity{associate.getBase()};
      // Register the temporary destruction after the call.
      preparedDummy.setExprAssociateCleanUp(
          associate.getFirBase(), associate.getMustFreeStrorageFlag());
    } else if (mustDoCopyInOut) {
      // Copy-in non contiguous variables.
      assert(entity.getType().isa<fir::BaseBoxType>() &&
             "expect non simply contiguous variables to be boxes");
      auto copyIn = builder.create<hlfir::CopyInOp>(
          loc, entity, /*var_is_present=*/mlir::Value{});
      entity = hlfir::Entity{copyIn.getCopiedIn()};
      // Register the copy-out after the call.
      preparedDummy.setCopyInCleanUp(
          copyIn.getCopiedIn(), copyIn.getWasCopied(),
          arg.mayBeModifiedByCall() ? copyIn.getVar() : mlir::Value{});
    }
  } else {
    // The actual is an expression value, place it into a temporary
    // and register the temporary destruction after the call.
    if (mustSetDynamicTypeToDummyType)
      TODO(loc, "passing polymorphic array expression to non polymorphic "
                "contiguous dummy");
    hlfir::AssociateOp associate = hlfir::genAssociateExpr(
        loc, builder, entity, dummyType, "adapt.valuebyref");
    entity = hlfir::Entity{associate.getBase()};
    preparedDummy.setExprAssociateCleanUp(associate.getFirBase(),
                                          associate.getMustFreeStrorageFlag());
  }

  // Step 3: now that the dummy argument storage has been prepared, package
  // it according to the interface.
  mlir::Value addr;
  if (dummyType.isa<fir::BoxCharType>()) {
    addr = hlfir::genVariableBoxChar(loc, builder, entity);
  } else if (dummyType.isa<fir::BaseBoxType>()) {
    entity = hlfir::genVariableBox(loc, builder, entity);
    // Ensures the box has the right attributes and that it holds an
    // addendum if needed.
    mlir::Type boxEleType =
        entity.getType().cast<fir::BaseBoxType>().getEleTy();
    // For now, assume it is not OK to pass the allocatable/pointer
    // descriptor to a non pointer/allocatable dummy. That is a strict
    // interpretation of 18.3.6 point 4 that stipulates the descriptor
    // has the dummy attributes in BIND(C) contexts.
    const bool actualBoxHasAllocatableOrPointerFlag =
        fir::isa_ref_type(boxEleType);
    // On the callee side, the current code generated for unlimited
    // polymorphic might unconditionally read the addendum. Intrinsic type
    // descriptors may not have an addendum, the rebox below will create a
    // descriptor with an addendum in such case.
    const bool actualBoxHasAddendum =
        fir::unwrapRefType(boxEleType).isa<fir::RecordType, mlir::NoneType>();
    const bool needToAddAddendum =
        fir::isUnlimitedPolymorphicType(dummyType) && !actualBoxHasAddendum;
    if (needToAddAddendum || actualBoxHasAllocatableOrPointerFlag)
      entity = hlfir::Entity{builder.create<fir::ReboxOp>(
          loc, dummyType, entity, /*shape=*/mlir::Value{},
          /*slice=*/mlir::Value{})};
    addr = entity;
  } else {
    addr = hlfir::genVariableRawAddress(loc, builder, entity);
  }
  preparedDummy.dummy = builder.createConvert(loc, dummyType, addr);
  return preparedDummy;
}

/// When dummy is not ALLOCATABLE, POINTER and is not passed in register,
/// prepare the actual argument according to the interface, taking care
/// of any optional aspect.
static PreparedDummyArgument prepareUserCallActualArgument(
    mlir::Location loc, fir::FirOpBuilder &builder,
    const PreparedActualArgument &preparedActual, mlir::Type dummyType,
    const Fortran::lower::CallerInterface::PassedEntity &arg,
    const Fortran::lower::SomeExpr &expr,
    Fortran::evaluate::FoldingContext &foldingContext) {
  if (!preparedActual.handleDynamicOptional())
    return preparePresentUserCallActualArgument(
        loc, builder, preparedActual, dummyType, arg, expr, foldingContext);

  // Conditional dummy argument preparation. The actual may be absent
  // at runtime, causing any addressing, copy, and packaging to have
  // undefined behavior.
  // To simplify the handling of this case, the "normal" dummy preparation
  // helper is used, except its generated code is wrapped inside a
  // fir.if(present).
  mlir::Value isPresent = preparedActual.getIsPresent();
  mlir::OpBuilder::InsertPoint insertPt = builder.saveInsertionPoint();

  // Code generated in a preparation block that will become the
  // "then" block in "if (present) then {} else {}". The reason
  // for this unusual if/then/else generation is that the number
  // and types of the if results will depend on how the argument
  // is prepared, and forecasting that here would be brittle.
  auto badIfOp = builder.create<fir::IfOp>(loc, dummyType, isPresent,
                                           /*withElseRegion=*/false);
  mlir::Block *preparationBlock = &badIfOp.getThenRegion().front();
  builder.setInsertionPointToStart(preparationBlock);
  PreparedDummyArgument unconditionalDummy =
      preparePresentUserCallActualArgument(
          loc, builder, preparedActual, dummyType, arg, expr, foldingContext);
  builder.restoreInsertionPoint(insertPt);

  // TODO: when forwarding an optional to an optional of the same kind
  // (i.e, unconditionalDummy.dummy was not created in preparationBlock),
  // the if/then/else generation could be skipped to improve the generated
  // code.

  // Now that the result types of the ifOp can be deduced, generate
  // the "real" ifOp (operation result types cannot be changed, so
  // badIfOp cannot be modified and used here).
  llvm::SmallVector<mlir::Type> ifOpResultTypes;
  ConditionallyPreparedDummy conditionalDummy(unconditionalDummy);
  auto ifOp = builder.create<fir::IfOp>(loc, conditionalDummy.getIfResulTypes(),
                                        isPresent,
                                        /*withElseRegion=*/true);
  // Move "preparationBlock" into the "then" of the new
  // fir.if operation and create fir.result propagating
  // unconditionalDummy.
  preparationBlock->moveBefore(&ifOp.getThenRegion().back());
  ifOp.getThenRegion().back().erase();
  builder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
  conditionalDummy.genThenResult(loc, builder);

  // Generate "else" branch with returning absent values.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  conditionalDummy.genElseResult(loc, builder);

  // Build dummy from IfOpResults.
  builder.setInsertionPointAfter(ifOp);
  PreparedDummyArgument result =
      conditionalDummy.getPreparedDummy(ifOp, unconditionalDummy);
  badIfOp->erase();
  return result;
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
  llvm::SmallVector<CallCleanUp> callCleanUps;
  for (auto [preparedActual, arg] :
       llvm::zip(loweredActuals, caller.getPassedArguments())) {
    mlir::Type argTy = callSiteType.getInput(arg.firArgument);
    if (!preparedActual) {
      // Optional dummy argument for which there is no actual argument.
      caller.placeInput(arg, builder.create<fir::AbsentOp>(loc, argTy));
      continue;
    }
    const auto *expr = arg.entity->UnwrapExpr();
    if (!expr)
      TODO(loc, "assumed type actual argument");

    switch (arg.passBy) {
    case PassBy::Value: {
      // True pass-by-value semantics.
      assert(!preparedActual->handleDynamicOptional() && "cannot be optional");
      hlfir::Entity actual = preparedActual->getActual(loc, builder);
      auto value = hlfir::loadTrivialScalar(loc, builder, actual);
      if (!value.isValue())
        TODO(loc, "Passing CPTR an CFUNCTPTR VALUE in HLFIR");
      caller.placeInput(arg, builder.createConvert(loc, argTy, value));
    } break;
    case PassBy::BaseAddressValueAttribute:
    case PassBy::CharBoxValueAttribute:
    case PassBy::Box:
    case PassBy::BaseAddress:
    case PassBy::BoxChar: {
      PreparedDummyArgument preparedDummy = prepareUserCallActualArgument(
          loc, builder, *preparedActual, argTy, arg, *expr,
          callContext.converter.getFoldingContext());
      if (preparedDummy.maybeCleanUp.has_value())
        callCleanUps.emplace_back(std::move(*preparedDummy.maybeCleanUp));
      caller.placeInput(arg, preparedDummy.dummy);
    } break;
    case PassBy::AddressAndLength:
      // PassBy::AddressAndLength is only used for character results. Results
      // are not handled here.
      fir::emitFatalError(
          loc, "unexpected PassBy::AddressAndLength for actual arguments");
      break;
    case PassBy::CharProcTuple: {
      hlfir::Entity actual = preparedActual->getActual(loc, builder);
      if (!fir::isCharacterProcedureTuple(actual.getType()))
        actual = fixProcedureDummyMismatch(loc, builder, actual, argTy);
      caller.placeInput(arg, actual);
    } break;
    case PassBy::MutableBox: {
      hlfir::Entity actual = preparedActual->getActual(loc, builder);
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
  for (auto cleanUp : callCleanUps)
    cleanUp.genCleanUp(loc, builder);

  if (!fir::getBase(result))
    return std::nullopt; // subroutine call.
  // TODO: "move" non pointer results into hlfir.expr.
  return extendedValueToHlfirEntity(loc, builder, result, ".tmp.func_result");
}

/// Lower calls to intrinsic procedures with actual arguments that have been
/// pre-lowered but have not yet been prepared according to the interface.
static std::optional<hlfir::EntityWithAttributes>
genIntrinsicRefCore(PreparedActualArguments &loweredActuals,
                    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                    const fir::IntrinsicArgumentLoweringRules *argLowering,
                    CallContext &callContext) {
  llvm::SmallVector<fir::ExtendedValue> operands;
  auto &stmtCtx = callContext.stmtCtx;
  auto &converter = callContext.converter;
  fir::FirOpBuilder &builder = callContext.getBuilder();
  mlir::Location loc = callContext.loc;
  for (auto arg : llvm::enumerate(loweredActuals)) {
    if (!arg.value()) {
      operands.emplace_back(fir::getAbsentIntrinsicArgument());
      continue;
    }
    if (arg.value()->handleDynamicOptional())
      TODO(loc, "intrinsic dynamically optional arguments");
    hlfir::Entity actual = arg.value()->getActual(loc, builder);
    if (!argLowering) {
      // No argument lowering instruction, lower by value.
      operands.emplace_back(
          Fortran::lower::convertToValue(loc, converter, actual, stmtCtx));
      continue;
    }
    // Helper to get the type of the Fortran expression in case it is a
    // computed value that must be placed in memory (logicals are computed as
    // i1, but must be placed in memory as fir.logical).
    auto getActualFortranElementType = [&]() {
      const Fortran::lower::SomeExpr *expr =
          callContext.procRef.UnwrapArgExpr(arg.index());
      assert(expr && "must be an expr");
      mlir::Type type = converter.genType(*expr);
      return hlfir::getFortranElementType(type);
    };
    // Ad-hoc argument lowering handling.
    fir::ArgLoweringRule argRules =
        fir::lowerIntrinsicArgumentAs(*argLowering, arg.index());
    switch (argRules.lowerAs) {
    case fir::LowerIntrinsicArgAs::Value:
      operands.emplace_back(
          Fortran::lower::convertToValue(loc, converter, actual, stmtCtx));
      continue;
    case fir::LowerIntrinsicArgAs::Addr:
      operands.emplace_back(Fortran::lower::convertToAddress(
          loc, converter, actual, stmtCtx, getActualFortranElementType()));
      continue;
    case fir::LowerIntrinsicArgAs::Box:
      operands.emplace_back(Fortran::lower::convertToBox(
          loc, converter, actual, stmtCtx, getActualFortranElementType()));
      continue;
    case fir::LowerIntrinsicArgAs::Inquired:
      // Place hlfir.expr in memory, and unbox fir.boxchar. Other entities
      // are translated to fir::ExtendedValue without transformation (notably,
      // pointers/allocatable are not dereferenced).
      // TODO: once lowering to FIR retires, UBOUND and LBOUND can be simplified
      // since the fir.box lowered here are now guaranteed to contain the local
      // lower bounds thanks to the hlfir.declare (the extra rebox can be
      // removed).
      operands.emplace_back(Fortran::lower::translateToExtendedValue(
          loc, builder, actual, stmtCtx));
      continue;
    }
    llvm_unreachable("bad switch");
  }
  // genIntrinsicCall needs the scalar type, even if this is a transformational
  // procedure returning an array.
  std::optional<mlir::Type> scalarResultType;
  if (callContext.resultType)
    scalarResultType = hlfir::getFortranElementType(*callContext.resultType);
  // Let the intrinsic library lower the intrinsic procedure call.
  auto [resultExv, mustBeFreed] =
      genIntrinsicCall(callContext.getBuilder(), loc, intrinsic.name,
                       scalarResultType, operands);
  if (!fir::getBase(resultExv))
    return std::nullopt;
  hlfir::EntityWithAttributes resultEntity = extendedValueToHlfirEntity(
      loc, builder, resultExv, ".tmp.intrinsic_result");
  // Move result into memory into an hlfir.expr since they are immutable from
  // that point, and the result storage is some temp.
  if (resultEntity.isVariable()) {
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

/// Lower calls to intrinsic procedures with actual arguments that have been
/// pre-lowered but have not yet been prepared according to the interface.
static std::optional<hlfir::EntityWithAttributes>
genHLFIRIntrinsicRefCore(PreparedActualArguments &loweredActuals,
                         const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                         const fir::IntrinsicArgumentLoweringRules *argLowering,
                         CallContext &callContext) {
  if (!useHlfirIntrinsicOps)
    return genIntrinsicRefCore(loweredActuals, intrinsic, argLowering, callContext);

  fir::FirOpBuilder &builder = callContext.getBuilder();
  mlir::Location loc = callContext.loc;

  auto getOperandVector = [&](PreparedActualArguments &loweredActuals) {
    llvm::SmallVector<mlir::Value> operands;
    operands.reserve(loweredActuals.size());

    for (size_t i = 0; i < loweredActuals.size(); ++i) {
      std::optional<PreparedActualArgument> arg = loweredActuals[i];
      if (!arg) {
        operands.emplace_back();
        continue;
      }
      hlfir::Entity actual = arg->getOriginalActual();
      mlir::Value valArg;

      fir::ArgLoweringRule argRules =
          fir::lowerIntrinsicArgumentAs(*argLowering, i);
      if (!argRules.handleDynamicOptional &&
          argRules.lowerAs != fir::LowerIntrinsicArgAs::Inquired)
        valArg = hlfir::derefPointersAndAllocatables(loc, builder, actual);
      else
        valArg = actual.getBase();

      operands.emplace_back(valArg);
    }
    return operands;
  };

  auto computeResultType = [&](mlir::Value argArray,
                               mlir::Type stmtResultType) -> mlir::Type {
    hlfir::ExprType::Shape resultShape;
    mlir::Type normalisedResult =
        hlfir::getFortranElementOrSequenceType(stmtResultType);
    mlir::Type elementType;
    if (auto array = normalisedResult.dyn_cast<fir::SequenceType>()) {
      resultShape = hlfir::ExprType::Shape{array.getShape()};
      elementType = array.getEleTy();
    } else {
      elementType = normalisedResult;
    }
    return hlfir::ExprType::get(builder.getContext(), resultShape, elementType,
                                /*polymorphic=*/false);
  };

  if (intrinsic.name == "sum") {
    llvm::SmallVector<mlir::Value> operands = getOperandVector(loweredActuals);
    assert(operands.size() == 3);
    mlir::Value array = operands[0];
    mlir::Value dim = operands[1];
    if (dim)
      dim = hlfir::loadTrivialScalar(loc, builder, hlfir::Entity{dim});
    mlir::Value mask = operands[2];
    mlir::Type resultTy = computeResultType(array, *callContext.resultType);
    // dim, mask can be NULL if these arguments were not given
    hlfir::SumOp sumOp =
        builder.create<hlfir::SumOp>(loc, resultTy, array, dim, mask);
    return {hlfir::EntityWithAttributes{sumOp.getResult()}};
  }
  if (intrinsic.name == "matmul") {
    llvm::SmallVector<mlir::Value> operands = getOperandVector(loweredActuals);
    mlir::Type resultTy =
        computeResultType(operands[0], *callContext.resultType);
    hlfir::MatmulOp matmulOp = builder.create<hlfir::MatmulOp>(
        loc, resultTy, operands[0], operands[1]);

    return {hlfir::EntityWithAttributes{matmulOp.getResult()}};
  }
  if (intrinsic.name == "transpose") {
    llvm::SmallVector<mlir::Value> operands = getOperandVector(loweredActuals);
    hlfir::ExprType::Shape resultShape;
    mlir::Type normalisedResult =
        hlfir::getFortranElementOrSequenceType(*callContext.resultType);
    auto array = normalisedResult.cast<fir::SequenceType>();
    llvm::ArrayRef<int64_t> arrayShape = array.getShape();
    assert(arrayShape.size() == 2 && "arguments to transpose have a rank of 2");
    mlir::Type elementType = array.getEleTy();
    resultShape.push_back(arrayShape[0]);
    resultShape.push_back(arrayShape[1]);
    mlir::Type resultTy = hlfir::ExprType::get(
        builder.getContext(), resultShape, elementType, /*polymorphic=*/false);
    hlfir::TransposeOp transposeOp =
        builder.create<hlfir::TransposeOp>(loc, resultTy, operands[0]);

    return {hlfir::EntityWithAttributes{transposeOp.getResult()}};
  }

  // TODO add hlfir operations for other transformational intrinsics here

  // fallback to calling the intrinsic via fir.call
  return genIntrinsicRefCore(loweredActuals, intrinsic, argLowering,
                             callContext);
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
    PreparedActualArgument *optionalWithShape;
    // 10.1.4 p5. Impure elemental procedures must be called in element order.
    bool mustBeOrdered = isImpure;
    for (unsigned i = 0; i < numArgs; ++i) {
      auto &preparedActual = loweredActuals[i];
      if (preparedActual) {
        hlfir::Entity actual = preparedActual->getOriginalActual();
        // Elemental procedure dummy arguments cannot be pointer/allocatables
        // (C15100), so it is safe to dereference any pointer or allocatable
        // actual argument now instead of doing this inside the elemental
        // region.
        actual = hlfir::derefPointersAndAllocatables(loc, builder, actual);
        // Better to load scalars outside of the loop when possible.
        if (!preparedActual->handleDynamicOptional() &&
            impl().canLoadActualArgumentBeforeLoop(i))
          actual = hlfir::loadTrivialScalar(loc, builder, actual);
        // TODO: merge shape instead of using the first one.
        if (!shape && actual.isArray()) {
          if (preparedActual->handleDynamicOptional())
            optionalWithShape = &*preparedActual;
          else
            shape = hlfir::genShape(loc, builder, actual);
        }
        // 15.8.3 p1. Elemental procedure with intent(out)/intent(inout)
        // arguments must be called in element order.
        if (impl().argMayBeModifiedByCall(i))
          mustBeOrdered = true;
        // Propagates pointer dereferences and scalar loads.
        preparedActual->setOriginalActual(actual);
      }
    }
    if (!shape && optionalWithShape) {
      // If all array operands appear in optional positions, then none of them
      // is allowed to be absent as per 15.5.2.12 point 3. (6). Just pick the
      // first operand.
      shape =
          hlfir::genShape(loc, builder, optionalWithShape->getOriginalActual());
      // TODO: There is an opportunity to add a runtime check here that
      // this array is present as required. Also, the optionality of all actual
      // could be checked and reset given the Fortran requirement.
      optionalWithShape->resetOptionalAspect();
    }
    assert(shape &&
           "elemental array calls must have at least one array arguments");
    if (mustBeOrdered)
      TODO(loc, "ordered elemental calls in HLFIR");
    // Push a new local scope so that any temps made inside the elemental
    // iterations are cleaned up inside the iterations.
    if (!callContext.resultType) {
      // Subroutine case. Generate call inside loop nest.
      auto [innerLoop, oneBasedIndicesVector] =
          hlfir::genLoopNest(loc, builder, shape);
      mlir::ValueRange oneBasedIndices = oneBasedIndicesVector;
      auto insPt = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(innerLoop.getBody());
      callContext.stmtCtx.pushScope();
      for (auto &preparedActual : loweredActuals)
        if (preparedActual)
          preparedActual->setElementalIndices(oneBasedIndices);
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
          preparedActual->setElementalIndices(oneBasedIndices);
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
      const fir::IntrinsicArgumentLoweringRules *argLowering, bool isFunction)
      : intrinsic{intrinsic}, argLowering{argLowering}, isFunction{isFunction} {
  }
  std::optional<hlfir::Entity>
  genElementalKernel(PreparedActualArguments &loweredActuals,
                     CallContext &callContext) {
    return genHLFIRIntrinsicRefCore(loweredActuals, intrinsic, argLowering,
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
      return hlfir::genCharLength(
          callContext.loc, callContext.getBuilder(),
          loweredActuals[0].value().getOriginalActual());
    // Character MIN/MAX is the min/max of the arguments length that are
    // present.
    TODO(callContext.loc,
         "compute elemental character min/max function result length in HLFIR");
  }

private:
  const Fortran::evaluate::SpecificIntrinsic &intrinsic;
  const fir::IntrinsicArgumentLoweringRules *argLowering;
  const bool isFunction;
};
} // namespace

static std::optional<mlir::Value>
genIsPresentIfArgMaybeAbsent(mlir::Location loc, hlfir::Entity actual,
                             const Fortran::lower::SomeExpr &expr,
                             CallContext &callContext,
                             bool passAsAllocatableOrPointer) {
  if (!Fortran::evaluate::MayBePassedAsAbsentOptional(
          expr, callContext.converter.getFoldingContext()))
    return std::nullopt;
  fir::FirOpBuilder &builder = callContext.getBuilder();
  if (!passAsAllocatableOrPointer &&
      Fortran::evaluate::IsAllocatableOrPointerObject(
          expr, callContext.converter.getFoldingContext())) {
    // Passing Allocatable/Pointer to non-pointer/non-allocatable OPTIONAL.
    // Fortran 2018 15.5.2.12 point 1: If unallocated/disassociated, it is
    // as if the argument was absent. The main care here is to not do a
    // copy-in/copy-out because the temp address, even though pointing to a
    // null size storage, would not be a nullptr and therefore the argument
    // would not be considered absent on the callee side. Note: if the
    // allocatable/pointer is also optional, it cannot be absent as per
    // 15.5.2.12 point 7. and 8. We rely on this to un-conditionally read
    // the allocatable/pointer descriptor here.
    mlir::Value addr = genVariableRawAddress(loc, builder, actual);
    return builder.genIsNotNullAddr(loc, addr);
  }
  // TODO: what if passing allocatable target to optional intent(in) pointer?
  // May fall into the category above if the allocatable is not optional.

  // Passing an optional to an optional.
  return builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), actual)
      .getResult();
}

/// Lower an intrinsic procedure reference.
static std::optional<hlfir::EntityWithAttributes>
genIntrinsicRef(const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                CallContext &callContext) {
  mlir::Location loc = callContext.loc;
  auto &converter = callContext.converter;
  if (Fortran::lower::intrinsicRequiresCustomOptionalHandling(
          callContext.procRef, intrinsic, converter))
    TODO(loc, "special cases of intrinsic with optional arguments");

  PreparedActualArguments loweredActuals;
  const fir::IntrinsicArgumentLoweringRules *argLowering =
      fir::getIntrinsicArgumentLowering(intrinsic.name);
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
    std::optional<mlir::Value> isPresent;
    if (argLowering) {
      fir::ArgLoweringRule argRules =
          fir::lowerIntrinsicArgumentAs(*argLowering, arg.index());
      if (argRules.handleDynamicOptional)
        isPresent =
            genIsPresentIfArgMaybeAbsent(loc, loweredActual, *expr, callContext,
                                         /*passAsAllocatableOrPointer=*/false);
    }
    loweredActuals.push_back(PreparedActualArgument{loweredActual, isPresent});
  }

  if (callContext.isElementalProcWithArrayArgs()) {
    // All intrinsic elemental functions are pure.
    const bool isFunction = callContext.resultType.has_value();
    return ElementalIntrinsicCallBuilder{intrinsic, argLowering, isFunction}
        .genElementalCall(loweredActuals, /*isImpure=*/!isFunction, callContext)
        .value();
  }
  std::optional<hlfir::EntityWithAttributes> result = genHLFIRIntrinsicRefCore(
      loweredActuals, intrinsic, argLowering, callContext);
  if (result && result->getType().isa<hlfir::ExprType>()) {
    fir::FirOpBuilder *bldr = &callContext.getBuilder();
    callContext.stmtCtx.attachCleanup(
        [=]() { bldr->create<hlfir::DestroyOp>(loc, *result); });
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
      if (Fortran::evaluate::UnwrapExpr<Fortran::evaluate::NullPointer>(
              *expr)) {
        if (arg.passBy !=
            Fortran::lower::CallerInterface::PassEntityBy::MutableBox) {
          assert(
              arg.isOptional() &&
              "NULL must be passed only to pointer, allocatable, or OPTIONAL");
          // Trying to lower NULL() outside of any context would lead to
          // trouble. NULL() here is equivalent to not providing the
          // actual argument.
          loweredActuals.emplace_back(std::nullopt);
          continue;
        }
      }

      auto loweredActual = Fortran::lower::convertExprToHLFIR(
          loc, callContext.converter, *expr, callContext.symMap,
          callContext.stmtCtx);
      std::optional<mlir::Value> isPresent;
      if (arg.isOptional())
        isPresent = genIsPresentIfArgMaybeAbsent(
            loc, loweredActual, *expr, callContext,
            arg.passBy ==
                Fortran::lower::CallerInterface::PassEntityBy::MutableBox);

      loweredActuals.emplace_back(
          PreparedActualArgument{loweredActual, isPresent});
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
