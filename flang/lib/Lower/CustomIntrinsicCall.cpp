//===-- CustomIntrinsicCall.cpp -------------------------------------------===//
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

#include "flang/Lower/CustomIntrinsicCall.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Semantics/tools.h"
#include <optional>

/// Is this a call to MIN or MAX intrinsic with arguments that may be absent at
/// runtime? This is a special case because MIN and MAX can have any number of
/// arguments.
static bool isMinOrMaxWithDynamicallyOptionalArg(
    llvm::StringRef name, const Fortran::evaluate::ProcedureRef &procRef) {
  if (name != "min" && name != "max")
    return false;
  const auto &args = procRef.arguments();
  std::size_t argSize = args.size();
  if (argSize <= 2)
    return false;
  for (std::size_t i = 2; i < argSize; ++i) {
    if (auto *expr =
            Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(args[i]))
      if (Fortran::evaluate::MayBePassedAsAbsentOptional(*expr))
        return true;
  }
  return false;
}

/// Is this a call to ISHFTC intrinsic with a SIZE argument that may be absent
/// at runtime? This is a special case because the SIZE value to be applied
/// when absent is not zero.
static bool isIshftcWithDynamicallyOptionalArg(
    llvm::StringRef name, const Fortran::evaluate::ProcedureRef &procRef) {
  if (name != "ishftc" || procRef.arguments().size() < 3)
    return false;
  auto *expr = Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(
      procRef.arguments()[2]);
  return expr && Fortran::evaluate::MayBePassedAsAbsentOptional(*expr);
}

/// Is this a call to ASSOCIATED where the TARGET is an OPTIONAL (but not a
/// deallocated allocatable or disassociated pointer)?
/// Subtle: contrary to other intrinsic optional arguments, disassociated
/// POINTER and unallocated ALLOCATABLE actual argument are not considered
/// absent here. This is because ASSOCIATED has special requirements for TARGET
/// actual arguments that are POINTERs. There is no precise requirements for
/// ALLOCATABLEs, but all existing Fortran compilers treat them similarly to
/// POINTERs. That is: unallocated TARGETs cause ASSOCIATED to rerun false.  The
/// runtime deals with the disassociated/unallocated case. Simply ensures that
/// TARGET that are OPTIONAL get conditionally emboxed here to convey the
/// optional aspect to the runtime.
static bool isAssociatedWithDynamicallyOptionalArg(
    llvm::StringRef name, const Fortran::evaluate::ProcedureRef &procRef) {
  if (name != "associated" || procRef.arguments().size() < 2)
    return false;
  auto *expr = Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(
      procRef.arguments()[1]);
  const Fortran::semantics::Symbol *sym{
      expr ? Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(expr)
           : nullptr};
  return (sym && Fortran::semantics::IsOptional(*sym));
}

bool Fortran::lower::intrinsicRequiresCustomOptionalHandling(
    const Fortran::evaluate::ProcedureRef &procRef,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    AbstractConverter &converter) {
  llvm::StringRef name = intrinsic.name;
  return isMinOrMaxWithDynamicallyOptionalArg(name, procRef) ||
         isIshftcWithDynamicallyOptionalArg(name, procRef) ||
         isAssociatedWithDynamicallyOptionalArg(name, procRef);
}

/// Generate the FIR+MLIR operations for the generic intrinsic \p name
/// with arguments \p args and the expected result type \p resultType.
/// Returned fir::ExtendedValue is the returned Fortran intrinsic value.
fir::ExtendedValue
Fortran::lower::genIntrinsicCall(fir::FirOpBuilder &builder, mlir::Location loc,
                                 llvm::StringRef name,
                                 std::optional<mlir::Type> resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args,
                                 Fortran::lower::StatementContext &stmtCtx,
                                 Fortran::lower::AbstractConverter *converter) {
  auto [result, mustBeFreed] =
      fir::genIntrinsicCall(builder, loc, name, resultType, args, converter);
  if (mustBeFreed) {
    mlir::Value addr = fir::getBase(result);
    if (auto *box = result.getBoxOf<fir::BoxValue>())
      addr =
          builder.create<fir::BoxAddrOp>(loc, box->getMemTy(), box->getAddr());
    fir::FirOpBuilder *bldr = &builder;
    stmtCtx.attachCleanup([=]() { bldr->create<fir::FreeMemOp>(loc, addr); });
  }
  return result;
}

static void prepareMinOrMaxArguments(
    const Fortran::evaluate::ProcedureRef &procRef,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    std::optional<mlir::Type> retTy,
    const Fortran::lower::OperandPrepare &prepareOptionalArgument,
    const Fortran::lower::OperandPrepareAs &prepareOtherArgument,
    Fortran::lower::AbstractConverter &converter) {
  assert(retTy && "MIN and MAX must have a return type");
  mlir::Type resultType = *retTy;
  mlir::Location loc = converter.getCurrentLocation();
  if (fir::isa_char(resultType))
    TODO(loc, "CHARACTER MIN and MAX with dynamically optional arguments");
  for (auto arg : llvm::enumerate(procRef.arguments())) {
    const auto *expr =
        Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg.value());
    if (!expr)
      continue;
    if (arg.index() <= 1 ||
        !Fortran::evaluate::MayBePassedAsAbsentOptional(*expr)) {
      // Non optional arguments.
      prepareOtherArgument(*expr, fir::LowerIntrinsicArgAs::Value);
    } else {
      // Dynamically optional arguments.
      // Subtle: even for scalar the if-then-else will be generated in the loop
      // nest because the then part will require the current extremum value that
      // may depend on previous array element argument and cannot be outlined.
      prepareOptionalArgument(*expr);
    }
  }
}

static fir::ExtendedValue
lowerMinOrMax(fir::FirOpBuilder &builder, mlir::Location loc,
              llvm::StringRef name, std::optional<mlir::Type> retTy,
              const Fortran::lower::OperandPresent &isPresentCheck,
              const Fortran::lower::OperandGetter &getOperand,
              std::size_t numOperands,
              Fortran::lower::StatementContext &stmtCtx) {
  assert(numOperands >= 2 && !isPresentCheck(0) && !isPresentCheck(1) &&
         "min/max must have at least two non-optional args");
  assert(retTy && "MIN and MAX must have a return type");
  mlir::Type resultType = *retTy;
  llvm::SmallVector<fir::ExtendedValue> args;
  const bool loadOperand = true;
  args.push_back(getOperand(0, loadOperand));
  args.push_back(getOperand(1, loadOperand));
  mlir::Value extremum = fir::getBase(
      genIntrinsicCall(builder, loc, name, resultType, args, stmtCtx));

  for (std::size_t opIndex = 2; opIndex < numOperands; ++opIndex) {
    if (std::optional<mlir::Value> isPresentRuntimeCheck =
            isPresentCheck(opIndex)) {
      // Argument is dynamically optional.
      extremum =
          builder
              .genIfOp(loc, {resultType}, *isPresentRuntimeCheck,
                       /*withElseRegion=*/true)
              .genThen([&]() {
                llvm::SmallVector<fir::ExtendedValue> args;
                args.emplace_back(extremum);
                args.emplace_back(getOperand(opIndex, loadOperand));
                fir::ExtendedValue newExtremum = genIntrinsicCall(
                    builder, loc, name, resultType, args, stmtCtx);
                builder.create<fir::ResultOp>(loc, fir::getBase(newExtremum));
              })
              .genElse([&]() { builder.create<fir::ResultOp>(loc, extremum); })
              .getResults()[0];
    } else {
      // Argument is know to be present at compile time.
      llvm::SmallVector<fir::ExtendedValue> args;
      args.emplace_back(extremum);
      args.emplace_back(getOperand(opIndex, loadOperand));
      extremum = fir::getBase(
          genIntrinsicCall(builder, loc, name, resultType, args, stmtCtx));
    }
  }
  return extremum;
}

static void prepareIshftcArguments(
    const Fortran::evaluate::ProcedureRef &procRef,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    std::optional<mlir::Type> retTy,
    const Fortran::lower::OperandPrepare &prepareOptionalArgument,
    const Fortran::lower::OperandPrepareAs &prepareOtherArgument,
    Fortran::lower::AbstractConverter &converter) {
  for (auto arg : llvm::enumerate(procRef.arguments())) {
    const auto *expr =
        Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg.value());
    assert(expr && "expected all ISHFTC argument to be textually present here");
    if (arg.index() == 2) {
      assert(Fortran::evaluate::MayBePassedAsAbsentOptional(*expr) &&
             "expected ISHFTC SIZE arg to be dynamically optional");
      prepareOptionalArgument(*expr);
    } else {
      // Non optional arguments.
      prepareOtherArgument(*expr, fir::LowerIntrinsicArgAs::Value);
    }
  }
}

static fir::ExtendedValue
lowerIshftc(fir::FirOpBuilder &builder, mlir::Location loc,
            llvm::StringRef name, std::optional<mlir::Type> retTy,
            const Fortran::lower::OperandPresent &isPresentCheck,
            const Fortran::lower::OperandGetter &getOperand,
            std::size_t numOperands,
            Fortran::lower::StatementContext &stmtCtx) {
  assert(numOperands == 3 && !isPresentCheck(0) && !isPresentCheck(1) &&
         isPresentCheck(2) &&
         "only ISHFTC SIZE arg is expected to be dynamically optional here");
  assert(retTy && "ISFHTC must have a return type");
  mlir::Type resultType = *retTy;
  llvm::SmallVector<fir::ExtendedValue> args;
  const bool loadOperand = true;
  args.push_back(getOperand(0, loadOperand));
  args.push_back(getOperand(1, loadOperand));
  auto iPC = isPresentCheck(2);
  assert(iPC.has_value());
  args.push_back(
      builder
          .genIfOp(loc, {resultType}, *iPC,
                   /*withElseRegion=*/true)
          .genThen([&]() {
            fir::ExtendedValue sizeExv = getOperand(2, loadOperand);
            mlir::Value size =
                builder.createConvert(loc, resultType, fir::getBase(sizeExv));
            builder.create<fir::ResultOp>(loc, size);
          })
          .genElse([&]() {
            mlir::Value bitSize = builder.createIntegerConstant(
                loc, resultType,
                mlir::cast<mlir::IntegerType>(resultType).getWidth());
            builder.create<fir::ResultOp>(loc, bitSize);
          })
          .getResults()[0]);
  return genIntrinsicCall(builder, loc, name, resultType, args, stmtCtx);
}

static void prepareAssociatedArguments(
    const Fortran::evaluate::ProcedureRef &procRef,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    std::optional<mlir::Type> retTy,
    const Fortran::lower::OperandPrepare &prepareOptionalArgument,
    const Fortran::lower::OperandPrepareAs &prepareOtherArgument,
    Fortran::lower::AbstractConverter &converter) {
  const auto *pointer = procRef.UnwrapArgExpr(0);
  const auto *optionalTarget = procRef.UnwrapArgExpr(1);
  assert(pointer && optionalTarget &&
         "expected call to associated with a target");
  prepareOtherArgument(*pointer, fir::LowerIntrinsicArgAs::Inquired);
  prepareOptionalArgument(*optionalTarget);
}

static fir::ExtendedValue
lowerAssociated(fir::FirOpBuilder &builder, mlir::Location loc,
                llvm::StringRef name, std::optional<mlir::Type> resultType,
                const Fortran::lower::OperandPresent &isPresentCheck,
                const Fortran::lower::OperandGetter &getOperand,
                std::size_t numOperands,
                Fortran::lower::StatementContext &stmtCtx) {
  assert(numOperands == 2 && "expect two arguments when TARGET is OPTIONAL");
  llvm::SmallVector<fir::ExtendedValue> args;
  args.push_back(getOperand(0, /*loadOperand=*/false));
  // Ensure a null descriptor is passed to the code lowering Associated if
  // TARGET is absent.
  fir::ExtendedValue targetExv = getOperand(1, /*loadOperand=*/false);
  mlir::Value targetBase = fir::getBase(targetExv);
  // subtle: isPresentCheck would test for an unallocated/disassociated target,
  // while the optionality of the target pointer/allocatable is what must be
  // checked here.
  mlir::Value isPresent =
      builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), targetBase);
  mlir::Type targetType = fir::unwrapRefType(targetBase.getType());
  mlir::Type targetValueType = fir::unwrapPassByRefType(targetType);
  mlir::Type boxType = mlir::isa<fir::BaseBoxType>(targetType)
                           ? targetType
                           : fir::BoxType::get(targetValueType);
  fir::BoxValue targetBox =
      builder
          .genIfOp(loc, {boxType}, isPresent,
                   /*withElseRegion=*/true)
          .genThen([&]() {
            mlir::Value box = builder.createBox(loc, targetExv);
            mlir::Value cast = builder.createConvert(loc, boxType, box);
            builder.create<fir::ResultOp>(loc, cast);
          })
          .genElse([&]() {
            mlir::Value absentBox = builder.create<fir::AbsentOp>(loc, boxType);
            builder.create<fir::ResultOp>(loc, absentBox);
          })
          .getResults()[0];
  args.emplace_back(std::move(targetBox));
  return genIntrinsicCall(builder, loc, name, resultType, args, stmtCtx);
}

void Fortran::lower::prepareCustomIntrinsicArgument(
    const Fortran::evaluate::ProcedureRef &procRef,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    std::optional<mlir::Type> retTy,
    const OperandPrepare &prepareOptionalArgument,
    const OperandPrepareAs &prepareOtherArgument,
    AbstractConverter &converter) {
  llvm::StringRef name = intrinsic.name;
  if (name == "min" || name == "max")
    return prepareMinOrMaxArguments(procRef, intrinsic, retTy,
                                    prepareOptionalArgument,
                                    prepareOtherArgument, converter);
  if (name == "associated")
    return prepareAssociatedArguments(procRef, intrinsic, retTy,
                                      prepareOptionalArgument,
                                      prepareOtherArgument, converter);
  assert(name == "ishftc" && "unexpected custom intrinsic argument call");
  return prepareIshftcArguments(procRef, intrinsic, retTy,
                                prepareOptionalArgument, prepareOtherArgument,
                                converter);
}

fir::ExtendedValue Fortran::lower::lowerCustomIntrinsic(
    fir::FirOpBuilder &builder, mlir::Location loc, llvm::StringRef name,
    std::optional<mlir::Type> retTy, const OperandPresent &isPresentCheck,
    const OperandGetter &getOperand, std::size_t numOperands,
    Fortran::lower::StatementContext &stmtCtx) {
  if (name == "min" || name == "max")
    return lowerMinOrMax(builder, loc, name, retTy, isPresentCheck, getOperand,
                         numOperands, stmtCtx);
  if (name == "associated")
    return lowerAssociated(builder, loc, name, retTy, isPresentCheck,
                           getOperand, numOperands, stmtCtx);
  assert(name == "ishftc" && "unexpected custom intrinsic call");
  return lowerIshftc(builder, loc, name, retTy, isPresentCheck, getOperand,
                     numOperands, stmtCtx);
}
