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
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Optimizer/Builder/Todo.h"

/// Is this a call to MIN or MAX intrinsic with arguments that may be absent at
/// runtime? This is a special case because MIN and MAX can have any number of
/// arguments.
static bool isMinOrMaxWithDynamicallyOptionalArg(
    llvm::StringRef name, const Fortran::evaluate::ProcedureRef &procRef,
    Fortran::evaluate::FoldingContext &foldingContex) {
  if (name != "min" && name != "max")
    return false;
  const auto &args = procRef.arguments();
  std::size_t argSize = args.size();
  if (argSize <= 2)
    return false;
  for (std::size_t i = 2; i < argSize; ++i) {
    if (auto *expr =
            Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(args[i]))
      if (Fortran::evaluate::MayBePassedAsAbsentOptional(*expr, foldingContex))
        return true;
  }
  return false;
}

/// Is this a call to ISHFTC intrinsic with a SIZE argument that may be absent
/// at runtime? This is a special case because the SIZE value to be applied
/// when absent is not zero.
static bool isIshftcWithDynamicallyOptionalArg(
    llvm::StringRef name, const Fortran::evaluate::ProcedureRef &procRef,
    Fortran::evaluate::FoldingContext &foldingContex) {
  if (name != "ishftc" || procRef.arguments().size() < 3)
    return false;
  auto *expr = Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(
      procRef.arguments()[2]);
  return expr &&
         Fortran::evaluate::MayBePassedAsAbsentOptional(*expr, foldingContex);
}

/// Is this a call to SYSTEM_CLOCK or RANDOM_SEED intrinsic with arguments that
/// may be absent at runtime? This are special cases because that aspect cannot
/// be delegated to the runtime via a null fir.box or address given the current
/// runtime entry point.
static bool isSystemClockOrRandomSeedWithOptionalArg(
    llvm::StringRef name, const Fortran::evaluate::ProcedureRef &procRef,
    Fortran::evaluate::FoldingContext &foldingContex) {
  if (name != "system_clock" && name != "random_seed")
    return false;
  for (const auto &arg : procRef.arguments()) {
    auto *expr = Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg);
    if (expr &&
        Fortran::evaluate::MayBePassedAsAbsentOptional(*expr, foldingContex))
      return true;
  }
  return false;
}

bool Fortran::lower::intrinsicRequiresCustomOptionalHandling(
    const Fortran::evaluate::ProcedureRef &procRef,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    AbstractConverter &converter) {
  llvm::StringRef name = intrinsic.name;
  Fortran::evaluate::FoldingContext &fldCtx = converter.getFoldingContext();
  return isMinOrMaxWithDynamicallyOptionalArg(name, procRef, fldCtx) ||
         isIshftcWithDynamicallyOptionalArg(name, procRef, fldCtx) ||
         isSystemClockOrRandomSeedWithOptionalArg(name, procRef, fldCtx);
}

static void prepareMinOrMaxArguments(
    const Fortran::evaluate::ProcedureRef &procRef,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    llvm::Optional<mlir::Type> retTy,
    const Fortran::lower::OperandPrepare &prepareOptionalArgument,
    const Fortran::lower::OperandPrepare &prepareOtherArgument,
    Fortran::lower::AbstractConverter &converter) {
  assert(retTy && "MIN and MAX must have a return type");
  mlir::Type resultType = retTy.getValue();
  mlir::Location loc = converter.getCurrentLocation();
  if (fir::isa_char(resultType))
    TODO(loc,
         "CHARACTER MIN and MAX lowering with dynamically optional arguments");
  for (auto arg : llvm::enumerate(procRef.arguments())) {
    const auto *expr =
        Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg.value());
    if (!expr)
      continue;
    if (arg.index() <= 1 || !Fortran::evaluate::MayBePassedAsAbsentOptional(
                                *expr, converter.getFoldingContext())) {
      // Non optional arguments.
      prepareOtherArgument(*expr);
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
              llvm::StringRef name, llvm::Optional<mlir::Type> retTy,
              const Fortran::lower::OperandPresent &isPresentCheck,
              const Fortran::lower::OperandGetter &getOperand,
              std::size_t numOperands,
              Fortran::lower::StatementContext &stmtCtx) {
  assert(numOperands >= 2 && !isPresentCheck(0) && !isPresentCheck(1) &&
         "min/max must have at least two non-optional args");
  assert(retTy && "MIN and MAX must have a return type");
  mlir::Type resultType = retTy.getValue();
  llvm::SmallVector<fir::ExtendedValue> args;
  args.push_back(getOperand(0));
  args.push_back(getOperand(1));
  mlir::Value extremum = fir::getBase(Fortran::lower::genIntrinsicCall(
      builder, loc, name, resultType, args, stmtCtx));

  for (std::size_t opIndex = 2; opIndex < numOperands; ++opIndex) {
    if (llvm::Optional<mlir::Value> isPresentRuntimeCheck =
            isPresentCheck(opIndex)) {
      // Argument is dynamically optional.
      extremum =
          builder
              .genIfOp(loc, {resultType}, isPresentRuntimeCheck.getValue(),
                       /*withElseRegion=*/true)
              .genThen([&]() {
                llvm::SmallVector<fir::ExtendedValue> args;
                args.emplace_back(extremum);
                args.emplace_back(getOperand(opIndex));
                fir::ExtendedValue newExtremum =
                    Fortran::lower::genIntrinsicCall(builder, loc, name,
                                                     resultType, args, stmtCtx);
                builder.create<fir::ResultOp>(loc, fir::getBase(newExtremum));
              })
              .genElse([&]() { builder.create<fir::ResultOp>(loc, extremum); })
              .getResults()[0];
    } else {
      // Argument is know to be present at compile time.
      llvm::SmallVector<fir::ExtendedValue> args;
      args.emplace_back(extremum);
      args.emplace_back(getOperand(opIndex));
      extremum = fir::getBase(Fortran::lower::genIntrinsicCall(
          builder, loc, name, resultType, args, stmtCtx));
    }
  }
  return extremum;
}

static void prepareIshftcArguments(
    const Fortran::evaluate::ProcedureRef &procRef,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    llvm::Optional<mlir::Type> retTy,
    const Fortran::lower::OperandPrepare &prepareOptionalArgument,
    const Fortran::lower::OperandPrepare &prepareOtherArgument,
    Fortran::lower::AbstractConverter &converter) {
  for (auto arg : llvm::enumerate(procRef.arguments())) {
    const auto *expr =
        Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg.value());
    assert(expr && "expected all ISHFTC argument to be textually present here");
    if (arg.index() == 2) {
      assert(Fortran::evaluate::MayBePassedAsAbsentOptional(
                 *expr, converter.getFoldingContext()) &&
             "expected ISHFTC SIZE arg to be dynamically optional");
      prepareOptionalArgument(*expr);
    } else {
      // Non optional arguments.
      prepareOtherArgument(*expr);
    }
  }
}

static fir::ExtendedValue
lowerIshftc(fir::FirOpBuilder &builder, mlir::Location loc,
            llvm::StringRef name, llvm::Optional<mlir::Type> retTy,
            const Fortran::lower::OperandPresent &isPresentCheck,
            const Fortran::lower::OperandGetter &getOperand,
            std::size_t numOperands,
            Fortran::lower::StatementContext &stmtCtx) {
  assert(numOperands == 3 && !isPresentCheck(0) && !isPresentCheck(1) &&
         isPresentCheck(2) &&
         "only ISHFTC SIZE arg is expected to be dynamically optional here");
  assert(retTy && "ISFHTC must have a return type");
  mlir::Type resultType = retTy.getValue();
  llvm::SmallVector<fir::ExtendedValue> args;
  args.push_back(getOperand(0));
  args.push_back(getOperand(1));
  args.push_back(builder
                     .genIfOp(loc, {resultType}, isPresentCheck(2).getValue(),
                              /*withElseRegion=*/true)
                     .genThen([&]() {
                       fir::ExtendedValue sizeExv = getOperand(2);
                       mlir::Value size = builder.createConvert(
                           loc, resultType, fir::getBase(sizeExv));
                       builder.create<fir::ResultOp>(loc, size);
                     })
                     .genElse([&]() {
                       mlir::Value bitSize = builder.createIntegerConstant(
                           loc, resultType,
                           resultType.cast<mlir::IntegerType>().getWidth());
                       builder.create<fir::ResultOp>(loc, bitSize);
                     })
                     .getResults()[0]);
  return Fortran::lower::genIntrinsicCall(builder, loc, name, resultType, args,
                                          stmtCtx);
}

void Fortran::lower::prepareCustomIntrinsicArgument(
    const Fortran::evaluate::ProcedureRef &procRef,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    llvm::Optional<mlir::Type> retTy,
    const OperandPrepare &prepareOptionalArgument,
    const OperandPrepare &prepareOtherArgument, AbstractConverter &converter) {
  llvm::StringRef name = intrinsic.name;
  if (name == "min" || name == "max")
    return prepareMinOrMaxArguments(procRef, intrinsic, retTy,
                                    prepareOptionalArgument,
                                    prepareOtherArgument, converter);
  if (name == "ishftc")
    return prepareIshftcArguments(procRef, intrinsic, retTy,
                                  prepareOptionalArgument, prepareOtherArgument,
                                  converter);
  TODO(converter.getCurrentLocation(),
       "unhandled dynamically optional arguments in SYSTEM_CLOCK or "
       "RANDOM_SEED");
}

fir::ExtendedValue Fortran::lower::lowerCustomIntrinsic(
    fir::FirOpBuilder &builder, mlir::Location loc, llvm::StringRef name,
    llvm::Optional<mlir::Type> retTy, const OperandPresent &isPresentCheck,
    const OperandGetter &getOperand, std::size_t numOperands,
    Fortran::lower::StatementContext &stmtCtx) {
  if (name == "min" || name == "max")
    return lowerMinOrMax(builder, loc, name, retTy, isPresentCheck, getOperand,
                         numOperands, stmtCtx);
  if (name == "ishftc")
    return lowerIshftc(builder, loc, name, retTy, isPresentCheck, getOperand,
                       numOperands, stmtCtx);
  TODO(loc, "unhandled dynamically optional arguments in SYSTEM_CLOCK or "
            "RANDOM_SEED");
}
