//===-- Intrinsics.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Intrinsics.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Runtime/extensions.h"
#include "flang/Runtime/misc-intrinsic.h"
#include "flang/Runtime/pointer.h"
#include "flang/Runtime/random.h"
#include "flang/Runtime/stop.h"
#include "flang/Runtime/time-intrinsic.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/Debug.h"
#include <optional>
#include <signal.h>

#define DEBUG_TYPE "flang-lower-runtime"

using namespace Fortran::runtime;

namespace {
/// Placeholder for real*16 version of RandomNumber Intrinsic
struct ForcedRandomNumberReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(RandomNumber16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::runtime::getModel<const char *>()(ctx);
      auto intTy = fir::runtime::getModel<int>()(ctx);
      ;
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy},
                                     mlir::NoneType::get(ctx));
    };
  }
};
} // namespace

mlir::Value fir::runtime::genAssociated(fir::FirOpBuilder &builder,
                                        mlir::Location loc, mlir::Value pointer,
                                        mlir::Value target) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PointerIsAssociatedWith)>(loc,
                                                                     builder);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, func.getFunctionType(), pointer, target);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

mlir::Value fir::runtime::genCpuTime(fir::FirOpBuilder &builder,
                                     mlir::Location loc) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CpuTime)>(loc, builder);
  return builder.create<fir::CallOp>(loc, func, std::nullopt).getResult(0);
}

void fir::runtime::genDateAndTime(fir::FirOpBuilder &builder,
                                  mlir::Location loc,
                                  std::optional<fir::CharBoxValue> date,
                                  std::optional<fir::CharBoxValue> time,
                                  std::optional<fir::CharBoxValue> zone,
                                  mlir::Value values) {
  mlir::func::FuncOp callee =
      fir::runtime::getRuntimeFunc<mkRTKey(DateAndTime)>(loc, builder);
  mlir::FunctionType funcTy = callee.getFunctionType();
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value zero;
  auto splitArg = [&](std::optional<fir::CharBoxValue> arg, mlir::Value &buffer,
                      mlir::Value &len) {
    if (arg) {
      buffer = arg->getBuffer();
      len = arg->getLen();
    } else {
      if (!zero)
        zero = builder.createIntegerConstant(loc, idxTy, 0);
      buffer = zero;
      len = zero;
    }
  };
  mlir::Value dateBuffer;
  mlir::Value dateLen;
  splitArg(date, dateBuffer, dateLen);
  mlir::Value timeBuffer;
  mlir::Value timeLen;
  splitArg(time, timeBuffer, timeLen);
  mlir::Value zoneBuffer;
  mlir::Value zoneLen;
  splitArg(zone, zoneBuffer, zoneLen);

  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(7));

  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, funcTy, dateBuffer, dateLen, timeBuffer, timeLen,
      zoneBuffer, zoneLen, sourceFile, sourceLine, values);
  builder.create<fir::CallOp>(loc, callee, args);
}

void fir::runtime::genEtime(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value values, mlir::Value time) {
  auto runtimeFunc = fir::runtime::getRuntimeFunc<mkRTKey(Etime)>(loc, builder);
  mlir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();

  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(3));

  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, runtimeFuncTy, values, time, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, runtimeFunc, args);
}

void fir::runtime::genFree(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value ptr) {
  auto runtimeFunc = fir::runtime::getRuntimeFunc<mkRTKey(Free)>(loc, builder);
  mlir::Type intPtrTy = builder.getIntPtrType();

  builder.create<fir::CallOp>(loc, runtimeFunc,
                              builder.createConvert(loc, intPtrTy, ptr));
}

mlir::Value fir::runtime::genGetGID(fir::FirOpBuilder &builder,
                                    mlir::Location loc) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(GetGID)>(loc, builder);

  return builder.create<fir::CallOp>(loc, runtimeFunc).getResult(0);
}

mlir::Value fir::runtime::genGetUID(fir::FirOpBuilder &builder,
                                    mlir::Location loc) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(GetUID)>(loc, builder);

  return builder.create<fir::CallOp>(loc, runtimeFunc).getResult(0);
}

mlir::Value fir::runtime::genMalloc(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value size) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Malloc)>(loc, builder);
  auto argTy = runtimeFunc.getArgumentTypes()[0];
  return builder
      .create<fir::CallOp>(loc, runtimeFunc,
                           builder.createConvert(loc, argTy, size))
      .getResult(0);
}

void fir::runtime::genRandomInit(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value repeatable,
                                 mlir::Value imageDistinct) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(RandomInit)>(loc, builder);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, func.getFunctionType(), repeatable, imageDistinct);
  builder.create<fir::CallOp>(loc, func, args);
}

void fir::runtime::genRandomNumber(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value harvest) {
  mlir::func::FuncOp func;
  auto boxEleTy = fir::dyn_cast_ptrOrBoxEleTy(harvest.getType());
  auto eleTy = fir::unwrapSequenceType(boxEleTy);
  if (eleTy.isF128()) {
    func = fir::runtime::getRuntimeFunc<ForcedRandomNumberReal16>(loc, builder);
  } else {
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomNumber)>(loc, builder);
  }

  mlir::FunctionType funcTy = func.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(2));
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, funcTy, harvest, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

void fir::runtime::genRandomSeed(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value size, mlir::Value put,
                                 mlir::Value get) {
  bool sizeIsPresent =
      !mlir::isa_and_nonnull<fir::AbsentOp>(size.getDefiningOp());
  bool putIsPresent =
      !mlir::isa_and_nonnull<fir::AbsentOp>(put.getDefiningOp());
  bool getIsPresent =
      !mlir::isa_and_nonnull<fir::AbsentOp>(get.getDefiningOp());
  mlir::func::FuncOp func;
  int staticArgCount = sizeIsPresent + putIsPresent + getIsPresent;
  if (staticArgCount == 0) {
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomSeedDefaultPut)>(loc,
                                                                       builder);
    builder.create<fir::CallOp>(loc, func);
    return;
  }
  mlir::FunctionType funcTy;
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine;
  mlir::Value argBox;
  llvm::SmallVector<mlir::Value> args;
  if (staticArgCount > 1) {
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomSeed)>(loc, builder);
    funcTy = func.getFunctionType();
    sourceLine =
        fir::factory::locationToLineNo(builder, loc, funcTy.getInput(4));
    args = fir::runtime::createArguments(builder, loc, funcTy, size, put, get,
                                         sourceFile, sourceLine);
    builder.create<fir::CallOp>(loc, func, args);
    return;
  }
  if (sizeIsPresent) {
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomSeedSize)>(loc, builder);
    argBox = size;
  } else if (putIsPresent) {
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomSeedPut)>(loc, builder);
    argBox = put;
  } else {
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomSeedGet)>(loc, builder);
    argBox = get;
  }
  funcTy = func.getFunctionType();
  sourceLine = fir::factory::locationToLineNo(builder, loc, funcTy.getInput(2));
  args = fir::runtime::createArguments(builder, loc, funcTy, argBox, sourceFile,
                                       sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// generate rename runtime call
void fir::runtime::genRename(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value path1, mlir::Value path2,
                             mlir::Value status) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Rename)>(loc, builder);
  mlir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();

  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(4));

  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, runtimeFuncTy, path1, path2,
                                    status, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, runtimeFunc, args);
}

/// generate runtime call to transfer intrinsic with no size argument
void fir::runtime::genTransfer(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value resultBox, mlir::Value sourceBox,
                               mlir::Value moldBox) {

  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(Transfer)>(loc, builder);
  mlir::FunctionType fTy = func.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, sourceBox, moldBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// generate runtime call to transfer intrinsic with size argument
void fir::runtime::genTransferSize(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value resultBox,
                                   mlir::Value sourceBox, mlir::Value moldBox,
                                   mlir::Value size) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(TransferSize)>(loc, builder);
  mlir::FunctionType fTy = func.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, sourceBox,
                                    moldBox, sourceFile, sourceLine, size);
  builder.create<fir::CallOp>(loc, func, args);
}

/// generate system_clock runtime call/s
/// all intrinsic arguments are optional and may appear here as mlir::Value{}
void fir::runtime::genSystemClock(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value count,
                                  mlir::Value rate, mlir::Value max) {
  auto makeCall = [&](mlir::func::FuncOp func, mlir::Value arg) {
    mlir::Type type = arg.getType();
    fir::IfOp ifOp{};
    const bool isOptionalArg =
        fir::valueHasFirAttribute(arg, fir::getOptionalAttrName());
    if (mlir::dyn_cast<fir::PointerType>(type) ||
        mlir::dyn_cast<fir::HeapType>(type)) {
      // Check for a disassociated pointer or an unallocated allocatable.
      assert(!isOptionalArg && "invalid optional argument");
      ifOp = builder.create<fir::IfOp>(loc, builder.genIsNotNullAddr(loc, arg),
                                       /*withElseRegion=*/false);
    } else if (isOptionalArg) {
      ifOp = builder.create<fir::IfOp>(
          loc, builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), arg),
          /*withElseRegion=*/false);
    }
    if (ifOp)
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::Type kindTy = func.getFunctionType().getInput(0);
    int integerKind = 8;
    if (auto intType =
            mlir::dyn_cast<mlir::IntegerType>(fir::unwrapRefType(type)))
      integerKind = intType.getWidth() / 8;
    mlir::Value kind = builder.createIntegerConstant(loc, kindTy, integerKind);
    mlir::Value res =
        builder.create<fir::CallOp>(loc, func, mlir::ValueRange{kind})
            .getResult(0);
    mlir::Value castRes =
        builder.createConvert(loc, fir::dyn_cast_ptrEleTy(type), res);
    builder.create<fir::StoreOp>(loc, castRes, arg);
    if (ifOp)
      builder.setInsertionPointAfter(ifOp);
  };
  using fir::runtime::getRuntimeFunc;
  if (count)
    makeCall(getRuntimeFunc<mkRTKey(SystemClockCount)>(loc, builder), count);
  if (rate)
    makeCall(getRuntimeFunc<mkRTKey(SystemClockCountRate)>(loc, builder), rate);
  if (max)
    makeCall(getRuntimeFunc<mkRTKey(SystemClockCountMax)>(loc, builder), max);
}

// CALL SIGNAL(NUMBER, HANDLER [, STATUS])
// The definition of the SIGNAL intrinsic allows HANDLER to be a function
// pointer or an integer. STATUS can be dynamically optional
void fir::runtime::genSignal(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value number, mlir::Value handler,
                             mlir::Value status) {
  assert(mlir::isa<mlir::IntegerType>(number.getType()));
  mlir::Type int64 = builder.getIntegerType(64);
  number = builder.create<fir::ConvertOp>(loc, int64, number);

  mlir::Type handlerUnwrappedTy = fir::unwrapRefType(handler.getType());
  if (mlir::isa_and_nonnull<mlir::IntegerType>(handlerUnwrappedTy)) {
    // pass the integer as a function pointer like one would to signal(2)
    handler = builder.create<fir::LoadOp>(loc, handler);
    mlir::Type fnPtrTy = fir::LLVMPointerType::get(
        mlir::FunctionType::get(handler.getContext(), {}, {}));
    handler = builder.create<fir::ConvertOp>(loc, fnPtrTy, handler);
  } else {
    assert(mlir::isa<fir::BoxProcType>(handler.getType()));
    handler = builder.create<fir::BoxAddrOp>(loc, handler);
  }

  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(Signal)>(loc, builder)};
  mlir::Value stat =
      builder.create<fir::CallOp>(loc, func, mlir::ValueRange{number, handler})
          ->getResult(0);

  // return status code via status argument (if present)
  if (status) {
    assert(mlir::isa<mlir::IntegerType>(fir::unwrapRefType(status.getType())));
    // status might be dynamically optional, so test if it is present
    mlir::Value isPresent =
        builder.create<IsPresentOp>(loc, builder.getI1Type(), status);
    builder.genIfOp(loc, /*results=*/{}, isPresent, /*withElseRegion=*/false)
        .genThen([&]() {
          stat = builder.create<fir::ConvertOp>(
              loc, fir::unwrapRefType(status.getType()), stat);
          builder.create<fir::StoreOp>(loc, stat, status);
        })
        .end();
  }
}

void fir::runtime::genSleep(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value seconds) {
  mlir::Type int64 = builder.getIntegerType(64);
  seconds = builder.create<fir::ConvertOp>(loc, int64, seconds);
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(Sleep)>(loc, builder)};
  builder.create<fir::CallOp>(loc, func, seconds);
}
