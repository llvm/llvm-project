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
    return [](aiir::AIIRContext *ctx) {
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::runtime::getModel<const char *>()(ctx);
      auto intTy = fir::runtime::getModel<int>()(ctx);
      ;
      return aiir::FunctionType::get(ctx, {boxTy, strTy, intTy}, {});
    };
  }
};
} // namespace

aiir::Value fir::runtime::genAssociated(fir::FirOpBuilder &builder,
                                        aiir::Location loc, aiir::Value pointer,
                                        aiir::Value target) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PointerIsAssociatedWith)>(loc,
                                                                     builder);
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, func.getFunctionType(), pointer, target);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

aiir::Value fir::runtime::genCpuTime(fir::FirOpBuilder &builder,
                                     aiir::Location loc) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CpuTime)>(loc, builder);
  return fir::CallOp::create(builder, loc, func, aiir::ValueRange{})
      .getResult(0);
}

void fir::runtime::genDateAndTime(fir::FirOpBuilder &builder,
                                  aiir::Location loc,
                                  std::optional<fir::CharBoxValue> date,
                                  std::optional<fir::CharBoxValue> time,
                                  std::optional<fir::CharBoxValue> zone,
                                  aiir::Value values) {
  aiir::func::FuncOp callee =
      fir::runtime::getRuntimeFunc<mkRTKey(DateAndTime)>(loc, builder);
  aiir::FunctionType funcTy = callee.getFunctionType();
  aiir::Type idxTy = builder.getIndexType();
  aiir::Value zero;
  auto splitArg = [&](std::optional<fir::CharBoxValue> arg, aiir::Value &buffer,
                      aiir::Value &len) {
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
  aiir::Value dateBuffer;
  aiir::Value dateLen;
  splitArg(date, dateBuffer, dateLen);
  aiir::Value timeBuffer;
  aiir::Value timeLen;
  splitArg(time, timeBuffer, timeLen);
  aiir::Value zoneBuffer;
  aiir::Value zoneLen;
  splitArg(zone, zoneBuffer, zoneLen);

  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(7));

  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, funcTy, dateBuffer, dateLen, timeBuffer, timeLen,
      zoneBuffer, zoneLen, sourceFile, sourceLine, values);
  fir::CallOp::create(builder, loc, callee, args);
}

aiir::Value fir::runtime::genDsecnds(fir::FirOpBuilder &builder,
                                     aiir::Location loc, aiir::Value refTime) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Dsecnds)>(loc, builder);

  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();

  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(2));

  llvm::SmallVector<aiir::Value> args = {refTime, sourceFile, sourceLine};
  args = fir::runtime::createArguments(builder, loc, runtimeFuncTy, args);

  return fir::CallOp::create(builder, loc, runtimeFunc, args).getResult(0);
}

void fir::runtime::genEtime(fir::FirOpBuilder &builder, aiir::Location loc,
                            aiir::Value values, aiir::Value time) {
  auto runtimeFunc = fir::runtime::getRuntimeFunc<mkRTKey(Etime)>(loc, builder);
  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();

  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(3));

  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, runtimeFuncTy, values, time, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, runtimeFunc, args);
}

void fir::runtime::genFlush(fir::FirOpBuilder &builder, aiir::Location loc,
                            aiir::Value unit) {
  auto runtimeFunc = fir::runtime::getRuntimeFunc<mkRTKey(Flush)>(loc, builder);
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, runtimeFunc.getFunctionType(), unit);

  fir::CallOp::create(builder, loc, runtimeFunc, args);
}

void fir::runtime::genFree(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::Value ptr) {
  auto runtimeFunc = fir::runtime::getRuntimeFunc<mkRTKey(Free)>(loc, builder);
  aiir::Type intPtrTy = builder.getIntPtrType();

  fir::CallOp::create(builder, loc, runtimeFunc,
                      builder.createConvert(loc, intPtrTy, ptr));
}

aiir::Value fir::runtime::genFseek(fir::FirOpBuilder &builder,
                                   aiir::Location loc, aiir::Value unit,
                                   aiir::Value offset, aiir::Value whence) {
  auto runtimeFunc = fir::runtime::getRuntimeFunc<mkRTKey(Fseek)>(loc, builder);
  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(2));
  llvm::SmallVector<aiir::Value> args =
      fir::runtime::createArguments(builder, loc, runtimeFuncTy, unit, offset,
                                    whence, sourceFile, sourceLine);
  return fir::CallOp::create(builder, loc, runtimeFunc, args).getResult(0);
  ;
}

aiir::Value fir::runtime::genFtell(fir::FirOpBuilder &builder,
                                   aiir::Location loc, aiir::Value unit) {
  auto runtimeFunc = fir::runtime::getRuntimeFunc<mkRTKey(Ftell)>(loc, builder);
  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();
  llvm::SmallVector<aiir::Value> args =
      fir::runtime::createArguments(builder, loc, runtimeFuncTy, unit);
  return fir::CallOp::create(builder, loc, runtimeFunc, args).getResult(0);
}

aiir::Value fir::runtime::genGetGID(fir::FirOpBuilder &builder,
                                    aiir::Location loc) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(GetGID)>(loc, builder);

  return fir::CallOp::create(builder, loc, runtimeFunc).getResult(0);
}

aiir::Value fir::runtime::genGetUID(fir::FirOpBuilder &builder,
                                    aiir::Location loc) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(GetUID)>(loc, builder);

  return fir::CallOp::create(builder, loc, runtimeFunc).getResult(0);
}

aiir::Value fir::runtime::genMalloc(fir::FirOpBuilder &builder,
                                    aiir::Location loc, aiir::Value size) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Malloc)>(loc, builder);
  auto argTy = runtimeFunc.getArgumentTypes()[0];
  return fir::CallOp::create(builder, loc, runtimeFunc,
                             builder.createConvert(loc, argTy, size))
      .getResult(0);
}

void fir::runtime::genRandomInit(fir::FirOpBuilder &builder, aiir::Location loc,
                                 aiir::Value repeatable,
                                 aiir::Value imageDistinct) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(RandomInit)>(loc, builder);
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, func.getFunctionType(), repeatable, imageDistinct);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genRandomNumber(fir::FirOpBuilder &builder,
                                   aiir::Location loc, aiir::Value harvest) {
  aiir::func::FuncOp func;
  auto boxEleTy = fir::dyn_cast_ptrOrBoxEleTy(harvest.getType());
  auto eleTy = fir::unwrapSequenceType(boxEleTy);
  if (eleTy.isF128()) {
    func = fir::runtime::getRuntimeFunc<ForcedRandomNumberReal16>(loc, builder);
  } else {
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomNumber)>(loc, builder);
  }

  aiir::FunctionType funcTy = func.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(2));
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, funcTy, harvest, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genRandomSeed(fir::FirOpBuilder &builder, aiir::Location loc,
                                 aiir::Value size, aiir::Value put,
                                 aiir::Value get) {
  bool sizeIsPresent =
      !aiir::isa_and_nonnull<fir::AbsentOp>(size.getDefiningOp());
  bool putIsPresent =
      !aiir::isa_and_nonnull<fir::AbsentOp>(put.getDefiningOp());
  bool getIsPresent =
      !aiir::isa_and_nonnull<fir::AbsentOp>(get.getDefiningOp());
  aiir::func::FuncOp func;
  int staticArgCount = sizeIsPresent + putIsPresent + getIsPresent;
  if (staticArgCount == 0) {
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomSeedDefaultPut)>(loc,
                                                                       builder);
    fir::CallOp::create(builder, loc, func);
    return;
  }
  aiir::FunctionType funcTy;
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine;
  aiir::Value argBox;
  llvm::SmallVector<aiir::Value> args;
  if (staticArgCount > 1) {
    func = fir::runtime::getRuntimeFunc<mkRTKey(RandomSeed)>(loc, builder);
    funcTy = func.getFunctionType();
    sourceLine =
        fir::factory::locationToLineNo(builder, loc, funcTy.getInput(4));
    args = fir::runtime::createArguments(builder, loc, funcTy, size, put, get,
                                         sourceFile, sourceLine);
    fir::CallOp::create(builder, loc, func, args);
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
  fir::CallOp::create(builder, loc, func, args);
}

/// generate rename runtime call
void fir::runtime::genRename(fir::FirOpBuilder &builder, aiir::Location loc,
                             aiir::Value path1, aiir::Value path2,
                             aiir::Value status) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Rename)>(loc, builder);
  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();

  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(4));

  llvm::SmallVector<aiir::Value> args =
      fir::runtime::createArguments(builder, loc, runtimeFuncTy, path1, path2,
                                    status, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, runtimeFunc, args);
}

aiir::Value fir::runtime::genSecnds(fir::FirOpBuilder &builder,
                                    aiir::Location loc, aiir::Value refTime) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Secnds)>(loc, builder);

  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();

  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(2));

  llvm::SmallVector<aiir::Value> args = {refTime, sourceFile, sourceLine};
  args = fir::runtime::createArguments(builder, loc, runtimeFuncTy, args);

  return fir::CallOp::create(builder, loc, runtimeFunc, args).getResult(0);
}

/// generate runtime call to time intrinsic
aiir::Value fir::runtime::genTime(fir::FirOpBuilder &builder,
                                  aiir::Location loc) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(time)>(loc, builder);
  return fir::CallOp::create(builder, loc, func, aiir::ValueRange{})
      .getResult(0);
}

/// generate runtime call to transfer intrinsic with no size argument
void fir::runtime::genTransfer(fir::FirOpBuilder &builder, aiir::Location loc,
                               aiir::Value resultBox, aiir::Value sourceBox,
                               aiir::Value moldBox) {

  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(Transfer)>(loc, builder);
  aiir::FunctionType fTy = func.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, sourceBox, moldBox, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

/// generate runtime call to transfer intrinsic with size argument
void fir::runtime::genTransferSize(fir::FirOpBuilder &builder,
                                   aiir::Location loc, aiir::Value resultBox,
                                   aiir::Value sourceBox, aiir::Value moldBox,
                                   aiir::Value size) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(TransferSize)>(loc, builder);
  aiir::FunctionType fTy = func.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  llvm::SmallVector<aiir::Value> args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, sourceBox,
                                    moldBox, sourceFile, sourceLine, size);
  fir::CallOp::create(builder, loc, func, args);
}

/// generate system_clock runtime call/s
/// all intrinsic arguments are optional and may appear here as aiir::Value{}
void fir::runtime::genSystemClock(fir::FirOpBuilder &builder,
                                  aiir::Location loc, aiir::Value count,
                                  aiir::Value rate, aiir::Value max) {
  auto makeCall = [&](aiir::func::FuncOp func, aiir::Value arg) {
    aiir::Type type = arg.getType();
    fir::IfOp ifOp{};
    const bool isOptionalArg =
        fir::valueHasFirAttribute(arg, fir::getOptionalAttrName());
    if (aiir::dyn_cast<fir::PointerType>(type) ||
        aiir::dyn_cast<fir::HeapType>(type)) {
      // Check for a disassociated pointer or an unallocated allocatable.
      assert(!isOptionalArg && "invalid optional argument");
      ifOp = fir::IfOp::create(builder, loc, builder.genIsNotNullAddr(loc, arg),
                               /*withElseRegion=*/false);
    } else if (isOptionalArg) {
      ifOp = fir::IfOp::create(
          builder, loc,
          fir::IsPresentOp::create(builder, loc, builder.getI1Type(), arg),
          /*withElseRegion=*/false);
    }
    if (ifOp)
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    aiir::Type kindTy = func.getFunctionType().getInput(0);
    int integerKind = 8;
    if (auto intType =
            aiir::dyn_cast<aiir::IntegerType>(fir::unwrapRefType(type)))
      integerKind = intType.getWidth() / 8;
    aiir::Value kind = builder.createIntegerConstant(loc, kindTy, integerKind);
    aiir::Value res =
        fir::CallOp::create(builder, loc, func, aiir::ValueRange{kind})
            .getResult(0);
    aiir::Value castRes =
        builder.createConvert(loc, fir::dyn_cast_ptrEleTy(type), res);
    fir::StoreOp::create(builder, loc, castRes, arg);
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
void fir::runtime::genSignal(fir::FirOpBuilder &builder, aiir::Location loc,
                             aiir::Value number, aiir::Value handler,
                             aiir::Value status) {
  assert(aiir::isa<aiir::IntegerType>(number.getType()));
  aiir::Type int64 = builder.getIntegerType(64);
  number = fir::ConvertOp::create(builder, loc, int64, number);

  aiir::Type handlerUnwrappedTy = fir::unwrapRefType(handler.getType());
  if (aiir::isa_and_nonnull<aiir::IntegerType>(handlerUnwrappedTy)) {
    // pass the integer as a function pointer like one would to signal(2)
    handler = fir::LoadOp::create(builder, loc, handler);
    aiir::Type fnPtrTy = fir::LLVMPointerType::get(
        aiir::FunctionType::get(handler.getContext(), {}, {}));
    handler = fir::ConvertOp::create(builder, loc, fnPtrTy, handler);
  } else {
    assert(aiir::isa<fir::BoxProcType>(handler.getType()));
    handler = fir::BoxAddrOp::create(builder, loc, handler);
  }

  aiir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(Signal)>(loc, builder)};
  aiir::Value stat =
      fir::CallOp::create(builder, loc, func, aiir::ValueRange{number, handler})
          ->getResult(0);

  // return status code via status argument (if present)
  if (status) {
    assert(aiir::isa<aiir::IntegerType>(fir::unwrapRefType(status.getType())));
    // status might be dynamically optional, so test if it is present
    aiir::Value isPresent =
        IsPresentOp::create(builder, loc, builder.getI1Type(), status);
    builder.genIfOp(loc, /*results=*/{}, isPresent, /*withElseRegion=*/false)
        .genThen([&]() {
          stat = fir::ConvertOp::create(
              builder, loc, fir::unwrapRefType(status.getType()), stat);
          fir::StoreOp::create(builder, loc, stat, status);
        })
        .end();
  }
}

void fir::runtime::genSleep(fir::FirOpBuilder &builder, aiir::Location loc,
                            aiir::Value seconds) {
  aiir::Type int64 = builder.getIntegerType(64);
  seconds = fir::ConvertOp::create(builder, loc, int64, seconds);
  aiir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(Sleep)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, seconds);
}

/// generate chdir runtime call
aiir::Value fir::runtime::genChdir(fir::FirOpBuilder &builder,
                                   aiir::Location loc, aiir::Value name) {
  aiir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(Chdir)>(loc, builder)};
  llvm::SmallVector<aiir::Value> args =
      fir::runtime::createArguments(builder, loc, func.getFunctionType(), name);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

aiir::Value fir::runtime::genIrand(fir::FirOpBuilder &builder,
                                   aiir::Location loc, aiir::Value i) {
  auto runtimeFunc = fir::runtime::getRuntimeFunc<mkRTKey(Irand)>(loc, builder);
  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();

  llvm::SmallVector<aiir::Value> args =
      fir::runtime::createArguments(builder, loc, runtimeFuncTy, i);
  return fir::CallOp::create(builder, loc, runtimeFunc, args).getResult(0);
}

aiir::Value fir::runtime::genRand(fir::FirOpBuilder &builder,
                                  aiir::Location loc, aiir::Value i) {
  auto runtimeFunc = fir::runtime::getRuntimeFunc<mkRTKey(Rand)>(loc, builder);
  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();

  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(2));

  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, runtimeFuncTy, i, sourceFile, sourceLine);
  return fir::CallOp::create(builder, loc, runtimeFunc, args).getResult(0);
}

void fir::runtime::genShowDescriptor(fir::FirOpBuilder &builder,
                                     aiir::Location loc, aiir::Value descAddr) {
  aiir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(ShowDescriptor)>(loc, builder)};
  fir::CallOp::create(builder, loc, func, descAddr);
}
