//===-- IO.cpp -- I/O statement lowering ----------------------------------===//
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

#include "flang/Lower/IO.h"
#include "../../runtime/io-api.h"
#include "RTBuilder.h"
#include "StatementContext.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/ComplexExpr.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-io"

using namespace Fortran::runtime::io;

#define mkIOKey(X) mkKey(IONAME(X))

namespace Fortran::lower {
/// Static table of IO runtime calls
///
/// This logical map contains the name and type builder function for each IO
/// runtime function listed in the tuple. This table is fully constructed at
/// compile-time. Use the `mkIOKey` macro to access the table.
static constexpr std::tuple<
    mkIOKey(BeginInternalArrayListOutput), mkIOKey(BeginInternalArrayListInput),
    mkIOKey(BeginInternalArrayFormattedOutput),
    mkIOKey(BeginInternalArrayFormattedInput), mkIOKey(BeginInternalListOutput),
    mkIOKey(BeginInternalListInput), mkIOKey(BeginInternalFormattedOutput),
    mkIOKey(BeginInternalFormattedInput), mkIOKey(BeginExternalListOutput),
    mkIOKey(BeginExternalListInput), mkIOKey(BeginExternalFormattedOutput),
    mkIOKey(BeginExternalFormattedInput), mkIOKey(BeginUnformattedOutput),
    mkIOKey(BeginUnformattedInput), mkIOKey(BeginAsynchronousOutput),
    mkIOKey(BeginAsynchronousInput), mkIOKey(BeginWait), mkIOKey(BeginWaitAll),
    mkIOKey(BeginClose), mkIOKey(BeginFlush), mkIOKey(BeginBackspace),
    mkIOKey(BeginEndfile), mkIOKey(BeginRewind), mkIOKey(BeginOpenUnit),
    mkIOKey(BeginOpenNewUnit), mkIOKey(BeginInquireUnit),
    mkIOKey(BeginInquireFile), mkIOKey(BeginInquireIoLength),
    mkIOKey(EnableHandlers), mkIOKey(SetAdvance), mkIOKey(SetBlank),
    mkIOKey(SetDecimal), mkIOKey(SetDelim), mkIOKey(SetPad), mkIOKey(SetPos),
    mkIOKey(SetRec), mkIOKey(SetRound), mkIOKey(SetSign),
    mkIOKey(OutputDescriptor), mkIOKey(InputDescriptor),
    mkIOKey(OutputUnformattedBlock), mkIOKey(InputUnformattedBlock),
    mkIOKey(OutputInteger64), mkIOKey(InputInteger), mkIOKey(OutputReal32),
    mkIOKey(InputReal32), mkIOKey(OutputReal64), mkIOKey(InputReal64),
    mkIOKey(OutputComplex32), mkIOKey(InputComplex32), mkIOKey(OutputComplex64),
    mkIOKey(InputComplex64), mkIOKey(OutputAscii), mkIOKey(InputAscii),
    mkIOKey(OutputLogical), mkIOKey(InputLogical), mkIOKey(SetAccess),
    mkIOKey(SetAction), mkIOKey(SetAsynchronous), mkIOKey(SetCarriagecontrol),
    mkIOKey(SetEncoding), mkIOKey(SetForm), mkIOKey(SetPosition),
    mkIOKey(SetRecl), mkIOKey(SetStatus), mkIOKey(SetFile), mkIOKey(GetNewUnit),
    mkIOKey(GetSize), mkIOKey(GetIoLength), mkIOKey(GetIoMsg),
    mkIOKey(InquireCharacter), mkIOKey(InquireLogical),
    mkIOKey(InquirePendingId), mkIOKey(InquireInteger64),
    mkIOKey(EndIoStatement)>
    newIOTable;
} // namespace Fortran::lower

namespace {
/// IO statements may require exceptional condition handling.  A statement that
/// encounters an exceptional condition may branch to a label given on an ERR
/// (error), END (end-of-file), or EOR (end-of-record) specifier.  An IOSTAT
/// specifier variable may be set to a value that indicates some condition,
/// and an IOMSG specifier variable may be set to a description of a condition.
struct ConditionSpecInfo {
  const Fortran::semantics::SomeExpr *ioStatExpr{};
  const Fortran::semantics::SomeExpr *ioMsgExpr{};
  bool hasErr{};
  bool hasEnd{};
  bool hasEor{};

  /// Check for any condition specifier that applies to specifier processing.
  bool hasErrorConditionSpec() const { return ioStatExpr != nullptr || hasErr; }

  /// Check for any condition specifier that applies to data transfer items
  /// in a PRINT, READ, WRITE, or WAIT statement.  (WAIT may be irrelevant.)
  bool hasTransferConditionSpec() const {
    return hasErrorConditionSpec() || hasEnd || hasEor;
  }

  /// Check for any condition specifier, including IOMSG.
  bool hasAnyConditionSpec() const {
    return hasTransferConditionSpec() || ioMsgExpr != nullptr;
  }
};
} // namespace

template <typename D>
static void genIoLoop(Fortran::lower::AbstractConverter &converter,
                      mlir::Value cookie, const D &ioImpliedDo,
                      bool isFormatted, bool checkResult, mlir::Value &ok,
                      bool inLoop, Fortran::lower::StatementContext &stmtCtx);

/// Helper function to retrieve the name of the IO function given the key `A`
template <typename A>
static constexpr const char *getName() {
  return std::get<A>(Fortran::lower::newIOTable).name;
}

/// Helper function to retrieve the type model signature builder of the IO
/// function as defined by the key `A`
template <typename A>
static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
  return std::get<A>(Fortran::lower::newIOTable).getTypeModel();
}

inline int64_t getLength(mlir::Type argTy) {
  return argTy.cast<fir::SequenceType>().getShape()[0];
}

/// Get (or generate) the MLIR FuncOp for a given IO runtime function.
template <typename E>
static mlir::FuncOp getIORuntimeFunc(mlir::Location loc,
                                     Fortran::lower::FirOpBuilder &builder) {
  auto name = getName<E>();
  auto func = builder.getNamedFunction(name);
  if (func)
    return func;
  auto funTy = getTypeModel<E>()(builder.getContext());
  func = builder.createFunction(loc, name, funTy);
  func->setAttr("fir.runtime", builder.getUnitAttr());
  func->setAttr("fir.io", builder.getUnitAttr());
  return func;
}

/// Generate calls to end an IO statement.  Return the IOSTAT value, if any.
/// It is the caller's responsibility to generate branches on that value.
static mlir::Value genEndIO(Fortran::lower::AbstractConverter &converter,
                            mlir::Location loc, mlir::Value cookie,
                            const ConditionSpecInfo &csi,
                            Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  if (csi.ioMsgExpr) {
    auto getIoMsg = getIORuntimeFunc<mkIOKey(GetIoMsg)>(loc, builder);
    auto ioMsgVar = converter.genExprAddr(csi.ioMsgExpr, stmtCtx, loc);
    builder.create<fir::CallOp>(
        loc, getIoMsg,
        mlir::ValueRange{
            cookie,
            builder.createConvert(loc, getIoMsg.getType().getInput(1),
                                  fir::getBase(ioMsgVar)),
            builder.createConvert(loc, getIoMsg.getType().getInput(2),
                                  fir::getLen(ioMsgVar))});
  }
  auto endIoStatement = getIORuntimeFunc<mkIOKey(EndIoStatement)>(loc, builder);
  auto call = builder.create<fir::CallOp>(loc, endIoStatement,
                                          mlir::ValueRange{cookie});
  if (csi.ioStatExpr) {
    auto ioStatVar =
        fir::getBase(converter.genExprAddr(csi.ioStatExpr, stmtCtx, loc));
    auto ioStatResult = builder.createConvert(
        loc, converter.genType(*csi.ioStatExpr), call.getResult(0));
    builder.create<fir::StoreOp>(loc, ioStatResult, ioStatVar);
  }
  return csi.hasTransferConditionSpec() ? call.getResult(0) : mlir::Value{};
}

/// Make the next call in the IO statement conditional on runtime result `ok`.
/// If a call returns `ok==false`, further suboperation calls for an I/O
/// statement will be skipped.  This may generate branch heavy, deeply nested
/// conditionals for I/O statements with a large number of suboperations.
static void makeNextConditionalOn(Fortran::lower::FirOpBuilder &builder,
                                  mlir::Location loc, bool checkResult,
                                  mlir::Value ok, bool inLoop = false) {
  if (!checkResult || !ok)
    // Either no I/O calls need to be checked, or this will be the first call.
    return;

  // A previous I/O call for a statement returned the bool `ok`.  If this call
  // is in a fir.iterate_while loop, the result must be propagated up to the
  // loop scope as an extra ifOp result. (The propagation is done in genIoLoop.)
  mlir::TypeRange resTy;
  if (inLoop)
    resTy = builder.getI1Type();
  auto ifOp = builder.create<fir::IfOp>(loc, resTy, ok,
                                        /*withElseRegion=*/inLoop);
  builder.setInsertionPointToStart(&ifOp.thenRegion().front());
}

/// Get the OutputXyz routine to output a value of the given type.
static mlir::FuncOp getOutputFunc(mlir::Location loc,
                                  Fortran::lower::FirOpBuilder &builder,
                                  mlir::Type type, bool isFormatted) {
  if (!isFormatted)
    return getIORuntimeFunc<mkIOKey(OutputDescriptor)>(loc, builder);
  if (auto ty = type.dyn_cast<mlir::IntegerType>())
    return ty.getWidth() == 1
               ? getIORuntimeFunc<mkIOKey(OutputLogical)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(OutputInteger64)>(loc, builder);
  if (auto ty = type.dyn_cast<mlir::FloatType>())
    return ty.getWidth() <= 32
               ? getIORuntimeFunc<mkIOKey(OutputReal32)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(OutputReal64)>(loc, builder);
  if (auto ty = type.dyn_cast<fir::ComplexType>())
    return ty.getFKind() <= 4
               ? getIORuntimeFunc<mkIOKey(OutputComplex32)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(OutputComplex64)>(loc, builder);
  if (type.isa<fir::LogicalType>())
    return getIORuntimeFunc<mkIOKey(OutputLogical)>(loc, builder);
  if (type.isa<fir::BoxType>())
    return getIORuntimeFunc<mkIOKey(OutputDescriptor)>(loc, builder);
  if (Fortran::lower::CharacterExprHelper::isCharacterScalar(type))
    return getIORuntimeFunc<mkIOKey(OutputAscii)>(loc, builder);
  // Use descriptors for arrays
  if (auto refTy = type.dyn_cast<fir::ReferenceType>())
    type = refTy.getEleTy();
  return getIORuntimeFunc<mkIOKey(OutputDescriptor)>(loc, builder);
}

/// Generate a sequence of output data transfer calls.
static void
genOutputItemList(Fortran::lower::AbstractConverter &converter,
                  mlir::Value cookie,
                  const std::list<Fortran::parser::OutputItem> &items,
                  bool isFormatted, bool checkResult, mlir::Value &ok,
                  bool inLoop, Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  for (auto &item : items) {
    if (const auto &impliedDo = std::get_if<1>(&item.u)) {
      genIoLoop(converter, cookie, impliedDo->value(), isFormatted, checkResult,
                ok, inLoop, stmtCtx);
      continue;
    }
    auto &pExpr = std::get<Fortran::parser::Expr>(item.u);
    auto loc = converter.genLocation(pExpr.source);
    makeNextConditionalOn(builder, loc, checkResult, ok, inLoop);

    const auto *expr = Fortran::semantics::GetExpr(pExpr);
    if (!expr)
      fir::emitFatalError(loc, "internal error: could not get evaluate::Expr");
    auto itemTy = converter.genType(*expr);
    auto outputFunc = getOutputFunc(loc, builder, itemTy, isFormatted);
    auto argType = outputFunc.getType().getInput(1);
    assert((isFormatted || argType.isa<fir::BoxType>()) &&
           "expect descriptor for unformatted IO runtime");
    llvm::SmallVector<mlir::Value> outputFuncArgs = {cookie};
    Fortran::lower::CharacterExprHelper helper{builder, loc};
    if (argType.isa<fir::BoxType>()) {
      auto exv = converter.genExprAddr(expr, stmtCtx, loc);
      auto box = builder.createBox(loc, exv);
      outputFuncArgs.push_back(builder.createConvert(loc, argType, box));
    } else if (helper.isCharacterScalar(itemTy)) {
      auto exv = converter.genExprAddr(expr, stmtCtx, loc);
      // scalar allocatable/pointer may also get here, not clear if
      // genExprAddr will lower them as CharBoxValue or BoxValue.
      if (!exv.getCharBox())
        llvm::report_fatal_error(
            "internal error: scalar character not in CharBox");
      outputFuncArgs.push_back(builder.createConvert(
          loc, outputFunc.getType().getInput(1), fir::getBase(exv)));
      outputFuncArgs.push_back(builder.createConvert(
          loc, outputFunc.getType().getInput(2), fir::getLen(exv)));
    } else {
      auto itemBox = converter.genExprValue(expr, stmtCtx, loc);
      auto itemValue = fir::getBase(itemBox);
      if (fir::isa_complex(itemTy)) {
        auto parts =
            Fortran::lower::ComplexExprHelper{builder, loc}.extractParts(
                itemValue);
        outputFuncArgs.push_back(parts.first);
        outputFuncArgs.push_back(parts.second);
      } else {
        itemValue = builder.createConvert(loc, argType, itemValue);
        outputFuncArgs.push_back(itemValue);
      }
    }
    ok = builder.create<fir::CallOp>(loc, outputFunc, outputFuncArgs)
             .getResult(0);
  }
}

/// Get the InputXyz routine to input a value of the given type.
static mlir::FuncOp getInputFunc(mlir::Location loc,
                                 Fortran::lower::FirOpBuilder &builder,
                                 mlir::Type type, bool isFormatted) {
  if (!isFormatted)
    return getIORuntimeFunc<mkIOKey(InputDescriptor)>(loc, builder);
  if (auto ty = type.dyn_cast<mlir::IntegerType>())
    return ty.getWidth() == 1
               ? getIORuntimeFunc<mkIOKey(InputLogical)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(InputInteger)>(loc, builder);
  if (auto ty = type.dyn_cast<mlir::FloatType>())
    return ty.getWidth() <= 32
               ? getIORuntimeFunc<mkIOKey(InputReal32)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(InputReal64)>(loc, builder);
  if (auto ty = type.dyn_cast<fir::ComplexType>())
    return ty.getFKind() <= 4
               ? getIORuntimeFunc<mkIOKey(InputComplex32)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(InputComplex64)>(loc, builder);
  if (type.isa<fir::LogicalType>())
    return getIORuntimeFunc<mkIOKey(InputLogical)>(loc, builder);
  if (type.isa<fir::BoxType>())
    return getIORuntimeFunc<mkIOKey(InputDescriptor)>(loc, builder);
  if (Fortran::lower::CharacterExprHelper::isCharacterScalar(type))
    return getIORuntimeFunc<mkIOKey(InputAscii)>(loc, builder);
  return getIORuntimeFunc<mkIOKey(InputDescriptor)>(loc, builder);
}

/// Generate a sequence of input data transfer calls.
static void genInputItemList(Fortran::lower::AbstractConverter &converter,
                             mlir::Value cookie,
                             const std::list<Fortran::parser::InputItem> &items,
                             bool isFormatted, bool checkResult,
                             mlir::Value &ok, bool inLoop,
                             Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  for (auto &item : items) {
    if (const auto &impliedDo = std::get_if<1>(&item.u)) {
      genIoLoop(converter, cookie, impliedDo->value(), isFormatted, checkResult,
                ok, inLoop, stmtCtx);
      continue;
    }
    auto &pVar = std::get<Fortran::parser::Variable>(item.u);
    auto loc = converter.genLocation(pVar.GetSource());
    makeNextConditionalOn(builder, loc, checkResult, ok, inLoop);
    Fortran::lower::CharacterExprHelper charHelper{builder, loc};
    auto itemBox =
        converter.genExprAddr(Fortran::semantics::GetExpr(pVar), stmtCtx, loc);
    auto itemAddr = fir::getBase(itemBox);
    auto itemTy = fir::dyn_cast_ptrEleTy(itemAddr.getType());
    if (!itemTy)
      fir::emitFatalError(loc, "internal error: unhandled input item type");
    auto inputFunc = getInputFunc(loc, builder, itemTy, isFormatted);
    auto argType = inputFunc.getType().getInput(1);
    if (argType.isa<fir::BoxType>())
      itemAddr = builder.createBox(loc, itemBox);
    itemAddr = builder.createConvert(loc, argType, itemAddr);
    llvm::SmallVector<mlir::Value> inputFuncArgs = {cookie, itemAddr};
    if (argType.isa<fir::BoxType>()) {
      // do nothing
    } else if (charHelper.isCharacterScalar(itemTy)) {
      auto len = fir::getLen(itemBox);
      inputFuncArgs.push_back(
          builder.createConvert(loc, inputFunc.getType().getInput(2), len));
    } else if (itemTy.isa<mlir::IntegerType>()) {
      inputFuncArgs.push_back(builder.create<mlir::ConstantOp>(
          loc, builder.getI32IntegerAttr(
                   itemTy.cast<mlir::IntegerType>().getWidth() / 8)));
    }
    ok =
        builder.create<fir::CallOp>(loc, inputFunc, inputFuncArgs).getResult(0);
  }
}

/// Generate an io-implied-do loop.
template <typename D>
static void genIoLoop(Fortran::lower::AbstractConverter &converter,
                      mlir::Value cookie, const D &ioImpliedDo,
                      bool isFormatted, bool checkResult, mlir::Value &ok,
                      bool inLoop, Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  makeNextConditionalOn(builder, loc, checkResult, ok, inLoop);
  const auto &itemList = std::get<0>(ioImpliedDo.t);
  const auto &control = std::get<1>(ioImpliedDo.t);
  const auto &loopSym = *control.name.thing.thing.symbol;
  auto loopVar = converter.getSymbolAddress(loopSym);
  auto genControlValue = [&](const Fortran::parser::ScalarIntExpr &expr) {
    auto v = fir::getBase(
        converter.genExprValue(*Fortran::semantics::GetExpr(expr), stmtCtx));
    return builder.createConvert(loc, builder.getIndexType(), v);
  };
  auto lowerValue = genControlValue(control.lower);
  auto upperValue = genControlValue(control.upper);
  auto stepValue = control.step.has_value()
                       ? genControlValue(*control.step)
                       : builder.create<mlir::ConstantIndexOp>(loc, 1);
  auto genItemList = [&](const D &ioImpliedDo) {
    Fortran::lower::StatementContext loopCtx;
    if constexpr (std::is_same_v<D, Fortran::parser::InputImpliedDo>)
      genInputItemList(converter, cookie, itemList, isFormatted, checkResult,
                       ok, /*inLoop=*/true, loopCtx);
    else
      genOutputItemList(converter, cookie, itemList, isFormatted, checkResult,
                        ok, /*inLoop=*/true, loopCtx);
  };
  if (!checkResult) {
    // No I/O call result checks - the loop is a fir.do_loop op.
    auto doLoopOp = builder.create<fir::DoLoopOp>(
        loc, lowerValue, upperValue, stepValue, /*unordered=*/false,
        /*finalCountValue=*/true);
    builder.setInsertionPointToStart(doLoopOp.getBody());
    auto lcv = builder.createConvert(loc, converter.genType(loopSym),
                                     doLoopOp.getInductionVar());
    builder.create<fir::StoreOp>(loc, lcv, loopVar);
    genItemList(ioImpliedDo);
    builder.setInsertionPointToEnd(doLoopOp.getBody());
    mlir::Value result = builder.create<mlir::AddIOp>(
        loc, doLoopOp.getInductionVar(), doLoopOp.step());
    builder.create<fir::ResultOp>(loc, result);
    builder.setInsertionPointAfter(doLoopOp);
    // The loop control variable may be used after the loop.
    lcv = builder.createConvert(loc, converter.genType(loopSym),
                                doLoopOp.getResult(0));
    builder.create<fir::StoreOp>(loc, lcv, loopVar);
    return;
  }
  // Check I/O call results - the loop is a fir.iterate_while op.
  if (!ok)
    ok = builder.createIntegerConstant(loc, builder.getI1Type(), 1);
  auto iterWhileOp = builder.create<fir::IterWhileOp>(
      loc, lowerValue, upperValue, stepValue, ok, /*finalCountValue*/ true);
  builder.setInsertionPointToStart(iterWhileOp.getBody());
  auto lcv = builder.createConvert(loc, converter.genType(loopSym),
                                   iterWhileOp.getInductionVar());
  builder.create<fir::StoreOp>(loc, lcv, loopVar);
  ok = iterWhileOp.getIterateVar();
  auto falseValue = builder.createIntegerConstant(loc, builder.getI1Type(), 0);
  genItemList(ioImpliedDo);
  // Unwind nested I/O call scopes, filling in true and false ResultOp's.
  for (auto *op = builder.getBlock()->getParentOp(); isa<fir::IfOp>(op);
       op = op->getBlock()->getParentOp()) {
    auto ifOp = dyn_cast<fir::IfOp>(op);
    auto *lastOp = &ifOp.thenRegion().front().back();
    builder.setInsertionPointAfter(lastOp);
    builder.create<fir::ResultOp>(loc, lastOp->getResult(0)); // runtime result
    builder.setInsertionPointToStart(&ifOp.elseRegion().front());
    builder.create<fir::ResultOp>(loc, falseValue); // known false result
  }
  builder.setInsertionPointToEnd(iterWhileOp.getBody());
  auto iterateResult = builder.getBlock()->back().getResult(0);
  auto inductionResult0 = iterWhileOp.getInductionVar();
  auto inductionResult1 =
      builder.create<mlir::AddIOp>(loc, inductionResult0, iterWhileOp.step());
  auto inductionResult = builder.create<mlir::SelectOp>(
      loc, iterateResult, inductionResult1, inductionResult0);
  llvm::SmallVector<mlir::Value> results = {inductionResult, iterateResult};
  builder.create<fir::ResultOp>(loc, results);
  ok = iterWhileOp.getResult(1);
  builder.setInsertionPointAfter(iterWhileOp);
  // The loop control variable may be used after the loop.
  lcv = builder.createConvert(loc, converter.genType(loopSym),
                              iterWhileOp.getResult(0));
  builder.create<fir::StoreOp>(loc, lcv, loopVar);
}

//===----------------------------------------------------------------------===//
// Default argument generation.
//===----------------------------------------------------------------------===//

static mlir::Value locToFilename(Fortran::lower::AbstractConverter &converter,
                                 mlir::Location loc, mlir::Type toType) {
  auto &builder = converter.getFirOpBuilder();
  return builder.createConvert(
      loc, toType, Fortran::lower::locationToFilename(builder, loc));
}

static mlir::Value locToLineNo(Fortran::lower::AbstractConverter &converter,
                               mlir::Location loc, mlir::Type toType) {
  return Fortran::lower::locationToLineNo(converter.getFirOpBuilder(), loc,
                                          toType);
}

static mlir::Value getDefaultScratch(Fortran::lower::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Type toType) {
  mlir::Value null =
      builder.create<mlir::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  return builder.createConvert(loc, toType, null);
}

static mlir::Value getDefaultScratchLen(Fortran::lower::FirOpBuilder &builder,
                                        mlir::Location loc, mlir::Type toType) {
  return builder.create<mlir::ConstantOp>(loc,
                                          builder.getIntegerAttr(toType, 0));
}

/// Generate a reference to a buffer and the length of buffer given
/// a character expression. An array expression will be cast to scalar
/// character as long as they are contiguous.
static std::tuple<mlir::Value, mlir::Value>
genBuffer(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const Fortran::semantics::SomeExpr &expr, mlir::Type strTy,
          mlir::Type lenTy, Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  auto exprAddr = converter.genExprAddr(expr, stmtCtx);
  Fortran::lower::CharacterExprHelper helper(builder, loc);
  using ValuePair = std::pair<mlir::Value, mlir::Value>;
  auto [buff, len] = exprAddr.match(
      [&](const fir::CharBoxValue &x) -> ValuePair {
        return {x.getBuffer(), x.getLen()};
      },
      [&](const fir::CharArrayBoxValue &x) -> ValuePair {
        auto scalar = helper.toScalarCharacter(x);
        return {scalar.getBuffer(), scalar.getLen()};
      },
      [&](const fir::BoxValue &) -> ValuePair {
        // May need to copy before after IO to handle contiguous
        // aspect. Not sure descriptor can get here though.
        TODO(loc, "character descriptor to contiguous buffer");
      },
      [&](const auto &) -> ValuePair {
        llvm::report_fatal_error(
            "internal error: IO buffer is not a character");
      });
  buff = builder.createConvert(loc, strTy, buff);
  len = builder.createConvert(loc, lenTy, len);
  return {buff, len};
}

/// Lower a string literal. Many arguments to the runtime are conveyed as
/// Fortran CHARACTER literals.
template <typename A>
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
lowerStringLit(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
               const A &syntax, mlir::Type strTy, mlir::Type lenTy,
               mlir::Type ty2 = {}) {
  auto &builder = converter.getFirOpBuilder();
  auto *expr = Fortran::semantics::GetExpr(syntax);
  if (!expr)
    fir::emitFatalError(loc, "internal error: null semantic expr in IO");
  Fortran::lower::StatementContext stmtCtx;
  auto [buff, len] = genBuffer(converter, loc, *expr, strTy, lenTy, stmtCtx);
  mlir::Value kind;
  if (ty2) {
    auto kindVal = expr->GetType().value().kind();
    kind = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ty2, kindVal));
  }
  return {buff, len, kind};
}

/// Pass the body of the FORMAT statement in as if it were a CHARACTER literal
/// constant. NB: This is the prescribed manner in which the front-end passes
/// this information to lowering.
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
lowerSourceTextAsStringLit(Fortran::lower::AbstractConverter &converter,
                           mlir::Location loc, llvm::StringRef text,
                           mlir::Type strTy, mlir::Type lenTy) {
  text = text.drop_front(text.find('('));
  text = text.take_front(text.rfind(')') + 1);
  auto &builder = converter.getFirOpBuilder();
  auto addrGlobalStringLit =
      fir::getBase(Fortran::lower::createStringLiteral(builder, loc, text));
  auto buff = builder.createConvert(loc, strTy, addrGlobalStringLit);
  auto len = builder.createIntegerConstant(loc, lenTy, text.size());
  return {buff, len, mlir::Value{}};
}

//===----------------------------------------------------------------------===//
// Handle I/O statement specifiers.
// These are threaded together for a single statement via the passed cookie.
//===----------------------------------------------------------------------===//

/// Generic to build an integral argument to the runtime.
template <typename A, typename B>
mlir::Value genIntIOOption(Fortran::lower::AbstractConverter &converter,
                           mlir::Location loc, mlir::Value cookie,
                           Fortran::lower::StatementContext &stmtCtx,
                           const B &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc = getIORuntimeFunc<A>(loc, builder);
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto expr = fir::getBase(converter.genExprValue(
      Fortran::semantics::GetExpr(spec.v), stmtCtx, loc));
  auto val = builder.createConvert(loc, ioFuncTy.getInput(1), expr);
  llvm::SmallVector<mlir::Value> ioArgs = {cookie, val};
  return builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

/// Generic to build a string argument to the runtime. This passes a CHARACTER
/// as a pointer to the buffer and a LEN parameter.
template <typename A, typename B>
mlir::Value genCharIOOption(Fortran::lower::AbstractConverter &converter,
                            mlir::Location loc, mlir::Value cookie,
                            const B &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc = getIORuntimeFunc<A>(loc, builder);
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(converter, loc, spec, ioFuncTy.getInput(1),
                            ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value> ioArgs = {cookie, std::get<0>(tup),
                                           std::get<1>(tup)};
  return builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <typename A>
mlir::Value genIOOption(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc, mlir::Value cookie,
                        Fortran::lower::StatementContext &stmtCtx,
                        const A &spec) {
  // default case: do nothing
  return {};
}

template <>
mlir::Value genIOOption<Fortran::parser::FileNameExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::FileNameExpr &spec) {
  auto &builder = converter.getFirOpBuilder();
  // has an extra KIND argument
  auto ioFunc = getIORuntimeFunc<mkIOKey(SetFile)>(loc, builder);
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(converter, loc, spec, ioFuncTy.getInput(1),
                            ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value> ioArgs{cookie, std::get<0>(tup),
                                        std::get<1>(tup)};
  return builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::ConnectSpec::CharExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::ConnectSpec::CharExpr &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc;
  switch (std::get<Fortran::parser::ConnectSpec::CharExpr::Kind>(spec.t)) {
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Access:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAccess)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Action:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAction)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Asynchronous:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAsynchronous)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Blank:
    ioFunc = getIORuntimeFunc<mkIOKey(SetBlank)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Decimal:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDecimal)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Delim:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDelim)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Encoding:
    ioFunc = getIORuntimeFunc<mkIOKey(SetEncoding)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Form:
    ioFunc = getIORuntimeFunc<mkIOKey(SetForm)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Pad:
    ioFunc = getIORuntimeFunc<mkIOKey(SetPad)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Position:
    ioFunc = getIORuntimeFunc<mkIOKey(SetPosition)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Round:
    ioFunc = getIORuntimeFunc<mkIOKey(SetRound)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Sign:
    ioFunc = getIORuntimeFunc<mkIOKey(SetSign)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Carriagecontrol:
    ioFunc = getIORuntimeFunc<mkIOKey(SetCarriagecontrol)>(loc, builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Convert:
    TODO(loc, "CONVERT not part of the runtime::io interface");
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Dispose:
    TODO(loc, "DISPOSE not part of the runtime::io interface");
  }
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(
      converter, loc, std::get<Fortran::parser::ScalarDefaultCharExpr>(spec.t),
      ioFuncTy.getInput(1), ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value> ioArgs = {cookie, std::get<0>(tup),
                                           std::get<1>(tup)};
  return builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::ConnectSpec::Recl>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::ConnectSpec::Recl &spec) {
  return genIntIOOption<mkIOKey(SetRecl)>(converter, loc, cookie, stmtCtx,
                                          spec);
}

template <>
mlir::Value genIOOption<Fortran::parser::StatusExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::StatusExpr &spec) {
  return genCharIOOption<mkIOKey(SetStatus)>(converter, loc, cookie, spec.v);
}

template <>
mlir::Value
genIOOption<Fortran::parser::Name>(Fortran::lower::AbstractConverter &converter,
                                   mlir::Location loc, mlir::Value cookie,
                                   Fortran::lower::StatementContext &stmtCtx,
                                   const Fortran::parser::Name &spec) {
  TODO(loc, "namelist");
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::CharExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::IoControlSpec::CharExpr &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc;
  switch (std::get<Fortran::parser::IoControlSpec::CharExpr::Kind>(spec.t)) {
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Advance:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAdvance)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Blank:
    ioFunc = getIORuntimeFunc<mkIOKey(SetBlank)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Decimal:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDecimal)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Delim:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDelim)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Pad:
    ioFunc = getIORuntimeFunc<mkIOKey(SetPad)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Round:
    ioFunc = getIORuntimeFunc<mkIOKey(SetRound)>(loc, builder);
    break;
  case Fortran::parser::IoControlSpec::CharExpr::Kind::Sign:
    ioFunc = getIORuntimeFunc<mkIOKey(SetSign)>(loc, builder);
    break;
  }
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(
      converter, loc, std::get<Fortran::parser::ScalarDefaultCharExpr>(spec.t),
      ioFuncTy.getInput(1), ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value> ioArgs = {cookie, std::get<0>(tup),
                                           std::get<1>(tup)};
  return builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Asynchronous>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::IoControlSpec::Asynchronous &spec) {
  return genCharIOOption<mkIOKey(SetAsynchronous)>(converter, loc, cookie,
                                                   spec.v);
}

template <>
mlir::Value genIOOption<Fortran::parser::IdVariable>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::IdVariable &spec) {
  TODO(loc, "asynchronous ID not implemented");
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Pos>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::IoControlSpec::Pos &spec) {
  return genIntIOOption<mkIOKey(SetPos)>(converter, loc, cookie, stmtCtx, spec);
}
template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Rec>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::IoControlSpec::Rec &spec) {
  return genIntIOOption<mkIOKey(SetRec)>(converter, loc, cookie, stmtCtx, spec);
}

//===----------------------------------------------------------------------===//
// Gather I/O statement condition specifier information (if any).
//===----------------------------------------------------------------------===//

template <typename SEEK, typename A>
static bool hasX(const A &list) {
  for (const auto &spec : list)
    if (std::holds_alternative<SEEK>(spec.u))
      return true;
  return false;
}

template <typename SEEK, typename A>
static bool hasMem(const A &stmt) {
  return hasX<SEEK>(stmt.v);
}

/// Get the sought expression from the specifier list.
template <typename SEEK, typename A>
static const Fortran::semantics::SomeExpr *getExpr(const A &stmt) {
  for (const auto &spec : stmt.v)
    if (auto *f = std::get_if<SEEK>(&spec.u))
      return Fortran::semantics::GetExpr(f->v);
  llvm::report_fatal_error("must have a file unit");
}

/// For each specifier, build the appropriate call, threading the cookie.
template <typename A>
static void threadSpecs(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc, mlir::Value cookie,
                        const A &specList, bool checkResult, mlir::Value &ok,
                        Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  for (const auto &spec : specList) {
    makeNextConditionalOn(builder, loc, checkResult, ok);
    ok = std::visit(Fortran::common::visitors{[&](const auto &x) {
                      return genIOOption(converter, loc, cookie, stmtCtx, x);
                    }},
                    spec.u);
  }
}

/// Most I/O statements allow one or more of five optional exception condition
/// handling specifiers: ERR, EOR, END, IOSTAT, and IOMSG. The first three
/// cause control flow to transfer to another statement. The final two return
/// information from the runtime, via a variable, about the nature of the
/// condition that occurred. These condition specifiers are handled here.
template <typename A>
static void
genConditionHandlerCall(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc, mlir::Value cookie,
                        const A &specList, ConditionSpecInfo &csi) {
  for (const auto &spec : specList) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::StatVariable &var) {
              csi.ioStatExpr = Fortran::semantics::GetExpr(var);
            },
            [&](const Fortran::parser::InquireSpec::IntVar &var) {
              if (std::get<Fortran::parser::InquireSpec::IntVar::Kind>(var.t) ==
                  Fortran::parser::InquireSpec::IntVar::Kind::Iostat)
                csi.ioStatExpr = Fortran::semantics::GetExpr(
                    std::get<Fortran::parser::ScalarIntVariable>(var.t));
            },
            [&](const Fortran::parser::MsgVariable &var) {
              csi.ioMsgExpr = Fortran::semantics::GetExpr(var);
            },
            [&](const Fortran::parser::InquireSpec::CharVar &var) {
              if (std::get<Fortran::parser::InquireSpec::CharVar::Kind>(
                      var.t) ==
                  Fortran::parser::InquireSpec::CharVar::Kind::Iomsg)
                csi.ioMsgExpr = Fortran::semantics::GetExpr(
                    std::get<Fortran::parser::ScalarDefaultCharVariable>(
                        var.t));
            },
            [&](const Fortran::parser::EndLabel &) { csi.hasEnd = true; },
            [&](const Fortran::parser::EorLabel &) { csi.hasEor = true; },
            [&](const Fortran::parser::ErrLabel &) { csi.hasErr = true; },
            [](const auto &) {}},
        spec.u);
  }
  if (!csi.hasAnyConditionSpec())
    return;
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp enableHandlers =
      getIORuntimeFunc<mkIOKey(EnableHandlers)>(loc, builder);
  mlir::Type boolType = enableHandlers.getType().getInput(1);
  auto boolValue = [&](bool specifierIsPresent) {
    return builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(boolType, specifierIsPresent));
  };
  llvm::SmallVector<mlir::Value> ioArgs = {cookie,
                                           boolValue(csi.ioStatExpr != nullptr),
                                           boolValue(csi.hasErr),
                                           boolValue(csi.hasEnd),
                                           boolValue(csi.hasEor),
                                           boolValue(csi.ioMsgExpr != nullptr)};
  builder.create<fir::CallOp>(loc, enableHandlers, ioArgs);
}

//===----------------------------------------------------------------------===//
// Data transfer helpers
//===----------------------------------------------------------------------===//

template <typename SEEK, typename A>
static bool hasIOControl(const A &stmt) {
  return hasX<SEEK>(stmt.controls);
}

template <typename SEEK, typename A>
static const auto *getIOControl(const A &stmt) {
  for (const auto &spec : stmt.controls)
    if (const auto *result = std::get_if<SEEK>(&spec.u))
      return result;
  return static_cast<const SEEK *>(nullptr);
}

/// returns true iff the expression in the parse tree is not really a format but
/// rather a namelist group
template <typename A>
static bool formatIsActuallyNamelist(const A &format) {
  if (auto *e = std::get_if<Fortran::parser::Expr>(&format.u)) {
    auto *expr = Fortran::semantics::GetExpr(*e);
    if (const Fortran::semantics::Symbol *y =
            Fortran::evaluate::UnwrapWholeSymbolDataRef(*expr))
      return y->has<Fortran::semantics::NamelistDetails>();
  }
  return false;
}

template <typename A>
static bool isDataTransferFormatted(const A &stmt) {
  if (stmt.format)
    return !formatIsActuallyNamelist(*stmt.format);
  return hasIOControl<Fortran::parser::Format>(stmt);
}
template <>
constexpr bool isDataTransferFormatted<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return true; // PRINT is always formatted
}

template <typename A>
static bool isDataTransferList(const A &stmt) {
  if (stmt.format)
    return std::holds_alternative<Fortran::parser::Star>(stmt.format->u);
  if (auto *mem = getIOControl<Fortran::parser::Format>(stmt))
    return std::holds_alternative<Fortran::parser::Star>(mem->u);
  return false;
}
template <>
bool isDataTransferList<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &stmt) {
  return std::holds_alternative<Fortran::parser::Star>(
      std::get<Fortran::parser::Format>(stmt.t).u);
}

template <typename A>
static bool isDataTransferInternal(const A &stmt) {
  if (stmt.iounit.has_value())
    return std::holds_alternative<Fortran::parser::Variable>(stmt.iounit->u);
  if (auto *unit = getIOControl<Fortran::parser::IoUnit>(stmt))
    return std::holds_alternative<Fortran::parser::Variable>(unit->u);
  return false;
}
template <>
constexpr bool isDataTransferInternal<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return false;
}

/// If the variable `var` is an array or of a KIND other than the default
/// (normally 1), then a descriptor is required by the runtime IO API. This
/// condition holds even in F77 sources.
static llvm::Optional<fir::ExtendedValue> getVariableBufferRequiredDescriptor(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::Variable &var,
    Fortran::lower::StatementContext &stmtCtx) {
  auto varBox = converter.genExprAddr(var.typedExpr->v.value(), stmtCtx);
  auto defCharKind = converter.getKindMap().defaultCharacterKind();
  auto varAddr = fir::getBase(varBox);
  if (Fortran::lower::CharacterExprHelper::getCharacterOrSequenceKind(
          varAddr.getType()) != defCharKind)
    return varBox;
  if (Fortran::lower::CharacterExprHelper::isArray(varAddr.getType()))
    return varBox;
  return llvm::None;
}

template <typename A>
static llvm::Optional<fir::ExtendedValue>
getIfDataTransferInternalRequiresDescriptor(
    Fortran::lower::AbstractConverter &converter, const A &stmt,
    Fortran::lower::StatementContext &stmtCtx) {
  if (stmt.iounit.has_value())
    if (auto *var = std::get_if<Fortran::parser::Variable>(&stmt.iounit->u))
      return getVariableBufferRequiredDescriptor(converter, *var, stmtCtx);
  if (auto *unit = getIOControl<Fortran::parser::IoUnit>(stmt))
    if (auto *var = std::get_if<Fortran::parser::Variable>(&unit->u))
      return getVariableBufferRequiredDescriptor(converter, *var, stmtCtx);
  return llvm::None;
}
template <>
inline llvm::Optional<fir::ExtendedValue>
getIfDataTransferInternalRequiresDescriptor<Fortran::parser::PrintStmt>(
    Fortran::lower::AbstractConverter &, const Fortran::parser::PrintStmt &,
    Fortran::lower::StatementContext &) {
  return llvm::None;
}

template <typename A>
static bool isDataTransferAsynchronous(const A &stmt) {
  if (auto *asynch =
          getIOControl<Fortran::parser::IoControlSpec::Asynchronous>(stmt)) {
    // FIXME: should contain a string of YES or NO
    TODO_NOLOC("asynchronous transfers not implemented in runtime");
  }
  return false;
}
template <>
constexpr bool isDataTransferAsynchronous<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return false;
}

template <typename A>
static bool isDataTransferNamelist(const A &stmt) {
  if (stmt.format)
    return formatIsActuallyNamelist(*stmt.format);
  return hasIOControl<Fortran::parser::Name>(stmt);
}
template <>
constexpr bool isDataTransferNamelist<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return false;
}

/// Lowers a format statment that uses an assigned variable label reference as
/// a select operation to allow for run-time selection of the format statement.
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
lowerReferenceAsStringSelect(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr,
    mlir::Type strTy, mlir::Type lenTy,
    Fortran::lower::StatementContext &stmtCtx) {
  // Possible optimization TODO: Instead of inlining a selectOp every time there
  // is a variable reference to a format statement, a function with the selectOp
  // could be generated to reduce code size. It is not clear if such an
  // optimization would be deployed very often or improve the object code
  // beyond, say, what GVN/GCM might produce.

  // Create the requisite blocks to inline a selectOp.
  auto &builder = converter.getFirOpBuilder();
  auto *startBlock = builder.getBlock();
  auto *endBlock = startBlock->splitBlock(builder.getInsertionPoint());
  auto *block = startBlock->splitBlock(builder.getInsertionPoint());
  builder.setInsertionPointToEnd(block);

  llvm::SmallVector<int64_t> indexList;
  llvm::SmallVector<mlir::Block *> blockList;

  auto symbol = GetLastSymbol(&expr);
  Fortran::lower::pft::LabelSet labels;
  [[maybe_unused]] auto foundLabelSet =
      converter.lookupLabelSet(*symbol, labels);
  assert(foundLabelSet && "Label not found in map");

  for (auto label : labels) {
    indexList.push_back(label);
    auto *eval = converter.lookupLabel(label);
    assert(eval && "Label is missing from the table");

    auto text = toStringRef(eval->position);
    mlir::Value stringRef;
    mlir::Value stringLen;
    if (eval->isA<Fortran::parser::FormatStmt>()) {
      assert(text.find('(') != llvm::StringRef::npos &&
             "FORMAT is unexpectedly ill-formed");
      // This is a format statement, so extract the spec from the text.
      auto stringLit =
          lowerSourceTextAsStringLit(converter, loc, text, strTy, lenTy);
      stringRef = std::get<0>(stringLit);
      stringLen = std::get<1>(stringLit);
    } else {
      // This is not a format statement, so use null.
      stringRef = builder.createConvert(
          loc, strTy,
          builder.createIntegerConstant(loc, builder.getIndexType(), 0));
      stringLen = builder.createIntegerConstant(loc, lenTy, 0);
    }

    // Pass the format string reference and the string length out of the select
    // statement.
    llvm::SmallVector<mlir::Value> args = {stringRef, stringLen};
    builder.create<mlir::BranchOp>(loc, endBlock, args);

    // Add block to the list of cases and make a new one.
    blockList.push_back(block);
    block = block->splitBlock(builder.getInsertionPoint());
    builder.setInsertionPointToEnd(block);
  }

  // Create the unit case which should result in an error.
  auto *unitBlock = block->splitBlock(builder.getInsertionPoint());
  builder.setInsertionPointToEnd(unitBlock);

  // Crash the program.
  builder.create<fir::UnreachableOp>(loc);

  // Add unit case to the select statement.
  blockList.push_back(unitBlock);

  // Lower the selectOp.
  builder.setInsertionPointToEnd(startBlock);
  auto label = fir::getBase(converter.genExprValue(&expr, stmtCtx, loc));
  builder.create<fir::SelectOp>(loc, label, indexList, blockList);

  builder.setInsertionPointToEnd(endBlock);
  endBlock->addArgument(strTy);
  endBlock->addArgument(lenTy);

  // Handle and return the string reference and length selected by the selectOp.
  auto buff = endBlock->getArgument(0);
  auto len = endBlock->getArgument(1);

  return {buff, len, mlir::Value{}};
}

/// Generate a reference to a format string.  There are four cases - a format
/// statement label, a character format expression, an integer that holds the
/// label of a format statement, and the * case.  The first three are done here.
/// The * case is done elsewhere.
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
genFormat(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const Fortran::parser::Format &format, mlir::Type strTy,
          mlir::Type lenTy, Fortran::lower::StatementContext &stmtCtx) {
  if (const auto *label = std::get_if<Fortran::parser::Label>(&format.u)) {
    // format statement label
    auto eval = converter.lookupLabel(*label);
    assert(eval && "FORMAT not found in PROCEDURE");
    return lowerSourceTextAsStringLit(
        converter, loc, toStringRef(eval->position), strTy, lenTy);
  }
  const auto *pExpr = std::get_if<Fortran::parser::Expr>(&format.u);
  assert(pExpr && "missing format expression");
  auto e = Fortran::semantics::GetExpr(*pExpr);
  if (Fortran::semantics::ExprHasTypeCategory(
          *e, Fortran::common::TypeCategory::Character))
    // character expression
    return lowerStringLit(converter, loc, *pExpr, strTy, lenTy);

  // integer variable containing an ASSIGN label
  assert(Fortran::semantics::ExprHasTypeCategory(
      *e, Fortran::common::TypeCategory::Integer));
  return lowerReferenceAsStringSelect(converter, loc, *e, strTy, lenTy,
                                      stmtCtx);
}

template <typename A>
std::tuple<mlir::Value, mlir::Value, mlir::Value>
getFormat(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const A &stmt, mlir::Type strTy, mlir::Type lenTy,
          Fortran ::lower::StatementContext &stmtCtx) {
  if (stmt.format && !formatIsActuallyNamelist(*stmt.format))
    return genFormat(converter, loc, *stmt.format, strTy, lenTy, stmtCtx);
  return genFormat(converter, loc, *getIOControl<Fortran::parser::Format>(stmt),
                   strTy, lenTy, stmtCtx);
}
template <>
std::tuple<mlir::Value, mlir::Value, mlir::Value>
getFormat<Fortran::parser::PrintStmt>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::parser::PrintStmt &stmt, mlir::Type strTy, mlir::Type lenTy,
    Fortran::lower::StatementContext &stmtCtx) {
  return genFormat(converter, loc, std::get<Fortran::parser::Format>(stmt.t),
                   strTy, lenTy, stmtCtx);
}

/// Get a buffer for an internal file data transfer.
template <typename A>
std::tuple<mlir::Value, mlir::Value>
getBuffer(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const A &stmt, mlir::Type strTy, mlir::Type lenTy,
          Fortran::lower::StatementContext &stmtCtx) {
  const Fortran::parser::IoUnit *iounit =
      stmt.iounit ? &*stmt.iounit : getIOControl<Fortran::parser::IoUnit>(stmt);
  if (iounit)
    if (auto *var = std::get_if<Fortran::parser::Variable>(&iounit->u))
      if (auto *expr = Fortran::semantics::GetExpr(*var))
        return genBuffer(converter, loc, *expr, strTy, lenTy, stmtCtx);
  llvm::report_fatal_error("failed to get IoUnit expr in lowering");
}

static mlir::Value genIOUnit(Fortran::lower::AbstractConverter &converter,
                             mlir::Location loc,
                             const Fortran::parser::IoUnit &iounit,
                             mlir::Type ty,
                             Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  if (auto *e = std::get_if<Fortran::parser::FileUnitNumber>(&iounit.u)) {
    auto ex = fir::getBase(
        converter.genExprValue(Fortran::semantics::GetExpr(*e), stmtCtx, loc));
    return builder.createConvert(loc, ty, ex);
  }
  return builder.create<mlir::ConstantOp>(
      loc, builder.getIntegerAttr(ty, Fortran::runtime::io::DefaultUnit));
}

template <typename A>
mlir::Value getIOUnit(Fortran::lower::AbstractConverter &converter,
                      mlir::Location loc, const A &stmt, mlir::Type ty,
                      Fortran::lower::StatementContext &stmtCtx) {
  if (stmt.iounit)
    return genIOUnit(converter, loc, *stmt.iounit, ty, stmtCtx);
  if (auto *iounit = getIOControl<Fortran::parser::IoUnit>(stmt))
    return genIOUnit(converter, loc, *iounit, ty, stmtCtx);
  auto &builder = converter.getFirOpBuilder();
  return builder.create<mlir::ConstantOp>(
      loc, builder.getIntegerAttr(ty, Fortran::runtime::io::DefaultUnit));
}

//===----------------------------------------------------------------------===//
// Generators for each I/O statement type.
//===----------------------------------------------------------------------===//

template <typename K, typename S>
static mlir::Value genBasicIOStmt(Fortran::lower::AbstractConverter &converter,
                                  const S &stmt) {
  auto &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  auto loc = converter.getCurrentLocation();
  auto beginFunc = getIORuntimeFunc<K>(loc, builder);
  mlir::FunctionType beginFuncTy = beginFunc.getType();
  auto unit = fir::getBase(converter.genExprValue(
      getExpr<Fortran::parser::FileUnitNumber>(stmt), stmtCtx, loc));
  auto un = builder.createConvert(loc, beginFuncTy.getInput(0), unit);
  auto file = locToFilename(converter, loc, beginFuncTy.getInput(1));
  auto line = locToLineNo(converter, loc, beginFuncTy.getInput(2));
  auto call = builder.create<fir::CallOp>(loc, beginFunc,
                                          mlir::ValueRange{un, file, line});
  auto cookie = call.getResult(0);
  ConditionSpecInfo csi;
  genConditionHandlerCall(converter, loc, cookie, stmt.v, csi);
  mlir::Value ok;
  auto insertPt = builder.saveInsertionPoint();
  threadSpecs(converter, loc, cookie, stmt.v, csi.hasErrorConditionSpec(), ok,
              stmtCtx);
  builder.restoreInsertionPoint(insertPt);
  return genEndIO(converter, converter.getCurrentLocation(), cookie, csi,
                  stmtCtx);
}

mlir::Value Fortran::lower::genBackspaceStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::BackspaceStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginBackspace)>(converter, stmt);
}

mlir::Value Fortran::lower::genEndfileStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EndfileStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginEndfile)>(converter, stmt);
}

mlir::Value
Fortran::lower::genFlushStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::FlushStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginFlush)>(converter, stmt);
}

mlir::Value
Fortran::lower::genRewindStatement(Fortran::lower::AbstractConverter &converter,
                                   const Fortran::parser::RewindStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginRewind)>(converter, stmt);
}

mlir::Value
Fortran::lower::genOpenStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::OpenStmt &stmt) {
  auto &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  mlir::FuncOp beginFunc;
  llvm::SmallVector<mlir::Value> beginArgs;
  auto loc = converter.getCurrentLocation();
  if (hasMem<Fortran::parser::FileUnitNumber>(stmt)) {
    beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenUnit)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    auto unit = fir::getBase(converter.genExprValue(
        getExpr<Fortran::parser::FileUnitNumber>(stmt), stmtCtx, loc));
    beginArgs.push_back(
        builder.createConvert(loc, beginFuncTy.getInput(0), unit));
    beginArgs.push_back(locToFilename(converter, loc, beginFuncTy.getInput(1)));
    beginArgs.push_back(locToLineNo(converter, loc, beginFuncTy.getInput(2)));
  } else {
    assert(hasMem<Fortran::parser::ConnectSpec::Newunit>(stmt));
    beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenNewUnit)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    beginArgs.push_back(locToFilename(converter, loc, beginFuncTy.getInput(0)));
    beginArgs.push_back(locToLineNo(converter, loc, beginFuncTy.getInput(1)));
  }
  auto cookie =
      builder.create<fir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
  ConditionSpecInfo csi;
  genConditionHandlerCall(converter, loc, cookie, stmt.v, csi);
  mlir::Value ok;
  auto insertPt = builder.saveInsertionPoint();
  threadSpecs(converter, loc, cookie, stmt.v, csi.hasErrorConditionSpec(), ok,
              stmtCtx);
  builder.restoreInsertionPoint(insertPt);
  return genEndIO(converter, loc, cookie, csi, stmtCtx);
}

mlir::Value
Fortran::lower::genCloseStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::CloseStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginClose)>(converter, stmt);
}

mlir::Value
Fortran::lower::genWaitStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::WaitStmt &stmt) {
  auto &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  auto loc = converter.getCurrentLocation();
  bool hasId = hasMem<Fortran::parser::IdExpr>(stmt);
  mlir::FuncOp beginFunc =
      hasId ? getIORuntimeFunc<mkIOKey(BeginWait)>(loc, builder)
            : getIORuntimeFunc<mkIOKey(BeginWaitAll)>(loc, builder);
  mlir::FunctionType beginFuncTy = beginFunc.getType();
  auto unit = fir::getBase(converter.genExprValue(
      getExpr<Fortran::parser::FileUnitNumber>(stmt), stmtCtx, loc));
  auto un = builder.createConvert(loc, beginFuncTy.getInput(0), unit);
  llvm::SmallVector<mlir::Value> args{un};
  if (hasId) {
    auto id = fir::getBase(converter.genExprValue(
        getExpr<Fortran::parser::IdExpr>(stmt), stmtCtx, loc));
    args.push_back(builder.createConvert(loc, beginFuncTy.getInput(1), id));
  }
  auto cookie = builder.create<fir::CallOp>(loc, beginFunc, args).getResult(0);
  ConditionSpecInfo csi;
  genConditionHandlerCall(converter, loc, cookie, stmt.v, csi);
  return genEndIO(converter, converter.getCurrentLocation(), cookie, csi,
                  stmtCtx);
}

//===----------------------------------------------------------------------===//
// Data transfer statements.
//
// There are several dimensions to the API with regard to data transfer
// statements that need to be considered.
//
//   - input (READ) vs. output (WRITE, PRINT)
//   - formatted vs. list vs. unformatted vs. namelist
//   - synchronous vs. asynchronous
//   - external vs. internal + default KIND vs. internal + other KIND
//===----------------------------------------------------------------------===//

// Determine the correct BeginXyz{In|Out}put api to invoke.
template <bool isInput>
mlir::FuncOp
getBeginDataTransfer(mlir::Location loc, Fortran::lower::FirOpBuilder &builder,
                     bool isFormatted, bool isList, bool isIntern,
                     bool isOtherIntern, bool isAsynch, bool isNml) {
  if constexpr (isInput) {
    if (isAsynch)
      return getIORuntimeFunc<mkIOKey(BeginAsynchronousInput)>(loc, builder);
    if (isFormatted) {
      if (isIntern) {
        if (isOtherIntern) {
          if (isList || isNml)
            return getIORuntimeFunc<mkIOKey(BeginInternalArrayListInput)>(
                loc, builder);
          return getIORuntimeFunc<mkIOKey(BeginInternalArrayFormattedInput)>(
              loc, builder);
        }
        if (isList || isNml)
          return getIORuntimeFunc<mkIOKey(BeginInternalListInput)>(loc,
                                                                   builder);
        return getIORuntimeFunc<mkIOKey(BeginInternalFormattedInput)>(loc,
                                                                      builder);
      }
      if (isList || isNml)
        return getIORuntimeFunc<mkIOKey(BeginExternalListInput)>(loc, builder);
      return getIORuntimeFunc<mkIOKey(BeginExternalFormattedInput)>(loc,
                                                                    builder);
    }
    return getIORuntimeFunc<mkIOKey(BeginUnformattedInput)>(loc, builder);
  } else {
    if (isAsynch)
      return getIORuntimeFunc<mkIOKey(BeginAsynchronousOutput)>(loc, builder);
    if (isFormatted) {
      if (isIntern) {
        if (isOtherIntern) {
          if (isList || isNml)
            return getIORuntimeFunc<mkIOKey(BeginInternalArrayListOutput)>(
                loc, builder);
          return getIORuntimeFunc<mkIOKey(BeginInternalArrayFormattedOutput)>(
              loc, builder);
        }
        if (isList || isNml)
          return getIORuntimeFunc<mkIOKey(BeginInternalListOutput)>(loc,
                                                                    builder);
        return getIORuntimeFunc<mkIOKey(BeginInternalFormattedOutput)>(loc,
                                                                       builder);
      }
      if (isList || isNml)
        return getIORuntimeFunc<mkIOKey(BeginExternalListOutput)>(loc, builder);
      return getIORuntimeFunc<mkIOKey(BeginExternalFormattedOutput)>(loc,
                                                                     builder);
    }
    return getIORuntimeFunc<mkIOKey(BeginUnformattedOutput)>(loc, builder);
  }
}

/// Generate the arguments of a BeginXyz call.
template <bool hasIOCtrl, typename A>
void genBeginCallArguments(llvm::SmallVectorImpl<mlir::Value> &ioArgs,
                           Fortran::lower::AbstractConverter &converter,
                           mlir::Location loc, const A &stmt,
                           mlir::FunctionType ioFuncTy, bool isFormatted,
                           bool isList, bool isIntern, bool isOtherIntern,
                           bool isAsynch, bool isNml,
                           const llvm::Optional<fir::ExtendedValue> &descRef,
                           Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  if constexpr (hasIOCtrl) {
    // READ/WRITE cases have a wide variety of argument permutations
    if (isAsynch || !isFormatted) {
      // unit (always first), ...
      ioArgs.push_back(getIOUnit(converter, loc, stmt,
                                 ioFuncTy.getInput(ioArgs.size()), stmtCtx));
      if (isAsynch) {
        // unknown-thingy, [buff, LEN]
        TODO(loc, "asynchrous");
      }
      return;
    }
    assert(isFormatted && "formatted data transfer");
    if (!isIntern) {
      if (isNml) {
        // namelist group, ...
        TODO(loc, "namelist");
      } else if (!isList) {
        // | [format, LEN], ...
        auto pair =
            getFormat(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                      ioFuncTy.getInput(ioArgs.size() + 1), stmtCtx);
        ioArgs.push_back(std::get<0>(pair));
        ioArgs.push_back(std::get<1>(pair));
      }
      // unit (always last)
      ioArgs.push_back(getIOUnit(converter, loc, stmt,
                                 ioFuncTy.getInput(ioArgs.size()), stmtCtx));
      return;
    }
    assert(isIntern && "internal data transfer");
    if (isNml || isOtherIntern) {
      // descriptor, ...
      assert(!isNml && "namelist is not implemented");
      assert(descRef.hasValue() && "descriptor value required");
      auto desc = builder.createBox(loc, *descRef);
      ioArgs.push_back(
          builder.createConvert(loc, ioFuncTy.getInput(ioArgs.size()), desc));
      if (isNml) {
        // namelist group, ...
        TODO(loc, "namelist");
      } else if (isOtherIntern && !isList) {
        // | [format, LEN], ...
        auto pair =
            getFormat(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                      ioFuncTy.getInput(ioArgs.size() + 1), stmtCtx);
        ioArgs.push_back(std::get<0>(pair));
        ioArgs.push_back(std::get<1>(pair));
      }
    } else {
      // | [buff, LEN], ...
      auto pair =
          getBuffer(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                    ioFuncTy.getInput(ioArgs.size() + 1), stmtCtx);
      ioArgs.push_back(std::get<0>(pair));
      ioArgs.push_back(std::get<1>(pair));
      if (!isList) {
        // [format, LEN], ...
        auto pair =
            getFormat(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                      ioFuncTy.getInput(ioArgs.size() + 1), stmtCtx);
        ioArgs.push_back(std::get<0>(pair));
        ioArgs.push_back(std::get<1>(pair));
      }
    }
    // [scratch, LEN] (always last)
    ioArgs.push_back(
        getDefaultScratch(builder, loc, ioFuncTy.getInput(ioArgs.size())));
    ioArgs.push_back(
        getDefaultScratchLen(builder, loc, ioFuncTy.getInput(ioArgs.size())));
  } else {
    if (!isList) {
      // [format, LEN], ...
      auto pair =
          getFormat(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                    ioFuncTy.getInput(ioArgs.size() + 1), stmtCtx);
      ioArgs.push_back(std::get<0>(pair));
      ioArgs.push_back(std::get<1>(pair));
    }
    // unit (always last)
    ioArgs.push_back(builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(ioArgs.size()),
                                    Fortran::runtime::io::DefaultUnit)));
  }
}

template <bool isInput, bool hasIOCtrl = true, typename A>
static mlir::Value
genDataTransferStmt(Fortran::lower::AbstractConverter &converter,
                    const A &stmt) {
  auto &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  auto loc = converter.getCurrentLocation();
  const bool isFormatted = isDataTransferFormatted(stmt);
  const bool isList = isFormatted ? isDataTransferList(stmt) : false;
  const bool isIntern = isDataTransferInternal(stmt);
  llvm::Optional<fir::ExtendedValue> descRef =
      isIntern ? getIfDataTransferInternalRequiresDescriptor(converter, stmt,
                                                             stmtCtx)
               : llvm::None;
  const bool isOtherIntern = descRef.hasValue();
  const bool isAsynch = isDataTransferAsynchronous(stmt);
  const bool isNml = isDataTransferNamelist(stmt);

  // Determine which BeginXyz call to make.
  mlir::FuncOp ioFunc =
      getBeginDataTransfer<isInput>(loc, builder, isFormatted, isList, isIntern,
                                    isOtherIntern, isAsynch, isNml);
  mlir::FunctionType ioFuncTy = ioFunc.getType();

  // Append BeginXyz call arguments.  File name and line number are always last.
  llvm::SmallVector<mlir::Value> ioArgs;
  genBeginCallArguments<hasIOCtrl>(ioArgs, converter, loc, stmt, ioFuncTy,
                                   isFormatted, isList, isIntern, isOtherIntern,
                                   isAsynch, isNml, descRef, stmtCtx);
  ioArgs.push_back(
      locToFilename(converter, loc, ioFuncTy.getInput(ioArgs.size())));
  ioArgs.push_back(
      locToLineNo(converter, loc, ioFuncTy.getInput(ioArgs.size())));

  // Arguments are done; call the BeginXyz function.
  mlir::Value cookie =
      builder.create<fir::CallOp>(loc, ioFunc, ioArgs).getResult(0);

  // Generate an EnableHandlers call and remaining specifier calls.
  ConditionSpecInfo csi;
  auto insertPt = builder.saveInsertionPoint();
  mlir::Value ok;
  if constexpr (hasIOCtrl) {
    genConditionHandlerCall(converter, loc, cookie, stmt.controls, csi);
    threadSpecs(converter, loc, cookie, stmt.controls,
                csi.hasErrorConditionSpec(), ok, stmtCtx);
  }

  // Generate data transfer list calls.
  if constexpr (isInput) // ReadStmt
    genInputItemList(converter, cookie, stmt.items, isFormatted,
                     csi.hasTransferConditionSpec(), ok, /*inLoop=*/false,
                     stmtCtx);
  else if constexpr (std::is_same_v<A, Fortran::parser::PrintStmt>)
    genOutputItemList(converter, cookie, std::get<1>(stmt.t), isFormatted,
                      csi.hasTransferConditionSpec(), ok,
                      /*inLoop=*/false, stmtCtx);
  else // WriteStmt
    genOutputItemList(converter, cookie, stmt.items, isFormatted,
                      csi.hasTransferConditionSpec(), ok,
                      /*inLoop=*/false, stmtCtx);
  stmtCtx.finalize();

  // Generate end statement call/s.
  builder.restoreInsertionPoint(insertPt);
  return genEndIO(converter, loc, cookie, csi, stmtCtx);
}

void Fortran::lower::genPrintStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::PrintStmt &stmt) {
  // PRINT does not take an io-control-spec. It only has a format specifier, so
  // it is a simplified case of WRITE.
  genDataTransferStmt</*isInput=*/false, /*ioCtrl=*/false>(converter, stmt);
}

mlir::Value
Fortran::lower::genWriteStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::WriteStmt &stmt) {
  return genDataTransferStmt</*isInput=*/false>(converter, stmt);
}

mlir::Value
Fortran::lower::genReadStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::ReadStmt &stmt) {
  return genDataTransferStmt</*isInput=*/true>(converter, stmt);
}

/// Get the file expression from the inquire spec list. Also return if the
/// expression is a file name.
static std::pair<const Fortran::semantics::SomeExpr *, bool>
getInquireFileExpr(const std::list<Fortran::parser::InquireSpec> *stmt) {
  if (!stmt)
    return {nullptr, /*filename?=*/false};
  for (const auto &spec : *stmt) {
    if (auto *f = std::get_if<Fortran::parser::FileUnitNumber>(&spec.u))
      return {Fortran::semantics::GetExpr(*f), /*filename?=*/false};
    if (auto *f = std::get_if<Fortran::parser::FileNameExpr>(&spec.u))
      return {Fortran::semantics::GetExpr(*f), /*filename?=*/true};
  }
  // semantics should have already caught this condition
  llvm::report_fatal_error("inquire spec must have a file");
}

/// Generate calls to the four distinct INQUIRE subhandlers. An INQUIRE may
/// return values of type CHARACTER, INTEGER, or LOGICAL. There is one
/// additional special case for INQUIRE with both PENDING and ID specifiers.
template <typename A>
static mlir::Value genInquireSpec(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc, mlir::Value cookie,
                                  mlir::Value idExpr, const A &var,
                                  Fortran::lower::StatementContext &stmtCtx) {
  // default case: do nothing
  return {};
}
/// Specialization for CHARACTER.
template <>
mlir::Value genInquireSpec<Fortran::parser::InquireSpec::CharVar>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, mlir::Value idExpr,
    const Fortran::parser::InquireSpec::CharVar &var,
    Fortran::lower::StatementContext &stmtCtx) {
  // IOMSG is handled with exception conditions
  if (std::get<Fortran::parser::InquireSpec::CharVar::Kind>(var.t) ==
      Fortran::parser::InquireSpec::CharVar::Kind::Iomsg)
    return {};
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp specFunc =
      getIORuntimeFunc<mkIOKey(InquireCharacter)>(loc, builder);
  auto specFuncTy = specFunc.getType();
  const auto *varExpr = Fortran::semantics::GetExpr(
      std::get<Fortran::parser::ScalarDefaultCharVariable>(var.t));
  auto str = converter.genExprAddr(varExpr, stmtCtx, loc);
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, specFuncTy.getInput(0), cookie),
      builder.createIntegerConstant(
          loc, specFuncTy.getInput(1),
          Fortran::runtime::io::HashInquiryKeyword(
              Fortran::parser::InquireSpec::CharVar::EnumToString(
                  std::get<Fortran::parser::InquireSpec::CharVar::Kind>(var.t))
                  .c_str())),
      builder.createConvert(loc, specFuncTy.getInput(2), fir::getBase(str)),
      builder.createConvert(loc, specFuncTy.getInput(3), fir::getLen(str))};
  return builder.create<fir::CallOp>(loc, specFunc, args).getResult(0);
}
/// Specialization for INTEGER.
template <>
mlir::Value genInquireSpec<Fortran::parser::InquireSpec::IntVar>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, mlir::Value idExpr,
    const Fortran::parser::InquireSpec::IntVar &var,
    Fortran::lower::StatementContext &stmtCtx) {
  // IOSTAT is handled with exception conditions
  if (std::get<Fortran::parser::InquireSpec::IntVar::Kind>(var.t) ==
      Fortran::parser::InquireSpec::IntVar::Kind::Iostat)
    return {};
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp specFunc =
      getIORuntimeFunc<mkIOKey(InquireInteger64)>(loc, builder);
  auto specFuncTy = specFunc.getType();
  const auto *varExpr = Fortran::semantics::GetExpr(
      std::get<Fortran::parser::ScalarIntVariable>(var.t));
  auto addr = fir::getBase(converter.genExprAddr(varExpr, stmtCtx, loc));
  auto eleTy = fir::dyn_cast_ptrEleTy(addr.getType());
  if (!eleTy)
    fir::emitFatalError(loc,
                        "internal error: expected a memory reference type");
  auto bitWidth = eleTy.cast<mlir::IntegerType>().getWidth();
  auto idxTy = builder.getIndexType();
  auto kind = builder.createIntegerConstant(loc, idxTy, bitWidth / 8);
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, specFuncTy.getInput(0), cookie),
      builder.createIntegerConstant(
          loc, specFuncTy.getInput(1),
          Fortran::runtime::io::HashInquiryKeyword(
              Fortran::parser::InquireSpec::IntVar::EnumToString(
                  std::get<Fortran::parser::InquireSpec::IntVar::Kind>(var.t))
                  .c_str())),
      builder.createConvert(loc, specFuncTy.getInput(2), addr),
      builder.createConvert(loc, specFuncTy.getInput(3), kind)};
  return builder.create<fir::CallOp>(loc, specFunc, args).getResult(0);
}
/// Specialization for LOGICAL and (PENDING + ID).
template <>
mlir::Value genInquireSpec<Fortran::parser::InquireSpec::LogVar>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, mlir::Value idExpr,
    const Fortran::parser::InquireSpec::LogVar &var,
    Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  auto logVarKind = std::get<Fortran::parser::InquireSpec::LogVar::Kind>(var.t);
  bool pendId =
      idExpr &&
      logVarKind == Fortran::parser::InquireSpec::LogVar::Kind::Pending;
  mlir::FuncOp specFunc =
      pendId ? getIORuntimeFunc<mkIOKey(InquirePendingId)>(loc, builder)
             : getIORuntimeFunc<mkIOKey(InquireLogical)>(loc, builder);
  auto specFuncTy = specFunc.getType();
  auto addr = fir::getBase(converter.genExprAddr(
      Fortran::semantics::GetExpr(
          std::get<Fortran::parser::Scalar<
              Fortran::parser::Logical<Fortran::parser::Variable>>>(var.t)),
      stmtCtx, loc));
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, specFuncTy.getInput(0), cookie)};
  if (pendId)
    args.push_back(builder.createConvert(loc, specFuncTy.getInput(1), idExpr));
  else
    args.push_back(builder.createIntegerConstant(
        loc, specFuncTy.getInput(1),
        Fortran::runtime::io::HashInquiryKeyword(
            Fortran::parser::InquireSpec::LogVar::EnumToString(logVarKind)
                .c_str())));
  args.push_back(builder.createConvert(loc, specFuncTy.getInput(2), addr));
  return builder.create<fir::CallOp>(loc, specFunc, args).getResult(0);
}

/// If there is an IdExpr in the list of inquire-specs, then lower it and return
/// the resulting Value. Otherwise, return null.
static mlir::Value
lowerIdExpr(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
            const std::list<Fortran::parser::InquireSpec> &ispecs,
            Fortran::lower::StatementContext &stmtCtx) {
  for (const auto &spec : ispecs)
    if (mlir::Value v = std::visit(
            Fortran::common::visitors{
                [&](const Fortran::parser::IdExpr &idExpr) {
                  return fir::getBase(converter.genExprValue(
                      Fortran::semantics::GetExpr(idExpr), stmtCtx, loc));
                },
                [](const auto &) { return mlir::Value{}; }},
            spec.u))
      return v;
  return {};
}

/// For each inquire-spec, build the appropriate call, threading the cookie.
static void threadInquire(Fortran::lower::AbstractConverter &converter,
                          mlir::Location loc, mlir::Value cookie,
                          const std::list<Fortran::parser::InquireSpec> &ispecs,
                          bool checkResult, mlir::Value &ok,
                          Fortran::lower::StatementContext &stmtCtx) {
  auto &builder = converter.getFirOpBuilder();
  mlir::Value idExpr = lowerIdExpr(converter, loc, ispecs, stmtCtx);
  for (const auto &spec : ispecs) {
    makeNextConditionalOn(builder, loc, checkResult, ok);
    ok = std::visit(Fortran::common::visitors{[&](const auto &x) {
                      return genInquireSpec(converter, loc, cookie, idExpr, x,
                                            stmtCtx);
                    }},
                    spec.u);
  }
}

mlir::Value Fortran::lower::genInquireStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::InquireStmt &stmt) {
  auto &builder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  auto loc = converter.getCurrentLocation();
  mlir::FuncOp beginFunc;
  ConditionSpecInfo csi;
  llvm::SmallVector<mlir::Value> beginArgs;
  const auto *list =
      std::get_if<std::list<Fortran::parser::InquireSpec>>(&stmt.u);
  auto exprPair = getInquireFileExpr(list);
  auto inquireFileUnit = [&]() -> bool {
    return exprPair.first && !exprPair.second;
  };
  auto inquireFileName = [&]() -> bool {
    return exprPair.first && exprPair.second;
  };

  // Determine which BeginInquire call to make.
  if (inquireFileUnit()) {
    // File unit call.
    beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireUnit)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    beginArgs = {builder.createConvert(loc, beginFuncTy.getInput(0),
                                       fir::getBase(converter.genExprValue(
                                           exprPair.first, stmtCtx, loc))),
                 locToFilename(converter, loc, beginFuncTy.getInput(1)),
                 locToLineNo(converter, loc, beginFuncTy.getInput(2))};
  } else if (inquireFileName()) {
    // Filename call.
    beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireFile)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    auto file = converter.genExprAddr(exprPair.first, stmtCtx, loc);
    beginArgs = {
        builder.createConvert(loc, beginFuncTy.getInput(0), fir::getBase(file)),
        builder.createConvert(loc, beginFuncTy.getInput(1), fir::getLen(file)),
        locToFilename(converter, loc, beginFuncTy.getInput(2)),
        locToLineNo(converter, loc, beginFuncTy.getInput(3))};
  } else {
    // INQUIRE IOLENGTH call.
    const auto *ioLength =
        std::get_if<Fortran::parser::InquireStmt::Iolength>(&stmt.u);
    assert(ioLength && "must have an io length");
    beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireIoLength)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    beginArgs = {locToFilename(converter, loc, beginFuncTy.getInput(0)),
                 locToLineNo(converter, loc, beginFuncTy.getInput(1))};
    // The IOLENGTH call is irregular enough to generate immediately here.
    auto cookie =
        builder.create<fir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
    genConditionHandlerCall(
        converter, loc, cookie,
        std::get<std::list<Fortran::parser::OutputItem>>(ioLength->t), csi);
    return genEndIO(converter, loc, cookie, csi, stmtCtx);
  }

  // Common handling for file {unit|name} cases.
  assert(list && "inquire-spec list must be present");
  auto cookie =
      builder.create<fir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
  genConditionHandlerCall(converter, loc, cookie, *list, csi);
  // Handle remaining arguments in specifier list.
  mlir::Value ok;
  auto insertPt = builder.saveInsertionPoint();
  threadInquire(converter, loc, cookie, *list, csi.hasErrorConditionSpec(), ok,
                stmtCtx);
  builder.restoreInsertionPoint(insertPt);
  // Generate end statement call.
  return genEndIO(converter, loc, cookie, csi, stmtCtx);
}
