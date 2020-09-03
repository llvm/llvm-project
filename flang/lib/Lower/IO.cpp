//===-- IO.cpp -- I/O statement lowering ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/IO.h"
#include "../../runtime/io-api.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/ComplexExpr.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Utils.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#define TODO() llvm_unreachable("not yet implemented")

using namespace Fortran::runtime::io;

#define NAMIFY_HELPER(X) #X
#define NAMIFY(X) NAMIFY_HELPER(IONAME(X))
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
    mkIOKey(BeginInternalFormattedInput), mkIOKey(BeginInternalNamelistOutput),
    mkIOKey(BeginInternalNamelistInput), mkIOKey(BeginExternalListOutput),
    mkIOKey(BeginExternalListInput), mkIOKey(BeginExternalFormattedOutput),
    mkIOKey(BeginExternalFormattedInput), mkIOKey(BeginUnformattedOutput),
    mkIOKey(BeginUnformattedInput), mkIOKey(BeginExternalNamelistOutput),
    mkIOKey(BeginExternalNamelistInput), mkIOKey(BeginAsynchronousOutput),
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
    mkIOKey(OutputComplex64), mkIOKey(OutputComplex32), mkIOKey(OutputAscii),
    mkIOKey(InputAscii), mkIOKey(OutputLogical), mkIOKey(InputLogical),
    mkIOKey(SetAccess), mkIOKey(SetAction), mkIOKey(SetAsynchronous),
    mkIOKey(SetEncoding), mkIOKey(SetForm), mkIOKey(SetPosition),
    mkIOKey(SetRecl), mkIOKey(SetStatus), mkIOKey(SetFile), mkIOKey(GetNewUnit),
    mkIOKey(GetSize), mkIOKey(GetIoLength), mkIOKey(GetIoMsg),
    mkIOKey(InquireCharacter), mkIOKey(InquireLogical),
    mkIOKey(InquirePendingId), mkIOKey(InquireInteger64),
    mkIOKey(EndIoStatement)>
    newIOTable;
} // namespace Fortran::lower

namespace {
struct ConditionSpecifierInfo {
  const Fortran::semantics::SomeExpr *ioStatExpr{};
  const Fortran::semantics::SomeExpr *ioMsgExpr{};
  bool hasErr{};
  bool hasEnd{};
  bool hasEor{};

  /// Check for any condition specifier that applies to specifier processing.
  bool hasErrorConditionSpecifier() const {
    return ioStatExpr != nullptr || hasErr;
  }
  /// Check for any condition specifier that applies to data transfer items
  /// in a PRINT, READ, WRITE, or WAIT statement.  (WAIT may be irrelevant.)
  bool hasTransferConditionSpecifier() const {
    return ioStatExpr != nullptr || hasErr || hasEnd || hasEor;
  }
  /// Check for any condition specifier, including IOMSG.
  bool hasAnyConditionSpecifier() const {
    return ioStatExpr != nullptr || ioMsgExpr != nullptr || hasErr || hasEnd ||
           hasEor;
  }
};
} // namespace

using namespace Fortran::lower;

/// Helper function to retrieve the name of the IO function given the key `A`
template <typename A>
static constexpr const char *getName() {
  return std::get<A>(newIOTable).name;
}

/// Helper function to retrieve the type model signature builder of the IO
/// function as defined by the key `A`
template <typename A>
static constexpr FuncTypeBuilderFunc getTypeModel() {
  return std::get<A>(newIOTable).getTypeModel();
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
  func.setAttr("fir.runtime", builder.getUnitAttr());
  func.setAttr("fir.io", builder.getUnitAttr());
  return func;
}

/// Generate calls to end an IO statement.  Return the IOSTAT value, if any.
/// It is the caller's responsibility to generate branches on that value.
static mlir::Value genEndIO(Fortran::lower::AbstractConverter &converter,
                            mlir::Location loc, mlir::Value cookie,
                            const ConditionSpecifierInfo &csi) {
  auto &builder = converter.getFirOpBuilder();
  if (csi.ioMsgExpr) {
    auto getIoMsg = getIORuntimeFunc<mkIOKey(GetIoMsg)>(loc, builder);
    auto ioMsgVar =
        Fortran::lower::CharacterExprHelper{builder, loc}.createUnboxChar(
            converter.genExprAddr(csi.ioMsgExpr, loc));
    llvm::SmallVector<mlir::Value, 3> args{
        cookie,
        builder.createConvert(loc, getIoMsg.getType().getInput(1),
                              ioMsgVar.first),
        builder.createConvert(loc, getIoMsg.getType().getInput(2),
                              ioMsgVar.second)};
    builder.create<mlir::CallOp>(loc, getIoMsg, args);
  }
  auto endIoStatement = getIORuntimeFunc<mkIOKey(EndIoStatement)>(loc, builder);
  llvm::SmallVector<mlir::Value, 1> endArgs{cookie};
  auto call = builder.create<mlir::CallOp>(loc, endIoStatement, endArgs);
  if (csi.ioStatExpr) {
    auto ioStatVar = converter.genExprAddr(csi.ioStatExpr, loc);
    auto ioStatResult = builder.createConvert(
        loc, converter.genType(*csi.ioStatExpr), call.getResult(0));
    builder.create<fir::StoreOp>(loc, ioStatResult, ioStatVar);
  }
  return csi.hasTransferConditionSpecifier() ? call.getResult(0)
                                             : mlir::Value{};
}

/// Make the next call in the IO statement conditional on runtime result `ok`.
/// If a call returns `ok==false`, further suboperation calls for an I/O
/// statement will be skipped.  This may generate branch heavy, deeply nested
/// conditionals for I/O statements with a large number of suboperations.
static void makeNextConditionalOn(Fortran::lower::FirOpBuilder &builder,
                                  mlir::Location loc,
                                  mlir::OpBuilder::InsertPoint &insertPt,
                                  bool checkResult, mlir::Value ok,
                                  bool inIterWhileLoop = false) {
  if (!checkResult || !ok)
    // Either I/O calls do not need to be checked, or the next I/O call is the
    // first potentially fallable call.
    return;
  // A previous I/O call for a statement returned the bool `ok`.  If this call
  // is in a fir.iterate_while loop, the result must be propagated up to the
  // loop scope.  That is done in genIoLoop, but it is enabled here.
  auto whereOp =
      inIterWhileLoop
          ? builder.create<fir::WhereOp>(loc, builder.getI1Type(), ok, true)
          : builder.create<fir::WhereOp>(loc, ok, /*withOtherwise=*/false);
  if (!insertPt.isSet())
    insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&whereOp.whereRegion().front());
}

template <typename D>
static void genIoLoop(Fortran::lower::AbstractConverter &converter,
                      mlir::Value cookie, const D &ioImpliedDo,
                      bool checkResult, mlir::Value &ok, bool inIterWhileLoop);

/// Get the OutputXyz routine to output a value of the given type.
static mlir::FuncOp getOutputFunc(mlir::Location loc,
                                  Fortran::lower::FirOpBuilder &builder,
                                  mlir::Type type) {
  if (auto ty = type.dyn_cast<mlir::IntegerType>())
    return ty.getWidth() == 1
               ? getIORuntimeFunc<mkIOKey(OutputLogical)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(OutputInteger64)>(loc, builder);
  if (auto ty = type.dyn_cast<mlir::FloatType>())
    return ty.getWidth() <= 32
               ? getIORuntimeFunc<mkIOKey(OutputReal32)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(OutputReal64)>(loc, builder);
  if (auto ty = type.dyn_cast<fir::CplxType>())
    return ty.getFKind() <= 4
               ? getIORuntimeFunc<mkIOKey(OutputComplex32)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(OutputComplex64)>(loc, builder);
  if (type.isa<fir::LogicalType>())
    return getIORuntimeFunc<mkIOKey(OutputLogical)>(loc, builder);
  if (type.isa<fir::BoxType>())
    return getIORuntimeFunc<mkIOKey(OutputDescriptor)>(loc, builder);
  if (Fortran::lower::CharacterExprHelper::isCharacter(type))
    return getIORuntimeFunc<mkIOKey(OutputAscii)>(loc, builder);
  // TODO: handle arrays
  mlir::emitError(loc, "output for entity type ") << type << " not implemented";
  return {};
}

/// Generate a sequence of output data transfer calls.
static void
genOutputItemList(Fortran::lower::AbstractConverter &converter,
                  mlir::Value cookie,
                  const std::list<Fortran::parser::OutputItem> &items,
                  mlir::OpBuilder::InsertPoint &insertPt, bool checkResult,
                  mlir::Value &ok, bool inIterWhileLoop) {
  auto &builder = converter.getFirOpBuilder();
  for (auto &item : items) {
    if (const auto &impliedDo = std::get_if<1>(&item.u)) {
      genIoLoop(converter, cookie, impliedDo->value(), checkResult, ok,
                inIterWhileLoop);
      continue;
    }
    auto &pExpr = std::get<Fortran::parser::Expr>(item.u);
    auto loc = converter.genLocation(pExpr.source);
    makeNextConditionalOn(builder, loc, insertPt, checkResult, ok,
                          inIterWhileLoop);
    auto itemValue =
        converter.genExprValue(Fortran::semantics::GetExpr(pExpr), loc);
    auto itemType = itemValue.getType();
    auto outputFunc = getOutputFunc(loc, builder, itemType);
    auto argType = outputFunc.getType().getInput(1);
    llvm::SmallVector<mlir::Value, 3> outputFuncArgs = {cookie};
    Fortran::lower::CharacterExprHelper helper{builder, loc};
    if (helper.isCharacter(itemType)) {
      auto dataLen = helper.materializeCharacter(itemValue);
      outputFuncArgs.push_back(builder.createConvert(
          loc, outputFunc.getType().getInput(1), dataLen.first));
      outputFuncArgs.push_back(builder.createConvert(
          loc, outputFunc.getType().getInput(2), dataLen.second));
    } else if (fir::isa_complex(itemType)) {
      auto parts = Fortran::lower::ComplexExprHelper{builder, loc}.extractParts(
          itemValue);
      outputFuncArgs.push_back(parts.first);
      outputFuncArgs.push_back(parts.second);
    } else {
      itemValue = builder.createConvert(loc, argType, itemValue);
      outputFuncArgs.push_back(itemValue);
    }
    ok = builder.create<mlir::CallOp>(loc, outputFunc, outputFuncArgs)
             .getResult(0);
  }
}

/// Get the InputXyz routine to input a value of the given type.
static mlir::FuncOp getInputFunc(mlir::Location loc,
                                 Fortran::lower::FirOpBuilder &builder,
                                 mlir::Type type) {
  if (auto ty = type.dyn_cast<mlir::IntegerType>())
    return ty.getWidth() == 1
               ? getIORuntimeFunc<mkIOKey(InputLogical)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(InputInteger)>(loc, builder);
  if (auto ty = type.dyn_cast<mlir::FloatType>())
    return ty.getWidth() <= 32
               ? getIORuntimeFunc<mkIOKey(InputReal32)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(InputReal64)>(loc, builder);
  if (auto ty = type.dyn_cast<fir::CplxType>())
    return ty.getFKind() <= 4
               ? getIORuntimeFunc<mkIOKey(InputReal32)>(loc, builder)
               : getIORuntimeFunc<mkIOKey(InputReal64)>(loc, builder);
  if (type.isa<fir::LogicalType>())
    return getIORuntimeFunc<mkIOKey(InputLogical)>(loc, builder);
  if (type.isa<fir::BoxType>())
    return getIORuntimeFunc<mkIOKey(InputDescriptor)>(loc, builder);
  if (Fortran::lower::CharacterExprHelper::isCharacter(type))
    return getIORuntimeFunc<mkIOKey(InputAscii)>(loc, builder);
  // TODO: handle arrays
  mlir::emitError(loc, "input for entity type ") << type << " not implemented";
  return {};
}

/// Generate a sequence of input data transfer calls.
static void genInputItemList(Fortran::lower::AbstractConverter &converter,
                             mlir::Value cookie,
                             const std::list<Fortran::parser::InputItem> &items,
                             mlir::OpBuilder::InsertPoint &insertPt,
                             bool checkResult, mlir::Value &ok,
                             bool inIterWhileLoop) {
  auto &builder = converter.getFirOpBuilder();
  for (auto &item : items) {
    if (const auto &impliedDo = std::get_if<1>(&item.u)) {
      genIoLoop(converter, cookie, impliedDo->value(), checkResult, ok,
                inIterWhileLoop);
      continue;
    }
    auto &pVar = std::get<Fortran::parser::Variable>(item.u);
    auto loc = converter.genLocation(pVar.GetSource());
    makeNextConditionalOn(builder, loc, insertPt, checkResult, ok,
                          inIterWhileLoop);
    auto itemAddr =
        converter.genExprAddr(Fortran::semantics::GetExpr(pVar), loc);
    auto itemType = itemAddr.getType().cast<fir::ReferenceType>().getEleTy();
    auto inputFunc = getInputFunc(loc, builder, itemType);
    auto argType = inputFunc.getType().getInput(1);
    auto originalItemAddr = itemAddr;
    mlir::Type complexPartType;
    if (itemType.isa<fir::CplxType>())
      complexPartType = builder.getRefType(
          Fortran::lower::ComplexExprHelper{builder, loc}.getComplexPartType(
              itemType));
    auto complexPartAddr = [&](int index) {
      return builder.create<fir::CoordinateOp>(
          loc, complexPartType, originalItemAddr,
          llvm::SmallVector<mlir::Value, 1>{builder.create<mlir::ConstantOp>(
              loc, builder.getI32IntegerAttr(index))});
    };
    if (complexPartType)
      itemAddr = complexPartAddr(0); // real part
    itemAddr = builder.createConvert(loc, argType, itemAddr);
    llvm::SmallVector<mlir::Value, 3> inputFuncArgs = {cookie, itemAddr};
    Fortran::lower::CharacterExprHelper helper{builder, loc};
    if (helper.isCharacter(itemType)) {
      auto len = helper.materializeCharacter(originalItemAddr).second;
      inputFuncArgs.push_back(
          builder.createConvert(loc, inputFunc.getType().getInput(2), len));
    } else if (itemType.isa<mlir::IntegerType>()) {
      inputFuncArgs.push_back(builder.create<mlir::ConstantOp>(
          loc, builder.getI32IntegerAttr(
                   itemType.cast<mlir::IntegerType>().getWidth() / 8)));
    }
    ok = builder.create<mlir::CallOp>(loc, inputFunc, inputFuncArgs)
             .getResult(0);
    if (complexPartType) { // imaginary part
      makeNextConditionalOn(builder, loc, insertPt, checkResult, ok,
                            inIterWhileLoop);
      inputFuncArgs = {cookie,
                       builder.createConvert(loc, argType, complexPartAddr(1))};
      ok = builder.create<mlir::CallOp>(loc, inputFunc, inputFuncArgs)
               .getResult(0);
    }
  }
}

/// Generate an io-implied-do loop.
template <typename D>
static void genIoLoop(Fortran::lower::AbstractConverter &converter,
                      mlir::Value cookie, const D &ioImpliedDo,
                      bool checkResult, mlir::Value &ok, bool inIterWhileLoop) {
  mlir::OpBuilder::InsertPoint insertPt;
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  makeNextConditionalOn(builder, loc, insertPt, checkResult, ok,
                        inIterWhileLoop);
  auto parentInsertPt = builder.saveInsertionPoint();
  const auto &itemList = std::get<0>(ioImpliedDo.t);
  const auto &control = std::get<1>(ioImpliedDo.t);
  const auto &loopSym = *control.name.thing.thing.symbol;
  auto loopVar = converter.getSymbolAddress(loopSym);
  auto genFIRLoopIndex = [&](const Fortran::parser::ScalarIntExpr &expr) {
    return builder.createConvert(
        loc, builder.getIndexType(),
        converter.genExprValue(*Fortran::semantics::GetExpr(expr)));
  };
  auto lowerValue = genFIRLoopIndex(control.lower);
  auto upperValue = genFIRLoopIndex(control.upper);
  auto stepValue = control.step.has_value()
                       ? genFIRLoopIndex(*control.step)
                       : builder.create<mlir::ConstantIndexOp>(loc, 1);
  auto genItemList = [&](const D &ioImpliedDo, bool inIterWhileLoop) {
    if constexpr (std::is_same_v<D, Fortran::parser::InputImpliedDo>)
      genInputItemList(converter, cookie, itemList, insertPt, checkResult, ok,
                       true);
    else
      genOutputItemList(converter, cookie, itemList, insertPt, checkResult, ok,
                        true);
  };
  if (!checkResult) {
    // No I/O call result checks - the loop is a fir.do_loop op.
    auto loopOp =
        builder.create<fir::LoopOp>(loc, lowerValue, upperValue, stepValue);
    builder.setInsertionPointToStart(loopOp.getBody());
    auto lcv = builder.createConvert(loc, converter.genType(loopSym),
                                     loopOp.getInductionVar());
    builder.create<fir::StoreOp>(loc, lcv, loopVar);
    insertPt = builder.saveInsertionPoint();
    genItemList(ioImpliedDo, false);
    builder.restoreInsertionPoint(parentInsertPt);
    return;
  }
  // Check I/O call results - the loop is a fir.iterate_while op.
  if (!ok)
    ok = builder.createIntegerConstant(loc, builder.getI1Type(), 1);
  fir::IterWhileOp iterWhileOp = builder.create<fir::IterWhileOp>(
      loc, lowerValue, upperValue, stepValue, ok);
  builder.setInsertionPointToStart(iterWhileOp.getBody());
  auto lcv = builder.createConvert(loc, converter.genType(loopSym),
                                   iterWhileOp.getInductionVar());
  builder.create<fir::StoreOp>(loc, lcv, loopVar);
  insertPt = builder.saveInsertionPoint();
  ok = iterWhileOp.getIterateVar();
  auto falseValue = builder.createIntegerConstant(loc, builder.getI1Type(), 0);
  genItemList(ioImpliedDo, true);
  // Unwind nested I/O call scopes, filling in true and false ResultOp's.
  for (auto *op = builder.getBlock()->getParentOp(); isa<fir::WhereOp>(op);
       op = op->getBlock()->getParentOp()) {
    auto whereOp = dyn_cast<fir::WhereOp>(op);
    auto *lastOp = &whereOp.whereRegion().front().back();
    builder.setInsertionPointAfter(lastOp);
    builder.create<fir::ResultOp>(loc, lastOp->getResult(0)); // runtime result
    builder.setInsertionPointToStart(&whereOp.otherRegion().front());
    builder.create<fir::ResultOp>(loc, falseValue); // known false result
  }
  builder.restoreInsertionPoint(insertPt);
  builder.create<fir::ResultOp>(loc, builder.getBlock()->back().getResult(0));
  ok = iterWhileOp.getResult(0);
  builder.restoreInsertionPoint(parentInsertPt);
}

//===----------------------------------------------------------------------===//
// Default argument generation.
//===----------------------------------------------------------------------===//

static mlir::Value getDefaultFilename(Fortran::lower::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Type toType) {
  mlir::Value null =
      builder.create<mlir::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  return builder.createConvert(loc, toType, null);
}

static mlir::Value getDefaultLineNo(Fortran::lower::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Type toType) {
  return builder.create<mlir::ConstantOp>(loc,
                                          builder.getIntegerAttr(toType, 0));
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

/// Lower a string literal. Many arguments to the runtime are conveyed as
/// Fortran CHARACTER literals.
template <typename A>
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
lowerStringLit(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
               const A &syntax, mlir::Type strTy, mlir::Type lenTy,
               mlir::Type ty2 = {}) {
  auto &builder = converter.getFirOpBuilder();
  auto *expr = Fortran::semantics::GetExpr(syntax);
  auto str = converter.genExprValue(expr, loc);
  Fortran::lower::CharacterExprHelper helper{builder, loc};
  auto dataLen = helper.materializeCharacter(str);
  auto buff = builder.createConvert(loc, strTy, dataLen.first);
  auto len = builder.createConvert(loc, lenTy, dataLen.second);
  if (ty2) {
    auto kindVal = helper.getCharacterKind(str.getType());
    auto kind = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ty2, kindVal));
    return {buff, len, kind};
  }
  return {buff, len, mlir::Value{}};
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
  auto lit = builder.createStringLit(
      loc, /*FIXME*/ fir::CharacterType::get(builder.getContext(), 1), text);
  auto data =
      Fortran::lower::CharacterExprHelper{builder, loc}.materializeCharacter(
          lit);
  auto buff = builder.createConvert(loc, strTy, data.first);
  auto len = builder.createConvert(loc, lenTy, data.second);
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
                           const B &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc = getIORuntimeFunc<A>(loc, builder);
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto expr = converter.genExprValue(Fortran::semantics::GetExpr(spec.v), loc);
  auto val = builder.createConvert(loc, ioFuncTy.getInput(1), expr);
  llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, val};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
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
  llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, std::get<0>(tup),
                                              std::get<1>(tup)};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <typename A>
mlir::Value genIOOption(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc, mlir::Value cookie, const A &spec) {
  // default case: do nothing
  return {};
}

template <>
mlir::Value genIOOption<Fortran::parser::FileNameExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::FileNameExpr &spec) {
  auto &builder = converter.getFirOpBuilder();
  // has an extra KIND argument
  auto ioFunc = getIORuntimeFunc<mkIOKey(SetFile)>(loc, builder);
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(converter, loc, spec, ioFuncTy.getInput(1),
                            ioFuncTy.getInput(2), ioFuncTy.getInput(3));
  llvm::SmallVector<mlir::Value, 4> ioArgs{cookie, std::get<0>(tup),
                                           std::get<1>(tup), std::get<2>(tup)};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::ConnectSpec::CharExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::ConnectSpec::CharExpr &spec) {
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
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Convert:
    llvm_unreachable("CONVERT not part of the runtime::io interface");
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Dispose:
    llvm_unreachable("DISPOSE not part of the runtime::io interface");
  }
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(
      converter, loc, std::get<Fortran::parser::ScalarDefaultCharExpr>(spec.t),
      ioFuncTy.getInput(1), ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, std::get<0>(tup),
                                              std::get<1>(tup)};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::ConnectSpec::Recl>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::ConnectSpec::Recl &spec) {
  return genIntIOOption<mkIOKey(SetRecl)>(converter, loc, cookie, spec);
}

template <>
mlir::Value genIOOption<Fortran::parser::StatusExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::StatusExpr &spec) {
  return genCharIOOption<mkIOKey(SetStatus)>(converter, loc, cookie, spec.v);
}

template <>
mlir::Value
genIOOption<Fortran::parser::Name>(Fortran::lower::AbstractConverter &converter,
                                   mlir::Location loc, mlir::Value cookie,
                                   const Fortran::parser::Name &spec) {
  // namelist
  llvm_unreachable("not implemented");
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::CharExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IoControlSpec::CharExpr &spec) {
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
  llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, std::get<0>(tup),
                                              std::get<1>(tup)};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Asynchronous>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie,
    const Fortran::parser::IoControlSpec::Asynchronous &spec) {
  return genCharIOOption<mkIOKey(SetAsynchronous)>(converter, loc, cookie,
                                                   spec.v);
}

template <>
mlir::Value genIOOption<Fortran::parser::IdVariable>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IdVariable &spec) {
  llvm_unreachable("asynchronous ID not implemented");
}

template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Pos>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IoControlSpec::Pos &spec) {
  return genIntIOOption<mkIOKey(SetPos)>(converter, loc, cookie, spec);
}
template <>
mlir::Value genIOOption<Fortran::parser::IoControlSpec::Rec>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::IoControlSpec::Rec &spec) {
  return genIntIOOption<mkIOKey(SetRec)>(converter, loc, cookie, spec);
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
  llvm_unreachable("must have a file unit");
}

/// For each specifier, build the appropriate call, threading the cookie, and
/// returning the insertion point as to the initial context. If there are no
/// specifiers, the insertion point is undefined.
template <typename A>
static mlir::OpBuilder::InsertPoint
threadSpecs(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
            mlir::Value cookie, const A &specList, bool checkResult,
            mlir::Value &ok) {
  auto &builder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insertPt;
  for (const auto &spec : specList) {
    makeNextConditionalOn(builder, loc, insertPt, checkResult, ok);
    ok = std::visit(Fortran::common::visitors{[&](const auto &x) {
                      return genIOOption(converter, loc, cookie, x);
                    }},
                    spec.u);
  }
  return insertPt;
}

template <typename A>
static void
genConditionHandlerCall(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc, mlir::Value cookie,
                        const A &specList, ConditionSpecifierInfo &csi) {
  for (const auto &spec : specList) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::StatVariable &msgVar) {
              csi.ioStatExpr = Fortran::semantics::GetExpr(msgVar);
            },
            [&](const Fortran::parser::MsgVariable &msgVar) {
              csi.ioMsgExpr = Fortran::semantics::GetExpr(msgVar);
            },
            [&](const Fortran::parser::EndLabel &) { csi.hasEnd = true; },
            [&](const Fortran::parser::EorLabel &) { csi.hasEor = true; },
            [&](const Fortran::parser::ErrLabel &) { csi.hasErr = true; },
            [](const auto &) {}},
        spec.u);
  }
  if (!csi.hasAnyConditionSpecifier())
    return;
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp enableHandlers =
      getIORuntimeFunc<mkIOKey(EnableHandlers)>(loc, builder);
  mlir::Type boolType = enableHandlers.getType().getInput(1);
  auto boolValue = [&](bool specifierIsPresent) {
    return builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(boolType, specifierIsPresent));
  };
  llvm::SmallVector<mlir::Value, 6> ioArgs = {
      cookie,
      boolValue(csi.ioStatExpr != nullptr),
      boolValue(csi.hasErr),
      boolValue(csi.hasEnd),
      boolValue(csi.hasEor),
      boolValue(csi.ioMsgExpr != nullptr)};
  builder.create<mlir::CallOp>(loc, enableHandlers, ioArgs);
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
/// rather a namelist variable.
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

static bool hasNonDefaultCharKind(const Fortran::parser::Variable &var) {
  // TODO
  return false;
}

template <typename A>
static bool isDataTransferInternalNotDefaultKind(const A &stmt) {
  // same as isDataTransferInternal, but the KIND of the expression is not the
  // default KIND.
  if (stmt.iounit.has_value())
    if (auto *var = std::get_if<Fortran::parser::Variable>(&stmt.iounit->u))
      return hasNonDefaultCharKind(*var);
  if (auto *unit = getIOControl<Fortran::parser::IoUnit>(stmt))
    if (auto *var = std::get_if<Fortran::parser::Variable>(&unit->u))
      return hasNonDefaultCharKind(*var);
  return false;
}
template <>
constexpr bool isDataTransferInternalNotDefaultKind<Fortran::parser::PrintStmt>(
    const Fortran::parser::PrintStmt &) {
  return false;
}

template <typename A>
static bool isDataTransferAsynchronous(const A &stmt) {
  if (auto *asynch =
          getIOControl<Fortran::parser::IoControlSpec::Asynchronous>(stmt)) {
    // FIXME: should contain a string of YES or NO
    llvm_unreachable("asynchronous transfers not implemented in runtime");
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

/// Generate a reference to a format string.  There are four cases - a format
/// statement label, a character format expression, an integer that holds the
/// label of a format statement, and the * case.  The first three are done here.
/// The * case is done elsewhere.
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
genFormat(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const Fortran::parser::Format &format, mlir::Type strTy,
          mlir::Type lenTy, Fortran::lower::pft::LabelEvalMap &labelMap,
          Fortran::lower::pft::SymbolLabelMap &assignMap) {
  if (const auto *label = std::get_if<Fortran::parser::Label>(&format.u)) {
    // format statement label
    auto iter = labelMap.find(*label);
    assert(iter != labelMap.end() && "FORMAT not found in PROCEDURE");
    return lowerSourceTextAsStringLit(
        converter, loc, toStringRef(iter->second->position), strTy, lenTy);
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
  // TODO - implement this
  llvm::report_fatal_error(
      "using a variable to reference a FORMAT statement; not implemented yet");
}

template <typename A>
std::tuple<mlir::Value, mlir::Value, mlir::Value>
getFormat(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const A &stmt, mlir::Type strTy, mlir::Type lenTy,
          Fortran::lower::pft::LabelEvalMap &labelMap,
          Fortran::lower::pft::SymbolLabelMap &assignMap) {
  if (stmt.format && !formatIsActuallyNamelist(*stmt.format))
    return genFormat(converter, loc, *stmt.format, strTy, lenTy, labelMap,
                     assignMap);
  return genFormat(converter, loc, *getIOControl<Fortran::parser::Format>(stmt),
                   strTy, lenTy, labelMap, assignMap);
}
template <>
std::tuple<mlir::Value, mlir::Value, mlir::Value>
getFormat<Fortran::parser::PrintStmt>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::parser::PrintStmt &stmt, mlir::Type strTy, mlir::Type lenTy,
    Fortran::lower::pft::LabelEvalMap &labelMap,
    Fortran::lower::pft::SymbolLabelMap &assignMap) {
  return genFormat(converter, loc, std::get<Fortran::parser::Format>(stmt.t),
                   strTy, lenTy, labelMap, assignMap);
}

static std::tuple<mlir::Value, mlir::Value, mlir::Value>
genBuffer(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const Fortran::parser::IoUnit &iounit, mlir::Type strTy,
          mlir::Type lenTy) {
  [[maybe_unused]] auto &var = std::get<Fortran::parser::Variable>(iounit.u);
  TODO();
}
template <typename A>
std::tuple<mlir::Value, mlir::Value, mlir::Value>
getBuffer(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
          const A &stmt, mlir::Type strTy, mlir::Type lenTy) {
  if (stmt.iounit)
    return genBuffer(converter, loc, *stmt.iounit, strTy, lenTy);
  return genBuffer(converter, loc, *getIOControl<Fortran::parser::IoUnit>(stmt),
                   strTy, lenTy);
}

template <typename A>
mlir::Value getDescriptor(Fortran::lower::AbstractConverter &converter,
                          mlir::Location loc, const A &stmt,
                          mlir::Type toType) {
  TODO();
}

static mlir::Value genIOUnit(Fortran::lower::AbstractConverter &converter,
                             mlir::Location loc,
                             const Fortran::parser::IoUnit &iounit,
                             mlir::Type ty) {
  auto &builder = converter.getFirOpBuilder();
  if (auto *e = std::get_if<Fortran::parser::FileUnitNumber>(&iounit.u)) {
    auto ex = converter.genExprValue(Fortran::semantics::GetExpr(*e), loc);
    return builder.createConvert(loc, ty, ex);
  }
  return builder.create<mlir::ConstantOp>(
      loc, builder.getIntegerAttr(ty, Fortran::runtime::io::DefaultUnit));
}

template <typename A>
mlir::Value getIOUnit(Fortran::lower::AbstractConverter &converter,
                      mlir::Location loc, const A &stmt, mlir::Type ty) {
  if (stmt.iounit)
    return genIOUnit(converter, loc, *stmt.iounit, ty);
  return genIOUnit(converter, loc, *getIOControl<Fortran::parser::IoUnit>(stmt),
                   ty);
}

//===----------------------------------------------------------------------===//
// Generators for each I/O statement type.
//===----------------------------------------------------------------------===//

template <typename K, typename S>
static mlir::Value genBasicIOStmt(Fortran::lower::AbstractConverter &converter,
                                  const S &stmt) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto beginFunc = getIORuntimeFunc<K>(loc, builder);
  mlir::FunctionType beginFuncTy = beginFunc.getType();
  auto unit = converter.genExprValue(
      getExpr<Fortran::parser::FileUnitNumber>(stmt), loc);
  auto un = builder.createConvert(loc, beginFuncTy.getInput(0), unit);
  auto file = getDefaultFilename(builder, loc, beginFuncTy.getInput(1));
  auto line = getDefaultLineNo(builder, loc, beginFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value, 4> args{un, file, line};
  auto cookie = builder.create<mlir::CallOp>(loc, beginFunc, args).getResult(0);
  ConditionSpecifierInfo csi{};
  genConditionHandlerCall(converter, loc, cookie, stmt.v, csi);
  mlir::Value ok{};
  auto insertPt = threadSpecs(converter, loc, cookie, stmt.v,
                              csi.hasErrorConditionSpecifier(), ok);
  if (insertPt.isSet())
    builder.restoreInsertionPoint(insertPt);
  return genEndIO(converter, converter.getCurrentLocation(), cookie, csi);
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
  mlir::FuncOp beginFunc;
  llvm::SmallVector<mlir::Value, 4> beginArgs;
  auto loc = converter.getCurrentLocation();
  if (hasMem<Fortran::parser::FileUnitNumber>(stmt)) {
    beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenUnit)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    auto unit = converter.genExprValue(
        getExpr<Fortran::parser::FileUnitNumber>(stmt), loc);
    beginArgs.push_back(
        builder.createConvert(loc, beginFuncTy.getInput(0), unit));
    beginArgs.push_back(
        getDefaultFilename(builder, loc, beginFuncTy.getInput(1)));
    beginArgs.push_back(
        getDefaultLineNo(builder, loc, beginFuncTy.getInput(2)));
  } else {
    assert(hasMem<Fortran::parser::ConnectSpec::Newunit>(stmt));
    beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenNewUnit)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    beginArgs.push_back(
        getDefaultFilename(builder, loc, beginFuncTy.getInput(0)));
    beginArgs.push_back(
        getDefaultLineNo(builder, loc, beginFuncTy.getInput(1)));
  }
  auto cookie =
      builder.create<mlir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
  ConditionSpecifierInfo csi{};
  genConditionHandlerCall(converter, loc, cookie, stmt.v, csi);
  mlir::Value ok{};
  auto insertPt = threadSpecs(converter, loc, cookie, stmt.v,
                              csi.hasErrorConditionSpecifier(), ok);
  if (insertPt.isSet())
    builder.restoreInsertionPoint(insertPt);
  return genEndIO(converter, loc, cookie, csi);
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
  auto loc = converter.getCurrentLocation();
  bool hasId = hasMem<Fortran::parser::IdExpr>(stmt);
  mlir::FuncOp beginFunc =
      hasId ? getIORuntimeFunc<mkIOKey(BeginWait)>(loc, builder)
            : getIORuntimeFunc<mkIOKey(BeginWaitAll)>(loc, builder);
  mlir::FunctionType beginFuncTy = beginFunc.getType();
  auto unit = converter.genExprValue(
      getExpr<Fortran::parser::FileUnitNumber>(stmt), loc);
  auto un = builder.createConvert(loc, beginFuncTy.getInput(0), unit);
  llvm::SmallVector<mlir::Value, 4> args{un};
  if (hasId) {
    auto id =
        converter.genExprValue(getExpr<Fortran::parser::IdExpr>(stmt), loc);
    args.push_back(builder.createConvert(loc, beginFuncTy.getInput(1), id));
  }
  auto cookie = builder.create<mlir::CallOp>(loc, beginFunc, args).getResult(0);
  ConditionSpecifierInfo csi{};
  genConditionHandlerCall(converter, loc, cookie, stmt.v, csi);
  return genEndIO(converter, converter.getCurrentLocation(), cookie, csi);
}

//===----------------------------------------------------------------------===//
// Data transfer statements.
//
// There are several dimensions to the API with regard to data transfer
// statements that need to be considered.
//
//   - input (READ) vs. output (WRITE, PRINT)
//   - formatted vs. list vs. unformatted
//   - synchronous vs. asynchronous
//   - namelist vs. list
//   - external vs. internal + default KIND vs. internal + other KIND
//===----------------------------------------------------------------------===//

// Determine the correct BeginXyz{In|Out}put api to invoke.
template <bool isInput>
mlir::FuncOp getBeginDataTransfer(mlir::Location loc, FirOpBuilder &builder,
                                  bool isFormatted, bool isList, bool isIntern,
                                  bool isOtherIntern, bool isAsynch,
                                  bool isNml) {
  if constexpr (isInput) {
    if (isAsynch)
      return getIORuntimeFunc<mkIOKey(BeginAsynchronousInput)>(loc, builder);
    if (isFormatted) {
      if (isIntern) {
        if (isNml)
          return getIORuntimeFunc<mkIOKey(BeginInternalNamelistInput)>(loc,
                                                                       builder);
        if (isOtherIntern) {
          if (isList)
            return getIORuntimeFunc<mkIOKey(BeginInternalArrayListInput)>(
                loc, builder);
          return getIORuntimeFunc<mkIOKey(BeginInternalArrayFormattedInput)>(
              loc, builder);
        }
        if (isList)
          return getIORuntimeFunc<mkIOKey(BeginInternalListInput)>(loc,
                                                                   builder);
        return getIORuntimeFunc<mkIOKey(BeginInternalFormattedInput)>(loc,
                                                                      builder);
      }
      if (isNml)
        return getIORuntimeFunc<mkIOKey(BeginExternalNamelistInput)>(loc,
                                                                     builder);
      if (isList)
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
        if (isNml)
          return getIORuntimeFunc<mkIOKey(BeginInternalNamelistOutput)>(
              loc, builder);
        if (isOtherIntern) {
          if (isList)
            return getIORuntimeFunc<mkIOKey(BeginInternalArrayListOutput)>(
                loc, builder);
          return getIORuntimeFunc<mkIOKey(BeginInternalArrayFormattedOutput)>(
              loc, builder);
        }
        if (isList)
          return getIORuntimeFunc<mkIOKey(BeginInternalListOutput)>(loc,
                                                                    builder);
        return getIORuntimeFunc<mkIOKey(BeginInternalFormattedOutput)>(loc,
                                                                       builder);
      }
      if (isNml)
        return getIORuntimeFunc<mkIOKey(BeginExternalNamelistOutput)>(loc,
                                                                      builder);
      if (isList)
        return getIORuntimeFunc<mkIOKey(BeginExternalListOutput)>(loc, builder);
      return getIORuntimeFunc<mkIOKey(BeginExternalFormattedOutput)>(loc,
                                                                     builder);
    }
    return getIORuntimeFunc<mkIOKey(BeginUnformattedOutput)>(loc, builder);
  }
}

/// Generate the arguments of a BeginXyz call.
template <bool hasIOCtrl, typename A>
void genBeginCallArguments(llvm::SmallVector<mlir::Value, 8> &ioArgs,
                           Fortran::lower::AbstractConverter &converter,
                           mlir::Location loc, const A &stmt,
                           mlir::FunctionType ioFuncTy, bool isFormatted,
                           bool isList, bool isIntern, bool isOtherIntern,
                           bool isAsynch, bool isNml,
                           Fortran::lower::pft::LabelEvalMap &labelMap,
                           Fortran::lower::pft::SymbolLabelMap &assignMap) {
  auto &builder = converter.getFirOpBuilder();
  if constexpr (hasIOCtrl) {
    // READ/WRITE cases have a wide variety of argument permutations
    if (isAsynch || !isFormatted) {
      // unit (always first), ...
      ioArgs.push_back(
          getIOUnit(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size())));
      if (isAsynch) {
        // unknown-thingy, [buff, LEN]
        llvm_unreachable("not implemented");
      }
      return;
    }
    assert(isFormatted && "formatted data transfer");
    if (!isIntern) {
      if (isNml) {
        // namelist group, ...
        llvm_unreachable("not implemented");
      } else if (!isList) {
        // | [format, LEN], ...
        auto pair = getFormat(
            converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
            ioFuncTy.getInput(ioArgs.size() + 1), labelMap, assignMap);
        ioArgs.push_back(std::get<0>(pair));
        ioArgs.push_back(std::get<1>(pair));
      }
      // unit (always last)
      ioArgs.push_back(
          getIOUnit(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size())));
      return;
    }
    assert(isIntern && "internal data transfer");
    if (isNml || isOtherIntern) {
      // descriptor, ...
      ioArgs.push_back(getDescriptor(converter, loc, stmt,
                                     ioFuncTy.getInput(ioArgs.size())));
      if (isNml) {
        // namelist group, ...
        llvm_unreachable("not implemented");
      } else if (isOtherIntern && !isList) {
        // | [format, LEN], ...
        auto pair = getFormat(
            converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
            ioFuncTy.getInput(ioArgs.size() + 1), labelMap, assignMap);
        ioArgs.push_back(std::get<0>(pair));
        ioArgs.push_back(std::get<1>(pair));
      }
    } else {
      // | [buff, LEN], ...
      auto pair =
          getBuffer(converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
                    ioFuncTy.getInput(ioArgs.size() + 1));
      ioArgs.push_back(std::get<0>(pair));
      ioArgs.push_back(std::get<1>(pair));
      if (!isList) {
        // [format, LEN], ...
        auto pair = getFormat(
            converter, loc, stmt, ioFuncTy.getInput(ioArgs.size()),
            ioFuncTy.getInput(ioArgs.size() + 1), labelMap, assignMap);
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
                    ioFuncTy.getInput(ioArgs.size() + 1), labelMap, assignMap);
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
genDataTransferStmt(Fortran::lower::AbstractConverter &converter, const A &stmt,
                    Fortran::lower::pft::LabelEvalMap &labelMap,
                    Fortran::lower::pft::SymbolLabelMap &assignMap) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  const bool isFormatted = isDataTransferFormatted(stmt);
  const bool isList = isFormatted ? isDataTransferList(stmt) : false;
  const bool isIntern = isDataTransferInternal(stmt);
  const bool isOtherIntern =
      isIntern ? isDataTransferInternalNotDefaultKind(stmt) : false;
  const bool isAsynch = isDataTransferAsynchronous(stmt);
  const bool isNml = isDataTransferNamelist(stmt);

  // Determine which BeginXyz call to make.
  mlir::FuncOp ioFunc =
      getBeginDataTransfer<isInput>(loc, builder, isFormatted, isList, isIntern,
                                    isOtherIntern, isAsynch, isNml);
  mlir::FunctionType ioFuncTy = ioFunc.getType();

  // Append BeginXyz call arguments.  File name and line number are always last.
  llvm::SmallVector<mlir::Value, 8> ioArgs;
  genBeginCallArguments<hasIOCtrl>(ioArgs, converter, loc, stmt, ioFuncTy,
                                   isFormatted, isList, isIntern, isOtherIntern,
                                   isAsynch, isNml, labelMap, assignMap);
  ioArgs.push_back(
      getDefaultFilename(builder, loc, ioFuncTy.getInput(ioArgs.size())));
  ioArgs.push_back(
      getDefaultLineNo(builder, loc, ioFuncTy.getInput(ioArgs.size())));

  // Arguments are done; call the BeginXyz function.
  mlir::Value cookie =
      builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);

  // Generate an EnableHandlers call and remaining specifier calls.
  ConditionSpecifierInfo csi;
  mlir::OpBuilder::InsertPoint insertPt;
  mlir::Value ok;
  if constexpr (hasIOCtrl) {
    genConditionHandlerCall(converter, loc, cookie, stmt.controls, csi);
    insertPt = threadSpecs(converter, loc, cookie, stmt.controls,
                           csi.hasErrorConditionSpecifier(), ok);
  }

  // Generate data transfer list calls.
  if constexpr (isInput) // ReadStmt
    genInputItemList(converter, cookie, stmt.items, insertPt,
                     csi.hasTransferConditionSpecifier(), ok, false);
  else if constexpr (std::is_same_v<A, Fortran::parser::PrintStmt>)
    genOutputItemList(converter, cookie, std::get<1>(stmt.t), insertPt,
                      csi.hasTransferConditionSpecifier(), ok, false);
  else // WriteStmt
    genOutputItemList(converter, cookie, stmt.items, insertPt,
                      csi.hasTransferConditionSpecifier(), ok, false);

  // Generate end statement call/s.
  if (insertPt.isSet())
    builder.restoreInsertionPoint(insertPt);
  return genEndIO(converter, loc, cookie, csi);
}

void Fortran::lower::genPrintStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::PrintStmt &stmt,
    Fortran::lower::pft::LabelEvalMap &labelMap,
    Fortran::lower::pft::SymbolLabelMap &assignMap) {
  // PRINT does not take an io-control-spec. It only has a format specifier, so
  // it is a simplified case of WRITE.
  genDataTransferStmt</*isInput=*/false, /*ioCtrl=*/false>(converter, stmt,
                                                           labelMap, assignMap);
}

mlir::Value Fortran::lower::genWriteStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::WriteStmt &stmt,
    Fortran::lower::pft::LabelEvalMap &labelMap,
    Fortran::lower::pft::SymbolLabelMap &assignMap) {
  return genDataTransferStmt</*isInput=*/false>(converter, stmt, labelMap,
                                                assignMap);
}

mlir::Value Fortran::lower::genReadStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::ReadStmt &stmt,
    Fortran::lower::pft::LabelEvalMap &labelMap,
    Fortran::lower::pft::SymbolLabelMap &assignMap) {
  return genDataTransferStmt</*isInput=*/true>(converter, stmt, labelMap,
                                               assignMap);
}

/// Get the file expression from the inquire spec list. Also return if the
/// expression is a file name.
static std::pair<const Fortran::semantics::SomeExpr *, bool>
getInquireFileExpr(const std::list<Fortran::parser::InquireSpec> *stmt) {
  if (!stmt)
    return {nullptr, false};
  for (const auto &spec : *stmt) {
    if (auto *f = std::get_if<Fortran::parser::FileUnitNumber>(&spec.u))
      return {Fortran::semantics::GetExpr(*f), false};
    if (auto *f = std::get_if<Fortran::parser::FileNameExpr>(&spec.u))
      return {Fortran::semantics::GetExpr(*f), true};
  }
  // semantics should have already caught this condition
  llvm_unreachable("inquire spec must have a file");
}

mlir::Value Fortran::lower::genInquireStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::InquireStmt &stmt) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  mlir::FuncOp beginFunc;
  mlir::Value cookie;
  ConditionSpecifierInfo csi{};
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
    auto unit = converter.genExprValue(exprPair.first, loc);
    auto un = builder.createConvert(loc, beginFuncTy.getInput(0), unit);
    auto file = getDefaultFilename(builder, loc, beginFuncTy.getInput(1));
    auto line = getDefaultLineNo(builder, loc, beginFuncTy.getInput(2));
    llvm::SmallVector<mlir::Value, 4> beginArgs{un, file, line};
    cookie =
        builder.create<mlir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
    // Handle remaining arguments in specifier list.
    genConditionHandlerCall(converter, loc, cookie, *list, csi);
  } else if (inquireFileName()) {
    // Filename call.
    beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireFile)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    auto file = converter.genExprValue(exprPair.first, loc);
    // Helper to query [BUFFER, LEN].
    Fortran::lower::CharacterExprHelper helper(builder, loc);
    auto dataLen = helper.materializeCharacter(file);
    auto buff =
        builder.createConvert(loc, beginFuncTy.getInput(0), dataLen.first);
    auto len =
        builder.createConvert(loc, beginFuncTy.getInput(1), dataLen.second);
    auto kindInt = helper.getCharacterKind(file.getType());
    mlir::Value kindValue =
        builder.createIntegerConstant(loc, beginFuncTy.getInput(2), kindInt);
    auto sourceFile = getDefaultFilename(builder, loc, beginFuncTy.getInput(3));
    auto line = getDefaultLineNo(builder, loc, beginFuncTy.getInput(4));
    llvm::SmallVector<mlir::Value, 5> beginArgs = {
        buff, len, kindValue, sourceFile, line,
    };
    cookie =
        builder.create<mlir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
    // Handle remaining arguments in specifier list.
    genConditionHandlerCall(converter, loc, cookie, *list, csi);
  } else {
    // Io length call.
    const auto *ioLength =
        std::get_if<Fortran::parser::InquireStmt::Iolength>(&stmt.u);
    assert(ioLength && "must have an io length");
    beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireIoLength)>(loc, builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    auto file = getDefaultFilename(builder, loc, beginFuncTy.getInput(0));
    auto line = getDefaultLineNo(builder, loc, beginFuncTy.getInput(1));
    llvm::SmallVector<mlir::Value, 4> beginArgs{file, line};
    cookie =
        builder.create<mlir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
    // Handle remaining arguments in output list.
    genConditionHandlerCall(
        converter, loc, cookie,
        std::get<std::list<Fortran::parser::OutputItem>>(ioLength->t), csi);
  }
  // Generate end statement call.
  return genEndIO(converter, loc, cookie, csi);
}
