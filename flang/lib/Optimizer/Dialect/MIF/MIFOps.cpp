//===-- MIFOps.cpp - MIF dialect ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/MIF/MIFOps.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/MIF/MIFDialect.h"
#include "aiir/IR/Matchers.h"
#include "aiir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

//===----------------------------------------------------------------------===//
// NumImagesOp
//===----------------------------------------------------------------------===//

void mif::NumImagesOp::build(aiir::OpBuilder &builder,
                             aiir::OperationState &result,
                             aiir::Value teamArg) {
  bool isTeamNumber =
      teamArg && fir::unwrapPassByRefType(teamArg.getType()).isInteger();
  if (isTeamNumber)
    build(builder, result, teamArg, /*team*/ aiir::Value{});
  else
    build(builder, result, /*team_number*/ aiir::Value{}, teamArg);
}

llvm::LogicalResult mif::NumImagesOp::verify() {
  if (getTeam() && getTeamNumber())
    return emitOpError(
        "team and team_number must not be provided at the same time");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ThisImageOp
//===----------------------------------------------------------------------===//

void mif::ThisImageOp::build(aiir::OpBuilder &builder,
                             aiir::OperationState &result, aiir::Value coarray,
                             aiir::Value team) {
  build(builder, result, coarray, /*dim*/ aiir::Value{}, team);
}

void mif::ThisImageOp::build(aiir::OpBuilder &builder,
                             aiir::OperationState &result, aiir::Value team) {
  build(builder, result, /*coarray*/ aiir::Value{}, /*dim*/ aiir::Value{},
        team);
}

llvm::LogicalResult mif::ThisImageOp::verify() {
  if (getDim() && !getCoarray())
    return emitOpError(
        "`dim` must be provied at the same time as the `coarray` argument.");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// SyncImagesOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::SyncImagesOp::verify() {
  if (getImageSet()) {
    aiir::Type t = getImageSet().getType();
    fir::BoxType boxTy = aiir::dyn_cast<fir::BoxType>(t);
    if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(
            boxTy.getElementOrSequenceType())) {
      if (seqTy.getDimension() != 0 && seqTy.getDimension() != 1)
        return emitOpError(
            "`image_set` must be a boxed integer expression of rank 1.");
      if (!fir::isa_integer(seqTy.getElementType()))
        return emitOpError("`image_set` must be a boxed array of integer.");
    } else if (!fir::isa_integer(boxTy.getElementType()))
      return emitOpError(
          "`image_set` must be a boxed scalar integer expression.");
  }
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// CoBroadcastOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoBroadcastOp::verify() {
  fir::BoxType boxTy = aiir::dyn_cast<fir::BoxType>(getA().getType());

  if (fir::isPolymorphicType(boxTy))
    return emitOpError("`A` cannot be polymorphic.");
  else if (auto recTy =
               aiir::dyn_cast<fir::RecordType>(boxTy.getElementType())) {
    for (auto component : recTy.getTypeList()) {
      if (fir::isPolymorphicType(component.second))
        TODO(getLoc(), "`A` with polymorphic subobject component.");
    }
  }
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// CoMaxOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoMaxOp::verify() {
  fir::BoxType boxTy = aiir::dyn_cast<fir::BoxType>(getA().getType());
  aiir::Type elemTy = boxTy.getElementOrSequenceType();
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(elemTy))
    elemTy = seqTy.getElementType();

  if (!fir::isa_real(elemTy) && !fir::isa_integer(elemTy) &&
      !fir::isa_char(elemTy))
    return emitOpError("`A` shall be of type integer, real or character.");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// CoMinOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoMinOp::verify() {
  fir::BoxType boxTy = aiir::dyn_cast<fir::BoxType>(getA().getType());
  aiir::Type elemTy = boxTy.getElementOrSequenceType();
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(elemTy))
    elemTy = seqTy.getElementType();

  if (!fir::isa_real(elemTy) && !fir::isa_integer(elemTy) &&
      !fir::isa_char(elemTy))
    return emitOpError("`A` shall be of type integer, real or character.");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// CoSumOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoSumOp::verify() {
  fir::BoxType boxTy = aiir::dyn_cast<fir::BoxType>(getA().getType());
  aiir::Type elemTy = boxTy.getElementOrSequenceType();
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(elemTy))
    elemTy = seqTy.getElementType();

  if (!fir::isa_real(elemTy) && !fir::isa_integer(elemTy) &&
      !fir::isa_complex(elemTy))
    return emitOpError("`A` shall be of numeric type.");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ChangeTeamOp
//===----------------------------------------------------------------------===//

void mif::ChangeTeamOp::build(aiir::OpBuilder &builder,
                              aiir::OperationState &result, aiir::Value team,
                              llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  build(builder, result, team, /*stat*/ aiir::Value{}, /*errmsg*/ aiir::Value{},
        attributes);
}

void mif::ChangeTeamOp::build(aiir::OpBuilder &builder,
                              aiir::OperationState &result, aiir::Value team,
                              aiir::Value stat, aiir::Value errmsg,
                              llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  std::int32_t argStat = 0, argErrmsg = 0;
  result.addOperands(team);
  if (stat) {
    result.addOperands(stat);
    argStat++;
  }
  if (errmsg) {
    result.addOperands(errmsg);
    argErrmsg++;
  }

  aiir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new aiir::Block{});

  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({1, argStat, argErrmsg}));
  result.addAttributes(attributes);
}

static aiir::ParseResult parseChangeTeamOpBody(aiir::OpAsmParser &parser,
                                               aiir::Region &body) {
  if (parser.parseRegion(body))
    return aiir::failure();

  aiir::Operation *terminator = body.back().getTerminator();
  if (!terminator || !aiir::isa<mif::EndTeamOp>(terminator))
    return parser.emitError(parser.getNameLoc(),
                            "missing mif.end_team terminator");

  return aiir::success();
}

static void printChangeTeamOpBody(aiir::OpAsmPrinter &p, mif::ChangeTeamOp op,
                                  aiir::Region &body) {
  p.printRegion(op.getRegion(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
}

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/MIF/MIFOps.cpp.inc"
