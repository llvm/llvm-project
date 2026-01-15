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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

//===----------------------------------------------------------------------===//
// NumImagesOp
//===----------------------------------------------------------------------===//

void mif::NumImagesOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result,
                             mlir::Value teamArg) {
  bool isTeamNumber =
      teamArg && fir::unwrapPassByRefType(teamArg.getType()).isInteger();
  if (isTeamNumber)
    build(builder, result, teamArg, /*team*/ mlir::Value{});
  else
    build(builder, result, /*team_number*/ mlir::Value{}, teamArg);
}

llvm::LogicalResult mif::NumImagesOp::verify() {
  if (getTeam() && getTeamNumber())
    return emitOpError(
        "team and team_number must not be provided at the same time");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ThisImageOp
//===----------------------------------------------------------------------===//

void mif::ThisImageOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value coarray,
                             mlir::Value team) {
  build(builder, result, coarray, /*dim*/ mlir::Value{}, team);
}

void mif::ThisImageOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value team) {
  build(builder, result, /*coarray*/ mlir::Value{}, /*dim*/ mlir::Value{},
        team);
}

llvm::LogicalResult mif::ThisImageOp::verify() {
  if (getDim() && !getCoarray())
    return emitOpError(
        "`dim` must be provied at the same time as the `coarray` argument.");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SyncImagesOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::SyncImagesOp::verify() {
  if (getImageSet()) {
    mlir::Type t = getImageSet().getType();
    fir::BoxType boxTy = mlir::dyn_cast<fir::BoxType>(t);
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(
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
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CoBroadcastOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoBroadcastOp::verify() {
  fir::BoxType boxTy = mlir::dyn_cast<fir::BoxType>(getA().getType());

  if (fir::isPolymorphicType(boxTy))
    return emitOpError("`A` cannot be polymorphic.");
  else if (auto recTy =
               mlir::dyn_cast<fir::RecordType>(boxTy.getElementType())) {
    for (auto component : recTy.getTypeList()) {
      if (fir::isPolymorphicType(component.second))
        TODO(getLoc(), "`A` with polymorphic subobject component.");
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CoMaxOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoMaxOp::verify() {
  fir::BoxType boxTy = mlir::dyn_cast<fir::BoxType>(getA().getType());
  mlir::Type elemTy = boxTy.getElementOrSequenceType();
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(elemTy))
    elemTy = seqTy.getElementType();

  if (!fir::isa_real(elemTy) && !fir::isa_integer(elemTy) &&
      !fir::isa_char(elemTy))
    return emitOpError("`A` shall be of type integer, real or character.");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CoMinOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoMinOp::verify() {
  fir::BoxType boxTy = mlir::dyn_cast<fir::BoxType>(getA().getType());
  mlir::Type elemTy = boxTy.getElementOrSequenceType();
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(elemTy))
    elemTy = seqTy.getElementType();

  if (!fir::isa_real(elemTy) && !fir::isa_integer(elemTy) &&
      !fir::isa_char(elemTy))
    return emitOpError("`A` shall be of type integer, real or character.");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CoSumOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoSumOp::verify() {
  fir::BoxType boxTy = mlir::dyn_cast<fir::BoxType>(getA().getType());
  mlir::Type elemTy = boxTy.getElementOrSequenceType();
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(elemTy))
    elemTy = seqTy.getElementType();

  if (!fir::isa_real(elemTy) && !fir::isa_integer(elemTy) &&
      !fir::isa_complex(elemTy))
    return emitOpError("`A` shall be of numeric type.");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ChangeTeamOp
//===----------------------------------------------------------------------===//

void mif::ChangeTeamOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result, mlir::Value team,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  build(builder, result, team, /*stat*/ mlir::Value{}, /*errmsg*/ mlir::Value{},
        attributes);
}

void mif::ChangeTeamOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result, mlir::Value team,
                              mlir::Value stat, mlir::Value errmsg,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
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

  mlir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new mlir::Block{});

  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({1, argStat, argErrmsg}));
  result.addAttributes(attributes);
}

static mlir::ParseResult parseChangeTeamOpBody(mlir::OpAsmParser &parser,
                                               mlir::Region &body) {
  if (parser.parseRegion(body))
    return mlir::failure();

  mlir::Operation *terminator = body.back().getTerminator();
  if (!terminator || !mlir::isa<mif::EndTeamOp>(terminator))
    return parser.emitError(parser.getNameLoc(),
                            "missing mif.end_team terminator");

  return mlir::success();
}

static void printChangeTeamOpBody(mlir::OpAsmPrinter &p, mif::ChangeTeamOp op,
                                  mlir::Region &body) {
  p.printRegion(op.getRegion(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
}

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/MIF/MIFOps.cpp.inc"
