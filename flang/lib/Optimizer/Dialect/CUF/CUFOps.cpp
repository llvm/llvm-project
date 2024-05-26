//===-- CUFOps.cpp --------------------------------------------------------===//
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

#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

static mlir::Type wrapAllocaResultType(mlir::Type intype) {
  if (mlir::isa<fir::ReferenceType>(intype))
    return {};
  return fir::ReferenceType::get(intype);
}

void cuf::AllocOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                         mlir::Type inType, llvm::StringRef uniqName,
                         llvm::StringRef bindcName,
                         cuf::DataAttributeAttr cudaAttr,
                         mlir::ValueRange typeparams, mlir::ValueRange shape,
                         llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  mlir::StringAttr nameAttr =
      uniqName.empty() ? mlir::StringAttr{} : builder.getStringAttr(uniqName);
  mlir::StringAttr bindcAttr =
      bindcName.empty() ? mlir::StringAttr{} : builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocaResultType(inType),
        mlir::TypeAttr::get(inType), nameAttr, bindcAttr, typeparams, shape,
        cudaAttr);
  result.addAttributes(attributes);
}

template <typename Op>
static mlir::LogicalResult checkCudaAttr(Op op) {
  if (op.getDataAttr() == cuf::DataAttribute::Device ||
      op.getDataAttr() == cuf::DataAttribute::Managed ||
      op.getDataAttr() == cuf::DataAttribute::Unified)
    return mlir::success();
  return op.emitOpError("expect device, managed or unified cuda attribute");
}

mlir::LogicalResult cuf::AllocOp::verify() { return checkCudaAttr(*this); }

//===----------------------------------------------------------------------===//
// FreeOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult cuf::FreeOp::verify() { return checkCudaAttr(*this); }

//===----------------------------------------------------------------------===//
// AllocateOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult cuf::AllocateOp::verify() {
  if (getPinned() && getStream())
    return emitOpError("pinned and stream cannot appears at the same time");
  if (!mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(getBox().getType())))
    return emitOpError(
        "expect box to be a reference to a class or box type value");
  if (getSource() &&
      !mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(getSource().getType())))
    return emitOpError(
        "expect source to be a reference to/or a class or box type value");
  if (getErrmsg() &&
      !mlir::isa<fir::BoxType>(fir::unwrapRefType(getErrmsg().getType())))
    return emitOpError(
        "expect errmsg to be a reference to/or a box type value");
  if (getErrmsg() && !getHasStat())
    return emitOpError("expect stat attribute when errmsg is provided");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DataTransferOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult cuf::DataTransferOp::verify() {
  mlir::Type srcTy = getSrc().getType();
  mlir::Type dstTy = getDst().getType();
  if ((fir::isa_ref_type(srcTy) && fir::isa_ref_type(dstTy)) ||
      (fir::isa_box_type(srcTy) && fir::isa_box_type(dstTy)))
    return mlir::success();
  if (fir::isa_trivial(srcTy) &&
      matchPattern(getSrc().getDefiningOp(), mlir::m_Constant()))
    return mlir::success();
  return emitOpError()
         << "expect src and dst to be both references or descriptors or src to "
            "be a constant";
}

//===----------------------------------------------------------------------===//
// DeallocateOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult cuf::DeallocateOp::verify() {
  if (!mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(getBox().getType())))
    return emitOpError(
        "expect box to be a reference to class or box type value");
  if (getErrmsg() &&
      !mlir::isa<fir::BoxType>(fir::unwrapRefType(getErrmsg().getType())))
    return emitOpError(
        "expect errmsg to be a reference to/or a box type value");
  if (getErrmsg() && !getHasStat())
    return emitOpError("expect stat attribute when errmsg is provided");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::Region *> cuf::KernelOp::getLoopRegions() {
  return {&getRegion()};
}

mlir::ParseResult parseCUFKernelValues(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &values,
    llvm::SmallVectorImpl<mlir::Type> &types) {
  if (mlir::succeeded(parser.parseOptionalStar()))
    return mlir::success();

  if (mlir::succeeded(parser.parseOptionalLParen())) {
    if (mlir::failed(parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::None, [&]() {
              if (parser.parseOperand(values.emplace_back()))
                return mlir::failure();
              return mlir::success();
            })))
      return mlir::failure();
    auto builder = parser.getBuilder();
    for (size_t i = 0; i < values.size(); i++) {
      types.emplace_back(builder.getI32Type());
    }
    if (parser.parseRParen())
      return mlir::failure();
  } else {
    if (parser.parseOperand(values.emplace_back()))
      return mlir::failure();
    auto builder = parser.getBuilder();
    types.emplace_back(builder.getI32Type());
    return mlir::success();
  }
  return mlir::success();
}

void printCUFKernelValues(mlir::OpAsmPrinter &p, mlir::Operation *op,
                          mlir::ValueRange values, mlir::TypeRange types) {
  if (values.empty())
    p << "*";

  if (values.size() > 1)
    p << "(";
  llvm::interleaveComma(values, p, [&p](mlir::Value v) { p << v; });
  if (values.size() > 1)
    p << ")";
}

mlir::ParseResult parseCUFKernelLoopControl(
    mlir::OpAsmParser &parser, mlir::Region &region,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &lowerbound,
    llvm::SmallVectorImpl<mlir::Type> &lowerboundType,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &upperbound,
    llvm::SmallVectorImpl<mlir::Type> &upperboundType,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &step,
    llvm::SmallVectorImpl<mlir::Type> &stepType) {

  llvm::SmallVector<mlir::OpAsmParser::Argument> inductionVars;
  if (parser.parseLParen() ||
      parser.parseArgumentList(inductionVars,
                               mlir::OpAsmParser::Delimiter::None,
                               /*allowType=*/true) ||
      parser.parseRParen() || parser.parseEqual() || parser.parseLParen() ||
      parser.parseOperandList(lowerbound, inductionVars.size(),
                              mlir::OpAsmParser::Delimiter::None) ||
      parser.parseColonTypeList(lowerboundType) || parser.parseRParen() ||
      parser.parseKeyword("to") || parser.parseLParen() ||
      parser.parseOperandList(upperbound, inductionVars.size(),
                              mlir::OpAsmParser::Delimiter::None) ||
      parser.parseColonTypeList(upperboundType) || parser.parseRParen() ||
      parser.parseKeyword("step") || parser.parseLParen() ||
      parser.parseOperandList(step, inductionVars.size(),
                              mlir::OpAsmParser::Delimiter::None) ||
      parser.parseColonTypeList(stepType) || parser.parseRParen())
    return mlir::failure();
  return parser.parseRegion(region, inductionVars);
}

void printCUFKernelLoopControl(
    mlir::OpAsmPrinter &p, mlir::Operation *op, mlir::Region &region,
    mlir::ValueRange lowerbound, mlir::TypeRange lowerboundType,
    mlir::ValueRange upperbound, mlir::TypeRange upperboundType,
    mlir::ValueRange steps, mlir::TypeRange stepType) {
  mlir::ValueRange regionArgs = region.front().getArguments();
  if (!regionArgs.empty()) {
    p << "(";
    llvm::interleaveComma(
        regionArgs, p, [&p](mlir::Value v) { p << v << " : " << v.getType(); });
    p << ") = (" << lowerbound << " : " << lowerboundType << ") to ("
      << upperbound << " : " << upperboundType << ") "
      << " step (" << steps << " : " << stepType << ") ";
  }
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

mlir::LogicalResult cuf::KernelOp::verify() {
  if (getLowerbound().size() != getUpperbound().size() ||
      getLowerbound().size() != getStep().size())
    return emitOpError(
        "expect same number of values in lowerbound, upperbound and step");

  return mlir::success();
}

// Tablegen operators

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/CUF/CUFOps.cpp.inc"
