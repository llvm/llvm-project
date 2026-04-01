//===-- CUFOps.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/Attributes.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/Matchers.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

static aiir::Type wrapAllocaResultType(aiir::Type intype) {
  if (aiir::isa<fir::ReferenceType>(intype))
    return {};
  return fir::ReferenceType::get(intype);
}

void cuf::AllocOp::build(aiir::OpBuilder &builder, aiir::OperationState &result,
                         aiir::Type inType, llvm::StringRef uniqName,
                         llvm::StringRef bindcName,
                         cuf::DataAttributeAttr cudaAttr,
                         aiir::ValueRange typeparams, aiir::ValueRange shape,
                         llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  aiir::StringAttr nameAttr =
      uniqName.empty() ? aiir::StringAttr{} : builder.getStringAttr(uniqName);
  aiir::StringAttr bindcAttr =
      bindcName.empty() ? aiir::StringAttr{} : builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocaResultType(inType),
        aiir::TypeAttr::get(inType), nameAttr, bindcAttr, typeparams, shape,
        cudaAttr);
  result.addAttributes(attributes);
}

template <typename Op>
static llvm::LogicalResult checkCudaAttr(Op op) {
  if (op.getDataAttr() == cuf::DataAttribute::Device ||
      op.getDataAttr() == cuf::DataAttribute::Managed ||
      op.getDataAttr() == cuf::DataAttribute::Unified ||
      op.getDataAttr() == cuf::DataAttribute::Pinned ||
      op.getDataAttr() == cuf::DataAttribute::Shared)
    return aiir::success();
  return op.emitOpError()
         << "expect device, managed, pinned or unified cuda attribute";
}

llvm::LogicalResult cuf::AllocOp::verify() { return checkCudaAttr(*this); }

//===----------------------------------------------------------------------===//
// FreeOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult cuf::FreeOp::verify() { return checkCudaAttr(*this); }

//===----------------------------------------------------------------------===//
// AllocateOp
//===----------------------------------------------------------------------===//

template <typename OpTy>
static llvm::LogicalResult checkStreamType(OpTy op) {
  if (!op.getStream())
    return aiir::success();
  if (auto refTy = aiir::dyn_cast<fir::ReferenceType>(op.getStream().getType()))
    if (!refTy.getEleTy().isInteger(64))
      return op.emitOpError("stream is expected to be an i64 reference");
  return aiir::success();
}

llvm::LogicalResult cuf::AllocateOp::verify() {
  if (getPinned() && getStream())
    return emitOpError("pinned and stream cannot appears at the same time");
  if (!aiir::isa<fir::BaseBoxType>(fir::unwrapRefType(getBox().getType())))
    return emitOpError(
        "expect box to be a reference to a class or box type value");
  if (getSource() &&
      !aiir::isa<fir::BaseBoxType>(fir::unwrapRefType(getSource().getType())))
    return emitOpError(
        "expect source to be a reference to/or a class or box type value");
  if (getErrmsg() &&
      !aiir::isa<fir::BoxType>(fir::unwrapRefType(getErrmsg().getType())))
    return emitOpError(
        "expect errmsg to be a reference to/or a box type value");
  if (getErrmsg() && !getHasStat())
    return emitOpError("expect stat attribute when errmsg is provided");
  return checkStreamType(*this);
}

//===----------------------------------------------------------------------===//
// DataTransferOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult cuf::DataTransferOp::verify() {
  aiir::Type srcTy = getSrc().getType();
  aiir::Type dstTy = getDst().getType();
  if (getShape()) {
    if (!fir::isa_ref_type(srcTy) && !fir::isa_ref_type(dstTy))
      return emitOpError()
             << "shape can only be specified on data transfer with references";
  }
  if ((fir::isa_ref_type(srcTy) && fir::isa_ref_type(dstTy)) ||
      (fir::isa_box_type(srcTy) && fir::isa_box_type(dstTy)) ||
      (fir::isa_ref_type(srcTy) && fir::isa_box_type(dstTy)) ||
      (fir::isa_box_type(srcTy) && fir::isa_ref_type(dstTy)))
    return aiir::success();
  if (fir::isa_trivial(srcTy) &&
      matchPattern(getSrc().getDefiningOp(), aiir::m_Constant()))
    return aiir::success();

  return emitOpError()
         << "expect src and dst to be references or descriptors or src to "
            "be a constant: "
         << srcTy << " - " << dstTy;
}

//===----------------------------------------------------------------------===//
// DeallocateOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult cuf::DeallocateOp::verify() {
  if (!aiir::isa<fir::BaseBoxType>(fir::unwrapRefType(getBox().getType())))
    return emitOpError(
        "expect box to be a reference to class or box type value");
  if (getErrmsg() &&
      !aiir::isa<fir::BoxType>(fir::unwrapRefType(getErrmsg().getType())))
    return emitOpError(
        "expect errmsg to be a reference to/or a box type value");
  if (getErrmsg() && !getHasStat())
    return emitOpError("expect stat attribute when errmsg is provided");
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// KernelLaunchOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult cuf::KernelLaunchOp::verify() {
  return checkStreamType(*this);
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<aiir::Region *> cuf::KernelOp::getLoopRegions() {
  return {&getRegion()};
}

aiir::ParseResult parseCUFKernelValues(
    aiir::OpAsmParser &parser,
    llvm::SmallVectorImpl<aiir::OpAsmParser::UnresolvedOperand> &values,
    llvm::SmallVectorImpl<aiir::Type> &types) {
  if (aiir::succeeded(parser.parseOptionalStar()))
    return aiir::success();

  if (aiir::succeeded(parser.parseOptionalLParen())) {
    if (aiir::failed(parser.parseCommaSeparatedList(
            aiir::AsmParser::Delimiter::None, [&]() {
              if (parser.parseOperand(values.emplace_back()))
                return aiir::failure();
              return aiir::success();
            })))
      return aiir::failure();
    auto builder = parser.getBuilder();
    for (size_t i = 0; i < values.size(); i++) {
      types.emplace_back(builder.getI32Type());
    }
    if (parser.parseRParen())
      return aiir::failure();
  } else {
    if (parser.parseOperand(values.emplace_back()))
      return aiir::failure();
    auto builder = parser.getBuilder();
    types.emplace_back(builder.getI32Type());
    return aiir::success();
  }
  return aiir::success();
}

void printCUFKernelValues(aiir::OpAsmPrinter &p, aiir::Operation *op,
                          aiir::ValueRange values, aiir::TypeRange types) {
  if (values.empty())
    p << "*";

  if (values.size() > 1)
    p << "(";
  llvm::interleaveComma(values, p, [&p](aiir::Value v) { p << v; });
  if (values.size() > 1)
    p << ")";
}

aiir::ParseResult parseCUFKernelLoopControl(
    aiir::OpAsmParser &parser, aiir::Region &region,
    llvm::SmallVectorImpl<aiir::OpAsmParser::UnresolvedOperand> &lowerbound,
    llvm::SmallVectorImpl<aiir::Type> &lowerboundType,
    llvm::SmallVectorImpl<aiir::OpAsmParser::UnresolvedOperand> &upperbound,
    llvm::SmallVectorImpl<aiir::Type> &upperboundType,
    llvm::SmallVectorImpl<aiir::OpAsmParser::UnresolvedOperand> &step,
    llvm::SmallVectorImpl<aiir::Type> &stepType) {

  llvm::SmallVector<aiir::OpAsmParser::Argument> inductionVars;
  if (parser.parseLParen() ||
      parser.parseArgumentList(inductionVars,
                               aiir::OpAsmParser::Delimiter::None,
                               /*allowType=*/true) ||
      parser.parseRParen() || parser.parseEqual() || parser.parseLParen() ||
      parser.parseOperandList(lowerbound, inductionVars.size(),
                              aiir::OpAsmParser::Delimiter::None) ||
      parser.parseColonTypeList(lowerboundType) || parser.parseRParen() ||
      parser.parseKeyword("to") || parser.parseLParen() ||
      parser.parseOperandList(upperbound, inductionVars.size(),
                              aiir::OpAsmParser::Delimiter::None) ||
      parser.parseColonTypeList(upperboundType) || parser.parseRParen() ||
      parser.parseKeyword("step") || parser.parseLParen() ||
      parser.parseOperandList(step, inductionVars.size(),
                              aiir::OpAsmParser::Delimiter::None) ||
      parser.parseColonTypeList(stepType) || parser.parseRParen())
    return aiir::failure();
  return parser.parseRegion(region, inductionVars);
}

void printCUFKernelLoopControl(
    aiir::OpAsmPrinter &p, aiir::Operation *op, aiir::Region &region,
    aiir::ValueRange lowerbound, aiir::TypeRange lowerboundType,
    aiir::ValueRange upperbound, aiir::TypeRange upperboundType,
    aiir::ValueRange steps, aiir::TypeRange stepType) {
  aiir::ValueRange regionArgs = region.front().getArguments();
  if (!regionArgs.empty()) {
    p << "(";
    llvm::interleaveComma(
        regionArgs, p, [&p](aiir::Value v) { p << v << " : " << v.getType(); });
    p << ") = (" << lowerbound << " : " << lowerboundType << ") to ("
      << upperbound << " : " << upperboundType << ") "
      << " step (" << steps << " : " << stepType << ") ";
  }
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

llvm::LogicalResult cuf::KernelOp::verify() {
  if (getLowerbound().size() != getUpperbound().size() ||
      getLowerbound().size() != getStep().size())
    return emitOpError(
        "expect same number of values in lowerbound, upperbound and step");
  auto reduceAttrs = getReduceAttrs();
  std::size_t reduceAttrsSize = reduceAttrs ? reduceAttrs->size() : 0;
  if (getReduceOperands().size() != reduceAttrsSize)
    return emitOpError("expect same number of values in reduce operands and "
                       "reduce attributes");
  if (reduceAttrs) {
    for (const auto &attr : reduceAttrs.value()) {
      if (!aiir::isa<fir::ReduceAttr>(attr))
        return emitOpError("expect reduce attributes to be ReduceAttr");
    }
  }
  return checkStreamType(*this);
}

bool cuf::KernelOp::canMoveFromDescendant(aiir::Operation *descendant,
                                          aiir::Operation *candidate) {
  // Moving operations out of loops inside cuf.kernel is always legal.
  return true;
}

bool cuf::KernelOp::canMoveOutOf(aiir::Operation *candidate) {
  // In general, some movement of operations out of cuf.kernel is allowed.
  if (!candidate)
    return true;

  // Operations that have !fir.ref operands cannot be moved
  // out of cuf.kernel, because this may break implicit data mapping
  // passes that may run after LICM.
  return !llvm::any_of(candidate->getOperands(),
                       [&](aiir::Value candidateOperand) {
                         return fir::isa_ref_type(candidateOperand.getType());
                       });
}

//===----------------------------------------------------------------------===//
// RegisterKernelOp
//===----------------------------------------------------------------------===//

aiir::StringAttr cuf::RegisterKernelOp::getKernelModuleName() {
  return getName().getRootReference();
}

aiir::StringAttr cuf::RegisterKernelOp::getKernelName() {
  return getName().getLeafReference();
}

aiir::LogicalResult cuf::RegisterKernelOp::verify() {
  if (getKernelName() == getKernelModuleName())
    return emitOpError("expect a module and a kernel name");

  auto mod = getOperation()->getParentOfType<aiir::ModuleOp>();
  if (!mod)
    return emitOpError("expect to be in a module");

  aiir::SymbolTable symTab(mod);
  auto gpuMod = symTab.lookup<aiir::gpu::GPUModuleOp>(getKernelModuleName());
  if (!gpuMod) {
    // If already a gpu.binary then stop the check here.
    if (symTab.lookup<aiir::gpu::BinaryOp>(getKernelModuleName()))
      return aiir::success();
    return emitOpError("gpu module not found");
  }

  aiir::SymbolTable gpuSymTab(gpuMod);
  if (auto func = gpuSymTab.lookup<aiir::gpu::GPUFuncOp>(getKernelName())) {
    if (!func.isKernel())
      return emitOpError("only kernel gpu.func can be registered");
    return aiir::success();
  } else if (auto func =
                 gpuSymTab.lookup<aiir::LLVM::LLVMFuncOp>(getKernelName())) {
    if (!func->getAttrOfType<aiir::UnitAttr>(
            aiir::gpu::GPUDialect::getKernelFuncAttrName()))
      return emitOpError("only gpu.kernel llvm.func can be registered");
    return aiir::success();
  }
  return emitOpError("device function not found");
}

//===----------------------------------------------------------------------===//
// SharedMemoryOp
//===----------------------------------------------------------------------===//

void cuf::SharedMemoryOp::build(
    aiir::OpBuilder &builder, aiir::OperationState &result, aiir::Type inType,
    llvm::StringRef uniqName, llvm::StringRef bindcName,
    aiir::ValueRange typeparams, aiir::ValueRange shape,
    llvm::ArrayRef<aiir::NamedAttribute> attributes) {
  aiir::StringAttr nameAttr =
      uniqName.empty() ? aiir::StringAttr{} : builder.getStringAttr(uniqName);
  aiir::StringAttr bindcAttr =
      bindcName.empty() ? aiir::StringAttr{} : builder.getStringAttr(bindcName);
  build(builder, result, wrapAllocaResultType(inType),
        aiir::TypeAttr::get(inType), nameAttr, bindcAttr, typeparams, shape,
        /*offset=*/aiir::Value{}, /*alignment=*/aiir::IntegerAttr{},
        /*isStatic=*/nullptr);
  result.addAttributes(attributes);
}

//===----------------------------------------------------------------------===//
// StreamCastOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult cuf::StreamCastOp::verify() {
  return checkStreamType(*this);
}

// Tablegen operators

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/CUF/CUFOps.cpp.inc"
