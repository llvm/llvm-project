//===- CIRDialect.cpp - MLIR CIR ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::cir;

#include "clang/CIR/Dialect/IR/CIROpsEnums.cpp.inc"
#include "clang/CIR/Dialect/IR/CIROpsStructs.cpp.inc"

#include "clang/CIR/Dialect/IR/CIROpsDialect.cpp.inc"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//
namespace {
struct CIROpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    if (auto structType = type.dyn_cast<StructType>()) {
      // TODO(cir): generate unique alias names for anonymous records.
      if (!structType.getName())
        return AliasResult::NoAlias;
      os << "ty_" << structType.getName();
      return AliasResult::OverridableAlias;
    }
    if (auto intType = type.dyn_cast<IntType>()) {
      os << intType.getAlias();
      return AliasResult::OverridableAlias;
    }
    if (auto voidType = type.dyn_cast<VoidType>()) {
      os << voidType.getAlias();
      return AliasResult::OverridableAlias;
    }

    return AliasResult::NoAlias;
  }

  AliasResult getAlias(Attribute attr, raw_ostream &os) const final {
    if (auto boolAttr = attr.dyn_cast<mlir::cir::BoolAttr>()) {
      os << (boolAttr.getValue() ? "true" : "false");
      return AliasResult::FinalAlias;
    }

    return AliasResult::NoAlias;
  }
};
} // namespace

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void cir::CIRDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "clang/CIR/Dialect/IR/CIROps.cpp.inc"
      >();
  addInterfaces<CIROpAsmDialectInterface>();
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Parses one of the keywords provided in the list `keywords` and returns the
// position of the parsed keyword in the list. If none of the keywords from the
// list is parsed, returns -1.
static int parseOptionalKeywordAlternative(AsmParser &parser,
                                           ArrayRef<StringRef> keywords) {
  for (auto en : llvm::enumerate(keywords)) {
    if (succeeded(parser.parseOptionalKeyword(en.value())))
      return en.index();
  }
  return -1;
}

namespace {
template <typename Ty> struct EnumTraits {};

#define REGISTER_ENUM_TYPE(Ty)                                                 \
  template <> struct EnumTraits<Ty> {                                          \
    static StringRef stringify(Ty value) { return stringify##Ty(value); }      \
    static unsigned getMaxEnumVal() { return getMaxEnumValFor##Ty(); }         \
  }
#define REGISTER_ENUM_TYPE_WITH_NS(NS, Ty)                                     \
  template <> struct EnumTraits<NS::Ty> {                                      \
    static StringRef stringify(NS::Ty value) {                                 \
      return NS::stringify##Ty(value);                                         \
    }                                                                          \
    static unsigned getMaxEnumVal() { return NS::getMaxEnumValFor##Ty(); }     \
  }

REGISTER_ENUM_TYPE(GlobalLinkageKind);
REGISTER_ENUM_TYPE_WITH_NS(sob, SignedOverflowBehavior);
} // namespace

/// Parse an enum from the keyword, or default to the provided default value.
/// The return type is the enum type by default, unless overriden with the
/// second template argument.
/// TODO: teach other places in this file to use this function.
template <typename EnumTy, typename RetTy = EnumTy>
static RetTy parseOptionalCIRKeyword(AsmParser &parser, EnumTy defaultValue) {
  SmallVector<StringRef, 10> names;
  for (unsigned i = 0, e = EnumTraits<EnumTy>::getMaxEnumVal(); i <= e; ++i)
    names.push_back(EnumTraits<EnumTy>::stringify(static_cast<EnumTy>(i)));

  int index = parseOptionalKeywordAlternative(parser, names);
  if (index == -1)
    return static_cast<RetTy>(defaultValue);
  return static_cast<RetTy>(index);
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

void AllocaOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState, ::mlir::Type addr,
                     ::mlir::Type allocaType, ::llvm::StringRef name,
                     ::mlir::IntegerAttr alignment) {
  odsState.addAttribute(getAllocaTypeAttrName(odsState.name),
                        ::mlir::TypeAttr::get(allocaType));
  odsState.addAttribute(getNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(name));
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addTypes(addr);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult checkConstantTypes(mlir::Operation *op, mlir::Type opType,
                                        mlir::Attribute attrType) {
  if (attrType.isa<ConstPtrAttr>()) {
    if (opType.isa<::mlir::cir::PointerType>())
      return success();
    return op->emitOpError("nullptr expects pointer type");
  }

  if (attrType.isa<ZeroAttr>()) {
    if (opType.isa<::mlir::cir::StructType, ::mlir::cir::ArrayType>())
      return success();
    return op->emitOpError("zero expects struct or array type");
  }

  if (attrType.isa<mlir::cir::BoolAttr>()) {
    if (!opType.isa<mlir::cir::BoolType>())
      return op->emitOpError("result type (")
             << opType << ") must be '!cir.bool' for '" << attrType << "'";
    return success();
  }

  if (attrType.isa<IntegerAttr, FloatAttr>()) {
    auto at = attrType.cast<TypedAttr>();
    if (at.getType() != opType) {
      return op->emitOpError("result type (")
             << opType << ") does not match value type (" << at.getType()
             << ")";
    }
    return success();
  }

  if (attrType.isa<SymbolRefAttr>()) {
    if (opType.isa<::mlir::cir::PointerType>())
      return success();
    return op->emitOpError("symbolref expects pointer type");
  }

  if (attrType.isa<mlir::cir::GlobalViewAttr>() ||
      attrType.isa<mlir::cir::TypeInfoAttr>() ||
      attrType.isa<mlir::cir::ConstArrayAttr>() ||
      attrType.isa<mlir::cir::ConstStructAttr>() ||
      attrType.isa<mlir::cir::VTableAttr>())
    return success();
  if (attrType.isa<mlir::cir::IntAttr>())
    return success();

  assert(attrType.isa<TypedAttr>() && "What else could we be looking at here?");
  return op->emitOpError("global with type ")
         << attrType.cast<TypedAttr>().getType() << " not supported";
}

LogicalResult ConstantOp::verify() {
  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  return checkConstantTypes(getOperation(), getType(), getValue());
}

static ParseResult parseConstantValue(OpAsmParser &parser,
                                      mlir::Attribute &valueAttr) {
  NamedAttrList attr;
  if (parser.parseAttribute(valueAttr, "value", attr).failed()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected constant attribute to match type");
  }

  return success();
}

// FIXME: create a CIRConstAttr and hide this away for both global
// initialization and cir.const operation.
static void printConstant(OpAsmPrinter &p, Attribute value) {
  p.printAttribute(value);
}

static void printConstantValue(OpAsmPrinter &p, cir::ConstantOp op,
                               Attribute value) {
  printConstant(p, value);
}

OpFoldResult ConstantOp::fold(FoldAdaptor /*adaptor*/) { return getValue(); }

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  auto resType = getResult().getType();
  auto srcType = getSrc().getType();

  switch (getKind()) {
  case cir::CastKind::int_to_bool: {
    if (!resType.isa<mlir::cir::BoolType>())
      return emitOpError() << "requires !cir.bool type for result";
    if (!srcType.isa<mlir::cir::IntType>())
      return emitOpError() << "requires integral type for result";
    return success();
  }
  case cir::CastKind::ptr_to_bool: {
    if (!resType.isa<mlir::cir::BoolType>())
      return emitOpError() << "requires !cir.bool type for result";
    if (!srcType.isa<mlir::cir::PointerType>())
      return emitOpError() << "requires pointer type for result";
    return success();
  }
  case cir::CastKind::integral: {
    if (!resType.isa<mlir::cir::IntType>())
      return emitOpError() << "requires !IntegerType for result";
    if (!srcType.isa<mlir::cir::IntType>())
      return emitOpError() << "requires !IntegerType for source";
    return success();
  }
  case cir::CastKind::array_to_ptrdecay: {
    auto arrayPtrTy = srcType.dyn_cast<mlir::cir::PointerType>();
    auto flatPtrTy = resType.dyn_cast<mlir::cir::PointerType>();
    if (!arrayPtrTy || !flatPtrTy)
      return emitOpError() << "requires !cir.ptr type for source and result";

    auto arrayTy = arrayPtrTy.getPointee().dyn_cast<mlir::cir::ArrayType>();
    if (!arrayTy)
      return emitOpError() << "requires !cir.array pointee";

    if (arrayTy.getEltType() != flatPtrTy.getPointee())
      return emitOpError()
             << "requires same type for array element and pointee result";
    return success();
  }
  case cir::CastKind::bitcast: {
    if (!srcType.dyn_cast<mlir::cir::PointerType>() ||
        !resType.dyn_cast<mlir::cir::PointerType>())
      return emitOpError() << "requires !cir.ptr type for source and result";
    return success();
  }
  case cir::CastKind::floating: {
    if (!srcType.dyn_cast<mlir::FloatType>() ||
        !resType.dyn_cast<mlir::FloatType>())
      return emitOpError() << "requries floating for source and result";
    return success();
  }
  case cir::CastKind::float_to_int: {
    if (!srcType.dyn_cast<mlir::FloatType>())
      return emitOpError() << "requires floating for source";
    if (!resType.dyn_cast<mlir::cir::IntType>())
      return emitOpError() << "requires !IntegerType for result";
    return success();
  }
  case cir::CastKind::int_to_ptr: {
    if (!srcType.dyn_cast<mlir::cir::IntType>())
      return emitOpError() << "requires integer for source";
    if (!resType.dyn_cast<mlir::cir::PointerType>())
      return emitOpError() << "requires pointer for result";
    return success();
  }
  case cir::CastKind::ptr_to_int: {
    if (!srcType.dyn_cast<mlir::cir::PointerType>())
      return emitOpError() << "requires pointer for source";
    if (!resType.dyn_cast<mlir::cir::IntType>())
      return emitOpError() << "requires integer for result";
    return success();
  }
  case cir::CastKind::float_to_bool: {
    if (!srcType.isa<mlir::FloatType>())
      return emitOpError() << "requires float for source";
    if (!resType.isa<mlir::cir::BoolType>())
      return emitOpError() << "requires !cir.bool for result";
    return success();
  }
  case cir::CastKind::bool_to_int: {
    if (!srcType.isa<mlir::cir::BoolType>())
      return emitOpError() << "requires !cir.bool for source";
    if (!resType.isa<mlir::cir::IntType>())
      return emitOpError() << "requires !cir.int for result";
    return success();
  }
  case cir::CastKind::int_to_float:
    if (!srcType.isa<mlir::cir::IntType>())
      return emitOpError() << "requires !cir.int for source";
    if (!resType.isa<mlir::FloatType>())
      return emitOpError() << "requires !cir.float for result";
    return success();
  }

  llvm_unreachable("Unknown CastOp kind?");
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult checkReturnAndFunction(ReturnOp op,
                                                  cir::FuncOp function) {
  // ReturnOps currently only have a single optional operand.
  if (op.getNumOperands() > 1)
    return op.emitOpError() << "expects at most 1 return operand";

  // Ensure returned type matches the function signature.
  auto expectedTy = function.getFunctionType().getReturnType();
  auto actualTy =
      (op.getNumOperands() == 0 ? mlir::cir::VoidType::get(op.getContext())
                                : op.getOperand(0).getType());
  if (actualTy != expectedTy)
    return op.emitOpError() << "returns " << actualTy
                            << " but enclosing function returns " << expectedTy;

  return mlir::success();
}

mlir::LogicalResult ReturnOp::verify() {
  // Returns can be present in multiple different scopes, get the
  // wrapping function and start from there.
  auto *fnOp = getOperation()->getParentOp();
  while (!isa<cir::FuncOp>(fnOp))
    fnOp = fnOp->getParentOp();

  // Make sure return types match function return type.
  if (checkReturnAndFunction(*this, cast<cir::FuncOp>(fnOp)).failed())
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ThrowOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ThrowOp::verify() {
  // For the no-rethrow version, it must have at least the exception pointer.
  if (rethrows())
    return success();

  if (getNumOperands() == 1) {
    if (!getTypeInfo())
      return emitOpError() << "'type_info' symbol attribute missing";
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

static LogicalResult checkBlockTerminator(OpAsmParser &parser,
                                          llvm::SMLoc parserLoc,
                                          std::optional<Location> l, Region *r,
                                          bool ensureTerm = true) {
  mlir::Builder &builder = parser.getBuilder();
  if (r->hasOneBlock()) {
    if (ensureTerm) {
      ::mlir::impl::ensureRegionTerminator(
          *r, builder, *l, [](OpBuilder &builder, Location loc) {
            OperationState state(loc, YieldOp::getOperationName());
            YieldOp::build(builder, state);
            return Operation::create(state);
          });
    } else {
      assert(r && "region must not be empty");
      Block &block = r->back();
      if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>()) {
        return parser.emitError(
            parser.getCurrentLocation(),
            "blocks are expected to be explicitly terminated");
      }
    }
    return success();
  }

  // Empty regions don't need any handling.
  auto &blocks = r->getBlocks();
  if (blocks.empty())
    return success();

  // Test that at least one block has a yield/return/throw terminator. We can
  // probably make this a bit more strict.
  for (Block &block : blocks) {
    if (block.empty())
      continue;
    auto &op = block.back();
    if (op.hasTrait<mlir::OpTrait::IsTerminator>() &&
        isa<YieldOp, ReturnOp, ThrowOp>(op)) {
      return success();
    }
  }

  parser.emitError(parserLoc,
                   "expected at least one block with cir.yield or cir.return");
  return failure();
}

ParseResult cir::IfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  Type boolType = ::mlir::cir::BoolType::get(builder.getContext());

  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, boolType, result.operands))
    return failure();

  // Parse the 'then' region.
  auto parseThenLoc = parser.getCurrentLocation();
  if (parser.parseRegion(*thenRegion, /*arguments=*/{},
                         /*argTypes=*/{}))
    return failure();
  if (checkBlockTerminator(parser, parseThenLoc, result.location, thenRegion)
          .failed())
    return failure();

  // If we find an 'else' keyword, parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    auto parseElseLoc = parser.getCurrentLocation();
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    if (checkBlockTerminator(parser, parseElseLoc, result.location, elseRegion)
            .failed())
      return failure();
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

bool shouldPrintTerm(mlir::Region &r) {
  if (!r.hasOneBlock())
    return true;
  auto *entryBlock = &r.front();
  if (entryBlock->empty())
    return false;
  if (isa<ReturnOp>(entryBlock->back()))
    return true;
  if (isa<ThrowOp>(entryBlock->back()))
    return true;
  YieldOp y = dyn_cast<YieldOp>(entryBlock->back());
  if (y && (!y.isPlain() || !y.getArgs().empty()))
    return true;
  return false;
}

void cir::IfOp::print(OpAsmPrinter &p) {
  p << " " << getCondition() << " ";
  auto &thenRegion = this->getThenRegion();
  p.printRegion(thenRegion,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/shouldPrintTerm(thenRegion));

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = this->getElseRegion();
  if (!elseRegion.empty()) {
    p << " else ";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/shouldPrintTerm(elseRegion));
  }

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

/// Default callback for IfOp builders. Inserts nothing for now.
void mlir::cir::buildTerminatedBody(OpBuilder &builder, Location loc) {}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void IfOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->getElseRegion();
  if (elseRegion->empty())
    elseRegion = nullptr;

  // Otherwise, the successor is dependent on the condition.
  // bool condition;
  // if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
  //   assert(0 && "not implemented");
  // condition = condAttr.getValue().isOneValue();
  // Add the successor regions using the condition.
  // regions.push_back(RegionSuccessor(condition ? &thenRegion() :
  // elseRegion));
  // return;
  // }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getThenRegion()));
  // If the else region does not exist, it is not a viable successor.
  if (elseRegion)
    regions.push_back(RegionSuccessor(elseRegion));
  return;
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 bool withElseRegion,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  assert(thenBuilder && "the builder callback for 'then' must be present");

  result.addOperands(cond);

  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  Region *elseRegion = result.addRegion();
  if (!withElseRegion)
    return;

  builder.createBlock(elseRegion);
  elseBuilder(builder, result.location);
}

LogicalResult IfOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

ParseResult cir::ScopeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create one region within 'scope'.
  result.regions.reserve(1);
  Region *scopeRegion = result.addRegion();
  auto loc = parser.getCurrentLocation();

  // Parse the scope region.
  if (parser.parseRegion(*scopeRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  if (checkBlockTerminator(parser, loc, result.location, scopeRegion).failed())
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void cir::ScopeOp::print(OpAsmPrinter &p) {
  p << ' ';
  auto &scopeRegion = this->getScopeRegion();
  p.printRegion(scopeRegion,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/shouldPrintTerm(scopeRegion));

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void ScopeOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  // The only region always branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getScopeRegion()));
}

void ScopeOp::build(
    OpBuilder &builder, OperationState &result,
    function_ref<void(OpBuilder &, Type &, Location)> scopeBuilder) {
  assert(scopeBuilder && "the builder callback for 'then' must be present");

  OpBuilder::InsertionGuard guard(builder);
  Region *scopeRegion = result.addRegion();
  builder.createBlock(scopeRegion);

  mlir::Type yieldTy;
  scopeBuilder(builder, yieldTy, result.location);

  if (yieldTy)
    result.addTypes(TypeRange{yieldTy});
}

void ScopeOp::build(OpBuilder &builder, OperationState &result,
                    function_ref<void(OpBuilder &, Location)> scopeBuilder) {
  assert(scopeBuilder && "the builder callback for 'then' must be present");
  OpBuilder::InsertionGuard guard(builder);
  Region *scopeRegion = result.addRegion();
  builder.createBlock(scopeRegion);
  scopeBuilder(builder, result.location);
}

LogicalResult ScopeOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

void TryOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                SmallVectorImpl<RegionSuccessor> &regions) {
  // The only region always branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(this->getODSResults(0)));
    return;
  }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getBody()));
}

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void TernaryOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  // The `true` and the `false` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(this->getODSResults(0)));
    return;
  }

  // Try optimize if we have more information
  // if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
  //   assert(0 && "not implemented");
  // }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getTrueRegion()));
  regions.push_back(RegionSuccessor(&getFalseRegion()));
  return;
}

void TernaryOp::build(OpBuilder &builder, OperationState &result, Value cond,
                      function_ref<void(OpBuilder &, Location)> trueBuilder,
                      function_ref<void(OpBuilder &, Location)> falseBuilder) {
  result.addOperands(cond);
  OpBuilder::InsertionGuard guard(builder);
  Region *trueRegion = result.addRegion();
  auto *block = builder.createBlock(trueRegion);
  trueBuilder(builder, result.location);
  Region *falseRegion = result.addRegion();
  builder.createBlock(falseRegion);
  falseBuilder(builder, result.location);

  auto yield = dyn_cast<YieldOp>(block->getTerminator());
  assert((yield && yield.getNumOperands() <= 1) &&
         "expected zero or one result type");
  if (yield.getNumOperands() == 1)
    result.addTypes(TypeRange{yield.getOperandTypes().front()});
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult YieldOp::verify() {
  auto isDominatedByLoopOrSwitch = [&](Operation *parentOp) {
    while (!llvm::isa<cir::FuncOp>(parentOp)) {
      if (llvm::isa<cir::SwitchOp, cir::LoopOp>(parentOp))
        return true;
      parentOp = parentOp->getParentOp();
    }

    emitOpError() << "shall be dominated by 'cir.loop' or 'cir.switch'";
    return false;
  };

  auto isDominatedByProperAwaitRegion = [&](Operation *parentOp,
                                            mlir::Region *currRegion) {
    while (!llvm::isa<cir::FuncOp>(parentOp)) {
      auto awaitOp = dyn_cast<cir::AwaitOp>(parentOp);
      if (awaitOp) {
        if (currRegion && currRegion == &awaitOp.getResume()) {
          emitOpError() << "kind 'nosuspend' can only be used in 'ready' and "
                           "'suspend' regions";
          return false;
        }
        return true;
      }

      currRegion = parentOp->getParentRegion();
      parentOp = parentOp->getParentOp();
    }

    emitOpError() << "shall be dominated by 'cir.await'";
    return false;
  };

  auto isDominatedByLoop = [](Operation *parentOp) {
    while (!llvm::isa<cir::FuncOp>(parentOp)) {
      if (llvm::isa<cir::LoopOp>(parentOp))
        return true;
      parentOp = parentOp->getParentOp();
    }
    return false;
  };

  if (isNoSuspend()) {
    if (!isDominatedByProperAwaitRegion(getOperation()->getParentOp(),
                                        getOperation()->getParentRegion()))
      return mlir::failure();
    return mlir::success();
  }

  if (isBreak()) {
    if (!isDominatedByLoopOrSwitch(getOperation()->getParentOp()))
      return mlir::failure();
    return mlir::success();
  }

  if (isContinue()) {
    if (!isDominatedByLoop(getOperation()->getParentOp()))
      return emitOpError() << "shall be dominated by 'cir.loop'";
    return mlir::success();
  }

  if (isFallthrough()) {
    if (!llvm::isa<SwitchOp>(getOperation()->getParentOp()))
      return emitOpError() << "fallthrough only expected within 'cir.switch'";
    return mlir::success();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BrOp
//===----------------------------------------------------------------------===//

mlir::SuccessorOperands BrOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return mlir::SuccessorOperands(getDestOperandsMutable());
}

Block *BrOp::getSuccessorForOperands(ArrayRef<Attribute>) { return getDest(); }

//===----------------------------------------------------------------------===//
// BrCondOp
//===----------------------------------------------------------------------===//

mlir::SuccessorOperands BrCondOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getDestOperandsTrueMutable()
                                      : getDestOperandsFalseMutable());
}

Block *BrCondOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  if (IntegerAttr condAttr = operands.front().dyn_cast_or_null<IntegerAttr>())
    return condAttr.getValue().isOne() ? getDestTrue() : getDestFalse();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

ParseResult
parseSwitchOp(OpAsmParser &parser,
              llvm::SmallVectorImpl<std::unique_ptr<::mlir::Region>> &regions,
              ::mlir::ArrayAttr &casesAttr,
              mlir::OpAsmParser::UnresolvedOperand &cond,
              mlir::Type &condType) {
  mlir::cir::IntType intCondType;
  SmallVector<mlir::Attribute, 4> cases;

  auto parseAndCheckRegion = [&]() -> ParseResult {
    // Parse region attached to case
    regions.emplace_back(new Region);
    Region &currRegion = *regions.back().get();
    auto parserLoc = parser.getCurrentLocation();
    if (parser.parseRegion(currRegion, /*arguments=*/{}, /*argTypes=*/{})) {
      regions.clear();
      return failure();
    }

    if (currRegion.empty()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "case region shall not be empty");
    }

    if (checkBlockTerminator(parser, parserLoc, std::nullopt, &currRegion,
                             /*ensureTerm=*/false)
            .failed())
      return failure();
    return success();
  };

  auto parseCase = [&]() -> ParseResult {
    auto loc = parser.getCurrentLocation();
    if (parser.parseKeyword("case").failed())
      return parser.emitError(loc, "expected 'case' keyword here");

    if (parser.parseLParen().failed())
      return parser.emitError(parser.getCurrentLocation(), "expected '('");

    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;

    //   case (equal, 20) {
    //   ...
    // 1. Get the case kind
    // 2. Get the value (next in list)

    // These needs to be in sync with CIROps.td
    if (parser.parseOptionalKeyword(&attrStr, {"default", "equal", "anyof"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult = parser.parseOptionalAttribute(
          attrVal, parser.getBuilder().getNoneType(), "kind", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      }
    }

    if (attrStr.empty()) {
      return parser.emitError(
          loc, "expected string or keyword containing one of the following "
               "enum values for attribute 'kind' [default, equal, anyof]");
    }

    auto attrOptional = ::mlir::cir::symbolizeCaseOpKind(attrStr.str());
    if (!attrOptional)
      return parser.emitError(loc, "invalid ")
             << "kind attribute specification: \"" << attrStr << '"';

    auto kindAttr = ::mlir::cir::CaseOpKindAttr::get(
        parser.getBuilder().getContext(), attrOptional.value());

    // `,` value or `,` [values,...]
    SmallVector<mlir::Attribute, 4> caseEltValueListAttr;
    mlir::ArrayAttr caseValueList;

    switch (kindAttr.getValue()) {
    case cir::CaseOpKind::Equal: {
      if (parser.parseComma().failed())
        return mlir::failure();
      int64_t val = 0;
      if (parser.parseInteger(val).failed())
        return ::mlir::failure();
      caseEltValueListAttr.push_back(mlir::cir::IntAttr::get(intCondType, val));
      break;
    }
    case cir::CaseOpKind::Anyof: {
      if (parser.parseComma().failed())
        return mlir::failure();
      if (parser.parseLSquare().failed())
        return mlir::failure();
      if (parser.parseCommaSeparatedList([&]() {
            int64_t val = 0;
            if (parser.parseInteger(val).failed())
              return ::mlir::failure();
            caseEltValueListAttr.push_back(
                mlir::cir::IntAttr::get(intCondType, val));
            return ::mlir::success();
          }))
        return mlir::failure();
      if (parser.parseRSquare().failed())
        return mlir::failure();
      break;
    }
    case cir::CaseOpKind::Default: {
      if (parser.parseRParen().failed())
        return parser.emitError(parser.getCurrentLocation(), "expected ')'");
      cases.push_back(cir::CaseAttr::get(
          parser.getContext(), parser.getBuilder().getArrayAttr({}), kindAttr));
      return parseAndCheckRegion();
    }
    }

    caseValueList = parser.getBuilder().getArrayAttr(caseEltValueListAttr);
    cases.push_back(
        cir::CaseAttr::get(parser.getContext(), caseValueList, kindAttr));
    if (succeeded(parser.parseOptionalColon())) {
      Type caseIntTy;
      if (parser.parseType(caseIntTy).failed())
        return parser.emitError(parser.getCurrentLocation(), "expected type");
      if (intCondType != caseIntTy)
        return parser.emitError(parser.getCurrentLocation(),
                                "expected a match with the condition type");
    }
    if (parser.parseRParen().failed())
      return parser.emitError(parser.getCurrentLocation(), "expected ')'");
    return parseAndCheckRegion();
  };

  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseOperand(cond))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseCustomTypeWithFallback(intCondType))
    return ::mlir::failure();
  condType = intCondType;
  if (parser.parseRParen())
    return ::mlir::failure();

  if (parser
          .parseCommaSeparatedList(OpAsmParser::Delimiter::Square, parseCase,
                                   " in cases list")
          .failed())
    return failure();

  casesAttr = parser.getBuilder().getArrayAttr(cases);
  return ::mlir::success();
}

void printSwitchOp(OpAsmPrinter &p, SwitchOp op,
                   mlir::MutableArrayRef<::mlir::Region> regions,
                   mlir::ArrayAttr casesAttr, mlir::Value condition,
                   mlir::Type condType) {
  int idx = 0, lastIdx = regions.size() - 1;

  p << "(";
  p << condition;
  p << " : ";
  p.printStrippedAttrOrType(condType);
  p << ") [";
  // FIXME: ideally we want some extra indentation for "cases" but too
  // cumbersome to pull it out now, since most handling is private. Perhaps
  // better improve overall mechanism.
  p.printNewline();
  for (auto &r : regions) {
    p << "case (";

    auto attr = casesAttr[idx].cast<CaseAttr>();
    auto kind = attr.getKind().getValue();
    assert((kind == CaseOpKind::Default || kind == CaseOpKind::Equal ||
            kind == CaseOpKind::Anyof) &&
           "unknown case");

    // Case kind
    p << stringifyCaseOpKind(kind);

    // Case value
    switch (kind) {
    case cir::CaseOpKind::Equal: {
      p << ", ";
      auto intAttr = attr.getValue()[0].cast<cir::IntAttr>();
      auto intAttrTy = intAttr.getType().cast<cir::IntType>();
      (intAttrTy.isSigned() ? p << intAttr.getSInt() : p << intAttr.getUInt());
      break;
    }
    case cir::CaseOpKind::Anyof: {
      p << ", [";
      llvm::interleaveComma(attr.getValue(), p, [&](const Attribute &a) {
        auto intAttr = a.cast<cir::IntAttr>();
        auto intAttrTy = intAttr.getType().cast<cir::IntType>();
        (intAttrTy.isSigned() ? p << intAttr.getSInt()
                              : p << intAttr.getUInt());
      });
      p << "] : ";
      auto typedAttr = attr.getValue()[0].dyn_cast<TypedAttr>();
      assert(typedAttr && "this should never not have a type!");
      p.printType(typedAttr.getType());
      break;
    }
    case cir::CaseOpKind::Default:
      break;
    }

    p << ") ";
    p.printRegion(r, /*printEntryBLockArgs=*/false,
                  /*printBlockTerminators=*/true);
    if (idx < lastIdx)
      p << ",";
    p.printNewline();
    idx++;
  }
  p << "]";
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes
/// that correspond to a constant value for each operand, or null if that
/// operand is not a constant.
void SwitchOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  // If any index all the underlying regions branch back to the parent
  // operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // for (auto &r : this->getRegions()) {
  // If we can figure out the case stmt we are landing, this can be
  // overly simplified.
  // bool condition;
  // if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
  //   assert(0 && "not implemented");
  //   (void)r;
  // condition = condAttr.getValue().isOneValue();
  // Add the successor regions using the condition.
  // regions.push_back(RegionSuccessor(condition ? &thenRegion() :
  // elseRegion));
  // return;
  // }
  // }

  // If the condition isn't constant, all regions may be executed.
  for (auto &r : this->getRegions())
    regions.push_back(RegionSuccessor(&r));
}

LogicalResult SwitchOp::verify() {
  if (getCases().has_value() && getCases()->size() != getNumRegions())
    return emitOpError("number of cases attributes and regions must match");
  return success();
}

void SwitchOp::build(
    OpBuilder &builder, OperationState &result, Value cond,
    function_ref<void(OpBuilder &, Location, OperationState &)> switchBuilder) {
  assert(switchBuilder && "the builder callback for regions must be present");
  OpBuilder::InsertionGuard guardSwitch(builder);
  result.addOperands({cond});
  switchBuilder(builder, result.location, result);
}

//===----------------------------------------------------------------------===//
// CatchOp
//===----------------------------------------------------------------------===//

ParseResult
parseCatchOp(OpAsmParser &parser,
             llvm::SmallVectorImpl<std::unique_ptr<::mlir::Region>> &regions,
             ::mlir::ArrayAttr &catchersAttr) {
  SmallVector<mlir::Attribute, 4> catchList;

  auto parseAndCheckRegion = [&]() -> ParseResult {
    // Parse region attached to catch
    regions.emplace_back(new Region);
    Region &currRegion = *regions.back().get();
    auto parserLoc = parser.getCurrentLocation();
    if (parser.parseRegion(currRegion, /*arguments=*/{}, /*argTypes=*/{})) {
      regions.clear();
      return failure();
    }

    if (currRegion.empty()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "catch region shall not be empty");
    }

    if (checkBlockTerminator(parser, parserLoc, std::nullopt, &currRegion,
                             /*ensureTerm=*/false)
            .failed())
      return failure();
    return success();
  };

  auto parseCatchEntry = [&]() -> ParseResult {
    mlir::Type exceptionType;
    mlir::Attribute exceptionTypeInfo;

    // cir.catch(..., [
    //   type (!cir.ptr<!u8i>, @type_info_char_star) {
    //     ...
    //   },
    //   all {
    //     ...
    //   }
    // ]
    ::llvm::StringRef attrStr;
    if (!parser.parseOptionalKeyword(&attrStr, {"all"})) {
      if (parser.parseKeyword("type").failed())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected 'type' keyword here");

      if (parser.parseLParen().failed())
        return parser.emitError(parser.getCurrentLocation(), "expected '('");

      if (parser.parseType(exceptionType).failed())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected valid exception type");
      if (parser.parseAttribute(exceptionTypeInfo).failed())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected valid RTTI info attribute");
      if (parser.parseRParen().failed())
        return parser.emitError(parser.getCurrentLocation(), "expected ')'");
    }
    catchList.push_back(mlir::cir::CatchEntryAttr::get(
        parser.getContext(), exceptionType, exceptionTypeInfo));

    return parseAndCheckRegion();
  };

  if (parser
          .parseCommaSeparatedList(OpAsmParser::Delimiter::Square,
                                   parseCatchEntry, " in catch list")
          .failed())
    return failure();

  catchersAttr = parser.getBuilder().getArrayAttr(catchList);
  return ::mlir::success();
}

void printCatchOp(OpAsmPrinter &p, CatchOp op,
                  mlir::MutableArrayRef<::mlir::Region> regions,
                  mlir::ArrayAttr catchList) {

  int currCatchIdx = 0;
  p << "[";
  llvm::interleaveComma(catchList, p, [&](const Attribute &a) {
    p.printNewline();
    p.increaseIndent();
    auto attr = a.cast<CatchEntryAttr>();
    auto exType = attr.getExceptionType();
    auto exRtti = attr.getExceptionTypeInfo();

    if (!exType) {
      p << "all";
    } else {
      p << "type (";
      p.printType(exType);
      p << ", ";
      p.printAttribute(exRtti);
      p << ") ";
    }
    p.printNewline();
    p.increaseIndent();
    p.printRegion(regions[currCatchIdx], /*printEntryBLockArgs=*/false,
                  /*printBlockTerminators=*/true);
    currCatchIdx++;
    p.decreaseIndent();
    p.decreaseIndent();
  });
  p << "]";
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes
/// that correspond to a constant value for each operand, or null if that
/// operand is not a constant.
void CatchOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  // If any index all the underlying regions branch back to the parent
  // operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // FIXME: optimize, ideas include:
  // - If we know a target function never throws a specific type, we can
  //   remove the catch handler.
  // - ???

  // If the condition isn't constant, all regions may be executed.
  for (auto &r : this->getRegions())
    regions.push_back(RegionSuccessor(&r));
}

void CatchOp::build(
    OpBuilder &builder, OperationState &result,
    function_ref<void(OpBuilder &, Location, OperationState &)> catchBuilder) {
  assert(catchBuilder && "the builder callback for regions must be present");
  OpBuilder::InsertionGuard guardCatch(builder);
  catchBuilder(builder, result.location, result);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

void LoopOp::build(OpBuilder &builder, OperationState &result,
                   cir::LoopOpKind kind,
                   function_ref<void(OpBuilder &, Location)> condBuilder,
                   function_ref<void(OpBuilder &, Location)> bodyBuilder,
                   function_ref<void(OpBuilder &, Location)> stepBuilder) {
  OpBuilder::InsertionGuard guard(builder);
  ::mlir::cir::LoopOpKindAttr kindAttr =
      cir::LoopOpKindAttr::get(builder.getContext(), kind);
  result.addAttribute(getKindAttrName(result.name), kindAttr);

  Region *condRegion = result.addRegion();
  builder.createBlock(condRegion);
  condBuilder(builder, result.location);

  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion);
  bodyBuilder(builder, result.location);

  Region *stepRegion = result.addRegion();
  builder.createBlock(stepRegion);
  stepBuilder(builder, result.location);
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes
/// that correspond to a constant value for each operand, or null if that
/// operand is not a constant.
void LoopOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  // If any index all the underlying regions branch back to the parent
  // operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // FIXME: we want to look at cond region for getting more accurate results
  // if the other regions will get a chance to execute.
  regions.push_back(RegionSuccessor(&this->getCond()));
  regions.push_back(RegionSuccessor(&this->getBody()));
  regions.push_back(RegionSuccessor(&this->getStep()));
}

llvm::SmallVector<Region *> LoopOp::getLoopRegions() { return {&getBody()}; }

LogicalResult LoopOp::verify() {
  // Cond regions should only terminate with plain 'cir.yield' or
  // 'cir.yield continue'.
  auto terminateError = [&]() {
    return emitOpError() << "cond region must be terminated with "
                            "'cir.yield' or 'cir.yield continue'";
  };

  auto &blocks = getCond().getBlocks();
  for (Block &block : blocks) {
    if (block.empty())
      continue;
    auto &op = block.back();
    if (isa<BrCondOp>(op))
      continue;
    if (!isa<YieldOp>(op))
      terminateError();
    auto y = cast<YieldOp>(op);
    if (!(y.isPlain() || y.isContinue()))
      terminateError();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static void printGlobalOpTypeAndInitialValue(OpAsmPrinter &p, GlobalOp op,
                                             TypeAttr type, Attribute initAttr,
                                             mlir::Region &ctorRegion,
                                             mlir::Region &dtorRegion) {
  auto printType = [&]() { p << ": " << type; };
  if (!op.isDeclaration()) {
    p << "= ";
    if (!ctorRegion.empty()) {
      p << "ctor ";
      printType();
      p << " ";
      p.printRegion(ctorRegion,
                    /*printEntryBlockArgs=*/false,
                    /*printBlockTerminators=*/false);
    } else {
      // This also prints the type...
      if (initAttr)
        printConstant(p, initAttr);
    }

    if (!dtorRegion.empty()) {
      p << " dtor ";
      p.printRegion(dtorRegion,
                    /*printEntryBlockArgs=*/false,
                    /*printBlockTerminators=*/false);
    }
  } else {
    printType();
  }
}

static ParseResult parseGlobalOpTypeAndInitialValue(OpAsmParser &parser,
                                                    TypeAttr &typeAttr,
                                                    Attribute &initialValueAttr,
                                                    mlir::Region &ctorRegion,
                                                    mlir::Region &dtorRegion) {
  mlir::Type opTy;
  if (parser.parseOptionalEqual().failed()) {
    // Absence of equal means a declaration, so we need to parse the type.
    //  cir.global @a : i32
    if (parser.parseColonType(opTy))
      return failure();
  } else {
    // Parse contructor, example:
    //  cir.global @rgb = ctor : type { ... }
    if (!parser.parseOptionalKeyword("ctor")) {
      if (parser.parseColonType(opTy))
        return failure();
      auto parseLoc = parser.getCurrentLocation();
      if (parser.parseRegion(ctorRegion, /*arguments=*/{}, /*argTypes=*/{}))
        return failure();
      if (!ctorRegion.hasOneBlock())
        return parser.emitError(parser.getCurrentLocation(),
                                "ctor region must have exactly one block");
      if (ctorRegion.back().empty())
        return parser.emitError(parser.getCurrentLocation(),
                                "ctor region shall not be empty");
      if (checkBlockTerminator(parser, parseLoc,
                               ctorRegion.back().back().getLoc(), &ctorRegion)
              .failed())
        return failure();
    } else {
      // Parse constant with initializer, examples:
      //  cir.global @y = 3.400000e+00 : f32
      //  cir.global @rgb = #cir.const_array<[...] : !cir.array<i8 x 3>>
      if (parseConstantValue(parser, initialValueAttr).failed())
        return failure();

      assert(initialValueAttr.isa<mlir::TypedAttr>() &&
             "Non-typed attrs shouldn't appear here.");
      auto typedAttr = initialValueAttr.cast<mlir::TypedAttr>();
      opTy = typedAttr.getType();
    }

    // Parse destructor, example:
    //   dtor { ... }
    if (!parser.parseOptionalKeyword("dtor")) {
      auto parseLoc = parser.getCurrentLocation();
      if (parser.parseRegion(dtorRegion, /*arguments=*/{}, /*argTypes=*/{}))
        return failure();
      if (!dtorRegion.hasOneBlock())
        return parser.emitError(parser.getCurrentLocation(),
                                "dtor region must have exactly one block");
      if (dtorRegion.back().empty())
        return parser.emitError(parser.getCurrentLocation(),
                                "dtor region shall not be empty");
      if (checkBlockTerminator(parser, parseLoc,
                               dtorRegion.back().back().getLoc(), &dtorRegion)
              .failed())
        return failure();
    }
  }

  typeAttr = TypeAttr::get(opTy);
  return success();
}

LogicalResult GlobalOp::verify() {
  // Verify that the initial value, if present, is either a unit attribute or
  // an attribute CIR supports.
  if (getInitialValue().has_value()) {
    if (checkConstantTypes(getOperation(), getSymType(), *getInitialValue())
            .failed())
      return failure();
  }

  // Verify that the constructor region, if present, has only one block which is
  // not empty.
  auto &ctorRegion = getCtorRegion();
  if (!ctorRegion.empty()) {
    if (!ctorRegion.hasOneBlock()) {
      return emitError() << "ctor region must have exactly one block.";
    }

    auto &block = ctorRegion.front();
    if (block.empty()) {
      return emitError() << "ctor region shall not be empty.";
    }
  }

  // Verify that the destructor region, if present, has only one block which is
  // not empty.
  auto &dtorRegion = getDtorRegion();
  if (!dtorRegion.empty()) {
    if (!dtorRegion.hasOneBlock()) {
      return emitError() << "dtor region must have exactly one block.";
    }

    auto &block = dtorRegion.front();
    if (block.empty()) {
      return emitError() << "dtor region shall not be empty.";
    }
  }

  if (std::optional<uint64_t> alignAttr = getAlignment()) {
    uint64_t alignment = alignAttr.value();
    if (!llvm::isPowerOf2_64(alignment))
      return emitError() << "alignment attribute value " << alignment
                         << " is not a power of 2";
  }

  switch (getLinkage()) {
  case GlobalLinkageKind::InternalLinkage:
  case GlobalLinkageKind::PrivateLinkage:
    if (isPublic())
      return emitError() << "public visibility not allowed with '"
                         << stringifyGlobalLinkageKind(getLinkage())
                         << "' linkage";
    break;
  case GlobalLinkageKind::ExternalLinkage:
  case GlobalLinkageKind::ExternalWeakLinkage:
  case GlobalLinkageKind::LinkOnceODRLinkage:
  case GlobalLinkageKind::LinkOnceAnyLinkage:
    // FIXME: mlir's concept of visibility gets tricky with LLVM ones,
    // for instance, symbol declarations cannot be "public", so we
    // have to mark them "private" to workaround the symbol verifier.
    if (isPrivate() && !isDeclaration())
      return emitError() << "private visibility not allowed with '"
                         << stringifyGlobalLinkageKind(getLinkage())
                         << "' linkage";
    break;
  default:
    emitError() << stringifyGlobalLinkageKind(getLinkage())
                << ": verifier not implemented\n";
    return failure();
  }

  // TODO: verify visibility for declarations?
  return success();
}

void GlobalOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     StringRef sym_name, Type sym_type, bool isConstant,
                     cir::GlobalLinkageKind linkage,
                     function_ref<void(OpBuilder &, Location)> ctorBuilder,
                     function_ref<void(OpBuilder &, Location)> dtorBuilder) {
  odsState.addAttribute(getSymNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getSymTypeAttrName(odsState.name),
                        ::mlir::TypeAttr::get(sym_type));
  if (isConstant)
    odsState.addAttribute(getConstantAttrName(odsState.name),
                          odsBuilder.getUnitAttr());

  ::mlir::cir::GlobalLinkageKindAttr linkageAttr =
      cir::GlobalLinkageKindAttr::get(odsBuilder.getContext(), linkage);
  odsState.addAttribute(getLinkageAttrName(odsState.name), linkageAttr);

  Region *ctorRegion = odsState.addRegion();
  if (ctorBuilder) {
    odsBuilder.createBlock(ctorRegion);
    ctorBuilder(odsBuilder, odsState.location);
  }

  Region *dtorRegion = odsState.addRegion();
  if (dtorBuilder) {
    odsBuilder.createBlock(dtorRegion);
    dtorBuilder(odsBuilder, odsState.location);
  }
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void GlobalOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  // The `ctor` and `dtor` regions always branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // Don't consider the ctor region if it is empty.
  Region *ctorRegion = &this->getCtorRegion();
  if (ctorRegion->empty())
    ctorRegion = nullptr;

  // Don't consider the dtor region if it is empty.
  Region *dtorRegion = &this->getCtorRegion();
  if (dtorRegion->empty())
    dtorRegion = nullptr;

  // If the condition isn't constant, both regions may be executed.
  if (ctorRegion)
    regions.push_back(RegionSuccessor(ctorRegion));
  if (dtorRegion)
    regions.push_back(RegionSuccessor(dtorRegion));
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

LogicalResult
GetGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify that the result type underlying pointer type matches the type of
  // the referenced cir.global or cir.func op.
  auto op = symbolTable.lookupNearestSymbolFrom(*this, getNameAttr());
  if (!(isa<GlobalOp>(op) || isa<FuncOp>(op)))
    return emitOpError("'")
           << getName()
           << "' does not reference a valid cir.global or cir.func";

  mlir::Type symTy;
  if (auto g = dyn_cast<GlobalOp>(op))
    symTy = g.getSymType();
  else if (auto f = dyn_cast<FuncOp>(op))
    symTy = f.getFunctionType();
  else
    llvm_unreachable("shall not get here");

  auto resultType = getAddr().getType().dyn_cast<PointerType>();
  if (!resultType || symTy != resultType.getPointee())
    return emitOpError("result type pointee type '")
           << resultType.getPointee() << "' does not match type " << symTy
           << " of the global @" << getName();
  return success();
}

//===----------------------------------------------------------------------===//
// VTableAddrPointOp
//===----------------------------------------------------------------------===//

LogicalResult
VTableAddrPointOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // vtable ptr is not coming from a symbol.
  if (!getName())
    return success();
  auto name = *getName();

  // Verify that the result type underlying pointer type matches the type of
  // the referenced cir.global or cir.func op.
  auto op = dyn_cast_or_null<GlobalOp>(
      symbolTable.lookupNearestSymbolFrom(*this, getNameAttr()));
  if (!op)
    return emitOpError("'")
           << name << "' does not reference a valid cir.global";
  auto init = op.getInitialValue();
  if (!init)
    return success();
  if (!isa<mlir::cir::VTableAttr>(*init))
    return emitOpError("Expected #cir.vtable in initializer for global '")
           << name << "'";
  return success();
}

LogicalResult cir::VTableAddrPointOp::verify() {
  // The operation uses either a symbol or a value to operate, but not both
  if (getName() && getSymAddr())
    return emitOpError("should use either a symbol or value, but not both");

  // If not a symbol, stick with the concrete type used for getSymAddr.
  if (getSymAddr())
    return success();

  auto resultType = getAddr().getType();
  auto intTy = mlir::cir::IntType::get(getContext(), 32, /*isSigned=*/false);
  auto fnTy = mlir::cir::FuncType::get({}, intTy);

  auto resTy = mlir::cir::PointerType::get(
      getContext(), mlir::cir::PointerType::get(getContext(), fnTy));

  if (resultType != resTy)
    return emitOpError("result type must be '")
           << resTy << "', but provided result type is '" << resultType << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// Returns the name used for the linkage attribute. This *must* correspond to
/// the name of the attribute in ODS.
static StringRef getLinkageAttrNameString() { return "linkage"; }

void cir::FuncOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, cir::FuncType type,
                        GlobalLinkageKind linkage,
                        ArrayRef<NamedAttribute> attrs,
                        ArrayRef<DictionaryAttr> argAttrs) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));
  result.addAttribute(
      getLinkageAttrNameString(),
      GlobalLinkageKindAttr::get(builder.getContext(), linkage));
  result.attributes.append(attrs.begin(), attrs.end());
  if (argAttrs.empty())
    return;

  function_interface_impl::addArgAndResultAttrs(
      builder, result, argAttrs,
      /*resultAttrs=*/std::nullopt, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

ParseResult cir::FuncOp::parse(OpAsmParser &parser, OperationState &state) {
  llvm::SMLoc loc = parser.getCurrentLocation();

  auto builtinNameAttr = getBuiltinAttrName(state.name);
  auto coroutineNameAttr = getCoroutineAttrName(state.name);
  auto lambdaNameAttr = getLambdaAttrName(state.name);
  auto visNameAttr = getSymVisibilityAttrName(state.name);
  auto noProtoNameAttr = getNoProtoAttrName(state.name);
  if (::mlir::succeeded(parser.parseOptionalKeyword(builtinNameAttr.strref())))
    state.addAttribute(builtinNameAttr, parser.getBuilder().getUnitAttr());
  if (::mlir::succeeded(
          parser.parseOptionalKeyword(coroutineNameAttr.strref())))
    state.addAttribute(coroutineNameAttr, parser.getBuilder().getUnitAttr());
  if (::mlir::succeeded(parser.parseOptionalKeyword(lambdaNameAttr.strref())))
    state.addAttribute(lambdaNameAttr, parser.getBuilder().getUnitAttr());
  if (parser.parseOptionalKeyword(noProtoNameAttr).succeeded())
    state.addAttribute(noProtoNameAttr, parser.getBuilder().getUnitAttr());

  // Default to external linkage if no keyword is provided.
  state.addAttribute(getLinkageAttrNameString(),
                     GlobalLinkageKindAttr::get(
                         parser.getContext(),
                         parseOptionalCIRKeyword<GlobalLinkageKind>(
                             parser, GlobalLinkageKind::ExternalLinkage)));

  ::llvm::StringRef visAttrStr;
  if (parser.parseOptionalKeyword(&visAttrStr, {"private", "public", "nested"})
          .succeeded()) {
    state.addAttribute(visNameAttr,
                       parser.getBuilder().getStringAttr(visAttrStr));
  }

  StringAttr nameAttr;
  SmallVector<OpAsmParser::Argument, 8> arguments;
  SmallVector<DictionaryAttr, 1> argAttrs;
  SmallVector<DictionaryAttr, 1> resultAttrs;
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/true, arguments, isVariadic, resultTypes,
          resultAttrs))
    return failure();

  for (auto &arg : arguments)
    argTypes.push_back(arg.type);

  if (resultTypes.size() > 1)
    return parser.emitError(loc, "functions only supports zero or one results");

  // Fetch return type or set it to void if empty/ommited.
  mlir::Type returnType =
      (resultTypes.empty() ? mlir::cir::VoidType::get(builder.getContext())
                           : resultTypes.front());

  // Build the function type.
  auto fnType = mlir::cir::FuncType::get(argTypes, returnType, isVariadic);
  if (!fnType)
    return failure();
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(fnType));

  // If additional attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(state.attributes))
    return failure();

  // Add the attributes to the function arguments.
  assert(resultAttrs.size() == resultTypes.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, arguments, resultAttrs, getArgAttrsAttrName(state.name),
      getResAttrsAttrName(state.name));

  bool hasAlias = false;
  auto aliaseeNameAttr = getAliaseeAttrName(state.name);
  if (::mlir::succeeded(parser.parseOptionalKeyword("alias"))) {
    if (parser.parseLParen().failed())
      return failure();
    StringAttr aliaseeAttr;
    if (parser.parseOptionalSymbolName(aliaseeAttr).failed())
      return failure();
    state.addAttribute(aliaseeNameAttr, FlatSymbolRefAttr::get(aliaseeAttr));
    if (parser.parseRParen().failed())
      return failure();
    hasAlias = true;
  }

  // If extra func attributes are present, parse them.
  NamedAttrList extraAttrs;
  if (::mlir::succeeded(parser.parseOptionalKeyword("extra"))) {
    if (parser.parseLParen().failed())
      return failure();
    if (parser.parseOptionalAttrDict(extraAttrs).failed())
      return failure();
    if (parser.parseRParen().failed())
      return failure();
  }
  state.addAttribute(getExtraAttrsAttrName(state.name),
                     mlir::cir::ExtraFuncAttributesAttr::get(
                         builder.getContext(),
                         extraAttrs.getDictionary(builder.getContext())));

  // Parse the optional function body.
  auto *body = state.addRegion();
  OptionalParseResult parseResult = parser.parseOptionalRegion(
      *body, arguments, /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
    if (hasAlias)
      parser.emitError(loc, "function alias shall not have a body");
    if (failed(*parseResult))
      return failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return parser.emitError(loc, "expected non-empty function body");
  }
  return success();
}

bool cir::FuncOp::isDeclaration() {
  auto aliasee = getAliasee();
  if (!aliasee)
    return isExternal();

  auto *modOp = getOperation()->getParentOp();
  auto targetFn = dyn_cast_or_null<mlir::cir::FuncOp>(
      mlir::SymbolTable::lookupSymbolIn(modOp, *aliasee));
  assert(targetFn && "expected aliasee to exist");
  return targetFn.isDeclaration();
}

::mlir::Region *cir::FuncOp::getCallableRegion() {
  auto aliasee = getAliasee();
  if (!aliasee)
    return isExternal() ? nullptr : &getBody();

  // Note that we forward the region from the original aliasee
  // function.
  auto *modOp = getOperation()->getParentOp();
  auto targetFn = dyn_cast_or_null<mlir::cir::FuncOp>(
      mlir::SymbolTable::lookupSymbolIn(modOp, *aliasee));
  assert(targetFn && "expected aliasee to exist");
  return targetFn.getCallableRegion();
}

void cir::FuncOp::print(OpAsmPrinter &p) {
  p << ' ';

  if (getBuiltin())
    p << "builtin ";

  if (getCoroutine())
    p << "coroutine ";

  if (getLambda())
    p << "lambda ";

  if (getNoProto())
    p << "no_proto ";

  if (getLinkage() != GlobalLinkageKind::ExternalLinkage)
    p << stringifyGlobalLinkageKind(getLinkage()) << ' ';

  auto vis = getVisibility();
  if (vis != mlir::SymbolTable::Visibility::Public)
    p << vis << " ";

  // Print function name, signature, and control.
  p.printSymbolName(getSymName());
  auto fnType = getFunctionType();
  SmallVector<Type, 1> resultTypes;
  if (!fnType.isVoid())
    function_interface_impl::printFunctionSignature(
        p, *this, fnType.getInputs(), fnType.isVarArg(),
        fnType.getReturnTypes());
  else
    function_interface_impl::printFunctionSignature(
        p, *this, fnType.getInputs(), fnType.isVarArg(), {});
  function_interface_impl::printFunctionAttributes(
      p, *this,
      {getSymVisibilityAttrName(), getAliaseeAttrName(),
       getFunctionTypeAttrName(), getLinkageAttrName(), getBuiltinAttrName(),
       getNoProtoAttrName(), getExtraAttrsAttrName()});

  if (auto aliaseeName = getAliasee()) {
    p << " alias(";
    p.printSymbolName(*aliaseeName);
    p << ")";
  }

  if (!getExtraAttrs().getElements().empty()) {
    p << " extra(";
    p.printOptionalAttrDict(getExtraAttrs().getElements().getValue());
    p << " )";
  }

  // Print the body if this is not an external function.
  Region &body = getOperation()->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
// attribute is present.  This can check for preconditions of the
// getNumArguments hook not failing.
LogicalResult cir::FuncOp::verifyType() {
  auto type = getFunctionType();
  if (!type.isa<cir::FuncType>())
    return emitOpError("requires '" + getFunctionTypeAttrName().str() +
                       "' attribute of function type");
  if (!getNoProto() && type.isVarArg() && type.getNumInputs() == 0)
    return emitError()
           << "prototyped function must have at least one non-variadic input";
  return success();
}

// Verifies linkage types
// - functions don't have 'common' linkage
// - external functions have 'external' or 'extern_weak' linkage
// - coroutine body must use at least one cir.await operation.
LogicalResult cir::FuncOp::verify() {
  if (getLinkage() == cir::GlobalLinkageKind::CommonLinkage)
    return emitOpError() << "functions cannot have '"
                         << stringifyGlobalLinkageKind(
                                cir::GlobalLinkageKind::CommonLinkage)
                         << "' linkage";

  if (isExternal()) {
    if (getLinkage() != cir::GlobalLinkageKind::ExternalLinkage &&
        getLinkage() != cir::GlobalLinkageKind::ExternalWeakLinkage)
      return emitOpError() << "external functions must have '"
                           << stringifyGlobalLinkageKind(
                                  cir::GlobalLinkageKind::ExternalLinkage)
                           << "' or '"
                           << stringifyGlobalLinkageKind(
                                  cir::GlobalLinkageKind::ExternalWeakLinkage)
                           << "' linkage";
    return success();
  }

  if (!isDeclaration() && getCoroutine()) {
    bool foundAwait = false;
    this->walk([&](Operation *op) {
      if (auto await = dyn_cast<AwaitOp>(op)) {
        foundAwait = true;
        return;
      }
    });
    if (!foundAwait)
      return emitOpError()
             << "coroutine body must use at least one cir.await op";
  }

  // Function alias should have an empty body.
  if (auto fn = getAliasee()) {
    if (fn && !getBody().empty())
      return emitOpError() << "a function alias '" << *fn
                           << "' must have empty body";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// Get the argument operands to the called function.
OperandRange cir::CallOp::getArgOperands() {
  return {arg_operand_begin(), arg_operand_end()};
}

MutableOperandRange cir::CallOp::getArgOperandsMutable() {
  return getOperandsMutable();
}

/// Return the callee of this operation
CallInterfaceCallable cir::CallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for this operation.
void cir::CallOp::setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
  if (auto calling =
          (*this)->getAttrOfType<mlir::SymbolRefAttr>(getCalleeAttrName()))
    (*this)->setAttr(getCalleeAttrName(), callee.get<mlir::SymbolRefAttr>());
  setOperand(0, callee.get<mlir::Value>());
}

LogicalResult
cir::CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Callee attribute only need on indirect calls.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return success();

  FuncOp fn =
      symbolTable.lookupNearestSymbolFrom<mlir::cir::FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee. Note that
  // argument-checking is disabled for functions without a prototype.
  auto fnType = fn.getFunctionType();
  if (!fn.getNoProto()) {
    if (!fnType.isVarArg() && getNumOperands() != fnType.getNumInputs())
      return emitOpError("incorrect number of operands for callee");

    if (fnType.isVarArg() && getNumOperands() < fnType.getNumInputs())
      return emitOpError("too few operands for callee");

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
      if (getOperand(i).getType() != fnType.getInput(i))
        return emitOpError("operand type mismatch: expected operand type ")
               << fnType.getInput(i) << ", but provided "
               << getOperand(i).getType() << " for operand number " << i;
  }

  // Void function must not return any results.
  if (fnType.isVoid() && getNumResults() != 0)
    return emitOpError("callee returns void but call has results");

  // Non-void function calls must return exactly one result.
  if (!fnType.isVoid() && getNumResults() != 1)
    return emitOpError("incorrect number of results for callee");

  // Parent function and return value types must match.
  if (!fnType.isVoid() && getResultTypes().front() != fnType.getReturnType()) {
    return emitOpError("result type mismatch: expected ")
           << fnType.getReturnType() << ", but provided "
           << getResult(0).getType();
  }

  return success();
}

::mlir::ParseResult CallOp::parse(::mlir::OpAsmParser &parser,
                                  ::mlir::OperationState &result) {
  mlir::FlatSymbolRefAttr calleeAttr;
  llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ops;
  llvm::SMLoc opsLoc;
  (void)opsLoc;
  llvm::ArrayRef<::mlir::Type> operandsTypes;
  llvm::ArrayRef<::mlir::Type> allResultTypes;

  // If we cannot parse a string callee, it means this is an indirect call.
  if (!parser.parseOptionalAttribute(calleeAttr, "callee", result.attributes)
           .has_value()) {
    OpAsmParser::UnresolvedOperand indirectVal;
    // Do not resolve right now, since we need to figure out the type
    if (parser.parseOperand(indirectVal).failed())
      return failure();
    ops.push_back(indirectVal);
  }

  if (parser.parseLParen())
    return ::mlir::failure();

  opsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ops))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType opsFnTy;
  if (parser.parseType(opsFnTy))
    return ::mlir::failure();
  operandsTypes = opsFnTy.getInputs();
  allResultTypes = opsFnTy.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(ops, operandsTypes, opsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CallOp::print(::mlir::OpAsmPrinter &state) {
  state << ' ';
  auto ops = getOperands();

  if (getCallee()) { // Direct calls
    state.printAttributeWithoutType(getCalleeAttr());
  } else { // Indirect calls
    state << ops.front();
    ops = ops.drop_front();
  }
  state << "(";
  state << ops;
  state << ")";
  llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("callee");
  state.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  state << ' ' << ":";
  state << ' ';
  state.printFunctionalType(getOperands().getTypes(),
                            getOperation()->getResultTypes());
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

LogicalResult UnaryOp::verify() {
  switch (getKind()) {
  case cir::UnaryOpKind::Inc:
    LLVM_FALLTHROUGH;
  case cir::UnaryOpKind::Dec: {
    // TODO: Consider looking at the memory interface instead of
    // LoadOp/StoreOp.
    auto loadOp = getInput().getDefiningOp<cir::LoadOp>();
    if (!loadOp)
      return emitOpError() << "requires input to be defined by a memory load";

    for (const auto user : getResult().getUsers()) {
      auto storeOp = dyn_cast<cir::StoreOp>(user);
      if (storeOp && storeOp.getAddr() == loadOp.getAddr())
        return success();
    }
    return emitOpError() << "requires result to be used by a memory store "
                            "to the same address as the input memory load";
  }
  case cir::UnaryOpKind::Plus:
  case cir::UnaryOpKind::Minus:
  case cir::UnaryOpKind::Not:
    // Nothing to verify.
    return success();
  }

  llvm_unreachable("Unknown UnaryOp kind?");
}

//===----------------------------------------------------------------------===//
// AwaitOp
//===----------------------------------------------------------------------===//

void AwaitOp::build(OpBuilder &builder, OperationState &result,
                    mlir::cir::AwaitKind kind,
                    function_ref<void(OpBuilder &, Location)> readyBuilder,
                    function_ref<void(OpBuilder &, Location)> suspendBuilder,
                    function_ref<void(OpBuilder &, Location)> resumeBuilder) {
  result.addAttribute(getKindAttrName(result.name),
                      cir::AwaitKindAttr::get(builder.getContext(), kind));
  {
    OpBuilder::InsertionGuard guard(builder);
    Region *readyRegion = result.addRegion();
    builder.createBlock(readyRegion);
    readyBuilder(builder, result.location);
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    Region *suspendRegion = result.addRegion();
    builder.createBlock(suspendRegion);
    suspendBuilder(builder, result.location);
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    Region *resumeRegion = result.addRegion();
    builder.createBlock(resumeRegion);
    resumeBuilder(builder, result.location);
  }
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes
/// that correspond to a constant value for each operand, or null if that
/// operand is not a constant.
void AwaitOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  // If any index all the underlying regions branch back to the parent
  // operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // FIXME: we want to look at cond region for getting more accurate results
  // if the other regions will get a chance to execute.
  regions.push_back(RegionSuccessor(&this->getReady()));
  regions.push_back(RegionSuccessor(&this->getSuspend()));
  regions.push_back(RegionSuccessor(&this->getResume()));
}

LogicalResult AwaitOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// CIR defined traits
//===----------------------------------------------------------------------===//

LogicalResult
mlir::OpTrait::impl::verifySameFirstOperandAndResultType(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 1)) || failed(verifyOneResult(op)))
    return failure();

  auto type = op->getResult(0).getType();
  auto opType = op->getOperand(0).getType();

  if (type != opType)
    return op->emitOpError()
           << "requires the same type for first operand and result";

  return success();
}

//===----------------------------------------------------------------------===//
// CIR attributes
// FIXME: move all of these to CIRAttrs.cpp
//===----------------------------------------------------------------------===//

LogicalResult mlir::cir::ConstArrayAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::Type type, Attribute attr) {

  if (!(attr.isa<mlir::ArrayAttr>() || attr.isa<mlir::StringAttr>()))
    return emitError() << "constant array expects ArrayAttr or StringAttr";

  if (auto strAttr = attr.dyn_cast<mlir::StringAttr>()) {
    mlir::cir::ArrayType at = type.cast<mlir::cir::ArrayType>();
    auto intTy = at.getEltType().dyn_cast<cir::IntType>();

    // TODO: add CIR type for char.
    if (!intTy || intTy.getWidth() != 8) {
      emitError() << "constant array element for string literals expects "
                     "!cir.int<u, 8> element type";
      return failure();
    }
    return success();
  }

  assert(attr.isa<mlir::ArrayAttr>());
  auto arrayAttr = attr.cast<mlir::ArrayAttr>();
  auto at = type.cast<ArrayType>();

  // Make sure both number of elements and subelement types match type.
  if (at.getSize() != arrayAttr.size())
    return emitError() << "constant array size should match type size";
  LogicalResult eltTypeCheck = success();
  arrayAttr.walkImmediateSubElements(
      [&](Attribute attr) {
        // Once we find a mismatch, stop there.
        if (eltTypeCheck.failed())
          return;
        auto typedAttr = attr.dyn_cast<TypedAttr>();
        if (!typedAttr || typedAttr.getType() != at.getEltType()) {
          eltTypeCheck = failure();
          emitError()
              << "constant array element should match array element type";
        }
      },
      [&](Type type) {});
  return eltTypeCheck;
}

::mlir::Attribute ConstArrayAttr::parse(::mlir::AsmParser &parser,
                                        ::mlir::Type type) {
  ::mlir::FailureOr<::mlir::Type> resultTy;
  ::mlir::FailureOr<Attribute> resultVal;
  ::llvm::SMLoc loc = parser.getCurrentLocation();
  (void)loc;
  // Parse literal '<'
  if (parser.parseLess())
    return {};

  // Parse variable 'value'
  resultVal = ::mlir::FieldParser<Attribute>::parse(parser);
  if (failed(resultVal)) {
    parser.emitError(
        parser.getCurrentLocation(),
        "failed to parse ConstArrayAttr parameter 'value' which is "
        "to be a `Attribute`");
    return {};
  }

  // ArrayAttrrs have per-element type, not the type of the array...
  if (resultVal->dyn_cast<ArrayAttr>()) {
    // Array has implicit type: infer from const array type.
    if (parser.parseOptionalColon().failed()) {
      resultTy = type;
    } else { // Array has explicit type: parse it.
      resultTy = ::mlir::FieldParser<::mlir::Type>::parse(parser);
      if (failed(resultTy)) {
        parser.emitError(
            parser.getCurrentLocation(),
            "failed to parse ConstArrayAttr parameter 'type' which is "
            "to be a `::mlir::Type`");
        return {};
      }
    }
  } else {
    assert(resultVal->isa<TypedAttr>() && "IDK");
    auto ta = resultVal->cast<TypedAttr>();
    resultTy = ta.getType();
    if (resultTy->isa<mlir::NoneType>()) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected type declaration for string literal");
      return {};
    }
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};
  return parser.getChecked<ConstArrayAttr>(loc, parser.getContext(),
                                           resultTy.value(), resultVal.value());
}

void ConstArrayAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getElts());
  printer << ">";
}

::mlir::Attribute SignedOverflowBehaviorAttr::parse(::mlir::AsmParser &parser,
                                                    ::mlir::Type type) {
  if (parser.parseLess())
    return {};
  auto behavior = parseOptionalCIRKeyword(
      parser, mlir::cir::sob::SignedOverflowBehavior::undefined);
  if (parser.parseGreater())
    return {};

  return SignedOverflowBehaviorAttr::get(parser.getContext(), behavior);
}

void SignedOverflowBehaviorAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  switch (getBehavior()) {
  case sob::SignedOverflowBehavior::undefined:
    printer << "undefined";
    break;
  case sob::SignedOverflowBehavior::defined:
    printer << "defined";
    break;
  case sob::SignedOverflowBehavior::trapping:
    printer << "trapping";
    break;
  }
  printer << ">";
}

LogicalResult TypeInfoAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::Type type, ::mlir::ArrayAttr typeinfoData) {

  if (mlir::cir::ConstStructAttr::verify(emitError, type, typeinfoData)
          .failed())
    return failure();

  for (auto &member : typeinfoData) {
    if (llvm::isa<GlobalViewAttr, IntAttr>(member))
      continue;
    emitError() << "expected GlobalViewAttr or IntAttr attribute";
    return failure();
  }

  return success();
}

LogicalResult
VTableAttr::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                   ::mlir::Type type, ::mlir::ArrayAttr vtableData) {
  auto sTy = type.dyn_cast_or_null<mlir::cir::StructType>();
  if (!sTy) {
    emitError() << "expected !cir.struct type result";
    return failure();
  }
  if (sTy.getMembers().size() != 1 || vtableData.size() != 1) {
    emitError() << "expected struct type with only one subtype";
    return failure();
  }

  auto arrayTy = sTy.getMembers()[0].dyn_cast<mlir::cir::ArrayType>();
  auto constArrayAttr = vtableData[0].dyn_cast<mlir::cir::ConstArrayAttr>();
  if (!arrayTy || !constArrayAttr) {
    emitError() << "expected struct type with one array element";
    return failure();
  }

  if (mlir::cir::ConstStructAttr::verify(emitError, type, vtableData).failed())
    return failure();

  LogicalResult eltTypeCheck = success();
  if (auto arrayElts = constArrayAttr.getElts().dyn_cast<ArrayAttr>()) {
    arrayElts.walkImmediateSubElements(
        [&](Attribute attr) {
          if (attr.isa<GlobalViewAttr>() || attr.isa<ConstPtrAttr>())
            return;
          emitError() << "expected GlobalViewAttr attribute";
          eltTypeCheck = failure();
        },
        [&](Type type) {});
    return eltTypeCheck;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult CopyOp::verify() {

  // A data layout is required for us to know the number of bytes to be copied.
  if (!getType().getPointee().hasTrait<DataLayoutTypeInterface::Trait>())
    return emitError() << "missing data layout for pointee type";

  if (getSrc() == getDst())
    return emitError() << "source and destination are the same";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MemCpyOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult MemCpyOp::verify() {
  auto voidPtr =
      cir::PointerType::get(getContext(), cir::VoidType::get(getContext()));

  if (!getLenTy().isUnsigned())
    return emitError() << "memcpy length must be an unsigned integer";

  if (getSrcTy() != voidPtr || getDstTy() != voidPtr)
    return emitError() << "memcpy src and dst must be void pointers";

  return mlir::success();
}

static bool isIncompleteType(mlir::Type typ) {
  if (auto ptr = typ.dyn_cast<PointerType>())
    return isIncompleteType(ptr.getPointee());
  else if (auto rec = typ.dyn_cast<StructType>())
    return rec.isIncomplete();
  return false;
}

//===----------------------------------------------------------------------===//
// GetMemberOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult GetMemberOp::verify() {

  const auto recordTy = getAddrTy().getPointee().dyn_cast<StructType>();
  if (!recordTy)
    return emitError() << "expected pointer to a record type";

  // FIXME: currently we bypass typechecking of incomplete types due to errors
  // in the codegen process. This should be removed once the codegen is fixed.
  if (isIncompleteType(recordTy))
    return mlir::success();

  if (recordTy.getMembers().size() <= getIndex())
    return emitError() << "member index out of bounds";

  // FIXME(cir): member type check is disabled for classes as the codegen for
  // these still need to be patched.
  // Also we bypass the typechecking for the fields of incomplete types.
  bool shouldSkipMemberTypeMismatch =
    recordTy.isClass() || isIncompleteType(recordTy.getMembers()[getIndex()]);

  if (!shouldSkipMemberTypeMismatch
      && recordTy.getMembers()[getIndex()] != getResultTy().getPointee())
    return emitError() << "member type mismatch";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROps.cpp.inc"
