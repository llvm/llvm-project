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

#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"

#include "clang/CIR/Dialect/IR/CIROpsDialect.cpp.inc"
#include "clang/CIR/Dialect/IR/CIROpsEnums.cpp.inc"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//
namespace {
struct CIROpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    if (auto recordType = dyn_cast<cir::RecordType>(type)) {
      StringAttr nameAttr = recordType.getName();
      if (!nameAttr)
        os << "rec_anon_" << recordType.getKindAsStr();
      else
        os << "rec_" << nameAttr.getValue();
      return AliasResult::OverridableAlias;
    }
    if (auto intType = dyn_cast<cir::IntType>(type)) {
      // We only provide alias for standard integer types (i.e. integer types
      // whose width is a power of 2 and at least 8).
      unsigned width = intType.getWidth();
      if (width < 8 || !llvm::isPowerOf2_32(width))
        return AliasResult::NoAlias;
      os << intType.getAlias();
      return AliasResult::OverridableAlias;
    }
    if (auto voidType = dyn_cast<cir::VoidType>(type)) {
      os << voidType.getAlias();
      return AliasResult::OverridableAlias;
    }

    return AliasResult::NoAlias;
  }

  AliasResult getAlias(Attribute attr, raw_ostream &os) const final {
    if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(attr)) {
      os << (boolAttr.getValue() ? "true" : "false");
      return AliasResult::FinalAlias;
    }
    if (auto bitfield = mlir::dyn_cast<cir::BitfieldInfoAttr>(attr)) {
      os << "bfi_" << bitfield.getName().str();
      return AliasResult::FinalAlias;
    }
    if (auto dynCastInfoAttr = mlir::dyn_cast<cir::DynamicCastInfoAttr>(attr)) {
      os << dynCastInfoAttr.getAlias();
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

void cir::CIRDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "clang/CIR/Dialect/IR/CIROps.cpp.inc"
      >();
  addInterfaces<CIROpAsmDialectInterface>();
}

Operation *cir::CIRDialect::materializeConstant(mlir::OpBuilder &builder,
                                                mlir::Attribute value,
                                                mlir::Type type,
                                                mlir::Location loc) {
  return builder.create<cir::ConstantOp>(loc, type,
                                         mlir::cast<mlir::TypedAttr>(value));
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Parses one of the keywords provided in the list `keywords` and returns the
// position of the parsed keyword in the list. If none of the keywords from the
// list is parsed, returns -1.
static int parseOptionalKeywordAlternative(AsmParser &parser,
                                           ArrayRef<llvm::StringRef> keywords) {
  for (auto en : llvm::enumerate(keywords)) {
    if (succeeded(parser.parseOptionalKeyword(en.value())))
      return en.index();
  }
  return -1;
}

namespace {
template <typename Ty> struct EnumTraits {};

#define REGISTER_ENUM_TYPE(Ty)                                                 \
  template <> struct EnumTraits<cir::Ty> {                                     \
    static llvm::StringRef stringify(cir::Ty value) {                          \
      return stringify##Ty(value);                                             \
    }                                                                          \
    static unsigned getMaxEnumVal() { return cir::getMaxEnumValFor##Ty(); }    \
  }

REGISTER_ENUM_TYPE(GlobalLinkageKind);
REGISTER_ENUM_TYPE(VisibilityKind);
REGISTER_ENUM_TYPE(SideEffect);
} // namespace

/// Parse an enum from the keyword, or default to the provided default value.
/// The return type is the enum type by default, unless overriden with the
/// second template argument.
template <typename EnumTy, typename RetTy = EnumTy>
static RetTy parseOptionalCIRKeyword(AsmParser &parser, EnumTy defaultValue) {
  llvm::SmallVector<llvm::StringRef, 10> names;
  for (unsigned i = 0, e = EnumTraits<EnumTy>::getMaxEnumVal(); i <= e; ++i)
    names.push_back(EnumTraits<EnumTy>::stringify(static_cast<EnumTy>(i)));

  int index = parseOptionalKeywordAlternative(parser, names);
  if (index == -1)
    return static_cast<RetTy>(defaultValue);
  return static_cast<RetTy>(index);
}

/// Parse an enum from the keyword, return failure if the keyword is not found.
template <typename EnumTy, typename RetTy = EnumTy>
static ParseResult parseCIRKeyword(AsmParser &parser, RetTy &result) {
  llvm::SmallVector<llvm::StringRef, 10> names;
  for (unsigned i = 0, e = EnumTraits<EnumTy>::getMaxEnumVal(); i <= e; ++i)
    names.push_back(EnumTraits<EnumTy>::stringify(static_cast<EnumTy>(i)));

  int index = parseOptionalKeywordAlternative(parser, names);
  if (index == -1)
    return failure();
  result = static_cast<RetTy>(index);
  return success();
}

// Check if a region's termination omission is valid and, if so, creates and
// inserts the omitted terminator into the region.
static LogicalResult ensureRegionTerm(OpAsmParser &parser, Region &region,
                                      SMLoc errLoc) {
  Location eLoc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  OpBuilder builder(parser.getBuilder().getContext());

  // Insert empty block in case the region is empty to ensure the terminator
  // will be inserted
  if (region.empty())
    builder.createBlock(&region);

  Block &block = region.back();
  // Region is properly terminated: nothing to do.
  if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>())
    return success();

  // Check for invalid terminator omissions.
  if (!region.hasOneBlock())
    return parser.emitError(errLoc,
                            "multi-block region must not omit terminator");

  // Terminator was omitted correctly: recreate it.
  builder.setInsertionPointToEnd(&block);
  builder.create<cir::YieldOp>(eLoc);
  return success();
}

// True if the region's terminator should be omitted.
static bool omitRegionTerm(mlir::Region &r) {
  const auto singleNonEmptyBlock = r.hasOneBlock() && !r.back().empty();
  const auto yieldsNothing = [&r]() {
    auto y = dyn_cast<cir::YieldOp>(r.back().getTerminator());
    return y && y.getArgs().empty();
  };
  return singleNonEmptyBlock && yieldsNothing();
}

void printVisibilityAttr(OpAsmPrinter &printer,
                         cir::VisibilityAttr &visibility) {
  switch (visibility.getValue()) {
  case cir::VisibilityKind::Hidden:
    printer << "hidden";
    break;
  case cir::VisibilityKind::Protected:
    printer << "protected";
    break;
  case cir::VisibilityKind::Default:
    break;
  }
}

void parseVisibilityAttr(OpAsmParser &parser, cir::VisibilityAttr &visibility) {
  cir::VisibilityKind visibilityKind =
      parseOptionalCIRKeyword(parser, cir::VisibilityKind::Default);
  visibility = cir::VisibilityAttr::get(parser.getContext(), visibilityKind);
}

//===----------------------------------------------------------------------===//
// CIR Custom Parsers/Printers
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseOmittedTerminatorRegion(mlir::OpAsmParser &parser,
                                                      mlir::Region &region) {
  auto regionLoc = parser.getCurrentLocation();
  if (parser.parseRegion(region))
    return failure();
  if (ensureRegionTerm(parser, region, regionLoc).failed())
    return failure();
  return success();
}

static void printOmittedTerminatorRegion(mlir::OpAsmPrinter &printer,
                                         cir::ScopeOp &op,
                                         mlir::Region &region) {
  printer.printRegion(region,
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/!omitRegionTerm(region));
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

void cir::AllocaOp::build(mlir::OpBuilder &odsBuilder,
                          mlir::OperationState &odsState, mlir::Type addr,
                          mlir::Type allocaType, llvm::StringRef name,
                          mlir::IntegerAttr alignment) {
  odsState.addAttribute(getAllocaTypeAttrName(odsState.name),
                        mlir::TypeAttr::get(allocaType));
  odsState.addAttribute(getNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(name));
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addTypes(addr);
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

LogicalResult cir::BreakOp::verify() {
  assert(!cir::MissingFeatures::switchOp());
  if (!getOperation()->getParentOfType<LoopOpInterface>() &&
      !getOperation()->getParentOfType<SwitchOp>())
    return emitOpError("must be within a loop");
  return success();
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

//===----------------------------------
// BranchOpTerminatorInterface Methods
//===----------------------------------

void cir::ConditionOp::getSuccessorRegions(
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions) {
  // TODO(cir): The condition value may be folded to a constant, narrowing
  // down its list of possible successors.

  // Parent is a loop: condition may branch to the body or to the parent op.
  if (auto loopOp = dyn_cast<LoopOpInterface>(getOperation()->getParentOp())) {
    regions.emplace_back(&loopOp.getBody(), loopOp.getBody().getArguments());
    regions.emplace_back(loopOp->getResults());
  }

  assert(!cir::MissingFeatures::awaitOp());
}

MutableOperandRange
cir::ConditionOp::getMutableSuccessorOperands(RegionBranchPoint point) {
  // No values are yielded to the successor region.
  return MutableOperandRange(getOperation(), 0, 0);
}

LogicalResult cir::ConditionOp::verify() {
  assert(!cir::MissingFeatures::awaitOp());
  if (!isa<LoopOpInterface>(getOperation()->getParentOp()))
    return emitOpError("condition must be within a conditional region");
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult checkConstantTypes(mlir::Operation *op, mlir::Type opType,
                                        mlir::Attribute attrType) {
  if (isa<cir::ConstPtrAttr>(attrType)) {
    if (!mlir::isa<cir::PointerType>(opType))
      return op->emitOpError(
          "pointer constant initializing a non-pointer type");
    return success();
  }

  if (isa<cir::ZeroAttr>(attrType)) {
    if (isa<cir::RecordType, cir::ArrayType, cir::VectorType, cir::ComplexType>(
            opType))
      return success();
    return op->emitOpError(
        "zero expects struct, array, vector, or complex type");
  }

  if (mlir::isa<cir::BoolAttr>(attrType)) {
    if (!mlir::isa<cir::BoolType>(opType))
      return op->emitOpError("result type (")
             << opType << ") must be '!cir.bool' for '" << attrType << "'";
    return success();
  }

  if (mlir::isa<cir::IntAttr, cir::FPAttr>(attrType)) {
    auto at = cast<TypedAttr>(attrType);
    if (at.getType() != opType) {
      return op->emitOpError("result type (")
             << opType << ") does not match value type (" << at.getType()
             << ")";
    }
    return success();
  }

  if (mlir::isa<cir::ConstArrayAttr, cir::ConstVectorAttr,
                cir::ConstComplexAttr, cir::ConstRecordAttr,
                cir::GlobalViewAttr, cir::PoisonAttr, cir::TypeInfoAttr,
                cir::VTableAttr>(attrType))
    return success();

  assert(isa<TypedAttr>(attrType) && "What else could we be looking at here?");
  return op->emitOpError("global with type ")
         << cast<TypedAttr>(attrType).getType() << " not yet supported";
}

LogicalResult cir::ConstantOp::verify() {
  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  return checkConstantTypes(getOperation(), getType(), getValue());
}

OpFoldResult cir::ConstantOp::fold(FoldAdaptor /*adaptor*/) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ContinueOp::verify() {
  if (!getOperation()->getParentOfType<LoopOpInterface>())
    return emitOpError("must be within a loop");
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult cir::CastOp::verify() {
  mlir::Type resType = getType();
  mlir::Type srcType = getSrc().getType();

  if (mlir::isa<cir::VectorType>(srcType) &&
      mlir::isa<cir::VectorType>(resType)) {
    // Use the element type of the vector to verify the cast kind. (Except for
    // bitcast, see below.)
    srcType = mlir::dyn_cast<cir::VectorType>(srcType).getElementType();
    resType = mlir::dyn_cast<cir::VectorType>(resType).getElementType();
  }

  switch (getKind()) {
  case cir::CastKind::int_to_bool: {
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    if (!mlir::isa<cir::IntType>(srcType))
      return emitOpError() << "requires !cir.int type for source";
    return success();
  }
  case cir::CastKind::ptr_to_bool: {
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    if (!mlir::isa<cir::PointerType>(srcType))
      return emitOpError() << "requires !cir.ptr type for source";
    return success();
  }
  case cir::CastKind::integral: {
    if (!mlir::isa<cir::IntType>(resType))
      return emitOpError() << "requires !cir.int type for result";
    if (!mlir::isa<cir::IntType>(srcType))
      return emitOpError() << "requires !cir.int type for source";
    return success();
  }
  case cir::CastKind::array_to_ptrdecay: {
    const auto arrayPtrTy = mlir::dyn_cast<cir::PointerType>(srcType);
    const auto flatPtrTy = mlir::dyn_cast<cir::PointerType>(resType);
    if (!arrayPtrTy || !flatPtrTy)
      return emitOpError() << "requires !cir.ptr type for source and result";

    // TODO(CIR): Make sure the AddrSpace of both types are equals
    return success();
  }
  case cir::CastKind::bitcast: {
    // Handle the pointer types first.
    auto srcPtrTy = mlir::dyn_cast<cir::PointerType>(srcType);
    auto resPtrTy = mlir::dyn_cast<cir::PointerType>(resType);

    if (srcPtrTy && resPtrTy) {
      return success();
    }

    return success();
  }
  case cir::CastKind::floating: {
    if (!mlir::isa<cir::FPTypeInterface>(srcType) ||
        !mlir::isa<cir::FPTypeInterface>(resType))
      return emitOpError() << "requires !cir.float type for source and result";
    return success();
  }
  case cir::CastKind::float_to_int: {
    if (!mlir::isa<cir::FPTypeInterface>(srcType))
      return emitOpError() << "requires !cir.float type for source";
    if (!mlir::dyn_cast<cir::IntType>(resType))
      return emitOpError() << "requires !cir.int type for result";
    return success();
  }
  case cir::CastKind::int_to_ptr: {
    if (!mlir::dyn_cast<cir::IntType>(srcType))
      return emitOpError() << "requires !cir.int type for source";
    if (!mlir::dyn_cast<cir::PointerType>(resType))
      return emitOpError() << "requires !cir.ptr type for result";
    return success();
  }
  case cir::CastKind::ptr_to_int: {
    if (!mlir::dyn_cast<cir::PointerType>(srcType))
      return emitOpError() << "requires !cir.ptr type for source";
    if (!mlir::dyn_cast<cir::IntType>(resType))
      return emitOpError() << "requires !cir.int type for result";
    return success();
  }
  case cir::CastKind::float_to_bool: {
    if (!mlir::isa<cir::FPTypeInterface>(srcType))
      return emitOpError() << "requires !cir.float type for source";
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    return success();
  }
  case cir::CastKind::bool_to_int: {
    if (!mlir::isa<cir::BoolType>(srcType))
      return emitOpError() << "requires !cir.bool type for source";
    if (!mlir::isa<cir::IntType>(resType))
      return emitOpError() << "requires !cir.int type for result";
    return success();
  }
  case cir::CastKind::int_to_float: {
    if (!mlir::isa<cir::IntType>(srcType))
      return emitOpError() << "requires !cir.int type for source";
    if (!mlir::isa<cir::FPTypeInterface>(resType))
      return emitOpError() << "requires !cir.float type for result";
    return success();
  }
  case cir::CastKind::bool_to_float: {
    if (!mlir::isa<cir::BoolType>(srcType))
      return emitOpError() << "requires !cir.bool type for source";
    if (!mlir::isa<cir::FPTypeInterface>(resType))
      return emitOpError() << "requires !cir.float type for result";
    return success();
  }
  case cir::CastKind::address_space: {
    auto srcPtrTy = mlir::dyn_cast<cir::PointerType>(srcType);
    auto resPtrTy = mlir::dyn_cast<cir::PointerType>(resType);
    if (!srcPtrTy || !resPtrTy)
      return emitOpError() << "requires !cir.ptr type for source and result";
    if (srcPtrTy.getPointee() != resPtrTy.getPointee())
      return emitOpError() << "requires two types differ in addrspace only";
    return success();
  }
  case cir::CastKind::float_to_complex: {
    if (!mlir::isa<cir::FPTypeInterface>(srcType))
      return emitOpError() << "requires !cir.float type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy)
      return emitOpError() << "requires !cir.complex type for result";
    if (srcType != resComplexTy.getElementType())
      return emitOpError() << "requires source type match result element type";
    return success();
  }
  case cir::CastKind::int_to_complex: {
    if (!mlir::isa<cir::IntType>(srcType))
      return emitOpError() << "requires !cir.int type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy)
      return emitOpError() << "requires !cir.complex type for result";
    if (srcType != resComplexTy.getElementType())
      return emitOpError() << "requires source type match result element type";
    return success();
  }
  case cir::CastKind::float_complex_to_real: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy)
      return emitOpError() << "requires !cir.complex type for source";
    if (!mlir::isa<cir::FPTypeInterface>(resType))
      return emitOpError() << "requires !cir.float type for result";
    if (srcComplexTy.getElementType() != resType)
      return emitOpError() << "requires source element type match result type";
    return success();
  }
  case cir::CastKind::int_complex_to_real: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy)
      return emitOpError() << "requires !cir.complex type for source";
    if (!mlir::isa<cir::IntType>(resType))
      return emitOpError() << "requires !cir.int type for result";
    if (srcComplexTy.getElementType() != resType)
      return emitOpError() << "requires source element type match result type";
    return success();
  }
  case cir::CastKind::float_complex_to_bool: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isFloatingPointComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for source";
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    return success();
  }
  case cir::CastKind::int_complex_to_bool: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isIntegerComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for source";
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    return success();
  }
  case cir::CastKind::float_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isFloatingPointComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy || !resComplexTy.isFloatingPointComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for result";
    return success();
  }
  case cir::CastKind::float_complex_to_int_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isFloatingPointComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy || !resComplexTy.isIntegerComplex())
      return emitOpError() << "requires integer !cir.complex type for result";
    return success();
  }
  case cir::CastKind::int_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isIntegerComplex())
      return emitOpError() << "requires integer !cir.complex type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy || !resComplexTy.isIntegerComplex())
      return emitOpError() << "requires integer !cir.complex type for result";
    return success();
  }
  case cir::CastKind::int_complex_to_float_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !srcComplexTy.isIntegerComplex())
      return emitOpError() << "requires integer !cir.complex type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy || !resComplexTy.isFloatingPointComplex())
      return emitOpError()
             << "requires floating point !cir.complex type for result";
    return success();
  }
  default:
    llvm_unreachable("Unknown CastOp kind?");
  }
}

static bool isIntOrBoolCast(cir::CastOp op) {
  auto kind = op.getKind();
  return kind == cir::CastKind::bool_to_int ||
         kind == cir::CastKind::int_to_bool || kind == cir::CastKind::integral;
}

static Value tryFoldCastChain(cir::CastOp op) {
  cir::CastOp head = op, tail = op;

  while (op) {
    if (!isIntOrBoolCast(op))
      break;
    head = op;
    op = head.getSrc().getDefiningOp<cir::CastOp>();
  }

  if (head == tail)
    return {};

  // if bool_to_int -> ...  -> int_to_bool: take the bool
  // as we had it was before all casts
  if (head.getKind() == cir::CastKind::bool_to_int &&
      tail.getKind() == cir::CastKind::int_to_bool)
    return head.getSrc();

  // if int_to_bool -> ...  -> int_to_bool: take the result
  // of the first one, as no other casts (and ext casts as well)
  // don't change the first result
  if (head.getKind() == cir::CastKind::int_to_bool &&
      tail.getKind() == cir::CastKind::int_to_bool)
    return head.getResult();

  return {};
}

OpFoldResult cir::CastOp::fold(FoldAdaptor adaptor) {
  if (mlir::isa_and_present<cir::PoisonAttr>(adaptor.getSrc())) {
    // Propagate poison value
    return cir::PoisonAttr::get(getContext(), getType());
  }

  if (getSrc().getType() == getType()) {
    switch (getKind()) {
    case cir::CastKind::integral: {
      // TODO: for sign differences, it's possible in certain conditions to
      // create a new attribute that's capable of representing the source.
      llvm::SmallVector<mlir::OpFoldResult, 1> foldResults;
      auto foldOrder = getSrc().getDefiningOp()->fold(foldResults);
      if (foldOrder.succeeded() && mlir::isa<mlir::Attribute>(foldResults[0]))
        return mlir::cast<mlir::Attribute>(foldResults[0]);
      return {};
    }
    case cir::CastKind::bitcast:
    case cir::CastKind::address_space:
    case cir::CastKind::float_complex:
    case cir::CastKind::int_complex: {
      return getSrc();
    }
    default:
      return {};
    }
  }
  return tryFoldCastChain(*this);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

mlir::OperandRange cir::CallOp::getArgOperands() {
  if (isIndirect())
    return getArgs().drop_front(1);
  return getArgs();
}

mlir::MutableOperandRange cir::CallOp::getArgOperandsMutable() {
  mlir::MutableOperandRange args = getArgsMutable();
  if (isIndirect())
    return args.slice(1, args.size() - 1);
  return args;
}

mlir::Value cir::CallOp::getIndirectCall() {
  assert(isIndirect());
  return getOperand(0);
}

/// Return the operand at index 'i'.
Value cir::CallOp::getArgOperand(unsigned i) {
  if (isIndirect())
    ++i;
  return getOperand(i);
}

/// Return the number of operands.
unsigned cir::CallOp::getNumArgOperands() {
  if (isIndirect())
    return this->getOperation()->getNumOperands() - 1;
  return this->getOperation()->getNumOperands();
}

static mlir::ParseResult parseCallCommon(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> ops;
  llvm::SMLoc opsLoc;
  mlir::FlatSymbolRefAttr calleeAttr;
  llvm::ArrayRef<mlir::Type> allResultTypes;

  // If we cannot parse a string callee, it means this is an indirect call.
  if (!parser
           .parseOptionalAttribute(calleeAttr, CIRDialect::getCalleeAttrName(),
                                   result.attributes)
           .has_value()) {
    OpAsmParser::UnresolvedOperand indirectVal;
    // Do not resolve right now, since we need to figure out the type
    if (parser.parseOperand(indirectVal).failed())
      return failure();
    ops.push_back(indirectVal);
  }

  if (parser.parseLParen())
    return mlir::failure();

  opsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ops))
    return mlir::failure();
  if (parser.parseRParen())
    return mlir::failure();

  if (parser.parseOptionalKeyword("nothrow").succeeded())
    result.addAttribute(CIRDialect::getNoThrowAttrName(),
                        mlir::UnitAttr::get(parser.getContext()));

  if (parser.parseOptionalKeyword("side_effect").succeeded()) {
    if (parser.parseLParen().failed())
      return failure();
    cir::SideEffect sideEffect;
    if (parseCIRKeyword<cir::SideEffect>(parser, sideEffect).failed())
      return failure();
    if (parser.parseRParen().failed())
      return failure();
    auto attr = cir::SideEffectAttr::get(parser.getContext(), sideEffect);
    result.addAttribute(CIRDialect::getSideEffectAttrName(), attr);
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  mlir::FunctionType opsFnTy;
  if (parser.parseType(opsFnTy))
    return mlir::failure();

  allResultTypes = opsFnTy.getResults();
  result.addTypes(allResultTypes);

  if (parser.resolveOperands(ops, opsFnTy.getInputs(), opsLoc, result.operands))
    return mlir::failure();

  return mlir::success();
}

static void printCallCommon(mlir::Operation *op,
                            mlir::FlatSymbolRefAttr calleeSym,
                            mlir::Value indirectCallee,
                            mlir::OpAsmPrinter &printer, bool isNothrow,
                            cir::SideEffect sideEffect) {
  printer << ' ';

  auto callLikeOp = mlir::cast<cir::CIRCallOpInterface>(op);
  auto ops = callLikeOp.getArgOperands();

  if (calleeSym) {
    // Direct calls
    printer.printAttributeWithoutType(calleeSym);
  } else {
    // Indirect calls
    assert(indirectCallee);
    printer << indirectCallee;
  }
  printer << "(" << ops << ")";

  if (isNothrow)
    printer << " nothrow";

  if (sideEffect != cir::SideEffect::All) {
    printer << " side_effect(";
    printer << stringifySideEffect(sideEffect);
    printer << ")";
  }

  printer.printOptionalAttrDict(op->getAttrs(),
                                {CIRDialect::getCalleeAttrName(),
                                 CIRDialect::getNoThrowAttrName(),
                                 CIRDialect::getSideEffectAttrName()});

  printer << " : ";
  printer.printFunctionalType(op->getOperands().getTypes(),
                              op->getResultTypes());
}

mlir::ParseResult cir::CallOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  return parseCallCommon(parser, result);
}

void cir::CallOp::print(mlir::OpAsmPrinter &p) {
  mlir::Value indirectCallee = isIndirect() ? getIndirectCall() : nullptr;
  cir::SideEffect sideEffect = getSideEffect();
  printCallCommon(*this, getCalleeAttr(), indirectCallee, p, getNothrow(),
                  sideEffect);
}

static LogicalResult
verifyCallCommInSymbolUses(mlir::Operation *op,
                           SymbolTableCollection &symbolTable) {
  auto fnAttr =
      op->getAttrOfType<FlatSymbolRefAttr>(CIRDialect::getCalleeAttrName());
  if (!fnAttr) {
    // This is an indirect call, thus we don't have to check the symbol uses.
    return mlir::success();
  }

  auto fn = symbolTable.lookupNearestSymbolFrom<cir::FuncOp>(op, fnAttr);
  if (!fn)
    return op->emitOpError() << "'" << fnAttr.getValue()
                             << "' does not reference a valid function";

  auto callIf = dyn_cast<cir::CIRCallOpInterface>(op);
  assert(callIf && "expected CIR call interface to be always available");

  // Verify that the operand and result types match the callee. Note that
  // argument-checking is disabled for functions without a prototype.
  auto fnType = fn.getFunctionType();
  if (!fn.getNoProto()) {
    unsigned numCallOperands = callIf.getNumArgOperands();
    unsigned numFnOpOperands = fnType.getNumInputs();

    if (!fnType.isVarArg() && numCallOperands != numFnOpOperands)
      return op->emitOpError("incorrect number of operands for callee");
    if (fnType.isVarArg() && numCallOperands < numFnOpOperands)
      return op->emitOpError("too few operands for callee");

    for (unsigned i = 0, e = numFnOpOperands; i != e; ++i)
      if (callIf.getArgOperand(i).getType() != fnType.getInput(i))
        return op->emitOpError("operand type mismatch: expected operand type ")
               << fnType.getInput(i) << ", but provided "
               << op->getOperand(i).getType() << " for operand number " << i;
  }

  assert(!cir::MissingFeatures::opCallCallConv());

  // Void function must not return any results.
  if (fnType.hasVoidReturn() && op->getNumResults() != 0)
    return op->emitOpError("callee returns void but call has results");

  // Non-void function calls must return exactly one result.
  if (!fnType.hasVoidReturn() && op->getNumResults() != 1)
    return op->emitOpError("incorrect number of results for callee");

  // Parent function and return value types must match.
  if (!fnType.hasVoidReturn() &&
      op->getResultTypes().front() != fnType.getReturnType()) {
    return op->emitOpError("result type mismatch: expected ")
           << fnType.getReturnType() << ", but provided "
           << op->getResult(0).getType();
  }

  return mlir::success();
}

LogicalResult
cir::CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyCallCommInSymbolUses(*this, symbolTable);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult checkReturnAndFunction(cir::ReturnOp op,
                                                  cir::FuncOp function) {
  // ReturnOps currently only have a single optional operand.
  if (op.getNumOperands() > 1)
    return op.emitOpError() << "expects at most 1 return operand";

  // Ensure returned type matches the function signature.
  auto expectedTy = function.getFunctionType().getReturnType();
  auto actualTy =
      (op.getNumOperands() == 0 ? cir::VoidType::get(op.getContext())
                                : op.getOperand(0).getType());
  if (actualTy != expectedTy)
    return op.emitOpError() << "returns " << actualTy
                            << " but enclosing function returns " << expectedTy;

  return mlir::success();
}

mlir::LogicalResult cir::ReturnOp::verify() {
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
// IfOp
//===----------------------------------------------------------------------===//

ParseResult cir::IfOp::parse(OpAsmParser &parser, OperationState &result) {
  // create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  mlir::Builder &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  Type boolType = cir::BoolType::get(builder.getContext());

  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, boolType, result.operands))
    return failure();

  // Parse 'then' region.
  mlir::SMLoc parseThenLoc = parser.getCurrentLocation();
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  if (ensureRegionTerm(parser, *thenRegion, parseThenLoc).failed())
    return failure();

  // If we find an 'else' keyword, parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    mlir::SMLoc parseElseLoc = parser.getCurrentLocation();
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    if (ensureRegionTerm(parser, *elseRegion, parseElseLoc).failed())
      return failure();
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void cir::IfOp::print(OpAsmPrinter &p) {
  p << " " << getCondition() << " ";
  mlir::Region &thenRegion = this->getThenRegion();
  p.printRegion(thenRegion,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!omitRegionTerm(thenRegion));

  // Print the 'else' regions if it exists and has a block.
  mlir::Region &elseRegion = this->getElseRegion();
  if (!elseRegion.empty()) {
    p << " else ";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!omitRegionTerm(elseRegion));
  }

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

/// Default callback for IfOp builders.
void cir::buildTerminatedBody(OpBuilder &builder, Location loc) {
  // add cir.yield to end of the block
  builder.create<cir::YieldOp>(loc);
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void cir::IfOp::getSuccessorRegions(mlir::RegionBranchPoint point,
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

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getThenRegion()));
  // If the else region does not exist, it is not a viable successor.
  if (elseRegion)
    regions.push_back(RegionSuccessor(elseRegion));

  return;
}

void cir::IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                      bool withElseRegion, BuilderCallbackRef thenBuilder,
                      BuilderCallbackRef elseBuilder) {
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

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes
/// that correspond to a constant value for each operand, or null if that
/// operand is not a constant.
void cir::ScopeOp::getSuccessorRegions(
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // The only region always branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(getODSResults(0)));
    return;
  }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getScopeRegion()));
}

void cir::ScopeOp::build(
    OpBuilder &builder, OperationState &result,
    function_ref<void(OpBuilder &, Type &, Location)> scopeBuilder) {
  assert(scopeBuilder && "the builder callback for 'then' must be present");

  OpBuilder::InsertionGuard guard(builder);
  Region *scopeRegion = result.addRegion();
  builder.createBlock(scopeRegion);
  assert(!cir::MissingFeatures::opScopeCleanupRegion());

  mlir::Type yieldTy;
  scopeBuilder(builder, yieldTy, result.location);

  if (yieldTy)
    result.addTypes(TypeRange{yieldTy});
}

void cir::ScopeOp::build(
    OpBuilder &builder, OperationState &result,
    function_ref<void(OpBuilder &, Location)> scopeBuilder) {
  assert(scopeBuilder && "the builder callback for 'then' must be present");
  OpBuilder::InsertionGuard guard(builder);
  Region *scopeRegion = result.addRegion();
  builder.createBlock(scopeRegion);
  assert(!cir::MissingFeatures::opScopeCleanupRegion());
  scopeBuilder(builder, result.location);
}

LogicalResult cir::ScopeOp::verify() {
  if (getRegion().empty()) {
    return emitOpError() << "cir.scope must not be empty since it should "
                            "include at least an implicit cir.yield ";
  }

  mlir::Block &lastBlock = getRegion().back();
  if (lastBlock.empty() || !lastBlock.mightHaveTerminator() ||
      !lastBlock.getTerminator()->hasTrait<OpTrait::IsTerminator>())
    return emitOpError() << "last block of cir.scope must be terminated";
  return success();
}

//===----------------------------------------------------------------------===//
// BrOp
//===----------------------------------------------------------------------===//

mlir::SuccessorOperands cir::BrOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return mlir::SuccessorOperands(getDestOperandsMutable());
}

Block *cir::BrOp::getSuccessorForOperands(ArrayRef<Attribute>) {
  return getDest();
}

//===----------------------------------------------------------------------===//
// BrCondOp
//===----------------------------------------------------------------------===//

mlir::SuccessorOperands cir::BrCondOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getDestOperandsTrueMutable()
                                      : getDestOperandsFalseMutable());
}

Block *cir::BrCondOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  if (IntegerAttr condAttr = dyn_cast_if_present<IntegerAttr>(operands.front()))
    return condAttr.getValue().isOne() ? getDestTrue() : getDestFalse();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// CaseOp
//===----------------------------------------------------------------------===//

void cir::CaseOp::getSuccessorRegions(
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }
  regions.push_back(RegionSuccessor(&getCaseRegion()));
}

void cir::CaseOp::build(OpBuilder &builder, OperationState &result,
                        ArrayAttr value, CaseOpKind kind,
                        OpBuilder::InsertPoint &insertPoint) {
  OpBuilder::InsertionGuard guardSwitch(builder);
  result.addAttribute("value", value);
  result.getOrAddProperties<Properties>().kind =
      cir::CaseOpKindAttr::get(builder.getContext(), kind);
  Region *caseRegion = result.addRegion();
  builder.createBlock(caseRegion);

  insertPoint = builder.saveInsertionPoint();
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

static ParseResult parseSwitchOp(OpAsmParser &parser, mlir::Region &regions,
                                 mlir::OpAsmParser::UnresolvedOperand &cond,
                                 mlir::Type &condType) {
  cir::IntType intCondType;

  if (parser.parseLParen())
    return mlir::failure();

  if (parser.parseOperand(cond))
    return mlir::failure();
  if (parser.parseColon())
    return mlir::failure();
  if (parser.parseCustomTypeWithFallback(intCondType))
    return mlir::failure();
  condType = intCondType;

  if (parser.parseRParen())
    return mlir::failure();
  if (parser.parseRegion(regions, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return mlir::success();
}

static void printSwitchOp(OpAsmPrinter &p, cir::SwitchOp op,
                          mlir::Region &bodyRegion, mlir::Value condition,
                          mlir::Type condType) {
  p << "(";
  p << condition;
  p << " : ";
  p.printStrippedAttrOrType(condType);
  p << ")";

  p << ' ';
  p.printRegion(bodyRegion, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

void cir::SwitchOp::getSuccessorRegions(
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &region) {
  if (!point.isParent()) {
    region.push_back(RegionSuccessor());
    return;
  }

  region.push_back(RegionSuccessor(&getBody()));
}

void cir::SwitchOp::build(OpBuilder &builder, OperationState &result,
                          Value cond, BuilderOpStateCallbackRef switchBuilder) {
  assert(switchBuilder && "the builder callback for regions must be present");
  OpBuilder::InsertionGuard guardSwitch(builder);
  Region *switchRegion = result.addRegion();
  builder.createBlock(switchRegion);
  result.addOperands({cond});
  switchBuilder(builder, result.location, result);
}

void cir::SwitchOp::collectCases(llvm::SmallVectorImpl<CaseOp> &cases) {
  walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    // Don't walk in nested switch op.
    if (isa<cir::SwitchOp>(op) && op != *this)
      return WalkResult::skip();

    if (auto caseOp = dyn_cast<cir::CaseOp>(op))
      cases.push_back(caseOp);

    return WalkResult::advance();
  });
}

bool cir::SwitchOp::isSimpleForm(llvm::SmallVectorImpl<CaseOp> &cases) {
  collectCases(cases);

  if (getBody().empty())
    return false;

  if (!isa<YieldOp>(getBody().front().back()))
    return false;

  if (!llvm::all_of(getBody().front(),
                    [](Operation &op) { return isa<CaseOp, YieldOp>(op); }))
    return false;

  return llvm::all_of(cases, [this](CaseOp op) {
    return op->getParentOfType<SwitchOp>() == *this;
  });
}

//===----------------------------------------------------------------------===//
// SwitchFlatOp
//===----------------------------------------------------------------------===//

void cir::SwitchFlatOp::build(OpBuilder &builder, OperationState &result,
                              Value value, Block *defaultDestination,
                              ValueRange defaultOperands,
                              ArrayRef<APInt> caseValues,
                              BlockRange caseDestinations,
                              ArrayRef<ValueRange> caseOperands) {

  std::vector<mlir::Attribute> caseValuesAttrs;
  for (const APInt &val : caseValues)
    caseValuesAttrs.push_back(cir::IntAttr::get(value.getType(), val));
  mlir::ArrayAttr attrs = ArrayAttr::get(builder.getContext(), caseValuesAttrs);

  build(builder, result, value, defaultOperands, caseOperands, attrs,
        defaultDestination, caseDestinations);
}

/// <cases> ::= `[` (case (`,` case )* )? `]`
/// <case>  ::= integer `:` bb-id (`(` ssa-use-and-type-list `)`)?
static ParseResult parseSwitchFlatOpCases(
    OpAsmParser &parser, Type flagType, mlir::ArrayAttr &caseValues,
    SmallVectorImpl<Block *> &caseDestinations,
    SmallVectorImpl<llvm::SmallVector<OpAsmParser::UnresolvedOperand>>
        &caseOperands,
    SmallVectorImpl<llvm::SmallVector<Type>> &caseOperandTypes) {
  if (failed(parser.parseLSquare()))
    return failure();
  if (succeeded(parser.parseOptionalRSquare()))
    return success();
  llvm::SmallVector<mlir::Attribute> values;

  auto parseCase = [&]() {
    int64_t value = 0;
    if (failed(parser.parseInteger(value)))
      return failure();

    values.push_back(cir::IntAttr::get(flagType, value));

    Block *destination;
    llvm::SmallVector<OpAsmParser::UnresolvedOperand> operands;
    llvm::SmallVector<Type> operandTypes;
    if (parser.parseColon() || parser.parseSuccessor(destination))
      return failure();
    if (!parser.parseOptionalLParen()) {
      if (parser.parseOperandList(operands, OpAsmParser::Delimiter::None,
                                  /*allowResultNumber=*/false) ||
          parser.parseColonTypeList(operandTypes) || parser.parseRParen())
        return failure();
    }
    caseDestinations.push_back(destination);
    caseOperands.emplace_back(operands);
    caseOperandTypes.emplace_back(operandTypes);
    return success();
  };
  if (failed(parser.parseCommaSeparatedList(parseCase)))
    return failure();

  caseValues = ArrayAttr::get(flagType.getContext(), values);

  return parser.parseRSquare();
}

static void printSwitchFlatOpCases(OpAsmPrinter &p, cir::SwitchFlatOp op,
                                   Type flagType, mlir::ArrayAttr caseValues,
                                   SuccessorRange caseDestinations,
                                   OperandRangeRange caseOperands,
                                   const TypeRangeRange &caseOperandTypes) {
  p << '[';
  p.printNewline();
  if (!caseValues) {
    p << ']';
    return;
  }

  size_t index = 0;
  llvm::interleave(
      llvm::zip(caseValues, caseDestinations),
      [&](auto i) {
        p << "  ";
        mlir::Attribute a = std::get<0>(i);
        p << mlir::cast<cir::IntAttr>(a).getValue();
        p << ": ";
        p.printSuccessorAndUseList(std::get<1>(i), caseOperands[index++]);
      },
      [&] {
        p << ',';
        p.printNewline();
      });
  p.printNewline();
  p << ']';
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static ParseResult parseConstantValue(OpAsmParser &parser,
                                      mlir::Attribute &valueAttr) {
  NamedAttrList attr;
  return parser.parseAttribute(valueAttr, "value", attr);
}

static void printConstant(OpAsmPrinter &p, Attribute value) {
  p.printAttribute(value);
}

mlir::LogicalResult cir::GlobalOp::verify() {
  // Verify that the initial value, if present, is either a unit attribute or
  // an attribute CIR supports.
  if (getInitialValue().has_value()) {
    if (checkConstantTypes(getOperation(), getSymType(), *getInitialValue())
            .failed())
      return failure();
  }

  // TODO(CIR): Many other checks for properties that haven't been upstreamed
  // yet.

  return success();
}

void cir::GlobalOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, llvm::StringRef sym_name,
    mlir::Type sym_type, bool isConstant, cir::GlobalLinkageKind linkage,
    function_ref<void(OpBuilder &, Location)> ctorBuilder,
    function_ref<void(OpBuilder &, Location)> dtorBuilder) {
  odsState.addAttribute(getSymNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getSymTypeAttrName(odsState.name),
                        mlir::TypeAttr::get(sym_type));
  if (isConstant)
    odsState.addAttribute(getConstantAttrName(odsState.name),
                          odsBuilder.getUnitAttr());

  cir::GlobalLinkageKindAttr linkageAttr =
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

  odsState.addAttribute(getGlobalVisibilityAttrName(odsState.name),
                        cir::VisibilityAttr::get(odsBuilder.getContext()));
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void cir::GlobalOp::getSuccessorRegions(
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
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

static void printGlobalOpTypeAndInitialValue(OpAsmPrinter &p, cir::GlobalOp op,
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
    //  cir.global @a : !cir.int<s, 32>
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
      if (ensureRegionTerm(parser, ctorRegion, parseLoc).failed())
        return failure();
    } else {
      // Parse constant with initializer, examples:
      //  cir.global @y = 3.400000e+00 : f32
      //  cir.global @rgb = #cir.const_array<[...] : !cir.array<i8 x 3>>
      if (parseConstantValue(parser, initialValueAttr).failed())
        return failure();

      assert(mlir::isa<mlir::TypedAttr>(initialValueAttr) &&
             "Non-typed attrs shouldn't appear here.");
      auto typedAttr = mlir::cast<mlir::TypedAttr>(initialValueAttr);
      opTy = typedAttr.getType();
    }

    // Parse destructor, example:
    //   dtor { ... }
    if (!parser.parseOptionalKeyword("dtor")) {
      auto parseLoc = parser.getCurrentLocation();
      if (parser.parseRegion(dtorRegion, /*arguments=*/{}, /*argTypes=*/{}))
        return failure();
      if (ensureRegionTerm(parser, dtorRegion, parseLoc).failed())
        return failure();
    }
  }

  typeAttr = TypeAttr::get(opTy);
  return success();
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

LogicalResult
cir::GetGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify that the result type underlying pointer type matches the type of
  // the referenced cir.global or cir.func op.
  mlir::Operation *op =
      symbolTable.lookupNearestSymbolFrom(*this, getNameAttr());
  if (op == nullptr || !(isa<GlobalOp>(op) || isa<FuncOp>(op)))
    return emitOpError("'")
           << getName()
           << "' does not reference a valid cir.global or cir.func";

  mlir::Type symTy;
  if (auto g = dyn_cast<GlobalOp>(op)) {
    symTy = g.getSymType();
    assert(!cir::MissingFeatures::addressSpace());
    assert(!cir::MissingFeatures::opGlobalThreadLocal());
  } else if (auto f = dyn_cast<FuncOp>(op)) {
    symTy = f.getFunctionType();
  } else {
    llvm_unreachable("Unexpected operation for GetGlobalOp");
  }

  auto resultType = dyn_cast<PointerType>(getAddr().getType());
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
cir::VTableAddrPointOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  StringRef name = getName();

  // Verify that the result type underlying pointer type matches the type of
  // the referenced cir.global.
  auto op =
      symbolTable.lookupNearestSymbolFrom<cir::GlobalOp>(*this, getNameAttr());
  if (!op)
    return emitOpError("'")
           << name << "' does not reference a valid cir.global";
  std::optional<mlir::Attribute> init = op.getInitialValue();
  if (!init)
    return success();
  if (!isa<cir::VTableAttr>(*init))
    return emitOpError("Expected #cir.vtable in initializer for global '")
           << name << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// VTTAddrPointOp
//===----------------------------------------------------------------------===//

LogicalResult
cir::VTTAddrPointOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // VTT ptr is not coming from a symbol.
  if (!getName())
    return success();
  StringRef name = *getName();

  // Verify that the result type underlying pointer type matches the type of
  // the referenced cir.global op.
  auto op =
      symbolTable.lookupNearestSymbolFrom<cir::GlobalOp>(*this, getNameAttr());
  if (!op)
    return emitOpError("'")
           << name << "' does not reference a valid cir.global";
  std::optional<mlir::Attribute> init = op.getInitialValue();
  if (!init)
    return success();
  if (!isa<cir::ConstArrayAttr>(*init))
    return emitOpError(
               "Expected constant array in initializer for global VTT '")
           << name << "'";
  return success();
}

LogicalResult cir::VTTAddrPointOp::verify() {
  // The operation uses either a symbol or a value to operate, but not both
  if (getName() && getSymAddr())
    return emitOpError("should use either a symbol or value, but not both");

  // If not a symbol, stick with the concrete type used for getSymAddr.
  if (getSymAddr())
    return success();

  mlir::Type resultType = getAddr().getType();
  mlir::Type resTy = cir::PointerType::get(
      cir::PointerType::get(cir::VoidType::get(getContext())));

  if (resultType != resTy)
    return emitOpError("result type must be ")
           << resTy << ", but provided result type is " << resultType;
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// Returns the name used for the linkage attribute. This *must* correspond to
/// the name of the attribute in ODS.
static llvm::StringRef getLinkageAttrNameString() { return "linkage"; }

void cir::FuncOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, FuncType type,
                        GlobalLinkageKind linkage) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));
  result.addAttribute(
      getLinkageAttrNameString(),
      GlobalLinkageKindAttr::get(builder.getContext(), linkage));
  result.addAttribute(getGlobalVisibilityAttrName(result.name),
                      cir::VisibilityAttr::get(builder.getContext()));
}

ParseResult cir::FuncOp::parse(OpAsmParser &parser, OperationState &state) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  mlir::Builder &builder = parser.getBuilder();

  mlir::StringAttr builtinNameAttr = getBuiltinAttrName(state.name);
  mlir::StringAttr coroutineNameAttr = getCoroutineAttrName(state.name);
  mlir::StringAttr lambdaNameAttr = getLambdaAttrName(state.name);
  mlir::StringAttr noProtoNameAttr = getNoProtoAttrName(state.name);
  mlir::StringAttr visNameAttr = getSymVisibilityAttrName(state.name);
  mlir::StringAttr visibilityNameAttr = getGlobalVisibilityAttrName(state.name);
  mlir::StringAttr dsoLocalNameAttr = getDsoLocalAttrName(state.name);

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

  cir::VisibilityAttr cirVisibilityAttr;
  parseVisibilityAttr(parser, cirVisibilityAttr);
  state.addAttribute(visibilityNameAttr, cirVisibilityAttr);

  if (parser.parseOptionalKeyword(dsoLocalNameAttr).succeeded())
    state.addAttribute(dsoLocalNameAttr, parser.getBuilder().getUnitAttr());

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return failure();
  llvm::SmallVector<OpAsmParser::Argument, 8> arguments;
  llvm::SmallVector<mlir::Type> resultTypes;
  llvm::SmallVector<DictionaryAttr> resultAttrs;
  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignatureWithArguments(
          parser, /*allowVariadic=*/true, arguments, isVariadic, resultTypes,
          resultAttrs))
    return failure();
  llvm::SmallVector<mlir::Type> argTypes;
  for (OpAsmParser::Argument &arg : arguments)
    argTypes.push_back(arg.type);

  if (resultTypes.size() > 1) {
    return parser.emitError(
        loc, "functions with multiple return types are not supported");
  }

  mlir::Type returnType =
      (resultTypes.empty() ? cir::VoidType::get(builder.getContext())
                           : resultTypes.front());

  cir::FuncType fnType = cir::FuncType::get(argTypes, returnType, isVariadic);
  if (!fnType)
    return failure();
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(fnType));

  bool hasAlias = false;
  mlir::StringAttr aliaseeNameAttr = getAliaseeAttrName(state.name);
  if (parser.parseOptionalKeyword("alias").succeeded()) {
    if (parser.parseLParen().failed())
      return failure();
    mlir::StringAttr aliaseeAttr;
    if (parser.parseOptionalSymbolName(aliaseeAttr).failed())
      return failure();
    state.addAttribute(aliaseeNameAttr, FlatSymbolRefAttr::get(aliaseeAttr));
    if (parser.parseRParen().failed())
      return failure();
    hasAlias = true;
  }

  auto parseGlobalDtorCtor =
      [&](StringRef keyword,
          llvm::function_ref<void(std::optional<int> prio)> createAttr)
      -> mlir::LogicalResult {
    if (mlir::succeeded(parser.parseOptionalKeyword(keyword))) {
      std::optional<int> priority;
      if (mlir::succeeded(parser.parseOptionalLParen())) {
        auto parsedPriority = mlir::FieldParser<int>::parse(parser);
        if (mlir::failed(parsedPriority))
          return parser.emitError(parser.getCurrentLocation(),
                                  "failed to parse 'priority', of type 'int'");
        priority = parsedPriority.value_or(int());
        // Parse literal ')'
        if (parser.parseRParen())
          return failure();
      }
      createAttr(priority);
    }
    return success();
  };

  if (parseGlobalDtorCtor("global_ctor", [&](std::optional<int> priority) {
        mlir::IntegerAttr globalCtorPriorityAttr =
            builder.getI32IntegerAttr(priority.value_or(65535));
        state.addAttribute(getGlobalCtorPriorityAttrName(state.name),
                           globalCtorPriorityAttr);
      }).failed())
    return failure();

  if (parseGlobalDtorCtor("global_dtor", [&](std::optional<int> priority) {
        mlir::IntegerAttr globalDtorPriorityAttr =
            builder.getI32IntegerAttr(priority.value_or(65535));
        state.addAttribute(getGlobalDtorPriorityAttrName(state.name),
                           globalDtorPriorityAttr);
      }).failed())
    return failure();

  // Parse the optional function body.
  auto *body = state.addRegion();
  OptionalParseResult parseResult = parser.parseOptionalRegion(
      *body, arguments, /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
    if (hasAlias)
      return parser.emitError(loc, "function alias shall not have a body");
    if (failed(*parseResult))
      return failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return parser.emitError(loc, "expected non-empty function body");
  }

  return success();
}

// This function corresponds to `llvm::GlobalValue::isDeclaration` and should
// have a similar implementation. We don't currently ifuncs or materializable
// functions, but those should be handled here as they are implemented.
bool cir::FuncOp::isDeclaration() {
  assert(!cir::MissingFeatures::supportIFuncAttr());

  std::optional<StringRef> aliasee = getAliasee();
  if (!aliasee)
    return getFunctionBody().empty();

  // Aliases are always definitions.
  return false;
}

mlir::Region *cir::FuncOp::getCallableRegion() {
  // TODO(CIR): This function will have special handling for aliases and a
  // check for an external function, once those features have been upstreamed.
  return &getBody();
}

void cir::FuncOp::print(OpAsmPrinter &p) {
  if (getBuiltin())
    p << " builtin";

  if (getCoroutine())
    p << " coroutine";

  if (getLambda())
    p << " lambda";

  if (getNoProto())
    p << " no_proto";

  if (getComdat())
    p << " comdat";

  if (getLinkage() != GlobalLinkageKind::ExternalLinkage)
    p << ' ' << stringifyGlobalLinkageKind(getLinkage());

  mlir::SymbolTable::Visibility vis = getVisibility();
  if (vis != mlir::SymbolTable::Visibility::Public)
    p << ' ' << vis;

  cir::VisibilityAttr cirVisibilityAttr = getGlobalVisibilityAttr();
  if (!cirVisibilityAttr.isDefault()) {
    p << ' ';
    printVisibilityAttr(p, cirVisibilityAttr);
  }

  if (getDsoLocal())
    p << " dso_local";

  p << ' ';
  p.printSymbolName(getSymName());
  cir::FuncType fnType = getFunctionType();
  function_interface_impl::printFunctionSignature(
      p, *this, fnType.getInputs(), fnType.isVarArg(), fnType.getReturnTypes());

  if (std::optional<StringRef> aliaseeName = getAliasee()) {
    p << " alias(";
    p.printSymbolName(*aliaseeName);
    p << ")";
  }

  if (auto globalCtorPriority = getGlobalCtorPriority()) {
    p << " global_ctor";
    if (globalCtorPriority.value() != 65535)
      p << "(" << globalCtorPriority.value() << ")";
  }

  if (auto globalDtorPriority = getGlobalDtorPriority()) {
    p << " global_dtor";
    if (globalDtorPriority.value() != 65535)
      p << "(" << globalDtorPriority.value() << ")";
  }

  // Print the body if this is not an external function.
  Region &body = getOperation()->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

mlir::LogicalResult cir::FuncOp::verify() {

  llvm::SmallSet<llvm::StringRef, 16> labels;
  llvm::SmallSet<llvm::StringRef, 16> gotos;

  getOperation()->walk([&](mlir::Operation *op) {
    if (auto lab = dyn_cast<cir::LabelOp>(op)) {
      labels.insert(lab.getLabel());
    } else if (auto goTo = dyn_cast<cir::GotoOp>(op)) {
      gotos.insert(goTo.getLabel());
    }
  });

  if (!labels.empty() || !gotos.empty()) {
    llvm::SmallSet<llvm::StringRef, 16> mismatched =
        llvm::set_difference(gotos, labels);

    if (!mismatched.empty())
      return emitOpError() << "goto/label mismatch";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BinOp
//===----------------------------------------------------------------------===//
LogicalResult cir::BinOp::verify() {
  bool noWrap = getNoUnsignedWrap() || getNoSignedWrap();
  bool saturated = getSaturated();

  if (!isa<cir::IntType>(getType()) && noWrap)
    return emitError()
           << "only operations on integer values may have nsw/nuw flags";

  bool noWrapOps = getKind() == cir::BinOpKind::Add ||
                   getKind() == cir::BinOpKind::Sub ||
                   getKind() == cir::BinOpKind::Mul;

  bool saturatedOps =
      getKind() == cir::BinOpKind::Add || getKind() == cir::BinOpKind::Sub;

  if (noWrap && !noWrapOps)
    return emitError() << "The nsw/nuw flags are applicable to opcodes: 'add', "
                          "'sub' and 'mul'";
  if (saturated && !saturatedOps)
    return emitError() << "The saturated flag is applicable to opcodes: 'add' "
                          "and 'sub'";
  if (noWrap && saturated)
    return emitError() << "The nsw/nuw flags and the saturated flag are "
                          "mutually exclusive";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

/// Given the region at `point`, or the parent operation if `point` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void cir::TernaryOp::getSuccessorRegions(
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // The `true` and the `false` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(this->getODSResults(0)));
    return;
  }

  // When branching from the parent operation, both the true and false
  // regions are considered possible successors
  regions.push_back(RegionSuccessor(&getTrueRegion()));
  regions.push_back(RegionSuccessor(&getFalseRegion()));
}

void cir::TernaryOp::build(
    OpBuilder &builder, OperationState &result, Value cond,
    function_ref<void(OpBuilder &, Location)> trueBuilder,
    function_ref<void(OpBuilder &, Location)> falseBuilder) {
  result.addOperands(cond);
  OpBuilder::InsertionGuard guard(builder);
  Region *trueRegion = result.addRegion();
  Block *block = builder.createBlock(trueRegion);
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
// SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::SelectOp::fold(FoldAdaptor adaptor) {
  mlir::Attribute condition = adaptor.getCondition();
  if (condition) {
    bool conditionValue = mlir::cast<cir::BoolAttr>(condition).getValue();
    return conditionValue ? getTrueValue() : getFalseValue();
  }

  // cir.select if %0 then x else x -> x
  mlir::Attribute trueValue = adaptor.getTrueValue();
  mlir::Attribute falseValue = adaptor.getFalseValue();
  if (trueValue == falseValue)
    return trueValue;
  if (getTrueValue() == getFalseValue())
    return getTrueValue();

  return {};
}

//===----------------------------------------------------------------------===//
// ShiftOp
//===----------------------------------------------------------------------===//
LogicalResult cir::ShiftOp::verify() {
  mlir::Operation *op = getOperation();
  auto op0VecTy = mlir::dyn_cast<cir::VectorType>(op->getOperand(0).getType());
  auto op1VecTy = mlir::dyn_cast<cir::VectorType>(op->getOperand(1).getType());
  if (!op0VecTy ^ !op1VecTy)
    return emitOpError() << "input types cannot be one vector and one scalar";

  if (op0VecTy) {
    if (op0VecTy.getSize() != op1VecTy.getSize())
      return emitOpError() << "input vector types must have the same size";

    auto opResultTy = mlir::dyn_cast<cir::VectorType>(getType());
    if (!opResultTy)
      return emitOpError() << "the type of the result must be a vector "
                           << "if it is vector shift";

    auto op0VecEleTy = mlir::cast<cir::IntType>(op0VecTy.getElementType());
    auto op1VecEleTy = mlir::cast<cir::IntType>(op1VecTy.getElementType());
    if (op0VecEleTy.getWidth() != op1VecEleTy.getWidth())
      return emitOpError()
             << "vector operands do not have the same elements sizes";

    auto resVecEleTy = mlir::cast<cir::IntType>(opResultTy.getElementType());
    if (op0VecEleTy.getWidth() != resVecEleTy.getWidth())
      return emitOpError() << "vector operands and result type do not have the "
                              "same elements sizes";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// LabelOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::LabelOp::verify() {
  mlir::Operation *op = getOperation();
  mlir::Block *blk = op->getBlock();
  if (&blk->front() != op)
    return emitError() << "must be the first operation in a block";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

LogicalResult cir::UnaryOp::verify() {
  switch (getKind()) {
  case cir::UnaryOpKind::Inc:
  case cir::UnaryOpKind::Dec:
  case cir::UnaryOpKind::Plus:
  case cir::UnaryOpKind::Minus:
  case cir::UnaryOpKind::Not:
    // Nothing to verify.
    return success();
  }

  llvm_unreachable("Unknown UnaryOp kind?");
}

static bool isBoolNot(cir::UnaryOp op) {
  return isa<cir::BoolType>(op.getInput().getType()) &&
         op.getKind() == cir::UnaryOpKind::Not;
}

// This folder simplifies the sequential boolean not operations.
// For instance, the next two unary operations will be eliminated:
//
// ```mlir
// %1 = cir.unary(not, %0) : !cir.bool, !cir.bool
// %2 = cir.unary(not, %1) : !cir.bool, !cir.bool
// ```
//
// and the argument of the first one (%0) will be used instead.
OpFoldResult cir::UnaryOp::fold(FoldAdaptor adaptor) {
  if (auto poison =
          mlir::dyn_cast_if_present<cir::PoisonAttr>(adaptor.getInput())) {
    // Propagate poison values
    return poison;
  }

  if (isBoolNot(*this))
    if (auto previous = getInput().getDefiningOp<cir::UnaryOp>())
      if (isBoolNot(previous))
        return previous.getInput();

  return {};
}

//===----------------------------------------------------------------------===//
// CopyOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::CopyOp::verify() {
  // A data layout is required for us to know the number of bytes to be copied.
  if (!getType().getPointee().hasTrait<DataLayoutTypeInterface::Trait>())
    return emitError() << "missing data layout for pointee type";

  if (getSrc() == getDst())
    return emitError() << "source and destination are the same";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GetMemberOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::GetMemberOp::verify() {
  const auto recordTy = dyn_cast<RecordType>(getAddrTy().getPointee());
  if (!recordTy)
    return emitError() << "expected pointer to a record type";

  if (recordTy.getMembers().size() <= getIndex())
    return emitError() << "member index out of bounds";

  if (recordTy.getMembers()[getIndex()] != getType().getPointee())
    return emitError() << "member type mismatch";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// VecCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::VecCreateOp::fold(FoldAdaptor adaptor) {
  if (llvm::any_of(getElements(), [](mlir::Value value) {
        return !value.getDefiningOp<cir::ConstantOp>();
      }))
    return {};

  return cir::ConstVectorAttr::get(
      getType(), mlir::ArrayAttr::get(getContext(), adaptor.getElements()));
}

LogicalResult cir::VecCreateOp::verify() {
  // Verify that the number of arguments matches the number of elements in the
  // vector, and that the type of all the arguments matches the type of the
  // elements in the vector.
  const cir::VectorType vecTy = getType();
  if (getElements().size() != vecTy.getSize()) {
    return emitOpError() << "operand count of " << getElements().size()
                         << " doesn't match vector type " << vecTy
                         << " element count of " << vecTy.getSize();
  }

  const mlir::Type elementType = vecTy.getElementType();
  for (const mlir::Value element : getElements()) {
    if (element.getType() != elementType) {
      return emitOpError() << "operand type " << element.getType()
                           << " doesn't match vector element type "
                           << elementType;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VecExtractOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::VecExtractOp::fold(FoldAdaptor adaptor) {
  const auto vectorAttr =
      llvm::dyn_cast_if_present<cir::ConstVectorAttr>(adaptor.getVec());
  if (!vectorAttr)
    return {};

  const auto indexAttr =
      llvm::dyn_cast_if_present<cir::IntAttr>(adaptor.getIndex());
  if (!indexAttr)
    return {};

  const mlir::ArrayAttr elements = vectorAttr.getElts();
  const uint64_t index = indexAttr.getUInt();
  if (index >= elements.size())
    return {};

  return elements[index];
}

//===----------------------------------------------------------------------===//
// VecCmpOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::VecCmpOp::fold(FoldAdaptor adaptor) {
  auto lhsVecAttr =
      mlir::dyn_cast_if_present<cir::ConstVectorAttr>(adaptor.getLhs());
  auto rhsVecAttr =
      mlir::dyn_cast_if_present<cir::ConstVectorAttr>(adaptor.getRhs());
  if (!lhsVecAttr || !rhsVecAttr)
    return {};

  mlir::Type inputElemTy =
      mlir::cast<cir::VectorType>(lhsVecAttr.getType()).getElementType();
  if (!isAnyIntegerOrFloatingPointType(inputElemTy))
    return {};

  cir::CmpOpKind opKind = adaptor.getKind();
  mlir::ArrayAttr lhsVecElhs = lhsVecAttr.getElts();
  mlir::ArrayAttr rhsVecElhs = rhsVecAttr.getElts();
  uint64_t vecSize = lhsVecElhs.size();

  SmallVector<mlir::Attribute, 16> elements(vecSize);
  bool isIntAttr = vecSize && mlir::isa<cir::IntAttr>(lhsVecElhs[0]);
  for (uint64_t i = 0; i < vecSize; i++) {
    mlir::Attribute lhsAttr = lhsVecElhs[i];
    mlir::Attribute rhsAttr = rhsVecElhs[i];
    int cmpResult = 0;
    switch (opKind) {
    case cir::CmpOpKind::lt: {
      if (isIntAttr) {
        cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() <
                    mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() <
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    case cir::CmpOpKind::le: {
      if (isIntAttr) {
        cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() <=
                    mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() <=
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    case cir::CmpOpKind::gt: {
      if (isIntAttr) {
        cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() >
                    mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() >
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    case cir::CmpOpKind::ge: {
      if (isIntAttr) {
        cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() >=
                    mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() >=
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    case cir::CmpOpKind::eq: {
      if (isIntAttr) {
        cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() ==
                    mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() ==
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    case cir::CmpOpKind::ne: {
      if (isIntAttr) {
        cmpResult = mlir::cast<cir::IntAttr>(lhsAttr).getSInt() !=
                    mlir::cast<cir::IntAttr>(rhsAttr).getSInt();
      } else {
        cmpResult = mlir::cast<cir::FPAttr>(lhsAttr).getValue() !=
                    mlir::cast<cir::FPAttr>(rhsAttr).getValue();
      }
      break;
    }
    }

    elements[i] = cir::IntAttr::get(getType().getElementType(), cmpResult);
  }

  return cir::ConstVectorAttr::get(
      getType(), mlir::ArrayAttr::get(getContext(), elements));
}

//===----------------------------------------------------------------------===//
// VecShuffleOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::VecShuffleOp::fold(FoldAdaptor adaptor) {
  auto vec1Attr =
      mlir::dyn_cast_if_present<cir::ConstVectorAttr>(adaptor.getVec1());
  auto vec2Attr =
      mlir::dyn_cast_if_present<cir::ConstVectorAttr>(adaptor.getVec2());
  if (!vec1Attr || !vec2Attr)
    return {};

  mlir::Type vec1ElemTy =
      mlir::cast<cir::VectorType>(vec1Attr.getType()).getElementType();

  mlir::ArrayAttr vec1Elts = vec1Attr.getElts();
  mlir::ArrayAttr vec2Elts = vec2Attr.getElts();
  mlir::ArrayAttr indicesElts = adaptor.getIndices();

  SmallVector<mlir::Attribute, 16> elements;
  elements.reserve(indicesElts.size());

  uint64_t vec1Size = vec1Elts.size();
  for (const auto &idxAttr : indicesElts.getAsRange<cir::IntAttr>()) {
    if (idxAttr.getSInt() == -1) {
      elements.push_back(cir::UndefAttr::get(vec1ElemTy));
      continue;
    }

    uint64_t idxValue = idxAttr.getUInt();
    elements.push_back(idxValue < vec1Size ? vec1Elts[idxValue]
                                           : vec2Elts[idxValue - vec1Size]);
  }

  return cir::ConstVectorAttr::get(
      getType(), mlir::ArrayAttr::get(getContext(), elements));
}

LogicalResult cir::VecShuffleOp::verify() {
  // The number of elements in the indices array must match the number of
  // elements in the result type.
  if (getIndices().size() != getResult().getType().getSize()) {
    return emitOpError() << ": the number of elements in " << getIndices()
                         << " and " << getResult().getType() << " don't match";
  }

  // The element types of the two input vectors and of the result type must
  // match.
  if (getVec1().getType().getElementType() !=
      getResult().getType().getElementType()) {
    return emitOpError() << ": element types of " << getVec1().getType()
                         << " and " << getResult().getType() << " don't match";
  }

  const uint64_t maxValidIndex =
      getVec1().getType().getSize() + getVec2().getType().getSize() - 1;
  if (llvm::any_of(
          getIndices().getAsRange<cir::IntAttr>(), [&](cir::IntAttr idxAttr) {
            return idxAttr.getSInt() != -1 && idxAttr.getUInt() > maxValidIndex;
          })) {
    return emitOpError() << ": index for __builtin_shufflevector must be "
                            "less than the total number of vector elements";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VecShuffleDynamicOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::VecShuffleDynamicOp::fold(FoldAdaptor adaptor) {
  mlir::Attribute vec = adaptor.getVec();
  mlir::Attribute indices = adaptor.getIndices();
  if (mlir::isa_and_nonnull<cir::ConstVectorAttr>(vec) &&
      mlir::isa_and_nonnull<cir::ConstVectorAttr>(indices)) {
    auto vecAttr = mlir::cast<cir::ConstVectorAttr>(vec);
    auto indicesAttr = mlir::cast<cir::ConstVectorAttr>(indices);

    mlir::ArrayAttr vecElts = vecAttr.getElts();
    mlir::ArrayAttr indicesElts = indicesAttr.getElts();

    const uint64_t numElements = vecElts.size();

    SmallVector<mlir::Attribute, 16> elements;
    elements.reserve(numElements);

    const uint64_t maskBits = llvm::NextPowerOf2(numElements - 1) - 1;
    for (const auto &idxAttr : indicesElts.getAsRange<cir::IntAttr>()) {
      uint64_t idxValue = idxAttr.getUInt();
      uint64_t newIdx = idxValue & maskBits;
      elements.push_back(vecElts[newIdx]);
    }

    return cir::ConstVectorAttr::get(
        getType(), mlir::ArrayAttr::get(getContext(), elements));
  }

  return {};
}

LogicalResult cir::VecShuffleDynamicOp::verify() {
  // The number of elements in the two input vectors must match.
  if (getVec().getType().getSize() !=
      mlir::cast<cir::VectorType>(getIndices().getType()).getSize()) {
    return emitOpError() << ": the number of elements in " << getVec().getType()
                         << " and " << getIndices().getType() << " don't match";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VecTernaryOp
//===----------------------------------------------------------------------===//

LogicalResult cir::VecTernaryOp::verify() {
  // Verify that the condition operand has the same number of elements as the
  // other operands.  (The automatic verification already checked that all
  // operands are vector types and that the second and third operands are the
  // same type.)
  if (getCond().getType().getSize() != getLhs().getType().getSize()) {
    return emitOpError() << ": the number of elements in "
                         << getCond().getType() << " and " << getLhs().getType()
                         << " don't match";
  }
  return success();
}

OpFoldResult cir::VecTernaryOp::fold(FoldAdaptor adaptor) {
  mlir::Attribute cond = adaptor.getCond();
  mlir::Attribute lhs = adaptor.getLhs();
  mlir::Attribute rhs = adaptor.getRhs();

  if (!mlir::isa_and_nonnull<cir::ConstVectorAttr>(cond) ||
      !mlir::isa_and_nonnull<cir::ConstVectorAttr>(lhs) ||
      !mlir::isa_and_nonnull<cir::ConstVectorAttr>(rhs))
    return {};
  auto condVec = mlir::cast<cir::ConstVectorAttr>(cond);
  auto lhsVec = mlir::cast<cir::ConstVectorAttr>(lhs);
  auto rhsVec = mlir::cast<cir::ConstVectorAttr>(rhs);

  mlir::ArrayAttr condElts = condVec.getElts();

  SmallVector<mlir::Attribute, 16> elements;
  elements.reserve(condElts.size());

  for (const auto &[idx, condAttr] :
       llvm::enumerate(condElts.getAsRange<cir::IntAttr>())) {
    if (condAttr.getSInt()) {
      elements.push_back(lhsVec.getElts()[idx]);
    } else {
      elements.push_back(rhsVec.getElts()[idx]);
    }
  }

  cir::VectorType vecTy = getLhs().getType();
  return cir::ConstVectorAttr::get(
      vecTy, mlir::ArrayAttr::get(getContext(), elements));
}

//===----------------------------------------------------------------------===//
// ComplexCreateOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexCreateOp::verify() {
  if (getType().getElementType() != getReal().getType()) {
    emitOpError()
        << "operand type of cir.complex.create does not match its result type";
    return failure();
  }

  return success();
}

OpFoldResult cir::ComplexCreateOp::fold(FoldAdaptor adaptor) {
  mlir::Attribute real = adaptor.getReal();
  mlir::Attribute imag = adaptor.getImag();
  if (!real || !imag)
    return {};

  // When both of real and imag are constants, we can fold the operation into an
  // `#cir.const_complex` operation.
  auto realAttr = mlir::cast<mlir::TypedAttr>(real);
  auto imagAttr = mlir::cast<mlir::TypedAttr>(imag);
  return cir::ConstComplexAttr::get(realAttr, imagAttr);
}

//===----------------------------------------------------------------------===//
// ComplexRealOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexRealOp::verify() {
  mlir::Type operandTy = getOperand().getType();
  if (auto complexOperandTy = mlir::dyn_cast<cir::ComplexType>(operandTy))
    operandTy = complexOperandTy.getElementType();

  if (getType() != operandTy) {
    emitOpError() << ": result type does not match operand type";
    return failure();
  }

  return success();
}

OpFoldResult cir::ComplexRealOp::fold(FoldAdaptor adaptor) {
  if (!mlir::isa<cir::ComplexType>(getOperand().getType()))
    return nullptr;

  if (auto complexCreateOp = getOperand().getDefiningOp<cir::ComplexCreateOp>())
    return complexCreateOp.getOperand(0);

  auto complex =
      mlir::cast_if_present<cir::ConstComplexAttr>(adaptor.getOperand());
  return complex ? complex.getReal() : nullptr;
}

//===----------------------------------------------------------------------===//
// ComplexImagOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexImagOp::verify() {
  mlir::Type operandTy = getOperand().getType();
  if (auto complexOperandTy = mlir::dyn_cast<cir::ComplexType>(operandTy))
    operandTy = complexOperandTy.getElementType();

  if (getType() != operandTy) {
    emitOpError() << ": result type does not match operand type";
    return failure();
  }

  return success();
}

OpFoldResult cir::ComplexImagOp::fold(FoldAdaptor adaptor) {
  if (!mlir::isa<cir::ComplexType>(getOperand().getType()))
    return nullptr;

  if (auto complexCreateOp = getOperand().getDefiningOp<cir::ComplexCreateOp>())
    return complexCreateOp.getOperand(1);

  auto complex =
      mlir::cast_if_present<cir::ConstComplexAttr>(adaptor.getOperand());
  return complex ? complex.getImag() : nullptr;
}

//===----------------------------------------------------------------------===//
// ComplexRealPtrOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexRealPtrOp::verify() {
  mlir::Type resultPointeeTy = getType().getPointee();
  cir::PointerType operandPtrTy = getOperand().getType();
  auto operandPointeeTy =
      mlir::cast<cir::ComplexType>(operandPtrTy.getPointee());

  if (resultPointeeTy != operandPointeeTy.getElementType()) {
    return emitOpError() << ": result type does not match operand type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ComplexImagPtrOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexImagPtrOp::verify() {
  mlir::Type resultPointeeTy = getType().getPointee();
  cir::PointerType operandPtrTy = getOperand().getType();
  auto operandPointeeTy =
      mlir::cast<cir::ComplexType>(operandPtrTy.getPointee());

  if (resultPointeeTy != operandPointeeTy.getElementType()) {
    return emitOpError()
           << "cir.complex.imag_ptr result type does not match operand type";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Bit manipulation operations
//===----------------------------------------------------------------------===//

static OpFoldResult
foldUnaryBitOp(mlir::Attribute inputAttr,
               llvm::function_ref<llvm::APInt(const llvm::APInt &)> func,
               bool poisonZero = false) {
  if (mlir::isa_and_present<cir::PoisonAttr>(inputAttr)) {
    // Propagate poison value
    return inputAttr;
  }

  auto input = mlir::dyn_cast_if_present<IntAttr>(inputAttr);
  if (!input)
    return nullptr;

  llvm::APInt inputValue = input.getValue();
  if (poisonZero && inputValue.isZero())
    return cir::PoisonAttr::get(input.getType());

  llvm::APInt resultValue = func(inputValue);
  return IntAttr::get(input.getType(), resultValue);
}

OpFoldResult BitClrsbOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    unsigned resultValue =
        inputValue.getBitWidth() - inputValue.getSignificantBits();
    return llvm::APInt(inputValue.getBitWidth(), resultValue);
  });
}

OpFoldResult BitClzOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(
      adaptor.getInput(),
      [](const llvm::APInt &inputValue) {
        unsigned resultValue = inputValue.countLeadingZeros();
        return llvm::APInt(inputValue.getBitWidth(), resultValue);
      },
      getPoisonZero());
}

OpFoldResult BitCtzOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(
      adaptor.getInput(),
      [](const llvm::APInt &inputValue) {
        return llvm::APInt(inputValue.getBitWidth(),
                           inputValue.countTrailingZeros());
      },
      getPoisonZero());
}

OpFoldResult BitFfsOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    unsigned trailingZeros = inputValue.countTrailingZeros();
    unsigned result =
        trailingZeros == inputValue.getBitWidth() ? 0 : trailingZeros + 1;
    return llvm::APInt(inputValue.getBitWidth(), result);
  });
}

OpFoldResult BitParityOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    return llvm::APInt(inputValue.getBitWidth(), inputValue.popcount() % 2);
  });
}

OpFoldResult BitPopcountOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    return llvm::APInt(inputValue.getBitWidth(), inputValue.popcount());
  });
}

OpFoldResult BitReverseOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    return inputValue.reverseBits();
  });
}

OpFoldResult ByteSwapOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    return inputValue.byteSwap();
  });
}

OpFoldResult RotateOp::fold(FoldAdaptor adaptor) {
  if (mlir::isa_and_present<cir::PoisonAttr>(adaptor.getInput()) ||
      mlir::isa_and_present<cir::PoisonAttr>(adaptor.getAmount())) {
    // Propagate poison values
    return cir::PoisonAttr::get(getType());
  }

  auto input = mlir::dyn_cast_if_present<IntAttr>(adaptor.getInput());
  auto amount = mlir::dyn_cast_if_present<IntAttr>(adaptor.getAmount());
  if (!input && !amount)
    return nullptr;

  // We could fold cir.rotate even if one of its two operands is not a constant:
  //   - `cir.rotate left/right %0, 0` could be folded into just %0 even if %0
  //     is not a constant.
  //   - `cir.rotate left/right 0/0b111...111, %0` could be folded into 0 or
  //     0b111...111 even if %0 is not a constant.

  llvm::APInt inputValue;
  if (input) {
    inputValue = input.getValue();
    if (inputValue.isZero() || inputValue.isAllOnes()) {
      // An input value of all 0s or all 1s will not change after rotation
      return input;
    }
  }

  uint64_t amountValue;
  if (amount) {
    amountValue = amount.getValue().urem(getInput().getType().getWidth());
    if (amountValue == 0) {
      // A shift amount of 0 will not change the input value
      return getInput();
    }
  }

  if (!input || !amount)
    return nullptr;

  assert(inputValue.getBitWidth() == getInput().getType().getWidth() &&
         "input value must have the same bit width as the input type");

  llvm::APInt resultValue;
  if (isRotateLeft())
    resultValue = inputValue.rotl(amountValue);
  else
    resultValue = inputValue.rotr(amountValue);

  return IntAttr::get(input.getContext(), input.getType(), resultValue);
}

//===----------------------------------------------------------------------===//
// InlineAsmOp
//===----------------------------------------------------------------------===//

void cir::InlineAsmOp::print(OpAsmPrinter &p) {
  p << '(' << getAsmFlavor() << ", ";
  p.increaseIndent();
  p.printNewline();

  llvm::SmallVector<std::string, 3> names{"out", "in", "in_out"};
  auto *nameIt = names.begin();
  auto *attrIt = getOperandAttrs().begin();

  for (mlir::OperandRange ops : getAsmOperands()) {
    p << *nameIt << " = ";

    p << '[';
    llvm::interleaveComma(llvm::make_range(ops.begin(), ops.end()), p,
                          [&](Value value) {
                            p.printOperand(value);
                            p << " : " << value.getType();
                            if (*attrIt)
                              p << " (maybe_memory)";
                            attrIt++;
                          });
    p << "],";
    p.printNewline();
    ++nameIt;
  }

  p << "{";
  p.printString(getAsmString());
  p << " ";
  p.printString(getConstraints());
  p << "}";
  p.decreaseIndent();
  p << ')';
  if (getSideEffects())
    p << " side_effects";

  std::array elidedAttrs{
      llvm::StringRef("asm_flavor"),        llvm::StringRef("asm_string"),
      llvm::StringRef("constraints"),       llvm::StringRef("operand_attrs"),
      llvm::StringRef("operands_segments"), llvm::StringRef("side_effects")};
  p.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);

  if (auto v = getRes())
    p << " -> " << v.getType();
}

void cir::InlineAsmOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             ArrayRef<ValueRange> asmOperands,
                             StringRef asmString, StringRef constraints,
                             bool sideEffects, cir::AsmFlavor asmFlavor,
                             ArrayRef<Attribute> operandAttrs) {
  // Set up the operands_segments for VariadicOfVariadic
  SmallVector<int32_t> segments;
  for (auto operandRange : asmOperands) {
    segments.push_back(operandRange.size());
    odsState.addOperands(operandRange);
  }

  odsState.addAttribute(
      "operands_segments",
      DenseI32ArrayAttr::get(odsBuilder.getContext(), segments));
  odsState.addAttribute("asm_string", odsBuilder.getStringAttr(asmString));
  odsState.addAttribute("constraints", odsBuilder.getStringAttr(constraints));
  odsState.addAttribute("asm_flavor",
                        AsmFlavorAttr::get(odsBuilder.getContext(), asmFlavor));

  if (sideEffects)
    odsState.addAttribute("side_effects", odsBuilder.getUnitAttr());

  odsState.addAttribute("operand_attrs", odsBuilder.getArrayAttr(operandAttrs));
}

ParseResult cir::InlineAsmOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  llvm::SmallVector<mlir::Attribute> operandAttrs;
  llvm::SmallVector<int32_t> operandsGroupSizes;
  std::string asmString, constraints;
  Type resType;
  MLIRContext *ctxt = parser.getBuilder().getContext();

  auto error = [&](const Twine &msg) -> LogicalResult {
    return parser.emitError(parser.getCurrentLocation(), msg);
  };

  auto expected = [&](const std::string &c) {
    return error("expected '" + c + "'");
  };

  if (parser.parseLParen().failed())
    return expected("(");

  auto flavor = FieldParser<AsmFlavor, AsmFlavor>::parse(parser);
  if (failed(flavor))
    return error("Unknown AsmFlavor");

  if (parser.parseComma().failed())
    return expected(",");

  auto parseValue = [&](Value &v) {
    OpAsmParser::UnresolvedOperand op;

    if (parser.parseOperand(op) || parser.parseColon())
      return error("can't parse operand");

    Type typ;
    if (parser.parseType(typ).failed())
      return error("can't parse operand type");
    llvm::SmallVector<mlir::Value> tmp;
    if (parser.resolveOperand(op, typ, tmp))
      return error("can't resolve operand");
    v = tmp[0];
    return mlir::success();
  };

  auto parseOperands = [&](llvm::StringRef name) {
    if (parser.parseKeyword(name).failed())
      return error("expected " + name + " operands here");
    if (parser.parseEqual().failed())
      return expected("=");
    if (parser.parseLSquare().failed())
      return expected("[");

    int size = 0;
    if (parser.parseOptionalRSquare().succeeded()) {
      operandsGroupSizes.push_back(size);
      if (parser.parseComma())
        return expected(",");
      return mlir::success();
    }

    auto parseOperand = [&]() {
      Value val;
      if (parseValue(val).succeeded()) {
        result.operands.push_back(val);
        size++;

        if (parser.parseOptionalLParen().failed()) {
          operandAttrs.push_back(mlir::Attribute());
          return mlir::success();
        }

        if (parser.parseKeyword("maybe_memory").succeeded()) {
          operandAttrs.push_back(mlir::UnitAttr::get(ctxt));
          if (parser.parseRParen())
            return expected(")");
          return mlir::success();
        } else {
          return expected("maybe_memory");
        }
      }
      return mlir::failure();
    };

    if (parser.parseCommaSeparatedList(parseOperand).failed())
      return mlir::failure();

    if (parser.parseRSquare().failed() || parser.parseComma().failed())
      return expected("]");
    operandsGroupSizes.push_back(size);
    return mlir::success();
  };

  if (parseOperands("out").failed() || parseOperands("in").failed() ||
      parseOperands("in_out").failed())
    return error("failed to parse operands");

  if (parser.parseLBrace())
    return expected("{");
  if (parser.parseString(&asmString))
    return error("asm string parsing failed");
  if (parser.parseString(&constraints))
    return error("constraints string parsing failed");
  if (parser.parseRBrace())
    return expected("}");
  if (parser.parseRParen())
    return expected(")");

  if (parser.parseOptionalKeyword("side_effects").succeeded())
    result.attributes.set("side_effects", UnitAttr::get(ctxt));

  if (parser.parseOptionalArrow().succeeded() &&
      parser.parseType(resType).failed())
    return mlir::failure();

  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return mlir::failure();

  result.attributes.set("asm_flavor", AsmFlavorAttr::get(ctxt, *flavor));
  result.attributes.set("asm_string", StringAttr::get(ctxt, asmString));
  result.attributes.set("constraints", StringAttr::get(ctxt, constraints));
  result.attributes.set("operand_attrs", ArrayAttr::get(ctxt, operandAttrs));
  result.getOrAddProperties<InlineAsmOp::Properties>().operands_segments =
      parser.getBuilder().getDenseI32ArrayAttr(operandsGroupSizes);
  if (resType)
    result.addTypes(TypeRange{resType});

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ThrowOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult cir::ThrowOp::verify() {
  // For the no-rethrow version, it must have at least the exception pointer.
  if (rethrows())
    return success();

  if (getNumOperands() != 0) {
    if (getTypeInfo())
      return success();
    return emitOpError() << "'type_info' symbol attribute missing";
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TypeInfoAttr
//===----------------------------------------------------------------------===//

LogicalResult cir::TypeInfoAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::Type type, ::mlir::ArrayAttr typeInfoData) {

  if (cir::ConstRecordAttr::verify(emitError, type, typeInfoData).failed())
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROps.cpp.inc"
