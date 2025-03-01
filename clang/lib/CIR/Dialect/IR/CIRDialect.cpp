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
#include "clang/AST/Attrs.inc"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <numeric>
#include <optional>
#include <set>

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

#include "clang/CIR/Dialect/IR/CIROpsEnums.cpp.inc"
#include "clang/CIR/Dialect/IR/CIROpsStructs.cpp.inc"

#include "clang/CIR/Dialect/IR/CIROpsDialect.cpp.inc"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include <clang/CIR/MissingFeatures.h>

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//
namespace {
struct CIROpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    if (auto structType = dyn_cast<cir::StructType>(type)) {
      StringAttr nameAttr = structType.getName();
      if (!nameAttr)
        os << "ty_anon_" << structType.getKindAsStr();
      else
        os << "ty_" << nameAttr.getValue();
      return AliasResult::OverridableAlias;
    }
    if (auto intType = dyn_cast<cir::IntType>(type)) {
      // We only provide alias for standard integer types (i.e. integer types
      // whose width is divisible by 8).
      if (intType.getWidth() % 8 != 0)
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
    if (auto extraFuncAttr =
            mlir::dyn_cast<cir::ExtraFuncAttributesAttr>(attr)) {
      os << "fn_attr";
      return AliasResult::FinalAlias;
    }
    if (auto cmpThreeWayInfoAttr =
            mlir::dyn_cast<cir::CmpThreeWayInfoAttr>(attr)) {
      os << cmpThreeWayInfoAttr.getAlias();
      return AliasResult::FinalAlias;
    }
    if (auto dynCastInfoAttr = mlir::dyn_cast<cir::DynamicCastInfoAttr>(attr)) {
      os << dynCastInfoAttr.getAlias();
      return AliasResult::FinalAlias;
    }
    return TypeSwitch<Attribute, AliasResult>(attr)
        .Case<cir::TBAAAttr, cir::TBAAOmnipotentCharAttr, cir::TBAAScalarAttr,
              cir::TBAAStructAttr, cir::TBAATagAttr>([&](auto attr) {
          os << decltype(attr)::getMnemonic();
          return AliasResult::OverridableAlias;
        })
        .Default([](Attribute) { return AliasResult::NoAlias; });
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
#define REGISTER_ENUM_TYPE_WITH_NS(NS, Ty)                                     \
  template <> struct EnumTraits<NS::Ty> {                                      \
    static llvm::StringRef stringify(NS::Ty value) {                           \
      return NS::stringify##Ty(value);                                         \
    }                                                                          \
    static unsigned getMaxEnumVal() { return NS::getMaxEnumValFor##Ty(); }     \
  }

REGISTER_ENUM_TYPE(GlobalLinkageKind);
REGISTER_ENUM_TYPE(VisibilityKind);
REGISTER_ENUM_TYPE(CallingConv);
REGISTER_ENUM_TYPE(SideEffect);
REGISTER_ENUM_TYPE_WITH_NS(cir::sob, SignedOverflowBehavior);
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
LogicalResult ensureRegionTerm(OpAsmParser &parser, Region &region,
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
bool omitRegionTerm(mlir::Region &r) {
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
  default:
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

static mlir::ParseResult
parseOmittedTerminatorRegion(mlir::OpAsmParser &parser,
                             mlir::Region &scopeRegion,
                             mlir::Region &cleanupRegion) {
  auto regionLoc = parser.getCurrentLocation();
  if (parser.parseRegion(scopeRegion))
    return failure();
  if (ensureRegionTerm(parser, scopeRegion, regionLoc).failed())
    return failure();

  // Parse optional cleanup region.
  if (parser.parseOptionalKeyword("cleanup").succeeded()) {
    if (parser.parseRegion(cleanupRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    if (ensureRegionTerm(parser, cleanupRegion, regionLoc).failed())
      return failure();
  }

  return success();
}

static void printOmittedTerminatorRegion(mlir::OpAsmPrinter &printer,
                                         cir::ScopeOp &op,
                                         mlir::Region &scopeRegion,
                                         mlir::Region &cleanupRegion) {
  printer.printRegion(scopeRegion,
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/!omitRegionTerm(scopeRegion));
  if (!op.getCleanupRegion().empty()) {
    printer << " cleanup ";
    printer.printRegion(
        cleanupRegion,
        /*printEntryBlockArgs=*/false,
        /*printBlockTerminators=*/!omitRegionTerm(cleanupRegion));
  }
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

void cir::AllocaOp::build(::mlir::OpBuilder &odsBuilder,
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
// BreakOp
//===----------------------------------------------------------------------===//

LogicalResult cir::BreakOp::verify() {
  if (!getOperation()->getParentOfType<LoopOpInterface>() &&
      !getOperation()->getParentOfType<SwitchOp>())
    return emitOpError("must be within a loop or switch");
  return success();
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===-----------------------------------------------------------------------===//

//===----------------------------------
// BranchOpTerminatorInterface Methods

void cir::ConditionOp::getSuccessorRegions(
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions) {
  // TODO(cir): The condition value may be folded to a constant, narrowing
  // down its list of possible successors.

  // Parent is a loop: condition may branch to the body or to the parent op.
  if (auto loopOp = dyn_cast<LoopOpInterface>(getOperation()->getParentOp())) {
    regions.emplace_back(&loopOp.getBody(), loopOp.getBody().getArguments());
    regions.emplace_back(loopOp->getResults());
  }

  // Parent is an await: condition may branch to resume or suspend regions.
  auto await = cast<AwaitOp>(getOperation()->getParentOp());
  regions.emplace_back(&await.getResume(), await.getResume().getArguments());
  regions.emplace_back(&await.getSuspend(), await.getSuspend().getArguments());
}

MutableOperandRange
cir::ConditionOp::getMutableSuccessorOperands(RegionBranchPoint point) {
  // No values are yielded to the successor region.
  return MutableOperandRange(getOperation(), 0, 0);
}

LogicalResult cir::ConditionOp::verify() {
  if (!isa<LoopOpInterface, AwaitOp>(getOperation()->getParentOp()))
    return emitOpError("condition must be within a conditional region");
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult checkConstantTypes(mlir::Operation *op, mlir::Type opType,
                                        mlir::Attribute attrType) {
  if (isa<cir::ConstPtrAttr>(attrType)) {
    if (::mlir::isa<cir::PointerType>(opType))
      return success();
    return op->emitOpError("nullptr expects pointer type");
  }

  if (isa<cir::DataMemberAttr, cir::MethodAttr>(attrType)) {
    // More detailed type verifications are already done in
    // DataMemberAttr::verify. Don't need to repeat here.
    return success();
  }

  if (isa<cir::ZeroAttr>(attrType)) {
    if (::mlir::isa<cir::StructType, cir::ArrayType, cir::ComplexType,
                    cir::VectorType>(opType))
      return success();
    return op->emitOpError("zero expects struct or array type");
  }

  if (isa<cir::UndefAttr>(attrType)) {
    if (!::mlir::isa<cir::VoidType>(opType))
      return success();
    return op->emitOpError("undef expects non-void type");
  }

  if (isa<cir::PoisonAttr>(attrType)) {
    if (!::mlir::isa<cir::VoidType>(opType))
      return success();
    return op->emitOpError("poison expects non-void type");
  }

  if (mlir::isa<cir::BoolAttr>(attrType)) {
    if (!mlir::isa<cir::BoolType>(opType))
      return op->emitOpError("result type (")
             << opType << ") must be '!cir.bool' for '" << attrType << "'";
    return success();
  }

  if (mlir::isa<cir::IntAttr, cir::FPAttr, cir::ComplexAttr>(attrType)) {
    auto at = cast<TypedAttr>(attrType);
    if (at.getType() != opType) {
      return op->emitOpError("result type (")
             << opType << ") does not match value type (" << at.getType()
             << ")";
    }
    return success();
  }

  if (isa<SymbolRefAttr>(attrType)) {
    if (::mlir::isa<cir::PointerType>(opType))
      return success();
    return op->emitOpError("symbolref expects pointer type");
  }

  if (mlir::isa<cir::GlobalViewAttr>(attrType) ||
      mlir::isa<cir::TypeInfoAttr>(attrType) ||
      mlir::isa<cir::ConstArrayAttr>(attrType) ||
      mlir::isa<cir::ConstVectorAttr>(attrType) ||
      mlir::isa<cir::ConstStructAttr>(attrType) ||
      mlir::isa<cir::VTableAttr>(attrType))
    return success();
  if (mlir::isa<cir::IntAttr>(attrType))
    return success();

  assert(isa<TypedAttr>(attrType) && "What else could we be looking at here?");
  return op->emitOpError("global with type ")
         << cast<TypedAttr>(attrType).getType() << " not supported";
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
  if (!this->getOperation()->getParentOfType<LoopOpInterface>())
    return emitOpError("must be within a loop");
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult cir::CastOp::verify() {
  auto resType = getResult().getType();
  auto srcType = getSrc().getType();

  if (mlir::isa<cir::VectorType>(srcType) &&
      mlir::isa<cir::VectorType>(resType)) {
    // Use the element type of the vector to verify the cast kind. (Except for
    // bitcast, see below.)
    srcType = mlir::dyn_cast<cir::VectorType>(srcType).getEltType();
    resType = mlir::dyn_cast<cir::VectorType>(resType).getEltType();
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
    auto arrayPtrTy = mlir::dyn_cast<cir::PointerType>(srcType);
    auto flatPtrTy = mlir::dyn_cast<cir::PointerType>(resType);
    if (!arrayPtrTy || !flatPtrTy)
      return emitOpError() << "requires !cir.ptr type for source and result";

    if (arrayPtrTy.getAddrSpace() != flatPtrTy.getAddrSpace()) {
      return emitOpError()
             << "requires same address space for source and result";
    }

    auto arrayTy = mlir::dyn_cast<cir::ArrayType>(arrayPtrTy.getPointee());
    if (!arrayTy)
      return emitOpError() << "requires !cir.array pointee";

    if (arrayTy.getEltType() != flatPtrTy.getPointee())
      return emitOpError()
             << "requires same type for array element and pointee result";
    return success();
  }
  case cir::CastKind::bitcast: {
    // Allow bitcast of structs for calling conventions.
    if (isa<StructType>(srcType) || isa<StructType>(resType))
      return success();

    // Handle the pointer types first.
    auto srcPtrTy = mlir::dyn_cast<cir::PointerType>(srcType);
    auto resPtrTy = mlir::dyn_cast<cir::PointerType>(resType);

    if (srcPtrTy && resPtrTy) {
      if (srcPtrTy.getAddrSpace() != resPtrTy.getAddrSpace()) {
        return emitOpError() << "result type address space does not match the "
                                "address space of the operand";
      }
      return success();
    }

    // Handle the data member pointer types.
    if (mlir::isa<cir::DataMemberType>(srcType) &&
        mlir::isa<cir::DataMemberType>(resType))
      return success();

    // Handle the pointer to member function types.
    if (mlir::isa<cir::MethodType>(srcType) &&
        mlir::isa<cir::MethodType>(resType))
      return success();

    // This is the only cast kind where we don't want vector types to decay
    // into the element type.
    if ((!mlir::isa<cir::VectorType>(getSrc().getType()) ||
         !mlir::isa<cir::VectorType>(getResult().getType())))
      return emitOpError()
             << "requires !cir.ptr or !cir.vector type for source and result";
    return success();
  }
  case cir::CastKind::floating: {
    if (!mlir::isa<cir::CIRFPTypeInterface>(srcType) ||
        !mlir::isa<cir::CIRFPTypeInterface>(resType))
      return emitOpError() << "requires !cir.float type for source and result";
    return success();
  }
  case cir::CastKind::float_to_int: {
    if (!mlir::isa<cir::CIRFPTypeInterface>(srcType))
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
    if (!mlir::isa<cir::CIRFPTypeInterface>(srcType))
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
    if (!mlir::isa<cir::CIRFPTypeInterface>(resType))
      return emitOpError() << "requires !cir.float type for result";
    return success();
  }
  case cir::CastKind::bool_to_float: {
    if (!mlir::isa<cir::BoolType>(srcType))
      return emitOpError() << "requires !cir.bool type for source";
    if (!mlir::isa<cir::CIRFPTypeInterface>(resType))
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
    if (!mlir::isa<cir::CIRFPTypeInterface>(srcType))
      return emitOpError() << "requires !cir.float type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy)
      return emitOpError() << "requires !cir.complex type for result";
    if (srcType != resComplexTy.getElementTy())
      return emitOpError() << "requires source type match result element type";
    return success();
  }
  case cir::CastKind::int_to_complex: {
    if (!mlir::isa<cir::IntType>(srcType))
      return emitOpError() << "requires !cir.int type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy)
      return emitOpError() << "requires !cir.complex type for result";
    if (srcType != resComplexTy.getElementTy())
      return emitOpError() << "requires source type match result element type";
    return success();
  }
  case cir::CastKind::float_complex_to_real: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy)
      return emitOpError() << "requires !cir.complex type for source";
    if (!mlir::isa<cir::CIRFPTypeInterface>(resType))
      return emitOpError() << "requires !cir.float type for result";
    if (srcComplexTy.getElementTy() != resType)
      return emitOpError() << "requires source element type match result type";
    return success();
  }
  case cir::CastKind::int_complex_to_real: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy)
      return emitOpError() << "requires !cir.complex type for source";
    if (!mlir::isa<cir::IntType>(resType))
      return emitOpError() << "requires !cir.int type for result";
    if (srcComplexTy.getElementTy() != resType)
      return emitOpError() << "requires source element type match result type";
    return success();
  }
  case cir::CastKind::float_complex_to_bool: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy ||
        !mlir::isa<cir::CIRFPTypeInterface>(srcComplexTy.getElementTy()))
      return emitOpError()
             << "requires !cir.complex<!cir.float> type for source";
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    return success();
  }
  case cir::CastKind::int_complex_to_bool: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !mlir::isa<cir::IntType>(srcComplexTy.getElementTy()))
      return emitOpError()
             << "requires !cir.complex<!cir.float> type for source";
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    return success();
  }
  case cir::CastKind::float_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy ||
        !mlir::isa<cir::CIRFPTypeInterface>(srcComplexTy.getElementTy()))
      return emitOpError()
             << "requires !cir.complex<!cir.float> type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy ||
        !mlir::isa<cir::CIRFPTypeInterface>(resComplexTy.getElementTy()))
      return emitOpError()
             << "requires !cir.complex<!cir.float> type for result";
    return success();
  }
  case cir::CastKind::float_complex_to_int_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy ||
        !mlir::isa<cir::CIRFPTypeInterface>(srcComplexTy.getElementTy()))
      return emitOpError()
             << "requires !cir.complex<!cir.float> type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy || !mlir::isa<cir::IntType>(resComplexTy.getElementTy()))
      return emitOpError() << "requires !cir.complex<!cir.int> type for result";
    return success();
  }
  case cir::CastKind::int_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !mlir::isa<cir::IntType>(srcComplexTy.getElementTy()))
      return emitOpError() << "requires !cir.complex<!cir.int> type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy || !mlir::isa<cir::IntType>(resComplexTy.getElementTy()))
      return emitOpError() << "requires !cir.complex<!cir.int> type for result";
    return success();
  }
  case cir::CastKind::int_complex_to_float_complex: {
    auto srcComplexTy = mlir::dyn_cast<cir::ComplexType>(srcType);
    if (!srcComplexTy || !mlir::isa<cir::IntType>(srcComplexTy.getElementTy()))
      return emitOpError() << "requires !cir.complex<!cir.int> type for source";
    auto resComplexTy = mlir::dyn_cast<cir::ComplexType>(resType);
    if (!resComplexTy ||
        !mlir::isa<cir::CIRFPTypeInterface>(resComplexTy.getElementTy()))
      return emitOpError()
             << "requires !cir.complex<!cir.float> type for result";
    return success();
  }
  case cir::CastKind::member_ptr_to_bool: {
    if (!mlir::isa<cir::DataMemberType, cir::MethodType>(srcType))
      return emitOpError()
             << "requires !cir.data_member or !cir.method type for source";
    if (!mlir::isa<cir::BoolType>(resType))
      return emitOpError() << "requires !cir.bool type for result";
    return success();
  }
  }

  llvm_unreachable("Unknown CastOp kind?");
}

bool isIntOrBoolCast(cir::CastOp op) {
  auto kind = op.getKind();
  return kind == cir::CastKind::bool_to_int ||
         kind == cir::CastKind::int_to_bool || kind == cir::CastKind::integral;
}

Value tryFoldCastChain(cir::CastOp op) {
  cir::CastOp head = op, tail = op;

  while (op) {
    if (!isIntOrBoolCast(op))
      break;
    head = op;
    op = dyn_cast_or_null<cir::CastOp>(head.getSrc().getDefiningOp());
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
  if (getSrc().getType() == getResult().getType()) {
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
  if (isBoolNot(*this))
    if (auto previous = dyn_cast_or_null<UnaryOp>(getInput().getDefiningOp()))
      if (isBoolNot(previous))
        return previous.getInput();

  return {};
}

//===----------------------------------------------------------------------===//
// DynamicCastOp
//===----------------------------------------------------------------------===//

LogicalResult cir::DynamicCastOp::verify() {
  auto resultPointeeTy = mlir::cast<cir::PointerType>(getType()).getPointee();
  if (!mlir::isa<cir::VoidType, cir::StructType>(resultPointeeTy))
    return emitOpError()
           << "cir.dyn_cast must produce a void ptr or struct ptr";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BaseDataMemberOp & DerivedDataMemberOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyMemberPtrCast(Operation *op, mlir::Value src,
                                         mlir::Type resultTy) {
  // Let the operand type be T1 C1::*, let the result type be T2 C2::*.
  // Verify that T1 and T2 are the same type.
  mlir::Type inputMemberTy;
  mlir::Type resultMemberTy;
  if (mlir::isa<cir::DataMemberType>(src.getType())) {
    inputMemberTy =
        mlir::cast<cir::DataMemberType>(src.getType()).getMemberTy();
    resultMemberTy = mlir::cast<cir::DataMemberType>(resultTy).getMemberTy();
  } else {
    inputMemberTy =
        mlir::cast<cir::MethodType>(src.getType()).getMemberFuncTy();
    resultMemberTy = mlir::cast<cir::MethodType>(resultTy).getMemberFuncTy();
  }
  if (inputMemberTy != resultMemberTy)
    return op->emitOpError()
           << "member types of the operand and the result do not match";

  return mlir::success();
}

LogicalResult cir::BaseDataMemberOp::verify() {
  return verifyMemberPtrCast(getOperation(), getSrc(), getType());
}

LogicalResult cir::DerivedDataMemberOp::verify() {
  return verifyMemberPtrCast(getOperation(), getSrc(), getType());
}

//===----------------------------------------------------------------------===//
// BaseMethodOp & DerivedMethodOp
//===----------------------------------------------------------------------===//

LogicalResult cir::BaseMethodOp::verify() {
  return verifyMemberPtrCast(getOperation(), getSrc(), getType());
}

LogicalResult cir::DerivedMethodOp::verify() {
  return verifyMemberPtrCast(getOperation(), getSrc(), getType());
}

//===----------------------------------------------------------------------===//
// ComplexCreateOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexCreateOp::verify() {
  if (getType().getElementTy() != getReal().getType()) {
    emitOpError()
        << "operand type of cir.complex.create does not match its result type";
    return failure();
  }

  return success();
}

OpFoldResult cir::ComplexCreateOp::fold(FoldAdaptor adaptor) {
  auto real = adaptor.getReal();
  auto imag = adaptor.getImag();

  if (!real || !imag)
    return nullptr;

  // When both of real and imag are constants, we can fold the operation into an
  // `cir.const #cir.complex` operation.

  auto realAttr = mlir::cast<mlir::TypedAttr>(real);
  auto imagAttr = mlir::cast<mlir::TypedAttr>(imag);
  assert(realAttr.getType() == imagAttr.getType() &&
         "real part and imag part should be of the same type");

  auto complexTy = cir::ComplexType::get(getContext(), realAttr.getType());
  return cir::ComplexAttr::get(complexTy, realAttr, imagAttr);
}

//===----------------------------------------------------------------------===//
// ComplexRealOp and ComplexImagOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexRealOp::verify() {
  if (getType() != getOperand().getType().getElementTy()) {
    emitOpError() << "cir.complex.real result type does not match operand type";
    return failure();
  }
  return success();
}

OpFoldResult cir::ComplexRealOp::fold(FoldAdaptor adaptor) {
  auto input = mlir::cast_if_present<cir::ComplexAttr>(adaptor.getOperand());
  if (input)
    return input.getReal();
  return nullptr;
}

LogicalResult cir::ComplexImagOp::verify() {
  if (getType() != getOperand().getType().getElementTy()) {
    emitOpError() << "cir.complex.imag result type does not match operand type";
    return failure();
  }
  return success();
}

OpFoldResult cir::ComplexImagOp::fold(FoldAdaptor adaptor) {
  auto input = mlir::cast_if_present<cir::ComplexAttr>(adaptor.getOperand());
  if (input)
    return input.getImag();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ComplexRealPtrOp and ComplexImagPtrOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexRealPtrOp::verify() {
  auto resultPointeeTy = mlir::cast<cir::PointerType>(getType()).getPointee();
  auto operandPtrTy = mlir::cast<cir::PointerType>(getOperand().getType());
  auto operandPointeeTy =
      mlir::cast<cir::ComplexType>(operandPtrTy.getPointee());

  if (resultPointeeTy != operandPointeeTy.getElementTy()) {
    emitOpError()
        << "cir.complex.real_ptr result type does not match operand type";
    return failure();
  }

  return success();
}

LogicalResult cir::ComplexImagPtrOp::verify() {
  auto resultPointeeTy = mlir::cast<cir::PointerType>(getType()).getPointee();
  auto operandPtrTy = mlir::cast<cir::PointerType>(getOperand().getType());
  auto operandPointeeTy =
      mlir::cast<cir::ComplexType>(operandPtrTy.getPointee());

  if (resultPointeeTy != operandPointeeTy.getElementTy()) {
    emitOpError()
        << "cir.complex.imag_ptr result type does not match operand type";
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

// TODO(CIR): The final interface here should include an argument for the
// SyncScope::ID.
void cir::LoadOp::setAtomic(cir::MemOrder order) {
  setMemOrder(order);
  if (cir::MissingFeatures::syncScopeID())
    llvm_unreachable("NYI");
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

// TODO(CIR): The final interface here should include an argument for the
// SyncScope::ID.
void cir::StoreOp::setAtomic(cir::MemOrder order) {
  setMemOrder(order);
  if (cir::MissingFeatures::syncScopeID())
    llvm_unreachable("NYI");
}

//===----------------------------------------------------------------------===//
// VecCreateOp
//===----------------------------------------------------------------------===//

LogicalResult cir::VecCreateOp::verify() {
  // Verify that the number of arguments matches the number of elements in the
  // vector, and that the type of all the arguments matches the type of the
  // elements in the vector.
  auto VecTy = getResult().getType();
  if (getElements().size() != VecTy.getSize()) {
    return emitOpError() << "operand count of " << getElements().size()
                         << " doesn't match vector type " << VecTy
                         << " element count of " << VecTy.getSize();
  }
  auto ElementType = VecTy.getEltType();
  for (auto Element : getElements()) {
    if (Element.getType() != ElementType) {
      return emitOpError() << "operand type " << Element.getType()
                           << " doesn't match vector element type "
                           << ElementType;
    }
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
  if (mlir::cast<cir::VectorType>(getCond().getType()).getSize() !=
      getVec1().getType().getSize()) {
    return emitOpError() << ": the number of elements in "
                         << getCond().getType() << " and "
                         << getVec1().getType() << " don't match";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VecShuffle
//===----------------------------------------------------------------------===//

LogicalResult cir::VecShuffleOp::verify() {
  // The number of elements in the indices array must match the number of
  // elements in the result type.
  if (getIndices().size() != getResult().getType().getSize()) {
    return emitOpError() << ": the number of elements in " << getIndices()
                         << " and " << getResult().getType() << " don't match";
  }
  // The element types of the two input vectors and of the result type must
  // match.
  if (getVec1().getType().getEltType() != getResult().getType().getEltType()) {
    return emitOpError() << ": element types of " << getVec1().getType()
                         << " and " << getResult().getType() << " don't match";
  }
  // The indices must all be integer constants
  if (not std::all_of(
          getIndices().begin(), getIndices().end(),
          [](mlir::Attribute attr) { return mlir::isa<cir::IntAttr>(attr); })) {
    return emitOpError() << "all index values must be integers";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VecShuffleDynamic
//===----------------------------------------------------------------------===//

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
// ThrowOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult cir::ThrowOp::verify() {
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

ParseResult cir::IfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  Type boolType = cir::BoolType::get(builder.getContext());

  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, boolType, result.operands))
    return failure();

  // Parse the 'then' region.
  auto parseThenLoc = parser.getCurrentLocation();
  if (parser.parseRegion(*thenRegion, /*arguments=*/{},
                         /*argTypes=*/{}))
    return failure();
  if (ensureRegionTerm(parser, *thenRegion, parseThenLoc).failed())
    return failure();

  // If we find an 'else' keyword, parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    auto parseElseLoc = parser.getCurrentLocation();
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
  auto &thenRegion = this->getThenRegion();
  p.printRegion(thenRegion,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!omitRegionTerm(thenRegion));

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = this->getElseRegion();
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
  // add cir.yield to the end of the block
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

void cir::IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
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

LogicalResult cir::IfOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void cir::ScopeOp::getSuccessorRegions(
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // The only region always branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(getODSResults(0)));
    return;
  }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getScopeRegion()));
  regions.push_back(RegionSuccessor(&getCleanupRegion()));
}

void cir::ScopeOp::build(
    OpBuilder &builder, OperationState &result,
    function_ref<void(OpBuilder &, Type &, Location)> scopeBuilder) {
  assert(scopeBuilder && "the builder callback for 'then' must be present");

  OpBuilder::InsertionGuard guard(builder);
  Region *scopeRegion = result.addRegion();
  builder.createBlock(scopeRegion);
  // cleanup region, do not eagarly create blocks, do it upon demand.
  (void)result.addRegion();

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
  // cleanup region, do not eagarly create blocks, do it upon demand.
  (void)result.addRegion();
  scopeBuilder(builder, result.location);
}

LogicalResult cir::ScopeOp::verify() {
  if (getScopeRegion().empty()) {
    return emitOpError() << "cir.scope must not be empty since it should "
                            "include at least an implicit cir.yield ";
  }

  if (getScopeRegion().back().empty() ||
      !getScopeRegion()
           .back()
           .getTerminator()
           ->hasTrait<OpTrait::IsTerminator>())
    return emitOpError() << "last block of cir.scope must be terminated";
  return success();
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

void cir::TryOp::build(
    OpBuilder &builder, OperationState &result,
    function_ref<void(OpBuilder &, Location)> tryBodyBuilder,
    function_ref<void(OpBuilder &, Location, OperationState &)> catchBuilder) {
  assert(tryBodyBuilder && "expected builder callback for 'cir.try' body");

  OpBuilder::InsertionGuard guard(builder);

  // Try body region
  Region *tryBodyRegion = result.addRegion();

  // Create try body region and set insertion point
  builder.createBlock(tryBodyRegion);
  tryBodyBuilder(builder, result.location);
  catchBuilder(builder, result.location, result);
}

mlir::Region *cir::TryOp::getCatchLastRegion() {
  unsigned numCatchRegions = getCatchRegions().size();
  assert(numCatchRegions && "expected at least one region");
  auto &lastRegion = getCatchRegions()[numCatchRegions - 1];
  return &lastRegion;
}

mlir::Block *cir::TryOp::getCatchUnwindEntryBlock() {
  return &getCatchLastRegion()->getBlocks().front();
}

mlir::Block *cir::TryOp::getCatchAllEntryBlock() {
  return &getCatchLastRegion()->getBlocks().front();
}

bool cir::TryOp::isCatchAllOnly() {
  mlir::ArrayAttr catchAttrList = getCatchTypesAttr();
  return catchAttrList.size() == 1 && isa<cir::CatchAllAttr>(catchAttrList[0]);
}

void cir::TryOp::getSuccessorRegions(
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // If any index all the underlying regions branch back to the parent
  // operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getTryRegion()));

  // FIXME: optimize, ideas include:
  // - If we know a target function never throws a specific type, we can
  //   remove the catch handler.
  for (auto &r : this->getCatchRegions())
    regions.push_back(RegionSuccessor(&r));
}

void printCatchRegions(OpAsmPrinter &p, cir::TryOp op,
                       mlir::MutableArrayRef<::mlir::Region> regions,
                       mlir::ArrayAttr catchList) {

  int currCatchIdx = 0;
  if (!catchList)
    return;
  p << "catch [";
  llvm::interleaveComma(catchList, p, [&](const Attribute &a) {
    auto exRtti = a;

    if (mlir::isa<cir::CatchUnwindAttr>(a)) {
      p.printAttribute(a);
      p << " ";
    } else if (!exRtti) {
      p << "all";
    } else {
      p << "type ";
      p.printAttribute(exRtti);
      p << " ";
    }
    p.printRegion(regions[currCatchIdx], /*printEntryBLockArgs=*/false,
                  /*printBlockTerminators=*/true);
    currCatchIdx++;
  });
  p << "]";
}

ParseResult parseCatchRegions(
    OpAsmParser &parser,
    llvm::SmallVectorImpl<std::unique_ptr<::mlir::Region>> &regions,
    ::mlir::ArrayAttr &catchersAttr) {
  llvm::SmallVector<mlir::Attribute, 4> catchList;

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

    if (!(currRegion.back().mightHaveTerminator() &&
          currRegion.back().getTerminator()))
      return parser.emitError(
          parserLoc, "blocks are expected to be explicitly terminated");

    return success();
  };

  auto parseCatchEntry = [&]() -> ParseResult {
    mlir::Type exceptionType;
    mlir::Attribute exceptionTypeInfo;

    // FIXME: support most recent syntax, currently broken.
    ::llvm::StringRef attrStr;
    if (!parser.parseOptionalKeyword(&attrStr, {"all"})) {
      if (parser.parseKeyword("type").failed())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected 'type' keyword here");
      if (parser.parseType(exceptionType).failed())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected valid exception type");
      if (parser.parseAttribute(exceptionTypeInfo).failed())
        return parser.emitError(parser.getCurrentLocation(),
                                "expected valid RTTI info attribute");
    }
    catchList.push_back(exceptionTypeInfo);
    return parseAndCheckRegion();
  };

  if (parser.parseKeyword("catch").failed())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'catch' keyword here");

  if (parser
          .parseCommaSeparatedList(OpAsmParser::Delimiter::Square,
                                   parseCatchEntry, " in catch list")
          .failed())
    return failure();

  catchersAttr = parser.getBuilder().getArrayAttr(catchList);
  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

/// Given the region at `index`, or the parent operation if `index` is None,
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

  // Try optimize if we have more information
  // if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
  //   assert(0 && "not implemented");
  // }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getTrueRegion()));
  regions.push_back(RegionSuccessor(&getFalseRegion()));
  return;
}

void cir::TernaryOp::build(
    OpBuilder &builder, OperationState &result, Value cond,
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
// SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::SelectOp::fold(FoldAdaptor adaptor) {
  auto condition = adaptor.getCondition();
  if (condition) {
    auto conditionValue = mlir::cast<cir::BoolAttr>(condition).getValue();
    return conditionValue ? getTrueValue() : getFalseValue();
  }

  // cir.select if %0 then x else x -> x
  auto trueValue = adaptor.getTrueValue();
  auto falseValue = adaptor.getFalseValue();
  if (trueValue && trueValue == falseValue)
    return trueValue;
  if (getTrueValue() == getFalseValue())
    return getTrueValue();

  return nullptr;
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

LogicalResult cir::CaseOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

ParseResult parseSwitchOp(OpAsmParser &parser, mlir::Region &regions,
                          mlir::OpAsmParser::UnresolvedOperand &cond,
                          mlir::Type &condType) {
  cir::IntType intCondType;

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
  if (parser.parseRegion(regions, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return ::mlir::success();
}

void printSwitchOp(OpAsmPrinter &p, cir::SwitchOp op, mlir::Region &bodyRegion,
                   mlir::Value condition, mlir::Type condType) {
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
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // If any index all the underlying regions branch back to the parent
  // operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  regions.push_back(RegionSuccessor(&getBody()));
}

LogicalResult cir::SwitchOp::verify() { return success(); }

void cir::SwitchOp::build(
    OpBuilder &builder, OperationState &result, Value cond,
    function_ref<void(OpBuilder &, Location, OperationState &)> switchBuilder) {
  assert(switchBuilder && "the builder callback for regions must be present");
  OpBuilder::InsertionGuard guardSwitch(builder);
  Region *swtichRegion = result.addRegion();
  builder.createBlock(swtichRegion);
  result.addOperands({cond});
  switchBuilder(builder, result.location, result);
}

void cir::SwitchOp::collectCases(llvm::SmallVector<CaseOp> &cases) {
  walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    // Don't walk in nested switch op.
    if (isa<cir::SwitchOp>(op) && op != *this)
      return WalkResult::skip();

    if (isa<cir::CaseOp>(op))
      cases.push_back(cast<cir::CaseOp>(*op));

    return WalkResult::advance();
  });
}

bool cir::SwitchOp::isSimpleForm(llvm::SmallVector<CaseOp> &cases) {
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
  for (auto &val : caseValues) {
    caseValuesAttrs.push_back(cir::IntAttr::get(value.getType(), val));
  }
  auto attrs = ArrayAttr::get(builder.getContext(), caseValuesAttrs);

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
// LoopOpInterface Methods
//===----------------------------------------------------------------------===//

void cir::DoWhileOp::getSuccessorRegions(
    ::mlir::RegionBranchPoint point,
    ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) {
  LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

::llvm::SmallVector<Region *> cir::DoWhileOp::getLoopRegions() {
  return {&getBody()};
}

void cir::WhileOp::getSuccessorRegions(
    ::mlir::RegionBranchPoint point,
    ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) {
  LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

::llvm::SmallVector<Region *> cir::WhileOp::getLoopRegions() {
  return {&getBody()};
}

void cir::ForOp::getSuccessorRegions(
    ::mlir::RegionBranchPoint point,
    ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) {
  LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

::llvm::SmallVector<Region *> cir::ForOp::getLoopRegions() {
  return {&getBody()};
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static ParseResult parseConstantValue(OpAsmParser &parser,
                                      mlir::Attribute &valueAttr) {
  NamedAttrList attr;
  return parser.parseAttribute(valueAttr, "value", attr);
}

// FIXME: create a CIRConstAttr and hide this away for both global
// initialization and cir.const operation.
static void printConstant(OpAsmPrinter &p, Attribute value) {
  p.printAttribute(value);
}

static ParseResult
parseGlobalOpAddrSpace(OpAsmParser &p, cir::AddressSpaceAttr &addrSpaceAttr) {
  return parseAddrSpaceAttribute(p, addrSpaceAttr);
}

static void printGlobalOpAddrSpace(OpAsmPrinter &p, cir::GlobalOp op,
                                   cir::AddressSpaceAttr addrSpaceAttr) {
  printAddrSpaceAttribute(p, addrSpaceAttr);
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
      if (!dtorRegion.hasOneBlock())
        return parser.emitError(parser.getCurrentLocation(),
                                "dtor region must have exactly one block");
      if (dtorRegion.back().empty())
        return parser.emitError(parser.getCurrentLocation(),
                                "dtor region shall not be empty");
      if (ensureRegionTerm(parser, dtorRegion, parseLoc).failed())
        return failure();
    }
  }

  typeAttr = TypeAttr::get(opTy);
  return success();
}

LogicalResult cir::GlobalOp::verify() {
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
  case GlobalLinkageKind::CommonLinkage:
  case GlobalLinkageKind::WeakAnyLinkage:
  case GlobalLinkageKind::WeakODRLinkage:
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

void cir::GlobalOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, llvm::StringRef sym_name,
    Type sym_type, bool isConstant, cir::GlobalLinkageKind linkage,
    cir::AddressSpaceAttr addrSpace,
    function_ref<void(OpBuilder &, Location)> ctorBuilder,
    function_ref<void(OpBuilder &, Location)> dtorBuilder) {
  odsState.addAttribute(getSymNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getSymTypeAttrName(odsState.name),
                        ::mlir::TypeAttr::get(sym_type));
  if (isConstant)
    odsState.addAttribute(getConstantAttrName(odsState.name),
                          odsBuilder.getUnitAttr());

  cir::GlobalLinkageKindAttr linkageAttr =
      cir::GlobalLinkageKindAttr::get(odsBuilder.getContext(), linkage);
  odsState.addAttribute(getLinkageAttrName(odsState.name), linkageAttr);

  if (addrSpace)
    odsState.addAttribute(getAddrSpaceAttrName(odsState.name), addrSpace);

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

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//

LogicalResult
cir::GetGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify that the result type underlying pointer type matches the type of
  // the referenced cir.global or cir.func op.
  auto *op = symbolTable.lookupNearestSymbolFrom(*this, getNameAttr());
  if (op == nullptr || !(isa<GlobalOp>(op) || isa<FuncOp>(op)))
    return emitOpError("'")
           << getName()
           << "' does not reference a valid cir.global or cir.func";

  mlir::Type symTy;
  cir::AddressSpaceAttr symAddrSpace{};
  if (auto g = dyn_cast<GlobalOp>(op)) {
    symTy = g.getSymType();
    symAddrSpace = g.getAddrSpaceAttr();
    // Verify that for thread local global access, the global needs to
    // be marked with tls bits.
    if (getTls() && !g.getTlsModel())
      return emitOpError("access to global not marked thread local");
  } else if (auto f = dyn_cast<FuncOp>(op))
    symTy = f.getFunctionType();
  else
    llvm_unreachable("shall not get here");

  auto resultType = dyn_cast<PointerType>(getAddr().getType());
  if (!resultType || symTy != resultType.getPointee())
    return emitOpError("result type pointee type '")
           << resultType.getPointee() << "' does not match type " << symTy
           << " of the global @" << getName();

  if (symAddrSpace != resultType.getAddrSpace()) {
    return emitOpError()
           << "result type address space does not match the address "
              "space of the global @"
           << getName();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VTableAddrPointOp
//===----------------------------------------------------------------------===//

LogicalResult
cir::VTableAddrPointOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
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
  if (!isa<cir::VTableAttr>(*init))
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
  auto intTy = cir::IntType::get(getContext(), 32, /*isSigned=*/false);
  auto fnTy = cir::FuncType::get({}, intTy);

  auto resTy = cir::PointerType::get(getContext(),
                                     cir::PointerType::get(getContext(), fnTy));

  if (resultType != resTy)
    return emitOpError("result type must be '")
           << resTy << "', but provided result type is '" << resultType << "'";
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
  if (!isa<cir::ConstArrayAttr>(*init))
    return emitOpError("Expected array in initializer for global VTT'")
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

  auto resultType = getAddr().getType();

  auto resTy = cir::PointerType::get(
      getContext(),
      cir::PointerType::get(getContext(), cir::VoidType::get(getContext())));

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
static llvm::StringRef getLinkageAttrNameString() { return "linkage"; }

void cir::FuncOp::build(OpBuilder &builder, OperationState &result,
                        llvm::StringRef name, cir::FuncType type,
                        GlobalLinkageKind linkage, CallingConv callingConv,
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
  result.addAttribute(getCallingConvAttrName(result.name),
                      CallingConvAttr::get(builder.getContext(), callingConv));
  result.addAttribute(getGlobalVisibilityAttrName(result.name),
                      cir::VisibilityAttr::get(builder.getContext()));

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
  auto visibilityNameAttr = getGlobalVisibilityAttrName(state.name);
  auto dsolocalNameAttr = getDsolocalAttrName(state.name);
  auto annotationsNameAttr = getAnnotationsAttrName(state.name);
  if (::mlir::succeeded(parser.parseOptionalKeyword(builtinNameAttr.strref())))
    state.addAttribute(builtinNameAttr, parser.getBuilder().getUnitAttr());
  if (::mlir::succeeded(
          parser.parseOptionalKeyword(coroutineNameAttr.strref())))
    state.addAttribute(coroutineNameAttr, parser.getBuilder().getUnitAttr());
  if (::mlir::succeeded(parser.parseOptionalKeyword(lambdaNameAttr.strref())))
    state.addAttribute(lambdaNameAttr, parser.getBuilder().getUnitAttr());
  if (parser.parseOptionalKeyword(noProtoNameAttr).succeeded())
    state.addAttribute(noProtoNameAttr, parser.getBuilder().getUnitAttr());

  // TODO: Missing comdat
  assert(!cir::MissingFeatures::setComdat());

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

  // TODO: It is unclear whether this is printed in the pretty-printer
  if (parser.parseOptionalKeyword(dsolocalNameAttr).succeeded())
    state.addAttribute(dsolocalNameAttr, parser.getBuilder().getUnitAttr());

  StringAttr nameAttr;
  llvm::SmallVector<OpAsmParser::Argument, 8> arguments;
  llvm::SmallVector<DictionaryAttr, 1> resultAttrs;
  llvm::SmallVector<Type, 8> argTypes;
  llvm::SmallVector<Type, 4> resultTypes;
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
      (resultTypes.empty() ? cir::VoidType::get(builder.getContext())
                           : resultTypes.front());

  // Build the function type.
  auto fnType = cir::FuncType::get(argTypes, returnType, isVariadic);
  if (!fnType)
    return failure();
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(fnType));

  {
    // Parse an OptionalAttr<ArrayAttr>:$annotations
    mlir::ArrayAttr annotations;
    // TODO: Is there a way to restrict the element type to cir.annotation?
    // parseOptionalAttribute takes a type, but unclear how to use this.
    if (auto oa = parser.parseOptionalAttribute(annotations); oa.has_value())
      state.addAttribute(annotationsNameAttr, annotations);
  }

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

  // Default to C calling convention if no keyword is provided.
  auto callConvNameAttr = getCallingConvAttrName(state.name);
  CallingConv callConv = CallingConv::C;
  if (parser.parseOptionalKeyword("cc").succeeded()) {
    if (parser.parseLParen().failed())
      return failure();
    if (parseCIRKeyword<CallingConv>(parser, callConv).failed())
      return parser.emitError(loc) << "unknown calling convention";
    if (parser.parseRParen().failed())
      return failure();
  }
  state.addAttribute(callConvNameAttr,
                     CallingConvAttr::get(parser.getContext(), callConv));

  auto parseGlobalDtorCtor =
      [&](llvm::StringRef keyword,
          llvm::function_ref<void(std::optional<int> prio)> createAttr)
      -> mlir::LogicalResult {
    if (::mlir::succeeded(parser.parseOptionalKeyword(keyword))) {
      std::optional<int> prio;
      if (mlir::succeeded(parser.parseOptionalLParen())) {
        auto parsedPrio = mlir::FieldParser<int>::parse(parser);
        if (mlir::failed(parsedPrio)) {
          return parser.emitError(parser.getCurrentLocation(),
                                  "failed to parse 'priority', of type 'int'");
          return failure();
        }
        prio = parsedPrio.value_or(int());
        // Parse literal ')'
        if (parser.parseRParen())
          return failure();
      }
      createAttr(prio);
    }
    return success();
  };

  if (parseGlobalDtorCtor("global_ctor", [&](std::optional<int> prio) {
        cir::GlobalCtorAttr globalCtorAttr =
            prio ? cir::GlobalCtorAttr::get(builder.getContext(), nameAttr,
                                            *prio)
                 : cir::GlobalCtorAttr::get(builder.getContext(), nameAttr);
        state.addAttribute(getGlobalCtorAttrName(state.name), globalCtorAttr);
      }).failed())
    return failure();

  if (parseGlobalDtorCtor("global_dtor", [&](std::optional<int> prio) {
        cir::GlobalDtorAttr globalDtorAttr =
            prio ? cir::GlobalDtorAttr::get(builder.getContext(), nameAttr,
                                            *prio)
                 : cir::GlobalDtorAttr::get(builder.getContext(), nameAttr);
        state.addAttribute(getGlobalDtorAttrName(state.name), globalDtorAttr);
      }).failed())
    return failure();

  Attribute extraAttrs;
  if (::mlir::succeeded(parser.parseOptionalKeyword("extra"))) {
    if (parser.parseLParen().failed())
      return failure();
    if (parser.parseAttribute(extraAttrs).failed())
      return failure();
    if (parser.parseRParen().failed())
      return failure();
  } else {
    NamedAttrList empty;
    extraAttrs = cir::ExtraFuncAttributesAttr::get(
        builder.getContext(), empty.getDictionary(builder.getContext()));
  }
  state.addAttribute(getExtraAttrsAttrName(state.name), extraAttrs);

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
  auto targetFn = dyn_cast_or_null<cir::FuncOp>(
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
  auto targetFn = dyn_cast_or_null<cir::FuncOp>(
      mlir::SymbolTable::lookupSymbolIn(modOp, *aliasee));
  assert(targetFn && "expected aliasee to exist");
  return targetFn.getCallableRegion();
}

void cir::FuncOp::print(OpAsmPrinter &p) {
  // When adding a specific keyword here, do not forget to omit it in
  // printFunctionAttributes below or there will be a syntax error when
  // parsing
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

  auto vis = getVisibility();
  if (vis != mlir::SymbolTable::Visibility::Public)
    p << ' ' << vis;

  auto cirVisibilityAttr = getGlobalVisibilityAttr();
  if (!cirVisibilityAttr.isDefault()) {
    p << ' ';
    printVisibilityAttr(p, cirVisibilityAttr);
  }

  // Print function name, signature, and control.
  p << ' ';
  p.printSymbolName(getSymName());
  auto fnType = getFunctionType();
  llvm::SmallVector<Type, 1> resultTypes;
  function_interface_impl::printFunctionSignature(
      p, *this, fnType.getInputs(), fnType.isVarArg(), fnType.getReturnTypes());

  if (mlir::ArrayAttr annotations = getAnnotationsAttr()) {
    p << ' ';
    p.printAttribute(annotations);
  }

  function_interface_impl::printFunctionAttributes(
      p, *this,
      // These are all omitted since they are custom printed already.
      {getAliaseeAttrName(), getBuiltinAttrName(), getCoroutineAttrName(),
       getDsolocalAttrName(), getExtraAttrsAttrName(),
       getFunctionTypeAttrName(), getGlobalCtorAttrName(),
       getGlobalDtorAttrName(), getLambdaAttrName(), getLinkageAttrName(),
       getCallingConvAttrName(), getNoProtoAttrName(),
       getSymVisibilityAttrName(), getArgAttrsAttrName(), getResAttrsAttrName(),
       getComdatAttrName(), getGlobalVisibilityAttrName(),
       getAnnotationsAttrName()});

  if (auto aliaseeName = getAliasee()) {
    p << " alias(";
    p.printSymbolName(*aliaseeName);
    p << ")";
  }

  if (getCallingConv() != CallingConv::C) {
    p << " cc(";
    p << stringifyCallingConv(getCallingConv());
    p << ")";
  }

  if (auto globalCtor = getGlobalCtorAttr()) {
    p << " global_ctor";
    if (!globalCtor.isDefaultPriority())
      p << "(" << globalCtor.getPriority() << ")";
  }

  if (auto globalDtor = getGlobalDtorAttr()) {
    p << " global_dtor";
    if (!globalDtor.isDefaultPriority())
      p << "(" << globalDtor.getPriority() << ")";
  }

  if (!getExtraAttrs().getElements().empty()) {
    p << " extra(";
    p.printAttributeWithoutType(getExtraAttrs());
    p << ")";
  }

  // Print the body if this is not an external function.
  Region &body = getOperation()->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
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

  std::set<llvm::StringRef> labels;
  std::set<llvm::StringRef> gotos;

  getOperation()->walk([&](mlir::Operation *op) {
    if (auto lab = dyn_cast<cir::LabelOp>(op)) {
      labels.emplace(lab.getLabel());
    } else if (auto goTo = dyn_cast<cir::GotoOp>(op)) {
      gotos.emplace(goTo.getLabel());
    }
  });

  std::vector<llvm::StringRef> mismatched;
  std::set_difference(gotos.begin(), gotos.end(), labels.begin(), labels.end(),
                      std::back_inserter(mismatched));

  if (!mismatched.empty())
    return emitOpError() << "goto/label mismatch";

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

mlir::Value cir::CallOp::getIndirectCall() {
  assert(isIndirect());
  return getOperand(0);
}

mlir::Operation::operand_iterator cir::CallOp::arg_operand_begin() {
  auto arg_begin = operand_begin();
  if (isIndirect())
    arg_begin++;
  return arg_begin;
}
mlir::Operation::operand_iterator cir::CallOp::arg_operand_end() {
  return operand_end();
}

/// Return the operand at index 'i', accounts for indirect call.
Value cir::CallOp::getArgOperand(unsigned i) {
  if (isIndirect())
    i++;
  return getOperand(i);
}
/// Return the number of operands, accounts for indirect call.
unsigned cir::CallOp::getNumArgOperands() {
  if (isIndirect())
    return this->getOperation()->getNumOperands() - 1;
  return this->getOperation()->getNumOperands();
}

static LogicalResult
verifyCallCommInSymbolUses(Operation *op, SymbolTableCollection &symbolTable) {
  // Callee attribute only need on indirect calls.
  auto fnAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return success();

  cir::FuncOp fn = symbolTable.lookupNearestSymbolFrom<cir::FuncOp>(op, fnAttr);
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

  // Calling convention must match.
  if (callIf.getCallingConv() != fn.getCallingConv())
    return op->emitOpError("calling convention mismatch: expected ")
           << stringifyCallingConv(fn.getCallingConv()) << ", but provided "
           << stringifyCallingConv(callIf.getCallingConv());

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

  return success();
}

static mlir::ParseResult
parseTryCallBranches(mlir::OpAsmParser &parser, mlir::OperationState &result,
                     llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>
                         &continueOperands,
                     llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>
                         &landingPadOperands,
                     llvm::SmallVectorImpl<mlir::Type> &continueTypes,
                     llvm::SmallVectorImpl<mlir::Type> &landingPadTypes,
                     llvm::SMLoc &continueOperandsLoc,
                     llvm::SMLoc &landingPadOperandsLoc) {
  mlir::Block *continueSuccessor = nullptr;
  mlir::Block *landingPadSuccessor = nullptr;

  if (parser.parseSuccessor(continueSuccessor))
    return mlir::failure();
  if (mlir::succeeded(parser.parseOptionalLParen())) {
    continueOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(continueOperands))
      return mlir::failure();
    if (parser.parseColon())
      return mlir::failure();

    if (parser.parseTypeList(continueTypes))
      return mlir::failure();
    if (parser.parseRParen())
      return mlir::failure();
  }
  if (parser.parseComma())
    return mlir::failure();

  if (parser.parseSuccessor(landingPadSuccessor))
    return mlir::failure();
  if (mlir::succeeded(parser.parseOptionalLParen())) {

    landingPadOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(landingPadOperands))
      return mlir::failure();
    if (parser.parseColon())
      return mlir::failure();

    if (parser.parseTypeList(landingPadTypes))
      return mlir::failure();
    if (parser.parseRParen())
      return mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();
    (void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return mlir::failure();
  }
  result.addSuccessors(continueSuccessor);
  result.addSuccessors(landingPadSuccessor);
  return mlir::success();
}

static ::mlir::ParseResult parseCallCommon(::mlir::OpAsmParser &parser,
                                           ::mlir::OperationState &result,
                                           llvm::StringRef extraAttrsAttrName,
                                           bool hasDestinationBlocks = false) {
  mlir::FlatSymbolRefAttr calleeAttr;
  llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ops;
  llvm::SMLoc opsLoc;
  (void)opsLoc;
  llvm::ArrayRef<::mlir::Type> operandsTypes;
  llvm::ArrayRef<::mlir::Type> allResultTypes;

  // Control flow related
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> continueOperands;
  llvm::SMLoc continueOperandsLoc;
  llvm::SmallVector<mlir::Type, 1> continueTypes;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> landingPadOperands;
  llvm::SMLoc landingPadOperandsLoc;
  llvm::SmallVector<mlir::Type, 1> landingPadTypes;

  bool hasExceptions = false;
  if (::mlir::succeeded(parser.parseOptionalKeyword("exception"))) {
    result.addAttribute("exception", parser.getBuilder().getUnitAttr());
    hasExceptions = true;
  }

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

  if (hasDestinationBlocks)
    if (parseTryCallBranches(parser, result, continueOperands,
                             landingPadOperands, continueTypes, landingPadTypes,
                             continueOperandsLoc, landingPadOperandsLoc)
            .failed())
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

  if (hasDestinationBlocks) {
    // The TryCall ODS layout is: cont, landing_pad, operands.
    llvm::copy(::llvm::ArrayRef<int32_t>(
                   {static_cast<int32_t>(continueOperands.size()),
                    static_cast<int32_t>(landingPadOperands.size()),
                    static_cast<int32_t>(ops.size())}),
               result.getOrAddProperties<cir::TryCallOp::Properties>()
                   .operandSegmentSizes.begin());
    if (parser.resolveOperands(continueOperands, continueTypes,
                               continueOperandsLoc, result.operands))
      return ::mlir::failure();
    if (parser.resolveOperands(landingPadOperands, landingPadTypes,
                               landingPadOperandsLoc, result.operands))
      return ::mlir::failure();
  }

  auto &builder = parser.getBuilder();
  if (parser.parseOptionalKeyword("cc").succeeded()) {
    if (parser.parseLParen().failed())
      return failure();
    cir::CallingConv callingConv;
    if (parseCIRKeyword<cir::CallingConv>(parser, callingConv).failed())
      return failure();
    if (parser.parseRParen().failed())
      return failure();
    result.addAttribute("calling_conv", cir::CallingConvAttr::get(
                                            builder.getContext(), callingConv));
  }

  if (parser.parseOptionalKeyword("side_effect").succeeded()) {
    if (parser.parseLParen().failed())
      return failure();
    cir::SideEffect sideEffect;
    if (parseCIRKeyword<cir::SideEffect>(parser, sideEffect).failed())
      return failure();
    if (parser.parseRParen().failed())
      return failure();
    result.addAttribute("side_effect", cir::SideEffectAttr::get(
                                           builder.getContext(), sideEffect));
  }

  Attribute extraAttrs;
  if (::mlir::succeeded(parser.parseOptionalKeyword("extra"))) {
    if (parser.parseLParen().failed())
      return failure();
    if (parser.parseAttribute(extraAttrs).failed())
      return failure();
    if (parser.parseRParen().failed())
      return failure();
  } else {
    NamedAttrList empty;
    extraAttrs = cir::ExtraFuncAttributesAttr::get(
        builder.getContext(), empty.getDictionary(builder.getContext()));
  }
  result.addAttribute(extraAttrsAttrName, extraAttrs);

  // If exception is present and there are cleanups, this should be latest thing
  // present (after all attributes, etc).
  mlir::Region *cleanupRegion = nullptr;
  if (!hasDestinationBlocks) // Regular cir.call
    cleanupRegion = result.addRegion();
  if (hasExceptions) {
    if (parser.parseOptionalKeyword("cleanup").succeeded()) {
      if (parser.parseRegion(*cleanupRegion, /*arguments=*/{}, /*argTypes=*/{}))
        return failure();
    }
  }

  return ::mlir::success();
}

void printCallCommon(Operation *op, mlir::Value indirectCallee,
                     mlir::FlatSymbolRefAttr flatSym,
                     ::mlir::OpAsmPrinter &state,
                     cir::ExtraFuncAttributesAttr extraAttrs,
                     cir::CallingConv callingConv, cir::SideEffect sideEffect,
                     ::mlir::UnitAttr exception = {},
                     mlir::Block *cont = nullptr,
                     mlir::Block *landingPad = nullptr) {
  state << ' ';

  auto callLikeOp = mlir::cast<cir::CIRCallOpInterface>(op);
  auto ops = callLikeOp.getArgOperands();

  if (exception)
    state << "exception ";

  if (flatSym) { // Direct calls
    state.printAttributeWithoutType(flatSym);
  } else { // Indirect calls
    assert(indirectCallee);
    state << indirectCallee;
  }
  state << "(";
  state << ops;
  state << ")";

  if (cont) {
    assert(landingPad && "expected two successors");
    auto tryCall = dyn_cast<cir::TryCallOp>(op);
    assert(tryCall && "regular calls do not branch");
    state << ' ' << tryCall.getCont();
    if (!tryCall.getContOperands().empty()) {
      state << "(";
      state << tryCall.getContOperands();
      state << ' ' << ":";
      state << ' ';
      state << tryCall.getContOperands().getTypes();
      state << ")";
    }
    state << ",";
    state << ' ';
    state << tryCall.getLandingPad();
    if (!tryCall.getLandingPadOperands().empty()) {
      state << "(";
      state << tryCall.getLandingPadOperands();
      state << ' ' << ":";
      state << ' ';
      state << tryCall.getLandingPadOperands().getTypes();
      state << ")";
    }
  }

  llvm::SmallVector<::llvm::StringRef, 4> elidedAttrs;
  elidedAttrs.push_back("callee");
  elidedAttrs.push_back("ast");
  elidedAttrs.push_back("extra_attrs");
  elidedAttrs.push_back("calling_conv");
  elidedAttrs.push_back("side_effect");
  elidedAttrs.push_back("exception");
  elidedAttrs.push_back("operandSegmentSizes");

  state.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  state << ' ' << ":";
  state << ' ';
  state.printFunctionalType(op->getOperands().getTypes(), op->getResultTypes());

  if (callingConv != cir::CallingConv::C) {
    state << " cc(";
    state << stringifyCallingConv(callingConv);
    state << ")";
  }

  if (sideEffect != cir::SideEffect::All) {
    state << " side_effect(";
    state << stringifySideEffect(sideEffect);
    state << ")";
  }

  if (!extraAttrs.getElements().empty()) {
    state << " extra(";
    state.printAttributeWithoutType(extraAttrs);
    state << ")";
  }

  // If exception is present and there are cleanups, this should be latest thing
  // present (after all attributes, etc).
  if (exception) {
    auto call = dyn_cast<cir::CallOp>(op);
    assert(call && "expected regular call");
    if (!call.getCleanup().empty()) {
      state << " cleanup ";
      state.printRegion(call.getCleanup());
    }
  }
}

LogicalResult
cir::CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyCallCommInSymbolUses(*this, symbolTable);
}

::mlir::ParseResult cir::CallOp::parse(::mlir::OpAsmParser &parser,
                                       ::mlir::OperationState &result) {

  return parseCallCommon(parser, result, getExtraAttrsAttrName(result.name));
}

void cir::CallOp::print(::mlir::OpAsmPrinter &state) {
  mlir::Value indirectCallee = isIndirect() ? getIndirectCall() : nullptr;
  cir::CallingConv callingConv = getCallingConv();
  cir::SideEffect sideEffect = getSideEffect();
  mlir::UnitAttr exception = getExceptionAttr();
  printCallCommon(*this, indirectCallee, getCalleeAttr(), state,
                  getExtraAttrs(), callingConv, sideEffect, exception);
}

//===----------------------------------------------------------------------===//
// TryCallOp
//===----------------------------------------------------------------------===//

mlir::Value cir::TryCallOp::getIndirectCall() {
  assert(isIndirect());
  return getOperand(0);
}

mlir::Operation::operand_iterator cir::TryCallOp::arg_operand_begin() {
  auto arg_begin = operand_begin();
  if (isIndirect())
    arg_begin++;
  return arg_begin;
}
mlir::Operation::operand_iterator cir::TryCallOp::arg_operand_end() {
  return operand_end();
}

/// Return the operand at index 'i', accounts for indirect call.
Value cir::TryCallOp::getArgOperand(unsigned i) {
  if (isIndirect())
    i++;
  return getOperand(i);
}
/// Return the number of operands, accounts for indirect call.
unsigned cir::TryCallOp::getNumArgOperands() {
  if (isIndirect())
    return this->getOperation()->getNumOperands() - 1;
  return this->getOperation()->getNumOperands();
}

LogicalResult
cir::TryCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyCallCommInSymbolUses(*this, symbolTable);
}

::mlir::ParseResult cir::TryCallOp::parse(::mlir::OpAsmParser &parser,
                                          ::mlir::OperationState &result) {

  return parseCallCommon(parser, result, getExtraAttrsAttrName(result.name),
                         /*hasDestinationBlocks=*/true);
}

void cir::TryCallOp::print(::mlir::OpAsmPrinter &state) {
  mlir::Value indirectCallee = isIndirect() ? getIndirectCall() : nullptr;
  cir::CallingConv callingConv = getCallingConv();
  cir::SideEffect sideEffect = getSideEffect();
  printCallCommon(*this, indirectCallee, getCalleeAttr(), state,
                  getExtraAttrs(), callingConv, sideEffect, {}, getCont(),
                  getLandingPad());
}

mlir::SuccessorOperands cir::TryCallOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  if (index == 0)
    return SuccessorOperands(getContOperandsMutable());
  if (index == 1)
    return SuccessorOperands(getLandingPadOperandsMutable());

  // index == 2
  return SuccessorOperands(getArgOperandsMutable());
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

//===----------------------------------------------------------------------===//
// AwaitOp
//===----------------------------------------------------------------------===//

void cir::AwaitOp::build(
    OpBuilder &builder, OperationState &result, cir::AwaitKind kind,
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
void cir::AwaitOp::getSuccessorRegions(
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
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

LogicalResult cir::AwaitOp::verify() {
  if (!isa<ConditionOp>(this->getReady().back().getTerminator()))
    return emitOpError("ready region must end with cir.condition");
  return success();
}

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

LogicalResult
mlir::OpTrait::impl::verifySameSecondOperandAndResultType(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 2)) || failed(verifyOneResult(op)))
    return failure();

  auto type = op->getResult(0).getType();
  auto opType = op->getOperand(1).getType();

  if (type != opType)
    return op->emitOpError()
           << "requires the same type for second operand and result";

  return success();
}

LogicalResult
mlir::OpTrait::impl::verifySameFirstSecondOperandAndResultType(Operation *op) {
  if (failed(verifyAtLeastNOperands(op, 3)) || failed(verifyOneResult(op)))
    return failure();

  auto checkType = op->getResult(0).getType();
  if (checkType != op->getOperand(0).getType() &&
      checkType != op->getOperand(1).getType())
    return op->emitOpError()
           << "requires the same type for first, second operand and result";

  return success();
}

//===----------------------------------------------------------------------===//
// CIR attributes
// FIXME: move all of these to CIRAttrs.cpp
//===----------------------------------------------------------------------===//

LogicalResult cir::ConstArrayAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::Type type, Attribute attr, int trailingZerosNum) {

  if (!(mlir::isa<mlir::ArrayAttr>(attr) || mlir::isa<mlir::StringAttr>(attr)))
    return emitError() << "constant array expects ArrayAttr or StringAttr";

  if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr)) {
    cir::ArrayType at = mlir::cast<cir::ArrayType>(type);
    auto intTy = mlir::dyn_cast<cir::IntType>(at.getEltType());

    // TODO: add CIR type for char.
    if (!intTy || intTy.getWidth() != 8) {
      emitError() << "constant array element for string literals expects "
                     "!cir.int<u, 8> element type";
      return failure();
    }
    return success();
  }

  assert(mlir::isa<mlir::ArrayAttr>(attr));
  auto arrayAttr = mlir::cast<mlir::ArrayAttr>(attr);
  auto at = mlir::cast<ArrayType>(type);

  // Make sure both number of elements and subelement types match type.
  if (at.getSize() != arrayAttr.size() + trailingZerosNum)
    return emitError() << "constant array size should match type size";
  return success();
}

::mlir::Attribute cir::ConstArrayAttr::parse(::mlir::AsmParser &parser,
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
  if (mlir::dyn_cast<ArrayAttr>(*resultVal)) {
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
    assert(mlir::isa<TypedAttr>(*resultVal) && "IDK");
    auto ta = mlir::cast<TypedAttr>(*resultVal);
    resultTy = ta.getType();
    if (mlir::isa<mlir::NoneType>(*resultTy)) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected type declaration for string literal");
      return {};
    }
  }

  auto zeros = 0;
  if (parser.parseOptionalComma().succeeded()) {
    if (parser.parseOptionalKeyword("trailing_zeros").succeeded()) {
      auto typeSize = mlir::cast<cir::ArrayType>(resultTy.value()).getSize();
      auto elts = resultVal.value();
      if (auto str = mlir::dyn_cast<mlir::StringAttr>(elts))
        zeros = typeSize - str.size();
      else
        zeros = typeSize - mlir::cast<mlir::ArrayAttr>(elts).size();
    } else {
      return {};
    }
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  return parser.getChecked<ConstArrayAttr>(
      loc, parser.getContext(), resultTy.value(), resultVal.value(), zeros);
}

void cir::ConstArrayAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getElts());
  if (getTrailingZerosNum())
    printer << ", trailing_zeros";
  printer << ">";
}

LogicalResult cir::ConstVectorAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::Type type, mlir::ArrayAttr arrayAttr) {

  if (!mlir::isa<cir::VectorType>(type)) {
    return emitError() << "type of cir::ConstVectorAttr is not a "
                          "cir::VectorType: "
                       << type;
  }
  auto vecType = mlir::cast<cir::VectorType>(type);

  // Do the number of elements match?
  if (vecType.getSize() != arrayAttr.size()) {
    return emitError()
           << "number of constant elements should match vector size";
  }
  // Do the types of the elements match?
  LogicalResult elementTypeCheck = success();
  arrayAttr.walkImmediateSubElements(
      [&](Attribute element) {
        if (elementTypeCheck.failed()) {
          // An earlier element didn't match
          return;
        }
        auto typedElement = mlir::dyn_cast<TypedAttr>(element);
        if (!typedElement || typedElement.getType() != vecType.getEltType()) {
          elementTypeCheck = failure();
          emitError() << "constant type should match vector element type";
        }
      },
      [&](Type) {});
  return elementTypeCheck;
}

::mlir::Attribute cir::ConstVectorAttr::parse(::mlir::AsmParser &parser,
                                              ::mlir::Type type) {
  ::mlir::FailureOr<::mlir::Type> resultType;
  ::mlir::FailureOr<ArrayAttr> resultValue;
  ::llvm::SMLoc loc = parser.getCurrentLocation();

  // Parse literal '<'
  if (parser.parseLess()) {
    return {};
  }

  // Parse variable 'value'
  resultValue = ::mlir::FieldParser<ArrayAttr>::parse(parser);
  if (failed(resultValue)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse ConstVectorAttr parameter 'value' as "
                     "an attribute");
    return {};
  }

  if (parser.parseOptionalColon().failed()) {
    resultType = type;
  } else {
    resultType = ::mlir::FieldParser<::mlir::Type>::parse(parser);
    if (failed(resultType)) {
      parser.emitError(parser.getCurrentLocation(),
                       "failed to parse ConstVectorAttr parameter 'type' as "
                       "an MLIR type");
      return {};
    }
  }

  // Parse literal '>'
  if (parser.parseGreater()) {
    return {};
  }

  return parser.getChecked<ConstVectorAttr>(
      loc, parser.getContext(), resultType.value(), resultValue.value());
}

void cir::ConstVectorAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getElts());
  printer << ">";
}

::mlir::Attribute
cir::SignedOverflowBehaviorAttr::parse(::mlir::AsmParser &parser,
                                       ::mlir::Type type) {
  if (parser.parseLess())
    return {};
  auto behavior = parseOptionalCIRKeyword(
      parser, cir::sob::SignedOverflowBehavior::undefined);
  if (parser.parseGreater())
    return {};

  return SignedOverflowBehaviorAttr::get(parser.getContext(), behavior);
}

void cir::SignedOverflowBehaviorAttr::print(::mlir::AsmPrinter &printer) const {
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

LogicalResult cir::TypeInfoAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::Type type, ::mlir::ArrayAttr typeinfoData) {

  if (cir::ConstStructAttr::verify(emitError, type, typeinfoData).failed())
    return failure();

  for (auto &member : typeinfoData) {
    if (llvm::isa<GlobalViewAttr, IntAttr>(member))
      continue;
    emitError() << "expected GlobalViewAttr or IntAttr attribute";
    return failure();
  }

  return success();
}

LogicalResult cir::VTableAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::Type type, ::mlir::ArrayAttr vtableData) {
  auto sTy = mlir::dyn_cast_if_present<cir::StructType>(type);
  if (!sTy) {
    emitError() << "expected !cir.struct type result";
    return failure();
  }
  if (sTy.getMembers().empty() || vtableData.empty()) {
    emitError() << "expected struct type with one or more subtype";
    return failure();
  }

  for (size_t i = 0; i < sTy.getMembers().size(); ++i) {

    auto arrayTy = mlir::dyn_cast<cir::ArrayType>(sTy.getMembers()[i]);
    auto constArrayAttr = mlir::dyn_cast<cir::ConstArrayAttr>(vtableData[i]);
    if (!arrayTy || !constArrayAttr) {
      emitError() << "expected struct type with one array element";
      return failure();
    }

    if (cir::ConstStructAttr::verify(emitError, type, vtableData).failed())
      return failure();

    LogicalResult eltTypeCheck = success();
    if (auto arrayElts = mlir::dyn_cast<ArrayAttr>(constArrayAttr.getElts())) {
      arrayElts.walkImmediateSubElements(
          [&](Attribute attr) {
            if (mlir::isa<GlobalViewAttr>(attr) ||
                mlir::isa<ConstPtrAttr>(attr))
              return;
            emitError() << "expected GlobalViewAttr attribute";
            eltTypeCheck = failure();
          },
          [&](Type type) {});
      if (eltTypeCheck.failed()) {
        return eltTypeCheck;
      }
    }
  }

  return success();
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

  const auto recordTy = dyn_cast<StructType>(getAddrTy().getPointee());
  if (!recordTy)
    return emitError() << "expected pointer to a record type";

  if (recordTy.getMembers().size() <= getIndex())
    return emitError() << "member index out of bounds";

  // FIXME(cir): member type check is disabled for classes as the codegen for
  // these still need to be patched.
  if (!recordTy.isClass() &&
      recordTy.getMembers()[getIndex()] != getResultTy().getPointee())
    return emitError() << "member type mismatch";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ExtractMemberOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::ExtractMemberOp::verify() {
  auto recordTy = mlir::cast<cir::StructType>(getRecord().getType());
  if (recordTy.getKind() == cir::StructType::Union)
    return emitError()
           << "cir.extract_member currently does not work on unions";
  if (recordTy.getMembers().size() <= getIndex())
    return emitError() << "member index out of bounds";
  if (recordTy.getMembers()[getIndex()] != getType())
    return emitError() << "member type mismatch";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// InsertMemberOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::InsertMemberOp::verify() {
  auto recordTy = mlir::cast<cir::StructType>(getRecord().getType());
  if (recordTy.getKind() == cir::StructType::Union)
    return emitError() << "cir.update_member currently does not work on unions";
  if (recordTy.getMembers().size() <= getIndex())
    return emitError() << "member index out of range";
  if (recordTy.getMembers()[getIndex()] != getValue().getType())
    return emitError() << "member type mismatch";
  // The op trait already checks that the types of $result and $record match.
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GetRuntimeMemberOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::GetRuntimeMemberOp::verify() {
  auto recordTy =
      cast<StructType>(cast<PointerType>(getAddr().getType()).getPointee());
  auto memberPtrTy = getMember().getType();

  if (recordTy != memberPtrTy.getClsTy()) {
    emitError() << "record type does not match the member pointer type";
    return mlir::failure();
  }

  if (getType().getPointee() != memberPtrTy.getMemberTy()) {
    emitError() << "result type does not match the member pointer type";
    return mlir::failure();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GetMethodOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::GetMethodOp::verify() {
  auto methodTy = getMethod().getType();

  // Assume objectTy is !cir.ptr<!T>
  auto objectPtrTy = mlir::cast<cir::PointerType>(getObject().getType());
  auto objectTy = objectPtrTy.getPointee();

  if (methodTy.getClsTy() != objectTy) {
    emitError() << "method class type and object type do not match";
    return mlir::failure();
  }

  // Assume methodFuncTy is !cir.func<!Ret (!Args)>
  auto calleePtrTy = mlir::cast<cir::PointerType>(getCallee().getType());
  auto calleeTy = mlir::cast<cir::FuncType>(calleePtrTy.getPointee());
  auto methodFuncTy = methodTy.getMemberFuncTy();

  // We verify at here that calleeTy is !cir.func<!Ret (!cir.ptr<!void>, !Args)>
  // Note that the first parameter type of the callee is !cir.ptr<!void> instead
  // of !cir.ptr<!T> because the "this" pointer may be adjusted before calling
  // the callee.

  if (methodFuncTy.getReturnType() != calleeTy.getReturnType()) {
    emitError() << "method return type and callee return type do not match";
    return mlir::failure();
  }

  auto calleeArgsTy = calleeTy.getInputs();
  auto methodFuncArgsTy = methodFuncTy.getInputs();

  if (calleeArgsTy.empty()) {
    emitError() << "callee parameter list lacks receiver object ptr";
    return mlir::failure();
  }

  auto calleeThisArgPtrTy = mlir::dyn_cast<cir::PointerType>(calleeArgsTy[0]);
  if (!calleeThisArgPtrTy ||
      !mlir::isa<cir::VoidType>(calleeThisArgPtrTy.getPointee())) {
    emitError() << "the first parameter of callee must be a void pointer";
    return mlir::failure();
  }

  if (calleeArgsTy.slice(1) != methodFuncArgsTy) {
    emitError() << "callee parameters and method parameters do not match";
    return mlir::failure();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// InlineAsmOp Definitions
//===----------------------------------------------------------------------===//

void cir::InlineAsmOp::print(OpAsmPrinter &p) {
  p << '(' << getAsmFlavor() << ", ";
  p.increaseIndent();
  p.printNewline();

  llvm::SmallVector<std::string, 3> names{"out", "in", "in_out"};
  auto nameIt = names.begin();
  auto attrIt = getOperandAttrs().begin();

  for (auto ops : getAsmOperands()) {
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

  llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("asm_flavor");
  elidedAttrs.push_back("asm_string");
  elidedAttrs.push_back("constraints");
  elidedAttrs.push_back("operand_attrs");
  elidedAttrs.push_back("operands_segments");
  elidedAttrs.push_back("side_effects");
  p.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);

  if (auto v = getRes())
    p << " -> " << v.getType();
}

ParseResult cir::InlineAsmOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  llvm::SmallVector<mlir::Attribute> operand_attrs;
  llvm::SmallVector<int32_t> operandsGroupSizes;
  std::string asm_string, constraints;
  Type resType;
  auto *ctxt = parser.getBuilder().getContext();

  auto error = [&](const Twine &msg) {
    parser.emitError(parser.getCurrentLocation(), msg);
    ;
    return mlir::failure();
  };

  auto expected = [&](const std::string &c) {
    return error("expected '" + c + "'");
  };

  if (parser.parseLParen().failed())
    return expected("(");

  auto flavor = mlir::FieldParser<AsmFlavor>::parse(parser);
  if (failed(flavor))
    return error("Unknown AsmFlavor");

  if (parser.parseComma().failed())
    return expected(",");

  auto parseValue = [&](Value &v) {
    OpAsmParser::UnresolvedOperand op;

    if (parser.parseOperand(op) || parser.parseColon())
      return mlir::failure();

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

    if (parser.parseCommaSeparatedList([&]() {
          Value val;
          if (parseValue(val).succeeded()) {
            result.operands.push_back(val);
            size++;

            if (parser.parseOptionalLParen().failed()) {
              operand_attrs.push_back(mlir::Attribute());
              return mlir::success();
            }

            if (parser.parseKeyword("maybe_memory").succeeded()) {
              operand_attrs.push_back(mlir::UnitAttr::get(ctxt));
              if (parser.parseRParen())
                return expected(")");
              return mlir::success();
            }
          }
          return mlir::failure();
        }))
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
  if (parser.parseString(&asm_string))
    return error("asm string parsing failed");
  if (parser.parseString(&constraints))
    return error("constraints string parsing failed");
  if (parser.parseRBrace())
    return expected("}");
  if (parser.parseRParen())
    return expected(")");

  if (parser.parseOptionalKeyword("side_effects").succeeded())
    result.attributes.set("side_effects", UnitAttr::get(ctxt));

  if (parser.parseOptionalArrow().failed())
    return mlir::failure();

  if (parser.parseType(resType).failed())
    return mlir::failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  result.attributes.set("asm_flavor", AsmFlavorAttr::get(ctxt, *flavor));
  result.attributes.set("asm_string", StringAttr::get(ctxt, asm_string));
  result.attributes.set("constraints", StringAttr::get(ctxt, constraints));
  result.attributes.set("operand_attrs", ArrayAttr::get(ctxt, operand_attrs));
  result.getOrAddProperties<InlineAsmOp::Properties>().operands_segments =
      parser.getBuilder().getDenseI32ArrayAttr(operandsGroupSizes);
  if (resType)
    result.addTypes(TypeRange{resType});

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Atomic Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::AtomicFetch::verify() {
  if (getBinop() == cir::AtomicFetchKind::Add ||
      getBinop() == cir::AtomicFetchKind::Sub)
    return mlir::success();

  if (!mlir::isa<cir::IntType>(getVal().getType()))
    return emitError() << "only operates on integer values";

  return mlir::success();
}

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

  bool complexOps =
      getKind() == cir::BinOpKind::Add || getKind() == cir::BinOpKind::Sub;
  if (isa<cir::ComplexType>(getType()) && !complexOps)
    return emitError()
           << "cir.binop can only represent 'add' and 'sub' on complex numbers";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ShiftOp Definitions
//===----------------------------------------------------------------------===//
LogicalResult cir::ShiftOp::verify() {
  mlir::Operation *op = getOperation();
  mlir::Type resType = getResult().getType();
  bool isOp0Vec = mlir::isa<cir::VectorType>(op->getOperand(0).getType());
  bool isOp1Vec = mlir::isa<cir::VectorType>(op->getOperand(1).getType());
  if (isOp0Vec != isOp1Vec)
    return emitOpError() << "input types cannot be one vector and one scalar";
  if (isOp1Vec && op->getOperand(1).getType() != resType) {
    return emitOpError() << "shift amount must have the type of the result "
                         << "if it is vector shift";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// LabelOp Definitions
//===----------------------------------------------------------------------===//

LogicalResult cir::LabelOp::verify() {
  auto *op = getOperation();
  auto *blk = op->getBlock();
  if (&blk->front() != op)
    return emitError() << "must be the first operation in a block";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// EhTypeIdOp
//===----------------------------------------------------------------------===//

LogicalResult
cir::EhTypeIdOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto op = symbolTable.lookupNearestSymbolFrom(*this, getTypeSymAttr());
  if (!isa<GlobalOp>(op))
    return emitOpError("'")
           << getTypeSym() << "' does not reference a valid cir.global";
  return success();
}

//===----------------------------------------------------------------------===//
// CatchParamOp
//===----------------------------------------------------------------------===//

LogicalResult cir::CatchParamOp::verify() {
  if (getExceptionPtr()) {
    auto kind = getKind();
    if (!kind || *kind != cir::CatchParamKind::begin)
      return emitOpError("needs 'begin' to work with exception pointer");
    return success();
  }
  if (!getKind() && !(*this)->getParentOfType<cir::TryOp>())
    return emitOpError("without 'kind' requires 'cir.try' surrounding scope");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROps.cpp.inc"
