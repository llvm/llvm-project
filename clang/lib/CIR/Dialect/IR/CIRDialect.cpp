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

#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "clang/CIR/Dialect/IR/CIROpsDialect.cpp.inc"
#include "clang/CIR/Dialect/IR/CIROpsEnums.cpp.inc"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//
namespace {
struct CIROpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    return AliasResult::NoAlias;
  }

  AliasResult getAlias(Attribute attr, raw_ostream &os) const final {
    if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(attr)) {
      os << (boolAttr.getValue() ? "true" : "false");
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

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

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
}

void cir::ScopeOp::build(
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

void cir::GlobalOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          llvm::StringRef sym_name, mlir::Type sym_type,
                          cir::GlobalLinkageKind linkage) {
  odsState.addAttribute(getSymNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getSymTypeAttrName(odsState.name),
                        mlir::TypeAttr::get(sym_type));

  cir::GlobalLinkageKindAttr linkageAttr =
      cir::GlobalLinkageKindAttr::get(odsBuilder.getContext(), linkage);
  odsState.addAttribute(getLinkageAttrName(odsState.name), linkageAttr);
}

static void printGlobalOpTypeAndInitialValue(OpAsmPrinter &p, cir::GlobalOp op,
                                             TypeAttr type,
                                             Attribute initAttr) {
  if (!op.isDeclaration()) {
    p << "= ";
    // This also prints the type...
    if (initAttr)
      printConstant(p, initAttr);
  } else {
    p << ": " << type;
  }
}

static ParseResult
parseGlobalOpTypeAndInitialValue(OpAsmParser &parser, TypeAttr &typeAttr,
                                 Attribute &initialValueAttr) {
  mlir::Type opTy;
  if (parser.parseOptionalEqual().failed()) {
    // Absence of equal means a declaration, so we need to parse the type.
    //  cir.global @a : !cir.int<s, 32>
    if (parser.parseColonType(opTy))
      return failure();
  } else {
    // Parse constant with initializer, examples:
    //  cir.global @y = #cir.fp<1.250000e+00> : !cir.double
    //  cir.global @rgb = #cir.const_array<[...] : !cir.array<i8 x 3>>
    if (parseConstantValue(parser, initialValueAttr).failed())
      return failure();

    assert(mlir::isa<mlir::TypedAttr>(initialValueAttr) &&
           "Non-typed attrs shouldn't appear here.");
    auto typedAttr = mlir::cast<mlir::TypedAttr>(initialValueAttr);
    opTy = typedAttr.getType();
  }

  typeAttr = TypeAttr::get(opTy);
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void cir::FuncOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, FuncType type) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));
}

ParseResult cir::FuncOp::parse(OpAsmParser &parser, OperationState &state) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  mlir::Builder &builder = parser.getBuilder();

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

  // Parse the optional function body.
  auto *body = state.addRegion();
  OptionalParseResult parseResult = parser.parseOptionalRegion(
      *body, arguments, /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return parser.emitError(loc, "expected non-empty function body");
  }

  return success();
}

bool cir::FuncOp::isDeclaration() {
  // TODO(CIR): This function will actually do something once external function
  // declarations and aliases are upstreamed.
  return false;
}

mlir::Region *cir::FuncOp::getCallableRegion() {
  // TODO(CIR): This function will have special handling for aliases and a
  // check for an external function, once those features have been upstreamed.
  return &getBody();
}

void cir::FuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  cir::FuncType fnType = getFunctionType();
  function_interface_impl::printFunctionSignature(
      p, *this, fnType.getInputs(), fnType.isVarArg(), fnType.getReturnTypes());

  // Print the body if this is not an external function.
  Region &body = getOperation()->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

// TODO(CIR): The properties of functions that require verification haven't
// been implemented yet.
mlir::LogicalResult cir::FuncOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROps.cpp.inc"
