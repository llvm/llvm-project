//===- CIROpsBuiltins.cpp - CIR ops in partition `Builtins` ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR ops in partition `Builtins`: the hand-written
// parts first, then the TableGen-generated definitions.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIROpsBuiltins.h"
#include "clang/CIR/Dialect/IR/CIROpsCore.h"

#include "CIROpDefinitions.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace cir;

// Verifies that the given operand is produced by an operation of type
// ExpectedProducerOp.
template <typename ExpectedProducerOp>
static LogicalResult verifyProducedBy(Operation *op, Value operand,
                                      StringRef operandName) {
  Operation *producer = operand.getDefiningOp();
  if (!producer || !isa<ExpectedProducerOp>(producer))
    return op->emitOpError()
           << "operand '" << operandName << "' must be produced by '"
           << ExpectedProducerOp::getOperationName() << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// AssumeOp
//===----------------------------------------------------------------------===//

void cir::printAssumeBundle(OpAsmPrinter &p, cir::AssumeOp op,
                            cir::AssumeBundleKindAttr kindAttr,
                            OperandRange bundleArgs, TypeRange bundleArgTypes) {
  cir::AssumeBundleKind kind = kindAttr.getValue();
  if (kind == cir::AssumeBundleKind::None)
    return;

  p << " " << cir::stringifyAssumeBundleKind(kind);
  if (bundleArgs.empty())
    return;

  p << "(";
  p.printOperands(bundleArgs);
  p << " : ";
  llvm::interleaveComma(bundleArgTypes, p);
  p << ")";
}

ParseResult cir::parseAssumeBundle(
    OpAsmParser &p, cir::AssumeBundleKindAttr &bundleKindAttr,
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> &bundleArgs,
    llvm::SmallVector<mlir::Type, 1> &bundleArgTypes) {
  StringRef keyword;
  auto loc = p.getCurrentLocation();
  if (failed(p.parseOptionalKeyword(&keyword))) {
    bundleKindAttr = cir::AssumeBundleKindAttr::get(
        p.getContext(), cir::AssumeBundleKind::None);
    return success();
  }

  std::optional<cir::AssumeBundleKind> parsedKind =
      cir::symbolizeAssumeBundleKind(keyword);
  if (!parsedKind)
    return p.emitError(loc, "unknown assume bundle kind '") << keyword << "'";

  bundleKindAttr = cir::AssumeBundleKindAttr::get(p.getContext(), *parsedKind);

  if (p.parseOptionalLParen())
    return success();

  if (p.parseOperandList(bundleArgs) || p.parseColon() ||
      p.parseTypeList(bundleArgTypes) || p.parseRParen())
    return failure();

  return success();
}

LogicalResult cir::AssumeOp::verify() {
  cir::AssumeBundleKind kind = getBundleKind();
  size_t numArgs = getBundleArgs().size();

  if (kind == cir::AssumeBundleKind::None) {
    if (numArgs != 0)
      return emitOpError("unexpected bundle operands for kind 'none'");
    return success();
  }

  if (numArgs == 0)
    return emitOpError("expected bundle operands for kind '")
           << cir::stringifyAssumeBundleKind(kind) << "'";

  switch (kind) {
  case cir::AssumeBundleKind::Align:
    if (numArgs != 2 && numArgs != 3)
      return emitOpError("align bundle expects 2 or 3 operands");
    break;
  case cir::AssumeBundleKind::SeparateStorage:
    if (numArgs != 2)
      return emitOpError("separate_storage bundle expects 2 operands");
    break;
  case cir::AssumeBundleKind::Dereferenceable:
    if (numArgs != 2)
      return emitOpError("dereferenceable bundle expects 2 operands");
    break;
  default:
    break;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp Definitions
//===----------------------------------------------------------------------===//

// Prints the pointer type(s) for a `cir.copy`. `src` and `dst` share the same
// pointee type but may differ in address space; a single type is printed when
// they match, otherwise both are printed (`src` first).
void cir::printCopyTypes(mlir::OpAsmPrinter &printer, mlir::Operation *,
                         mlir::Type srcType, mlir::Type dstType) {
  printer.printType(srcType);
  if (srcType != dstType) {
    printer << ", ";
    printer.printType(dstType);
  }
}

mlir::ParseResult cir::parseCopyTypes(mlir::OpAsmParser &parser,
                                      mlir::Type &srcType,
                                      mlir::Type &dstType) {
  if (parser.parseType(srcType))
    return mlir::failure();
  if (parser.parseOptionalComma().succeeded()) {
    if (parser.parseType(dstType))
      return mlir::failure();
  } else {
    dstType = srcType;
  }
  return mlir::success();
}

LogicalResult cir::CopyOp::verify() {
  // The pointee types of `src` and `dst` are guaranteed to match by the
  // SameOperandsPointeeType trait; they may still differ in address space.

  // A data layout is required for us to know the number of bytes to be copied.
  if (!getType().getPointee().hasTrait<DataLayoutTypeInterface::Trait>())
    return emitError() << "missing data layout for pointee type";

  if (getSkipTailPadding() &&
      !mlir::isa<cir::RecordType>(getType().getPointee()))
    return emitError()
           << "skip_tail_padding is only valid for record pointee types";

  return mlir::success();
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
                            if (mlir::isa<mlir::UnitAttr>(*attrIt))
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

  p.printOptionalAttrDict(
      getOperation()->getDiscardableAttrDictionary().getValue());

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
          operandAttrs.push_back(mlir::DictionaryAttr::get(ctxt));
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

  if (parser.parseOptionalAttrDict(result.attributes).failed())
    return mlir::failure();

  if (parser.parseOptionalArrow().succeeded() &&
      parser.parseType(resType).failed())
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

void InlineAsmOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  // If we have any side effects (that is, we're volatile asm), add a read and
  // write memory effect. We do this the same as the llvm dialect InlineAsmOp.
  if (getSideEffects()) {
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
  }
}

//===----------------------------------------------------------------------===//
// AtomicFetchOp
//===----------------------------------------------------------------------===//

LogicalResult cir::AtomicFetchOp::verify() {
  if (getBinop() != cir::AtomicFetchKind::Add &&
      getBinop() != cir::AtomicFetchKind::Sub &&
      getBinop() != cir::AtomicFetchKind::Max &&
      getBinop() != cir::AtomicFetchKind::Min &&
      getBinop() != cir::AtomicFetchKind::Maximum &&
      getBinop() != cir::AtomicFetchKind::Minimum &&
      getBinop() != cir::AtomicFetchKind::MaximumNum &&
      getBinop() != cir::AtomicFetchKind::MinimumNum &&
      !mlir::isa<cir::IntType>(getVal().getType()))
    return emitError("only atomic add, sub, max, min, maximum, minimum, "
                     "maximum_num, and minimum_num operation could operate on "
                     "floating-point values");

  if ((getBinop() == cir::AtomicFetchKind::Maximum ||
       getBinop() == cir::AtomicFetchKind::Minimum ||
       getBinop() == cir::AtomicFetchKind::MaximumNum ||
       getBinop() == cir::AtomicFetchKind::MinimumNum) &&
      !mlir::isa<cir::FPTypeInterface>(getVal().getType()))
    return emitError("atomic maximum, minimum, maximum_num, and minimum_num "
                     "operation could only operate on floating-point values");

  return success();
}

//===----------------------------------------------------------------------===//
// LifetimeStartOp & LifetimeEndOp
//===----------------------------------------------------------------------===//

LogicalResult cir::LifetimeStartOp::verify() {
  return verifyProducedBy<cir::AllocaOp>(*this, getPtr(), "ptr");
}

LogicalResult cir::LifetimeEndOp::verify() {
  return verifyProducedBy<cir::AllocaOp>(*this, getPtr(), "ptr");
}

//===----------------------------------------------------------------------===//
// MemChrOp
//===----------------------------------------------------------------------===//

/// Reads a fundamental integer width from a signless i32 attribute.
static std::optional<unsigned> getRecordedIntegerWidth(mlir::Attribute attr) {
  auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
  if (!intAttr || !intAttr.getType().isSignlessInteger(32))
    return std::nullopt;
  int64_t width = intAttr.getInt();
  if (width < 0 ||
      !cir::isValidFundamentalIntWidth(static_cast<unsigned>(width)))
    return std::nullopt;
  return static_cast<unsigned>(width);
}

LogicalResult cir::MemChrOp::verify() {
  auto moduleOp = (*this)->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return emitOpError("expects an enclosing module");

  // libc memchr uses pointers in the target's default address space.
  if (mlir::cast<cir::PointerType>(getSrc().getType()).getAddrSpace())
    return emitOpError("src must be in the default address space");

  auto checkWidth = [&](cir::IntType type, llvm::StringRef operandName,
                        llvm::StringRef attrName) -> LogicalResult {
    mlir::Attribute attr = moduleOp->getAttr(attrName);
    if (!attr)
      return emitOpError("expects the module to record ") << attrName;
    std::optional<unsigned> width = getRecordedIntegerWidth(attr);
    if (!width)
      return emitOpError("requires ")
             << attrName
             << " to be a signless i32 holding a fundamental integer width";
    if (type.getWidth() != *width)
      return emitOpError() << operandName << " must have the width recorded in "
                           << attrName;
    return success();
  };

  if (failed(checkWidth(getPattern().getType(), "pattern",
                        cir::CIRDialect::getIntTypeWidthAttrName())))
    return failure();
  return checkWidth(getLen().getType(), "len",
                    cir::CIRDialect::getSizeTypeWidthAttrName());
}

//===----------------------------------------------------------------------===//
// Standard library op signature matching
//===----------------------------------------------------------------------===//

bool cir::StdFindOp::signatureMatches(mlir::TypeRange operands,
                                      mlir::TypeRange results) {
  if (operands.size() != getNumArgs() || results.size() != 1)
    return false;
  mlir::Type iterTy = operands[0];
  return iterTy == operands[1] && iterTy == operands[2] && iterTy == results[0];
}

//===----------------------------------------------------------------------===//
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsBuiltins.cpp.inc"
