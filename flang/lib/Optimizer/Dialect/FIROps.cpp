//===-- FIROps.cpp --------------------------------------------------------===//
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

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace fir;

/// Return true if a sequence type is of some incomplete size or a record type
/// is malformed or contains an incomplete sequence type. An incomplete sequence
/// type is one with more unknown extents in the type than have been provided
/// via `dynamicExtents`. Sequence types with an unknown rank are incomplete by
/// definition.
static bool verifyInType(mlir::Type inType,
                         llvm::SmallVectorImpl<llvm::StringRef> &visited,
                         unsigned dynamicExtents = 0) {
  if (auto st = inType.dyn_cast<fir::SequenceType>()) {
    auto shape = st.getShape();
    if (shape.size() == 0)
      return true;
    for (std::size_t i = 0, end{shape.size()}; i < end; ++i) {
      if (shape[i] != fir::SequenceType::getUnknownExtent())
        continue;
      if (dynamicExtents-- == 0)
        return true;
    }
  } else if (auto rt = inType.dyn_cast<fir::RecordType>()) {
    // don't recurse if we're already visiting this one
    if (llvm::is_contained(visited, rt.getName()))
      return false;
    // keep track of record types currently being visited
    visited.push_back(rt.getName());
    for (auto &field : rt.getTypeList())
      if (verifyInType(field.second, visited))
        return true;
    visited.pop_back();
  } else if (auto rt = inType.dyn_cast<fir::PointerType>()) {
    return verifyInType(rt.getEleTy(), visited);
  }
  return false;
}

static bool verifyRecordLenParams(mlir::Type inType, unsigned numLenParams) {
  if (numLenParams > 0) {
    if (auto rt = inType.dyn_cast<fir::RecordType>())
      return numLenParams != rt.getNumLenParams();
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

mlir::Type fir::AllocaOp::getAllocatedType() {
  return getType().cast<ReferenceType>().getEleTy();
}

/// Create a legal memory reference as return type
mlir::Type fir::AllocaOp::wrapResultType(mlir::Type intype) {
  // FIR semantics: memory references to memory references are disallowed
  if (intype.isa<ReferenceType>())
    return {};
  return ReferenceType::get(intype);
}

mlir::Type fir::AllocaOp::getRefTy(mlir::Type ty) {
  return ReferenceType::get(ty);
}

//===----------------------------------------------------------------------===//
// AllocMemOp
//===----------------------------------------------------------------------===//

mlir::Type fir::AllocMemOp::getAllocatedType() {
  return getType().cast<HeapType>().getEleTy();
}

mlir::Type fir::AllocMemOp::getRefTy(mlir::Type ty) {
  return HeapType::get(ty);
}

/// Create a legal heap reference as return type
mlir::Type fir::AllocMemOp::wrapResultType(mlir::Type intype) {
  // Fortran semantics: C852 an entity cannot be both ALLOCATABLE and POINTER
  // 8.5.3 note 1 prohibits ALLOCATABLE procedures as well
  // FIR semantics: one may not allocate a memory reference value
  if (intype.isa<ReferenceType>() || intype.isa<HeapType>() ||
      intype.isa<PointerType>() || intype.isa<FunctionType>())
    return {};
  return HeapType::get(intype);
}

//===----------------------------------------------------------------------===//
// ArrayCoorOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::ArrayCoorOp op) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(op.memref().getType());
  auto arrTy = eleTy.dyn_cast<fir::SequenceType>();
  if (!arrTy)
    return op.emitOpError("must be a reference to an array");
  auto arrDim = arrTy.getDimension();

  if (auto shapeOp = op.shape()) {
    auto shapeTy = shapeOp.getType();
    unsigned shapeTyRank = 0;
    if (auto s = shapeTy.dyn_cast<fir::ShapeType>()) {
      shapeTyRank = s.getRank();
    } else if (auto ss = shapeTy.dyn_cast<fir::ShapeShiftType>()) {
      shapeTyRank = ss.getRank();
    } else {
      auto s = shapeTy.cast<fir::ShiftType>();
      shapeTyRank = s.getRank();
      if (!op.memref().getType().isa<fir::BoxType>())
        return op.emitOpError("shift can only be provided with fir.box memref");
    }
    if (arrDim && arrDim != shapeTyRank)
      return op.emitOpError("rank of dimension mismatched");
    if (shapeTyRank != op.indices().size())
      return op.emitOpError("number of indices do not match dim rank");
  }

  if (auto sliceOp = op.slice())
    if (auto sliceTy = sliceOp.getType().dyn_cast<fir::SliceType>())
      if (sliceTy.getRank() != arrDim)
        return op.emitOpError("rank of dimension in slice mismatched");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ArrayLoadOp
//===----------------------------------------------------------------------===//

std::vector<mlir::Value> fir::ArrayLoadOp::getExtents() {
  if (auto sh = shape())
    if (auto *op = sh.getDefiningOp()) {
      if (auto shOp = dyn_cast<fir::ShapeOp>(op))
        return shOp.getExtents();
      return cast<fir::ShapeShiftOp>(op).getExtents();
    }
  return {};
}

static mlir::LogicalResult verify(fir::ArrayLoadOp op) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(op.memref().getType());
  auto arrTy = eleTy.dyn_cast<fir::SequenceType>();
  if (!arrTy)
    return op.emitOpError("must be a reference to an array");
  auto arrDim = arrTy.getDimension();

  if (auto shapeOp = op.shape()) {
    auto shapeTy = shapeOp.getType();
    unsigned shapeTyRank = 0;
    if (auto s = shapeTy.dyn_cast<fir::ShapeType>()) {
      shapeTyRank = s.getRank();
    } else if (auto ss = shapeTy.dyn_cast<fir::ShapeShiftType>()) {
      shapeTyRank = ss.getRank();
    } else {
      auto s = shapeTy.cast<fir::ShiftType>();
      shapeTyRank = s.getRank();
      if (!op.memref().getType().isa<fir::BoxType>())
        return op.emitOpError("shift can only be provided with fir.box memref");
    }
    if (arrDim && arrDim != shapeTyRank)
      return op.emitOpError("rank of dimension mismatched");
  }

  if (auto sliceOp = op.slice())
    if (auto sliceTy = sliceOp.getType().dyn_cast<fir::SliceType>())
      if (sliceTy.getRank() != arrDim)
        return op.emitOpError("rank of dimension in slice mismatched");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BoxAddrOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult fir::BoxAddrOp::fold(llvm::ArrayRef<mlir::Attribute> opnds) {
  if (auto v = val().getDefiningOp()) {
    if (auto box = dyn_cast<fir::EmboxOp>(v))
      return box.memref();
    if (auto box = dyn_cast<fir::EmboxCharOp>(v))
      return box.memref();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BoxCharLenOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult
fir::BoxCharLenOp::fold(llvm::ArrayRef<mlir::Attribute> opnds) {
  if (auto v = val().getDefiningOp()) {
    if (auto box = dyn_cast<fir::EmboxCharOp>(v))
      return box.len();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// BoxDimsOp
//===----------------------------------------------------------------------===//

/// Get the result types packed in a tuple tuple
mlir::Type fir::BoxDimsOp::getTupleType() {
  // note: triple, but 4 is nearest power of 2
  llvm::SmallVector<mlir::Type, 4> triple{
      getResult(0).getType(), getResult(1).getType(), getResult(2).getType()};
  return mlir::TupleType::get(getContext(), triple);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

mlir::FunctionType fir::CallOp::getFunctionType() {
  return mlir::FunctionType::get(getContext(), getOperandTypes(),
                                 getResultTypes());
}

static void printCallOp(mlir::OpAsmPrinter &p, fir::CallOp &op) {
  auto callee = op.callee();
  bool isDirect = callee.hasValue();
  p << op.getOperationName() << ' ';
  if (isDirect)
    p << callee.getValue();
  else
    p << op.getOperand(0);
  p << '(' << op->getOperands().drop_front(isDirect ? 0 : 1) << ')';
  p.printOptionalAttrDict(op->getAttrs(), {fir::CallOp::calleeAttrName()});
  auto resultTypes{op.getResultTypes()};
  llvm::SmallVector<Type, 8> argTypes(
      llvm::drop_begin(op.getOperandTypes(), isDirect ? 0 : 1));
  p << " : " << FunctionType::get(op.getContext(), argTypes, resultTypes);
}

static mlir::ParseResult parseCallOp(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 8> operands;
  if (parser.parseOperandList(operands))
    return mlir::failure();

  mlir::NamedAttrList attrs;
  mlir::SymbolRefAttr funcAttr;
  bool isDirect = operands.empty();
  if (isDirect)
    if (parser.parseAttribute(funcAttr, fir::CallOp::calleeAttrName(), attrs))
      return mlir::failure();

  Type type;
  if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColon() ||
      parser.parseType(type))
    return mlir::failure();

  auto funcType = type.dyn_cast<mlir::FunctionType>();
  if (!funcType)
    return parser.emitError(parser.getNameLoc(), "expected function type");
  if (isDirect) {
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return mlir::failure();
  } else {
    auto funcArgs =
        llvm::ArrayRef<mlir::OpAsmParser::OperandType>(operands).drop_front();
    if (parser.resolveOperand(operands[0], funcType, result.operands) ||
        parser.resolveOperands(funcArgs, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return mlir::failure();
  }
  result.addTypes(funcType.getResults());
  result.attributes = attrs;
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CmpfOp
//===----------------------------------------------------------------------===//

// Note: getCmpFPredicateNames() is inline static in StandardOps/IR/Ops.cpp
mlir::CmpFPredicate fir::CmpfOp::getPredicateByName(llvm::StringRef name) {
  auto pred = mlir::symbolizeCmpFPredicate(name);
  assert(pred.hasValue() && "invalid predicate name");
  return pred.getValue();
}

void fir::buildCmpFOp(OpBuilder &builder, OperationState &result,
                      CmpFPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(builder.getI1Type());
  result.addAttribute(
      CmpfOp::getPredicateAttrName(),
      builder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

template <typename OPTY>
static void printCmpOp(OpAsmPrinter &p, OPTY op) {
  p << op.getOperationName() << ' ';
  auto predSym = mlir::symbolizeCmpFPredicate(
      op->template getAttrOfType<mlir::IntegerAttr>(
            OPTY::getPredicateAttrName())
          .getInt());
  assert(predSym.hasValue() && "invalid symbol value for predicate");
  p << '"' << mlir::stringifyCmpFPredicate(predSym.getValue()) << '"' << ", ";
  p.printOperand(op.lhs());
  p << ", ";
  p.printOperand(op.rhs());
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{OPTY::getPredicateAttrName()});
  p << " : " << op.lhs().getType();
}

static void printCmpfOp(OpAsmPrinter &p, CmpfOp op) { printCmpOp(p, op); }

template <typename OPTY>
static mlir::ParseResult parseCmpOp(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> ops;
  mlir::NamedAttrList attrs;
  mlir::Attribute predicateNameAttr;
  mlir::Type type;
  if (parser.parseAttribute(predicateNameAttr, OPTY::getPredicateAttrName(),
                            attrs) ||
      parser.parseComma() || parser.parseOperandList(ops, 2) ||
      parser.parseOptionalAttrDict(attrs) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands))
    return failure();

  if (!predicateNameAttr.isa<mlir::StringAttr>())
    return parser.emitError(parser.getNameLoc(),
                            "expected string comparison predicate attribute");

  // Rewrite string attribute to an enum value.
  llvm::StringRef predicateName =
      predicateNameAttr.cast<mlir::StringAttr>().getValue();
  auto predicate = fir::CmpfOp::getPredicateByName(predicateName);
  auto builder = parser.getBuilder();
  mlir::Type i1Type = builder.getI1Type();
  attrs.set(OPTY::getPredicateAttrName(),
            builder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
  result.attributes = attrs;
  result.addTypes({i1Type});
  return success();
}

mlir::ParseResult fir::parseCmpfOp(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  return parseCmpOp<fir::CmpfOp>(parser, result);
}

//===----------------------------------------------------------------------===//
// CmpcOp
//===----------------------------------------------------------------------===//

void fir::buildCmpCOp(OpBuilder &builder, OperationState &result,
                      CmpFPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(builder.getI1Type());
  result.addAttribute(
      fir::CmpcOp::getPredicateAttrName(),
      builder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

static void printCmpcOp(OpAsmPrinter &p, fir::CmpcOp op) { printCmpOp(p, op); }

mlir::ParseResult fir::parseCmpcOp(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  return parseCmpOp<fir::CmpcOp>(parser, result);
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void fir::ConvertOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
}

mlir::OpFoldResult fir::ConvertOp::fold(llvm::ArrayRef<mlir::Attribute> opnds) {
  if (value().getType() == getType())
    return value();
  if (matchPattern(value(), m_Op<fir::ConvertOp>())) {
    auto inner = cast<fir::ConvertOp>(value().getDefiningOp());
    // (convert (convert 'a : logical -> i1) : i1 -> logical) ==> forward 'a
    if (auto toTy = getType().dyn_cast<fir::LogicalType>())
      if (auto fromTy = inner.value().getType().dyn_cast<fir::LogicalType>())
        if (inner.getType().isa<mlir::IntegerType>() && (toTy == fromTy))
          return inner.value();
    // (convert (convert 'a : i1 -> logical) : logical -> i1) ==> forward 'a
    if (auto toTy = getType().dyn_cast<mlir::IntegerType>())
      if (auto fromTy = inner.value().getType().dyn_cast<mlir::IntegerType>())
        if (inner.getType().isa<fir::LogicalType>() && (toTy == fromTy) &&
            (fromTy.getWidth() == 1))
          return inner.value();
  }
  return {};
}

bool fir::ConvertOp::isIntegerCompatible(mlir::Type ty) {
  return ty.isa<mlir::IntegerType>() || ty.isa<mlir::IndexType>() ||
         ty.isa<fir::IntegerType>() || ty.isa<fir::LogicalType>();
}

bool fir::ConvertOp::isFloatCompatible(mlir::Type ty) {
  return ty.isa<mlir::FloatType>() || ty.isa<fir::RealType>();
}

bool fir::ConvertOp::isPointerCompatible(mlir::Type ty) {
  return ty.isa<fir::ReferenceType>() || ty.isa<fir::PointerType>() ||
         ty.isa<fir::HeapType>() || ty.isa<mlir::MemRefType>() ||
         ty.isa<mlir::FunctionType>() || ty.isa<fir::TypeDescType>();
}

//===----------------------------------------------------------------------===//
// CoordinateOp
//===----------------------------------------------------------------------===//

static void print(mlir::OpAsmPrinter &p, fir::CoordinateOp op) {
  p << op.getOperationName() << ' ' << op.ref() << ", " << op.coor();
  p.printOptionalAttrDict(op->getAttrs(), /*elideAttrs=*/{"baseType"});
  p << " : ";
  p.printFunctionalType(op.getOperandTypes(), op->getResultTypes());
}

static mlir::ParseResult parseCoordinateCustom(mlir::OpAsmParser &parser,
                                               mlir::OperationState &result) {
  mlir::OpAsmParser::OperandType memref;
  if (parser.parseOperand(memref) || parser.parseComma())
    return mlir::failure();
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 8> coorOperands;
  if (parser.parseOperandList(coorOperands))
    return mlir::failure();
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 16> allOperands;
  allOperands.push_back(memref);
  allOperands.append(coorOperands.begin(), coorOperands.end());
  mlir::FunctionType funcTy;
  auto loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(funcTy) ||
      parser.resolveOperands(allOperands, funcTy.getInputs(), loc,
                             result.operands))
    return failure();
  parser.addTypesToList(funcTy.getResults(), result.types);
  result.addAttribute("baseType", mlir::TypeAttr::get(funcTy.getInput(0)));
  return mlir::success();
}

static mlir::LogicalResult verify(fir::CoordinateOp op) {
  auto refTy = op.ref().getType();
  if (fir::isa_ref_type(refTy)) {
    auto eleTy = fir::dyn_cast_ptrEleTy(refTy);
    if (auto arrTy = eleTy.dyn_cast<fir::SequenceType>()) {
      if (arrTy.hasUnknownShape())
        return op.emitOpError("cannot find coordinate in unknown shape");
      if (arrTy.getConstantRows() < arrTy.getDimension() - 1)
        return op.emitOpError("cannot find coordinate with unknown extents");
    }
    if (!(fir::isa_aggregate(eleTy) || fir::isa_complex(eleTy) ||
          fir::isa_char_string(eleTy)))
      return op.emitOpError("cannot apply coordinate_of to this type");
  }
  // Recovering a LEN type parameter only makes sense from a boxed value. For a
  // bare reference, the LEN type parameters must be passed as additional
  // arguments to `op`.
  for (auto co : op.coor())
    if (dyn_cast_or_null<fir::LenParamIndexOp>(co.getDefiningOp())) {
      if (op.getNumOperands() != 2)
        return op.emitOpError("len_param_index must be last argument");
      if (!op.ref().getType().isa<BoxType>())
        return op.emitOpError("len_param_index must be used on box type");
    }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

mlir::FunctionType fir::DispatchOp::getFunctionType() {
  return mlir::FunctionType::get(getContext(), getOperandTypes(),
                                 getResultTypes());
}

//===----------------------------------------------------------------------===//
// DispatchTableOp
//===----------------------------------------------------------------------===//

void fir::DispatchTableOp::appendTableEntry(mlir::Operation *op) {
  assert(mlir::isa<fir::DTEntryOp>(*op) && "operation must be a DTEntryOp");
  auto &block = getBlock();
  block.getOperations().insert(block.end(), op);
}

//===----------------------------------------------------------------------===//
// EmboxOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::EmboxOp op) {
  auto eleTy = fir::dyn_cast_ptrEleTy(op.memref().getType());
  bool isArray = false;
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>()) {
    eleTy = seqTy.getEleTy();
    isArray = true;
  }
  if (op.hasLenParams()) {
    auto lenPs = op.numLenParams();
    if (auto rt = eleTy.dyn_cast<fir::RecordType>()) {
      if (lenPs != rt.getNumLenParams())
        return op.emitOpError("number of LEN params does not correspond"
                              " to the !fir.type type");
    } else if (auto strTy = eleTy.dyn_cast<fir::CharacterType>()) {
      if (strTy.getLen() != fir::CharacterType::unknownLen())
        return op.emitOpError("CHARACTER already has static LEN");
    } else {
      return op.emitOpError("LEN parameters require CHARACTER or derived type");
    }
    for (auto lp : op.lenParams())
      if (!fir::isa_integer(lp.getType()))
        return op.emitOpError("LEN parameters must be integral type");
  }
  if (op.getShape() && !isArray)
    return op.emitOpError("shape must not be provided for a scalar");
  if (op.getSlice() && !isArray)
    return op.emitOpError("slice must not be provided for a scalar");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GenTypeDescOp
//===----------------------------------------------------------------------===//

void fir::GenTypeDescOp::build(OpBuilder &, OperationState &result,
                               mlir::TypeAttr inty) {
  result.addAttribute("in_type", inty);
  result.addTypes(TypeDescType::get(inty.getValue()));
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

static ParseResult parseGlobalOp(OpAsmParser &parser, OperationState &result) {
  // Parse the optional linkage
  llvm::StringRef linkage;
  auto &builder = parser.getBuilder();
  if (mlir::succeeded(parser.parseOptionalKeyword(&linkage))) {
    if (fir::GlobalOp::verifyValidLinkage(linkage))
      return mlir::failure();
    mlir::StringAttr linkAttr = builder.getStringAttr(linkage);
    result.addAttribute(fir::GlobalOp::linkageAttrName(), linkAttr);
  }

  // Parse the name as a symbol reference attribute.
  mlir::SymbolRefAttr nameAttr;
  if (parser.parseAttribute(nameAttr, fir::GlobalOp::symbolAttrName(),
                            result.attributes))
    return mlir::failure();
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(nameAttr.getRootReference()));

  bool simpleInitializer = false;
  if (mlir::succeeded(parser.parseOptionalLParen())) {
    Attribute attr;
    if (parser.parseAttribute(attr, fir::GlobalOp::initValAttrName(),
                              result.attributes) ||
        parser.parseRParen())
      return mlir::failure();
    simpleInitializer = true;
  }

  if (succeeded(parser.parseOptionalKeyword("constant"))) {
    // if "constant" keyword then mark this as a constant, not a variable
    result.addAttribute(fir::GlobalOp::constantAttrName(),
                        builder.getUnitAttr());
  }

  mlir::Type globalType;
  if (parser.parseColonType(globalType))
    return mlir::failure();

  result.addAttribute(fir::GlobalOp::typeAttrName(),
                      mlir::TypeAttr::get(globalType));

  if (simpleInitializer) {
    result.addRegion();
  } else {
    // Parse the optional initializer body.
    auto parseResult = parser.parseOptionalRegion(
        *result.addRegion(), /*arguments=*/llvm::None, /*argTypes=*/llvm::None);
    if (parseResult.hasValue() && mlir::failed(*parseResult))
      return mlir::failure();
  }

  return mlir::success();
}

void fir::GlobalOp::appendInitialValue(mlir::Operation *op) {
  getBlock().getOperations().push_back(op);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, bool isConstant, Type type,
                          Attribute initialVal, StringAttr linkage,
                          ArrayRef<NamedAttribute> attrs) {
  result.addRegion();
  result.addAttribute(typeAttrName(), mlir::TypeAttr::get(type));
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(symbolAttrName(), builder.getSymbolRefAttr(name));
  if (isConstant)
    result.addAttribute(constantAttrName(), builder.getUnitAttr());
  if (initialVal)
    result.addAttribute(initValAttrName(), initialVal);
  if (linkage)
    result.addAttribute(linkageAttrName(), linkage);
  result.attributes.append(attrs.begin(), attrs.end());
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, Type type, Attribute initialVal,
                          StringAttr linkage, ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, type, {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, bool isConstant, Type type,
                          StringAttr linkage, ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isConstant, type, {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, Type type, StringAttr linkage,
                          ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, type, {}, linkage, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, bool isConstant, Type type,
                          ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isConstant, type, StringAttr{}, attrs);
}

void fir::GlobalOp::build(mlir::OpBuilder &builder, OperationState &result,
                          StringRef name, Type type,
                          ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, /*isConstant=*/false, type, attrs);
}

mlir::ParseResult fir::GlobalOp::verifyValidLinkage(StringRef linkage) {
  // Supporting only a subset of the LLVM linkage types for now
  static const char *validNames[] = {"common", "internal", "linkonce", "weak"};
  return mlir::success(llvm::is_contained(validNames, linkage));
}

//===----------------------------------------------------------------------===//
// InsertValueOp
//===----------------------------------------------------------------------===//

static bool checkIsIntegerConstant(mlir::Value v, int64_t conVal) {
  if (auto c = dyn_cast_or_null<mlir::ConstantOp>(v.getDefiningOp())) {
    auto attr = c.getValue();
    if (auto iattr = attr.dyn_cast<mlir::IntegerAttr>())
      return iattr.getInt() == conVal;
  }
  return false;
}
static bool isZero(mlir::Value v) { return checkIsIntegerConstant(v, 0); }
static bool isOne(mlir::Value v) { return checkIsIntegerConstant(v, 1); }

// Undo some complex patterns created in the front-end and turn them back into
// complex ops.
template <typename FltOp, typename CpxOp>
struct UndoComplexPattern : public mlir::RewritePattern {
  UndoComplexPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern("fir.insert_value", 2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto insval = dyn_cast_or_null<fir::InsertValueOp>(op);
    if (!insval || !insval.getType().isa<fir::ComplexType>())
      return mlir::failure();
    auto insval2 =
        dyn_cast_or_null<fir::InsertValueOp>(insval.adt().getDefiningOp());
    if (!insval2 || !isa<fir::UndefOp>(insval2.adt().getDefiningOp()))
      return mlir::failure();
    auto binf = dyn_cast_or_null<FltOp>(insval.val().getDefiningOp());
    auto binf2 = dyn_cast_or_null<FltOp>(insval2.val().getDefiningOp());
    if (!binf || !binf2 || insval.coor().size() != 1 ||
        !isOne(insval.coor()[0]) || insval2.coor().size() != 1 ||
        !isZero(insval2.coor()[0]))
      return mlir::failure();
    auto eai =
        dyn_cast_or_null<fir::ExtractValueOp>(binf.lhs().getDefiningOp());
    auto ebi =
        dyn_cast_or_null<fir::ExtractValueOp>(binf.rhs().getDefiningOp());
    auto ear =
        dyn_cast_or_null<fir::ExtractValueOp>(binf2.lhs().getDefiningOp());
    auto ebr =
        dyn_cast_or_null<fir::ExtractValueOp>(binf2.rhs().getDefiningOp());
    if (!eai || !ebi || !ear || !ebr || ear.adt() != eai.adt() ||
        ebr.adt() != ebi.adt() || eai.coor().size() != 1 ||
        !isOne(eai.coor()[0]) || ebi.coor().size() != 1 ||
        !isOne(ebi.coor()[0]) || ear.coor().size() != 1 ||
        !isZero(ear.coor()[0]) || ebr.coor().size() != 1 ||
        !isZero(ebr.coor()[0]))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<CpxOp>(op, ear.adt(), ebr.adt());
    return mlir::success();
  }
};

void fir::InsertValueOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &results, mlir::MLIRContext *context) {
  results.insert<UndoComplexPattern<mlir::AddFOp, fir::AddcOp>,
                 UndoComplexPattern<mlir::SubFOp, fir::SubcOp>>(context);
}

//===----------------------------------------------------------------------===//
// IterWhileOp
//===----------------------------------------------------------------------===//

void fir::IterWhileOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value lb,
                             mlir::Value ub, mlir::Value step,
                             mlir::Value iterate, bool finalCountValue,
                             mlir::ValueRange iterArgs,
                             llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands({lb, ub, step, iterate});
  if (finalCountValue) {
    result.addTypes(builder.getIndexType());
    result.addAttribute(finalValueAttrName(), builder.getUnitAttr());
  }
  result.addTypes(iterate.getType());
  result.addOperands(iterArgs);
  for (auto v : iterArgs)
    result.addTypes(v.getType());
  mlir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block{});
  bodyRegion->front().addArgument(builder.getIndexType());
  bodyRegion->front().addArgument(iterate.getType());
  bodyRegion->front().addArguments(iterArgs.getTypes());
  result.addAttributes(attributes);
}

static mlir::ParseResult parseIterWhileOp(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::OperandType inductionVariable, lb, ub, step;
  if (parser.parseLParen() || parser.parseRegionArgument(inductionVariable) ||
      parser.parseEqual())
    return mlir::failure();

  // Parse loop bounds.
  auto indexType = builder.getIndexType();
  auto i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.parseRParen() ||
      parser.resolveOperand(step, indexType, result.operands))
    return mlir::failure();

  mlir::OpAsmParser::OperandType iterateVar, iterateInput;
  if (parser.parseKeyword("and") || parser.parseLParen() ||
      parser.parseRegionArgument(iterateVar) || parser.parseEqual() ||
      parser.parseOperand(iterateInput) || parser.parseRParen() ||
      parser.resolveOperand(iterateInput, i1Type, result.operands))
    return mlir::failure();

  // Parse the initial iteration arguments.
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> regionArgs;
  auto prependCount = false;

  // Induction variable.
  regionArgs.push_back(inductionVariable);
  regionArgs.push_back(iterateVar);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
    llvm::SmallVector<mlir::Type, 4> regionTypes;
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(regionTypes))
      return failure();
    if (regionTypes.size() == operands.size() + 2)
      prependCount = true;
    llvm::ArrayRef<mlir::Type> resTypes = regionTypes;
    resTypes = prependCount ? resTypes.drop_front(2) : resTypes;
    // Resolve input operands.
    for (auto operand_type : llvm::zip(operands, resTypes))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return failure();
    if (prependCount) {
      result.addTypes(regionTypes);
    } else {
      result.addTypes(i1Type);
      result.addTypes(resTypes);
    }
  } else if (succeeded(parser.parseOptionalArrow())) {
    llvm::SmallVector<mlir::Type, 4> typeList;
    if (parser.parseLParen() || parser.parseTypeList(typeList) ||
        parser.parseRParen())
      return failure();
    // Type list must be "(index, i1)".
    if (typeList.size() != 2 || !typeList[0].isa<mlir::IndexType>() ||
        !typeList[1].isSignlessInteger(1))
      return failure();
    result.addTypes(typeList);
    prependCount = true;
  } else {
    result.addTypes(i1Type);
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return mlir::failure();

  llvm::SmallVector<mlir::Type, 4> argTypes;
  // Induction variable (hidden)
  if (prependCount)
    result.addAttribute(IterWhileOp::finalValueAttrName(),
                        builder.getUnitAttr());
  else
    argTypes.push_back(indexType);
  // Loop carried variables (including iterate)
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  auto *body = result.addRegion();
  if (regionArgs.size() != argTypes.size())
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  if (parser.parseRegion(*body, regionArgs, argTypes))
    return failure();

  fir::IterWhileOp::ensureTerminator(*body, builder, result.location);

  return mlir::success();
}

static mlir::LogicalResult verify(fir::IterWhileOp op) {
  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = op.getBody();
  if (!body->getArgument(1).getType().isInteger(1))
    return op.emitOpError(
        "expected body second argument to be an index argument for "
        "the induction variable");
  if (!body->getArgument(0).getType().isIndex())
    return op.emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = op.getNumResults();
  if (op.finalValue()) {
    // Result type must be "(index, i1, ...)".
    if (!op.getResult(0).getType().isa<mlir::IndexType>())
      return op.emitOpError("result #0 expected to be index");
    if (!op.getResult(1).getType().isSignlessInteger(1))
      return op.emitOpError("result #1 expected to be i1");
    opNumResults--;
  } else {
    // iterate_while always returns the early exit induction value.
    // Result type must be "(i1, ...)"
    if (!op.getResult(0).getType().isSignlessInteger(1))
      return op.emitOpError("result #0 expected to be i1");
  }
  if (opNumResults == 0)
    return mlir::failure();
  if (op.getNumIterOperands() != opNumResults)
    return op.emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (op.getNumRegionIterArgs() != opNumResults)
    return op.emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = op.getIterOperands();
  auto iterArgs = op.getRegionIterArgs();
  auto opResults =
      op.finalValue() ? op.getResults().drop_front() : op.getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter region arg and defined value";

    i++;
  }
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::IterWhileOp op) {
  p << fir::IterWhileOp::getOperationName() << " (" << op.getInductionVar()
    << " = " << op.lowerBound() << " to " << op.upperBound() << " step "
    << op.step() << ") and (";
  assert(op.hasIterOperands());
  auto regionArgs = op.getRegionIterArgs();
  auto operands = op.getIterOperands();
  p << regionArgs.front() << " = " << *operands.begin() << ")";
  if (regionArgs.size() > 1) {
    p << " iter_args(";
    llvm::interleaveComma(
        llvm::zip(regionArgs.drop_front(), operands.drop_front()), p,
        [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
    p << ") -> (";
    llvm::interleaveComma(
        llvm::drop_begin(op.getResultTypes(), op.finalValue() ? 0 : 1), p);
    p << ")";
  } else if (op.finalValue()) {
    p << " -> (" << op.getResultTypes() << ')';
  }
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     {IterWhileOp::finalValueAttrName()});
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

mlir::Region &fir::IterWhileOp::getLoopBody() { return region(); }

bool fir::IterWhileOp::isDefinedOutsideOfLoop(mlir::Value value) {
  return !region().isAncestor(value.getParentRegion());
}

mlir::LogicalResult
fir::IterWhileOp::moveOutOfLoop(llvm::ArrayRef<mlir::Operation *> ops) {
  for (auto op : ops)
    op->moveBefore(*this);
  return success();
}

mlir::BlockArgument fir::IterWhileOp::iterArgToBlockArg(mlir::Value iterArg) {
  for (auto i : llvm::enumerate(initArgs()))
    if (iterArg == i.value())
      return region().front().getArgument(i.index() + 1);
  return {};
}

void fir::IterWhileOp::resultToSourceOps(
    llvm::SmallVectorImpl<mlir::Value> &results, unsigned resultNum) {
  auto oper = finalValue() ? resultNum + 1 : resultNum;
  auto *term = region().front().getTerminator();
  if (oper < term->getNumOperands())
    results.push_back(term->getOperand(oper));
}

mlir::Value fir::IterWhileOp::blockArgToSourceOp(unsigned blockArgNum) {
  if (blockArgNum > 0 && blockArgNum <= initArgs().size())
    return initArgs()[blockArgNum - 1];
  return {};
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

/// Get the element type of a reference like type; otherwise null
static mlir::Type elementTypeOf(mlir::Type ref) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(ref)
      .Case<ReferenceType, PointerType, HeapType>(
          [](auto type) { return type.getEleTy(); })
      .Default([](mlir::Type) { return mlir::Type{}; });
}

mlir::ParseResult fir::LoadOp::getElementOf(mlir::Type &ele, mlir::Type ref) {
  if ((ele = elementTypeOf(ref)))
    return mlir::success();
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// DoLoopOp
//===----------------------------------------------------------------------===//

void fir::DoLoopOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result, mlir::Value lb,
                          mlir::Value ub, mlir::Value step, bool unordered,
                          bool finalCountValue, mlir::ValueRange iterArgs,
                          llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands({lb, ub, step});
  result.addOperands(iterArgs);
  if (finalCountValue) {
    result.addTypes(builder.getIndexType());
    result.addAttribute(finalValueAttrName(), builder.getUnitAttr());
  }
  for (auto v : iterArgs)
    result.addTypes(v.getType());
  mlir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block{});
  if (iterArgs.empty() && !finalCountValue)
    DoLoopOp::ensureTerminator(*bodyRegion, builder, result.location);
  bodyRegion->front().addArgument(builder.getIndexType());
  bodyRegion->front().addArguments(iterArgs.getTypes());
  if (unordered)
    result.addAttribute(unorderedAttrName(), builder.getUnitAttr());
  result.addAttributes(attributes);
}

static mlir::ParseResult parseDoLoopOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  mlir::OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(inductionVariable) || parser.parseEqual())
    return mlir::failure();

  // Parse loop bounds.
  auto indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return failure();

  if (mlir::succeeded(parser.parseOptionalKeyword("unordered")))
    result.addAttribute(fir::DoLoopOp::unorderedAttrName(),
                        builder.getUnitAttr());

  // Parse the optional initial iteration arguments.
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> regionArgs, operands;
  llvm::SmallVector<mlir::Type, 4> argTypes;
  auto prependCount = false;
  regionArgs.push_back(inductionVariable);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
    if (result.types.size() == operands.size() + 1)
      prependCount = true;
    // Resolve input operands.
    llvm::ArrayRef<mlir::Type> resTypes = result.types;
    for (auto operand_type :
         llvm::zip(operands, prependCount ? resTypes.drop_front() : resTypes))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return failure();
  } else if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseKeyword("index"))
      return failure();
    result.types.push_back(indexType);
    prependCount = true;
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return mlir::failure();

  // Induction variable.
  if (prependCount)
    result.addAttribute(DoLoopOp::finalValueAttrName(), builder.getUnitAttr());
  else
    argTypes.push_back(indexType);
  // Loop carried variables
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  auto *body = result.addRegion();
  if (regionArgs.size() != argTypes.size())
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  if (parser.parseRegion(*body, regionArgs, argTypes))
    return failure();

  DoLoopOp::ensureTerminator(*body, builder, result.location);

  return mlir::success();
}

fir::DoLoopOp fir::getForInductionVarOwner(mlir::Value val) {
  auto ivArg = val.dyn_cast<mlir::BlockArgument>();
  if (!ivArg)
    return {};
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingInst = ivArg.getOwner()->getParentOp();
  return dyn_cast_or_null<fir::DoLoopOp>(containingInst);
}

// Lifted from loop.loop
static mlir::LogicalResult verify(fir::DoLoopOp op) {
  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = op.getBody();
  if (!body->getArgument(0).getType().isIndex())
    return op.emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = op.getNumResults();
  if (opNumResults == 0)
    return success();

  if (op.finalValue()) {
    if (op.unordered())
      return op.emitOpError("unordered loop has no final value");
    opNumResults--;
  }
  if (op.getNumIterOperands() != opNumResults)
    return op.emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (op.getNumRegionIterArgs() != opNumResults)
    return op.emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = op.getIterOperands();
  auto iterArgs = op.getRegionIterArgs();
  auto opResults =
      op.finalValue() ? op.getResults().drop_front() : op.getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter region arg and defined value";

    i++;
  }
  return success();
}

static void print(mlir::OpAsmPrinter &p, fir::DoLoopOp op) {
  bool printBlockTerminators = false;
  p << fir::DoLoopOp::getOperationName() << ' ' << op.getInductionVar() << " = "
    << op.lowerBound() << " to " << op.upperBound() << " step " << op.step();
  if (op.unordered())
    p << " unordered";
  if (op.hasIterOperands()) {
    p << " iter_args(";
    auto regionArgs = op.getRegionIterArgs();
    auto operands = op.getIterOperands();
    llvm::interleaveComma(llvm::zip(regionArgs, operands), p, [&](auto it) {
      p << std::get<0>(it) << " = " << std::get<1>(it);
    });
    p << ") -> (" << op.getResultTypes() << ')';
    printBlockTerminators = true;
  } else if (op.finalValue()) {
    p << " -> " << op.getResultTypes();
    printBlockTerminators = true;
  }
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     {fir::DoLoopOp::unorderedAttrName(),
                                      fir::DoLoopOp::finalValueAttrName()});
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false,
                printBlockTerminators);
}

mlir::Region &fir::DoLoopOp::getLoopBody() { return region(); }

bool fir::DoLoopOp::isDefinedOutsideOfLoop(mlir::Value value) {
  return !region().isAncestor(value.getParentRegion());
}

mlir::LogicalResult
fir::DoLoopOp::moveOutOfLoop(llvm::ArrayRef<mlir::Operation *> ops) {
  for (auto op : ops)
    op->moveBefore(*this);
  return success();
}

/// Translate a value passed as an iter_arg to the corresponding block
/// argument in the body of the loop.
mlir::BlockArgument fir::DoLoopOp::iterArgToBlockArg(mlir::Value iterArg) {
  for (auto i : llvm::enumerate(initArgs()))
    if (iterArg == i.value())
      return region().front().getArgument(i.index() + 1);
  return {};
}

/// Translate the result vector (by index number) to the corresponding value
/// to the `fir.result` Op.
void fir::DoLoopOp::resultToSourceOps(
    llvm::SmallVectorImpl<mlir::Value> &results, unsigned resultNum) {
  auto oper = finalValue() ? resultNum + 1 : resultNum;
  auto *term = region().front().getTerminator();
  if (oper < term->getNumOperands())
    results.push_back(term->getOperand(oper));
}

/// Translate the block argument (by index number) to the corresponding value
/// passed as an iter_arg to the parent DoLoopOp.
mlir::Value fir::DoLoopOp::blockArgToSourceOp(unsigned blockArgNum) {
  if (blockArgNum > 0 && blockArgNum <= initArgs().size())
    return initArgs()[blockArgNum - 1];
  return {};
}

//===----------------------------------------------------------------------===//
// ReboxOp
//===----------------------------------------------------------------------===//

/// Get the scalar type related to a fir.box type.
/// Example: return f32 for !fir.box<!fir.heap<!fir.array<?x?xf32>>.
static mlir::Type getBoxScalarEleTy(mlir::Type boxTy) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(boxTy);
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
    return seqTy.getEleTy();
  return eleTy;
}

/// Get the rank from a !fir.box type
static unsigned getBoxRank(mlir::Type boxTy) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(boxTy);
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
    return seqTy.getDimension();
  return 0;
}

static mlir::LogicalResult verify(fir::ReboxOp op) {
  auto inputBoxTy = op.box().getType();
  if (fir::isa_unknown_size_box(inputBoxTy))
    return op.emitOpError("box operand must not have unknown rank or type");
  auto outBoxTy = op.getType();
  if (fir::isa_unknown_size_box(outBoxTy))
    return op.emitOpError("result type must not have unknown rank or type");
  auto inputRank = getBoxRank(inputBoxTy);
  auto inputEleTy = getBoxScalarEleTy(inputBoxTy);
  auto outRank = getBoxRank(outBoxTy);
  auto outEleTy = getBoxScalarEleTy(outBoxTy);

  if (auto slice = op.slice()) {
    // Slicing case
    if (slice.getType().cast<fir::SliceType>().getRank() != inputRank)
      return op.emitOpError("slice operand rank must match box operand rank");
    if (auto shape = op.shape()) {
      if (auto shiftTy = shape.getType().dyn_cast<fir::ShiftType>()) {
        if (shiftTy.getRank() != inputRank)
          return op.emitOpError("shape operand and input box ranks must match "
                                "when there is a slice");
      } else {
        return op.emitOpError("shape operand must absent or be a fir.shift "
                              "when there is a slice");
      }
    }
    if (auto sliceOp = slice.getDefiningOp()) {
      auto slicedRank = mlir::cast<fir::SliceOp>(sliceOp).getOutRank();
      if (slicedRank != outRank)
        return op.emitOpError("result type rank and rank after applying slice "
                              "operand must match");
    }
  } else {
    // Reshaping case
    unsigned shapeRank = inputRank;
    if (auto shape = op.shape()) {
      auto ty = shape.getType();
      if (auto shapeTy = ty.dyn_cast<fir::ShapeType>()) {
        shapeRank = shapeTy.getRank();
      } else if (auto shapeShiftTy = ty.dyn_cast<fir::ShapeShiftType>()) {
        shapeRank = shapeShiftTy.getRank();
      } else {
        auto shiftTy = ty.cast<fir::ShiftType>();
        shapeRank = shiftTy.getRank();
        if (shapeRank != inputRank)
          return op.emitOpError("shape operand and input box ranks must match "
                                "when the shape is a fir.shift");
      }
    }
    if (shapeRank != outRank)
      return op.emitOpError("result type and shape operand ranks must match");
  }

  if (inputEleTy != outEleTy)
    // TODO: check that outBoxTy is a parent type of inputBoxTy for derived
    // types.
    if (!inputEleTy.isa<fir::RecordType>())
      return op.emitOpError(
          "op input and output element types must match for intrinsic types");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ResultOp
//===----------------------------------------------------------------------===//

static mlir::LogicalResult verify(fir::ResultOp op) {
  auto *parentOp = op->getParentOp();
  auto results = parentOp->getResults();
  auto operands = op->getOperands();

  if (parentOp->getNumResults() != op.getNumOperands())
    return op.emitOpError() << "parent of result must have same arity";
  for (auto e : llvm::zip(results, operands))
    if (std::get<0>(e).getType() != std::get<1>(e).getType())
      return op.emitOpError()
             << "types mismatch between result op and its parent";
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

static constexpr llvm::StringRef getCompareOffsetAttr() {
  return "compare_operand_offsets";
}

static constexpr llvm::StringRef getTargetOffsetAttr() {
  return "target_operand_offsets";
}

template <typename A, typename... AdditionalArgs>
static A getSubOperands(unsigned pos, A allArgs,
                        mlir::DenseIntElementsAttr ranges,
                        AdditionalArgs &&... additionalArgs) {
  unsigned start = 0;
  for (unsigned i = 0; i < pos; ++i)
    start += (*(ranges.begin() + i)).getZExtValue();
  return allArgs.slice(start, (*(ranges.begin() + pos)).getZExtValue(),
                       std::forward<AdditionalArgs>(additionalArgs)...);
}

static mlir::MutableOperandRange
getMutableSuccessorOperands(unsigned pos, mlir::MutableOperandRange operands,
                            StringRef offsetAttr) {
  Operation *owner = operands.getOwner();
  NamedAttribute targetOffsetAttr =
      *owner->getAttrDictionary().getNamed(offsetAttr);
  return getSubOperands(
      pos, operands, targetOffsetAttr.second.cast<DenseIntElementsAttr>(),
      mlir::MutableOperandRange::OperandSegment(pos, targetOffsetAttr));
}

static unsigned denseElementsSize(mlir::DenseIntElementsAttr attr) {
  return attr.getNumElements();
}

llvm::Optional<mlir::OperandRange> fir::SelectOp::getCompareOperands(unsigned) {
  return {};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

llvm::Optional<mlir::MutableOperandRange>
fir::SelectOp::getMutableSuccessorOperands(unsigned oper) {
  return ::getMutableSuccessorOperands(oper, targetArgsMutable(),
                                       getTargetOffsetAttr());
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                    unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

unsigned fir::SelectOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

//===----------------------------------------------------------------------===//
// SelectCaseOp
//===----------------------------------------------------------------------===//

llvm::Optional<mlir::OperandRange>
fir::SelectCaseOp::getCompareOperands(unsigned cond) {
  auto a = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getCompareOffsetAttr());
  return {getSubOperands(cond, compareArgs(), a)};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectCaseOp::getCompareOperands(llvm::ArrayRef<mlir::Value> operands,
                                      unsigned cond) {
  auto a = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getCompareOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(cond, getSubOperands(1, operands, segments), a)};
}

llvm::Optional<mlir::MutableOperandRange>
fir::SelectCaseOp::getMutableSuccessorOperands(unsigned oper) {
  return ::getMutableSuccessorOperands(oper, targetArgsMutable(),
                                       getTargetOffsetAttr());
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectCaseOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

// parser for fir.select_case Op
static mlir::ParseResult parseSelectCase(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::OpAsmParser::OperandType selector;
  mlir::Type type;
  if (parseSelector(parser, result, selector, type))
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute, 8> attrs;
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 8> opers;
  llvm::SmallVector<mlir::Block *, 8> dests;
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 8>, 8> destArgs;
  llvm::SmallVector<int32_t, 8> argOffs;
  int32_t offSize = 0;
  while (true) {
    mlir::Attribute attr;
    mlir::Block *dest;
    llvm::SmallVector<mlir::Value, 8> destArg;
    mlir::NamedAttrList temp;
    if (parser.parseAttribute(attr, "a", temp) || isValidCaseAttr(attr) ||
        parser.parseComma())
      return mlir::failure();
    attrs.push_back(attr);
    if (attr.dyn_cast_or_null<mlir::UnitAttr>()) {
      argOffs.push_back(0);
    } else if (attr.dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
      mlir::OpAsmParser::OperandType oper1;
      mlir::OpAsmParser::OperandType oper2;
      if (parser.parseOperand(oper1) || parser.parseComma() ||
          parser.parseOperand(oper2) || parser.parseComma())
        return mlir::failure();
      opers.push_back(oper1);
      opers.push_back(oper2);
      argOffs.push_back(2);
      offSize += 2;
    } else {
      mlir::OpAsmParser::OperandType oper;
      if (parser.parseOperand(oper) || parser.parseComma())
        return mlir::failure();
      opers.push_back(oper);
      argOffs.push_back(1);
      ++offSize;
    }
    if (parser.parseSuccessorAndUseList(dest, destArg))
      return mlir::failure();
    dests.push_back(dest);
    destArgs.push_back(destArg);
    if (mlir::succeeded(parser.parseOptionalRSquare()))
      break;
    if (parser.parseComma())
      return mlir::failure();
  }
  result.addAttribute(fir::SelectCaseOp::getCasesAttr(),
                      parser.getBuilder().getArrayAttr(attrs));
  if (parser.resolveOperands(opers, type, result.operands))
    return mlir::failure();
  llvm::SmallVector<int32_t, 8> targOffs;
  int32_t toffSize = 0;
  const auto count = dests.size();
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    result.addSuccessors(dests[i]);
    result.addOperands(destArgs[i]);
    auto argSize = destArgs[i].size();
    targOffs.push_back(argSize);
    toffSize += argSize;
  }
  auto &bld = parser.getBuilder();
  result.addAttribute(fir::SelectCaseOp::getOperandSegmentSizeAttr(),
                      bld.getI32VectorAttr({1, offSize, toffSize}));
  result.addAttribute(getCompareOffsetAttr(), bld.getI32VectorAttr(argOffs));
  result.addAttribute(getTargetOffsetAttr(), bld.getI32VectorAttr(targOffs));
  return mlir::success();
}

unsigned fir::SelectCaseOp::compareOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getCompareOffsetAttr()));
}

unsigned fir::SelectCaseOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

void fir::SelectCaseOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              mlir::Value selector,
                              llvm::ArrayRef<mlir::Attribute> compareAttrs,
                              llvm::ArrayRef<mlir::ValueRange> cmpOperands,
                              llvm::ArrayRef<mlir::Block *> destinations,
                              llvm::ArrayRef<mlir::ValueRange> destOperands,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  result.addOperands(selector);
  result.addAttribute(getCasesAttr(), builder.getArrayAttr(compareAttrs));
  llvm::SmallVector<int32_t, 8> operOffs;
  int32_t operSize = 0;
  for (auto attr : compareAttrs) {
    if (attr.isa<fir::ClosedIntervalAttr>()) {
      operOffs.push_back(2);
      operSize += 2;
    } else if (attr.isa<mlir::UnitAttr>()) {
      operOffs.push_back(0);
    } else {
      operOffs.push_back(1);
      ++operSize;
    }
  }
  for (auto ops : cmpOperands)
    result.addOperands(ops);
  result.addAttribute(getCompareOffsetAttr(),
                      builder.getI32VectorAttr(operOffs));
  const auto count = destinations.size();
  for (auto d : destinations)
    result.addSuccessors(d);
  const auto opCount = destOperands.size();
  llvm::SmallVector<int32_t, 8> argOffs;
  int32_t sumArgs = 0;
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    if (i < opCount) {
      result.addOperands(destOperands[i]);
      const auto argSz = destOperands[i].size();
      argOffs.push_back(argSz);
      sumArgs += argSz;
    } else {
      argOffs.push_back(0);
    }
  }
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr({1, operSize, sumArgs}));
  result.addAttribute(getTargetOffsetAttr(), builder.getI32VectorAttr(argOffs));
  result.addAttributes(attributes);
}

/// This builder has a slightly simplified interface in that the list of
/// operands need not be partitioned by the builder. Instead the operands are
/// partitioned here, before being passed to the default builder. This
/// partitioning is unchecked, so can go awry on bad input.
void fir::SelectCaseOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result,
                              mlir::Value selector,
                              llvm::ArrayRef<mlir::Attribute> compareAttrs,
                              llvm::ArrayRef<mlir::Value> cmpOpList,
                              llvm::ArrayRef<mlir::Block *> destinations,
                              llvm::ArrayRef<mlir::ValueRange> destOperands,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  llvm::SmallVector<mlir::ValueRange, 16> cmpOpers;
  auto iter = cmpOpList.begin();
  for (auto &attr : compareAttrs) {
    if (attr.isa<fir::ClosedIntervalAttr>()) {
      cmpOpers.push_back(mlir::ValueRange({iter, iter + 2}));
      iter += 2;
    } else if (attr.isa<UnitAttr>()) {
      cmpOpers.push_back(mlir::ValueRange{});
    } else {
      cmpOpers.push_back(mlir::ValueRange({iter, iter + 1}));
      ++iter;
    }
  }
  build(builder, result, selector, compareAttrs, cmpOpers, destinations,
        destOperands, attributes);
}

//===----------------------------------------------------------------------===//
// SelectRankOp
//===----------------------------------------------------------------------===//

llvm::Optional<mlir::OperandRange>
fir::SelectRankOp::getCompareOperands(unsigned) {
  return {};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectRankOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

llvm::Optional<mlir::MutableOperandRange>
fir::SelectRankOp::getMutableSuccessorOperands(unsigned oper) {
  return ::getMutableSuccessorOperands(oper, targetArgsMutable(),
                                       getTargetOffsetAttr());
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectRankOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

unsigned fir::SelectRankOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

//===----------------------------------------------------------------------===//
// SelectTypeOp
//===----------------------------------------------------------------------===//

llvm::Optional<mlir::OperandRange>
fir::SelectTypeOp::getCompareOperands(unsigned) {
  return {};
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectTypeOp::getCompareOperands(llvm::ArrayRef<mlir::Value>, unsigned) {
  return {};
}

llvm::Optional<mlir::MutableOperandRange>
fir::SelectTypeOp::getMutableSuccessorOperands(unsigned oper) {
  return ::getMutableSuccessorOperands(oper, targetArgsMutable(),
                                       getTargetOffsetAttr());
}

llvm::Optional<llvm::ArrayRef<mlir::Value>>
fir::SelectTypeOp::getSuccessorOperands(llvm::ArrayRef<mlir::Value> operands,
                                        unsigned oper) {
  auto a =
      (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(getTargetOffsetAttr());
  auto segments = (*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getOperandSegmentSizeAttr());
  return {getSubOperands(oper, getSubOperands(2, operands, segments), a)};
}

static ParseResult parseSelectType(OpAsmParser &parser,
                                   OperationState &result) {
  mlir::OpAsmParser::OperandType selector;
  mlir::Type type;
  if (parseSelector(parser, result, selector, type))
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute, 8> attrs;
  llvm::SmallVector<mlir::Block *, 8> dests;
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 8>, 8> destArgs;
  while (true) {
    mlir::Attribute attr;
    mlir::Block *dest;
    llvm::SmallVector<mlir::Value, 8> destArg;
    mlir::NamedAttrList temp;
    if (parser.parseAttribute(attr, "a", temp) || parser.parseComma() ||
        parser.parseSuccessorAndUseList(dest, destArg))
      return mlir::failure();
    attrs.push_back(attr);
    dests.push_back(dest);
    destArgs.push_back(destArg);
    if (mlir::succeeded(parser.parseOptionalRSquare()))
      break;
    if (parser.parseComma())
      return mlir::failure();
  }
  auto &bld = parser.getBuilder();
  result.addAttribute(fir::SelectTypeOp::getCasesAttr(),
                      bld.getArrayAttr(attrs));
  llvm::SmallVector<int32_t, 8> argOffs;
  int32_t offSize = 0;
  const auto count = dests.size();
  for (std::remove_const_t<decltype(count)> i = 0; i != count; ++i) {
    result.addSuccessors(dests[i]);
    result.addOperands(destArgs[i]);
    auto argSize = destArgs[i].size();
    argOffs.push_back(argSize);
    offSize += argSize;
  }
  result.addAttribute(fir::SelectTypeOp::getOperandSegmentSizeAttr(),
                      bld.getI32VectorAttr({1, 0, offSize}));
  result.addAttribute(getTargetOffsetAttr(), bld.getI32VectorAttr(argOffs));
  return mlir::success();
}

unsigned fir::SelectTypeOp::targetOffsetSize() {
  return denseElementsSize((*this)->getAttrOfType<mlir::DenseIntElementsAttr>(
      getTargetOffsetAttr()));
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

/// Return the output rank of a slice op. The output rank must be between 1 and
/// the rank of the array being sliced (inclusive).
unsigned fir::SliceOp::getOutputRank(mlir::ValueRange triples) {
  unsigned rank = 0;
  if (!triples.empty()) {
    for (unsigned i = 1, end = triples.size(); i < end; i += 3) {
      auto op = triples[i].getDefiningOp();
      if (!mlir::isa_and_nonnull<fir::UndefOp>(op))
        ++rank;
    }
    assert(rank > 0);
  }
  return rank;
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

mlir::Type fir::StoreOp::elementType(mlir::Type refType) {
  if (auto ref = refType.dyn_cast<ReferenceType>())
    return ref.getEleTy();
  if (auto ref = refType.dyn_cast<PointerType>())
    return ref.getEleTy();
  if (auto ref = refType.dyn_cast<HeapType>())
    return ref.getEleTy();
  return {};
}

//===----------------------------------------------------------------------===//
// StringLitOp
//===----------------------------------------------------------------------===//

bool fir::StringLitOp::isWideValue() {
  auto eleTy = getType().cast<fir::SequenceType>().getEleTy();
  return eleTy.cast<fir::CharacterType>().getFKind() != 1;
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void fir::IfOp::build(mlir::OpBuilder &builder, OperationState &result,
                      mlir::Value cond, bool withElseRegion) {
  build(builder, result, llvm::None, cond, withElseRegion);
}

void fir::IfOp::build(mlir::OpBuilder &builder, OperationState &result,
                      mlir::TypeRange resultTypes, mlir::Value cond,
                      bool withElseRegion) {
  result.addOperands(cond);
  result.addTypes(resultTypes);

  mlir::Region *thenRegion = result.addRegion();
  thenRegion->push_back(new mlir::Block());
  if (resultTypes.empty())
    IfOp::ensureTerminator(*thenRegion, builder, result.location);

  mlir::Region *elseRegion = result.addRegion();
  if (withElseRegion) {
    elseRegion->push_back(new mlir::Block());
    if (resultTypes.empty())
      IfOp::ensureTerminator(*elseRegion, builder, result.location);
  }
}

static mlir::ParseResult parseIfOp(OpAsmParser &parser,
                                   OperationState &result) {
  result.regions.reserve(2);
  mlir::Region *thenRegion = result.addRegion();
  mlir::Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType cond;
  mlir::Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return mlir::failure();

  if (parser.parseOptionalArrowTypeList(result.types))
    return mlir::failure();

  if (parser.parseRegion(*thenRegion, {}, {}))
    return mlir::failure();
  IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  if (mlir::succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion, {}, {}))
      return mlir::failure();
    IfOp::ensureTerminator(*elseRegion, parser.getBuilder(), result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  return mlir::success();
}

static LogicalResult verify(fir::IfOp op) {
  if (op.getNumResults() != 0 && op.elseRegion().empty())
    return op.emitOpError("must have an else block if defining values");

  return mlir::success();
}

static void print(mlir::OpAsmPrinter &p, fir::IfOp op) {
  bool printBlockTerminators = false;
  p << fir::IfOp::getOperationName() << ' ' << op.condition();
  if (!op.results().empty()) {
    p << " -> (" << op.getResultTypes() << ')';
    printBlockTerminators = true;
  }
  p.printRegion(op.thenRegion(), /*printEntryBlockArgs=*/false,
                printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &otherReg = op.elseRegion();
  if (!otherReg.empty()) {
    p << " else";
    p.printRegion(otherReg, /*printEntryBlockArgs=*/false,
                  printBlockTerminators);
  }
  p.printOptionalAttrDict(op->getAttrs());
}

void fir::IfOp::resultToSourceOps(llvm::SmallVectorImpl<mlir::Value> &results,
                                  unsigned resultNum) {
  auto *term = thenRegion().front().getTerminator();
  if (resultNum < term->getNumOperands())
    results.push_back(term->getOperand(resultNum));
  term = elseRegion().front().getTerminator();
  if (resultNum < term->getNumOperands())
    results.push_back(term->getOperand(resultNum));
}

//===----------------------------------------------------------------------===//

mlir::ParseResult fir::isValidCaseAttr(mlir::Attribute attr) {
  if (attr.dyn_cast_or_null<mlir::UnitAttr>() ||
      attr.dyn_cast_or_null<ClosedIntervalAttr>() ||
      attr.dyn_cast_or_null<PointIntervalAttr>() ||
      attr.dyn_cast_or_null<LowerBoundAttr>() ||
      attr.dyn_cast_or_null<UpperBoundAttr>())
    return mlir::success();
  return mlir::failure();
}

unsigned fir::getCaseArgumentOffset(llvm::ArrayRef<mlir::Attribute> cases,
                                    unsigned dest) {
  unsigned o = 0;
  for (unsigned i = 0; i < dest; ++i) {
    auto &attr = cases[i];
    if (!attr.dyn_cast_or_null<mlir::UnitAttr>()) {
      ++o;
      if (attr.dyn_cast_or_null<ClosedIntervalAttr>())
        ++o;
    }
  }
  return o;
}

mlir::ParseResult fir::parseSelector(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result,
                                     mlir::OpAsmParser::OperandType &selector,
                                     mlir::Type &type) {
  if (parser.parseOperand(selector) || parser.parseColonType(type) ||
      parser.resolveOperand(selector, type, result.operands) ||
      parser.parseLSquare())
    return mlir::failure();
  return mlir::success();
}

/// Generic pretty-printer of a binary operation
static void printBinaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 2 && "binary op must have two operands");
  assert(op->getNumResults() == 1 && "binary op must have one result");

  p << op->getName() << ' ' << op->getOperand(0) << ", " << op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getResult(0).getType();
}

/// Generic pretty-printer of an unary operation
static void printUnaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 1 && "unary op must have one operand");
  assert(op->getNumResults() == 1 && "unary op must have one result");

  p << op->getName() << ' ' << op->getOperand(0);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getResult(0).getType();
}

bool fir::isReferenceLike(mlir::Type type) {
  return type.isa<fir::ReferenceType>() || type.isa<fir::HeapType>() ||
         type.isa<fir::PointerType>();
}

mlir::FuncOp fir::createFuncOp(mlir::Location loc, mlir::ModuleOp module,
                               StringRef name, mlir::FunctionType type,
                               llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  if (auto f = module.lookupSymbol<mlir::FuncOp>(name))
    return f;
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  modBuilder.setInsertionPoint(module.getBody()->getTerminator());
  auto result = modBuilder.create<mlir::FuncOp>(loc, name, type, attrs);
  result.setVisibility(mlir::SymbolTable::Visibility::Private);
  return result;
}

fir::GlobalOp fir::createGlobalOp(mlir::Location loc, mlir::ModuleOp module,
                                  StringRef name, mlir::Type type,
                                  llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  if (auto g = module.lookupSymbol<fir::GlobalOp>(name))
    return g;
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  auto result = modBuilder.create<fir::GlobalOp>(loc, name, type, attrs);
  result.setVisibility(mlir::SymbolTable::Visibility::Private);
  return result;
}

bool fir::valueHasFirAttribute(mlir::Value value,
                               llvm::StringRef attributeName) {
  // If this is a fir.box that was loaded, the fir attributes will be on the
  // related fir.ref<fir.box> creation.
  if (value.getType().isa<fir::BoxType>())
    if (auto definingOp = value.getDefiningOp())
      if (auto loadOp = mlir::dyn_cast<fir::LoadOp>(definingOp))
        value = loadOp.memref();
  // If this is a function argument, look in the argument attributes.
  if (auto blockArg = value.dyn_cast<mlir::BlockArgument>()) {
    if (blockArg.getOwner() && blockArg.getOwner()->isEntryBlock())
      if (auto funcOp =
              mlir::dyn_cast<mlir::FuncOp>(blockArg.getOwner()->getParentOp()))
        if (funcOp.getArgAttr(blockArg.getArgNumber(), attributeName))
          return true;
    return false;
  }

  if (auto definingOp = value.getDefiningOp()) {
    // If this is an allocated value, look at the allocation attributes.
    if (mlir::isa<fir::AllocMemOp>(definingOp) ||
        mlir::isa<AllocaOp>(definingOp))
      return definingOp->hasAttr(attributeName);
    // If this is an imported global, look at AddrOfOp and GlobalOp attributes.
    // Both operations are looked at because use/host associated variable (the
    // AddrOfOp) can have ASYNCHRONOUS/VOLATILE attributes even if the ultimate
    // entity (the globalOp) does not have them.
    if (auto addressOfOp = mlir::dyn_cast<fir::AddrOfOp>(definingOp)) {
      if (addressOfOp->hasAttr(attributeName))
        return true;
      if (auto module = definingOp->getParentOfType<mlir::ModuleOp>())
        if (auto globalOp =
                module.lookupSymbol<fir::GlobalOp>(addressOfOp.symbol()))
          return globalOp->hasAttr(attributeName);
    }
  }
  // TODO: Construct associated entities attributes. Decide where the fir
  // attributes must be placed/looked for in this case.
  return false;
}

// Tablegen operators

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FIROps.cpp.inc"
