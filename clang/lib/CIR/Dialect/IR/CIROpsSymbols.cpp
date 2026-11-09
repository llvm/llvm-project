//===- CIROpsSymbols.cpp - CIR ops in partition `Symbols` -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR ops that define or refer to symbols.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIROpsSymbols.h"
#include "clang/CIR/Dialect/IR/CIROpsControlFlow.h"
#include "clang/CIR/Dialect/IR/CIROpsCoroutine.h"

#include "CIROpDefinitions.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace cir;

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
REGISTER_ENUM_TYPE(CallingConv);
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

//===----------------------------------------------------------------------===//
// InlineKindAttr (FIXME: remove once FuncOp uses assembly format)
//===----------------------------------------------------------------------===//

static ParseResult parseInlineKindAttr(OpAsmParser &parser,
                                       cir::InlineKindAttr &inlineKindAttr) {
  // Static list of possible inline kind keywords
  static constexpr llvm::StringRef keywords[] = {"no_inline", "always_inline",
                                                 "inline_hint"};

  // Parse the inline kind keyword (optional)
  llvm::StringRef keyword;
  if (parser.parseOptionalKeyword(&keyword, keywords).failed()) {
    // Not an inline kind keyword, leave inlineKindAttr empty
    return success();
  }

  // Parse the enum value from the keyword
  auto inlineKindResult = ::cir::symbolizeEnum<::cir::InlineKind>(keyword);
  if (!inlineKindResult) {
    return parser.emitError(parser.getCurrentLocation(), "expected one of [")
           << llvm::join(llvm::ArrayRef(keywords), ", ")
           << "] for inlineKind, got: " << keyword;
  }

  inlineKindAttr =
      ::cir::InlineKindAttr::get(parser.getContext(), *inlineKindResult);
  return success();
}

static void printInlineKindAttr(OpAsmPrinter &p,
                                cir::InlineKindAttr inlineKindAttr) {
  if (inlineKindAttr) {
    p << " " << stringifyInlineKind(inlineKindAttr.getValue());
  }
}

//===----------------------------------------------------------------------===//
// LocalInitOp
//===----------------------------------------------------------------------===//

LogicalResult
cir::LocalInitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  cir::GlobalOp global = getReferencedGlobal(symbolTable);
  if (!global)
    return emitOpError("'")
           << getGlobalName() << "' does not reference a valid cir.global";

  if (getTls() && !global.getTlsModel())
    return emitOpError("access to global not marked thread local");

  if (!global.getDynamicInitGuard().has_value())
    return emitOpError("static_local attribute mismatch");

  return success();
}

//===----------------------------------------------------------------------===//
// RegisterExitDtorOp
//===----------------------------------------------------------------------===//

/// Returns true if \p op is nested in the ctor region of a cir.global or
/// cir.local_init without crossing a cir.func or a cir.register_exit_dtor.
static bool isInVarCtorRegion(mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    mlir::Region *ctorRegion = nullptr;
    if (auto global = mlir::dyn_cast<cir::GlobalOp>(parent))
      ctorRegion = &global.getCtorRegion();
    else if (auto localInit = mlir::dyn_cast<cir::LocalInitOp>(parent))
      ctorRegion = &localInit.getCtorRegion();
    else if (mlir::isa<cir::FuncOp, cir::RegisterExitDtorOp>(parent))
      return false;
    if (ctorRegion)
      return ctorRegion->isAncestor(op->getParentRegion());
  }
  return false;
}

LogicalResult cir::RegisterExitDtorOp::verify() {
  mlir::Block &body = getBody().front();
  if (body.without_terminator().empty())
    return emitOpError("body must destroy the object");
  if (body.getTerminator()->getNumOperands())
    return emitOpError("body must not yield a value");
  if (!isInVarCtorRegion(*this))
    return emitOpError("must be in the ctor region of a cir.global or "
                       "cir.local_init");
  return success();
}

LogicalResult
cir::RegisterExitDtorOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global = symbolTable.lookupNearestSymbolFrom<cir::GlobalOp>(
      *this, getObjectAttr());
  if (!global)
    return emitOpError("'")
           << getObject() << "' does not reference a valid cir.global";
  if (!global.getCtorRegion().empty() || !global.getDtorRegion().empty())
    return emitOpError("object must not have ctor or dtor regions");
  return success();
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

static mlir::ParseResult
parseTryCallDestinations(mlir::OpAsmParser &parser,
                         mlir::OperationState &result) {
  mlir::Block *normalDestSuccessor;
  if (parser.parseSuccessor(normalDestSuccessor))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  mlir::Block *unwindDestSuccessor;
  if (parser.parseSuccessor(unwindDestSuccessor))
    return mlir::failure();

  result.addSuccessors(normalDestSuccessor);
  result.addSuccessors(unwindDestSuccessor);
  return mlir::success();
}

/// Reject an attribute in an explicit attribute dictionary whose value is not
/// the attribute class expected for its name.  Where an op declares the
/// attribute it is stored as a property, so a value of the wrong class would
/// be dropped without a diagnostic.
static ParseResult checkEffectAttrKinds(mlir::OpAsmParser &parser,
                                        llvm::SMLoc loc,
                                        const mlir::NamedAttrList &attrs) {
  if (mlir::Attribute effects =
          attrs.get(CIRDialect::getMemoryEffectsAttrName()))
    if (!mlir::isa<cir::MemoryEffectsAttr>(effects))
      return parser.emitError(loc, "attribute '")
             << CIRDialect::getMemoryEffectsAttrName()
             << "' must be a #cir.memory_effects attribute";

  if (mlir::Attribute uwtable = attrs.get(CIRDialect::getUwtableAttrName()))
    if (!mlir::isa<cir::UnwindTableKindAttr>(uwtable))
      return parser.emitError(loc, "attribute '")
             << CIRDialect::getUwtableAttrName()
             << "' must be a #cir.uwtable attribute";

  if (mlir::Attribute framePointer =
          attrs.get(CIRDialect::getFramePointerAttrName()))
    if (!mlir::isa<cir::FramePointerKindAttr>(framePointer))
      return parser.emitError(loc, "attribute '")
             << CIRDialect::getFramePointerAttrName()
             << "' must be a #cir.frame_pointer attribute";

  for (llvm::StringRef name :
       {CIRDialect::getNoUnwindAttrName(), CIRDialect::getWillReturnAttrName(),
        CIRDialect::getMustProgressAttrName()})
    if (mlir::Attribute flag = attrs.get(name))
      if (!mlir::isa<mlir::UnitAttr>(flag))
        return parser.emitError(loc, "attribute '")
               << name << "' must be a unit attribute";

  return mlir::success();
}

static mlir::ParseResult parseCallCommon(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result,
                                         bool hasDestinationBlocks = false) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> ops;
  llvm::SMLoc opsLoc;
  mlir::FlatSymbolRefAttr calleeAttr;

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

  if (hasDestinationBlocks &&
      parseTryCallDestinations(parser, result).failed()) {
    return ::mlir::failure();
  }

  if (parser.parseOptionalKeyword("cc").succeeded()) {
    cir::CallingConv callingConv;
    if (parser.parseLParen().failed())
      return failure();
    if (parseCIRKeyword<cir::CallingConv>(parser, callingConv).failed())
      return parser.emitError(parser.getCurrentLocation(),
                              "unknown calling convention");
    if (parser.parseRParen().failed())
      return failure();
    result.addAttribute(
        CIRDialect::getCallingConvAttrName(),
        cir::CallingConvAttr::get(parser.getContext(), callingConv));
  }

  if (parser.parseOptionalKeyword("musttail").succeeded())
    result.addAttribute(CIRDialect::getMustTailAttrName(),
                        mlir::UnitAttr::get(parser.getContext()));

  if (parser.parseOptionalKeyword("nothrow").succeeded())
    result.addAttribute(CIRDialect::getNoThrowAttrName(),
                        mlir::UnitAttr::get(parser.getContext()));

  if (parser.parseOptionalKeyword("nounwind").succeeded())
    result.addAttribute(CIRDialect::getNoUnwindAttrName(),
                        mlir::UnitAttr::get(parser.getContext()));

  if (parser.parseOptionalKeyword("willreturn").succeeded())
    result.addAttribute(CIRDialect::getWillReturnAttrName(),
                        mlir::UnitAttr::get(parser.getContext()));

  llvm::SMLoc attrsLoc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  if (checkEffectAttrKinds(parser, attrsLoc, result.attributes).failed())
    return ::mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  SmallVector<Type> argTypes;
  SmallVector<DictionaryAttr> argAttrs;
  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  if (call_interface_impl::parseFunctionSignature(parser, argTypes, argAttrs,
                                                  resultTypes, resultAttrs))
    return mlir::failure();

  if (resultTypes.size() > 1 || resultAttrs.size() > 1)
    return parser.emitError(
        parser.getCurrentLocation(),
        "functions with multiple return types are not supported");

  result.addTypes(resultTypes);

  if (parser.resolveOperands(ops, argTypes, opsLoc, result.operands))
    return mlir::failure();

  if (!resultAttrs.empty() && resultAttrs[0])
    result.addAttribute(
        CIRDialect::getResAttrsAttrName(),
        mlir::ArrayAttr::get(parser.getContext(), {resultAttrs[0]}));

  // ArrayAttr requires a vector of 'Attribute', so we have to do the conversion
  // here into a separate collection.
  llvm::SmallVector<Attribute> convertedArgAttrs;
  bool argAttrsEmpty = true;

  llvm::transform(argAttrs, std::back_inserter(convertedArgAttrs),
                  [&](DictionaryAttr da) -> mlir::Attribute {
                    if (da)
                      argAttrsEmpty = false;
                    return da;
                  });

  if (!argAttrsEmpty) {
    llvm::ArrayRef argAttrsRef = convertedArgAttrs;
    if (!calleeAttr) {
      // Fixup for indirect calls, which get an extra entry in the 'args' for
      // the indirect type, which doesn't get attributes.
      argAttrsRef = argAttrsRef.drop_front();
    }
    result.addAttribute(CIRDialect::getArgAttrsAttrName(),
                        mlir::ArrayAttr::get(parser.getContext(), argAttrsRef));
  }

  return mlir::success();
}

static void printCallCommon(mlir::Operation *op,
                            mlir::FlatSymbolRefAttr calleeSym,
                            mlir::Value indirectCallee,
                            mlir::OpAsmPrinter &printer, bool isNothrow,
                            ArrayAttr argAttrs, ArrayAttr resAttrs,
                            mlir::Block *normalDest = nullptr,
                            mlir::Block *unwindDest = nullptr) {
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

  if (normalDest) {
    assert(unwindDest && "expected two successors");
    auto tryCall = cast<cir::TryCallOp>(op);
    printer << ' ' << tryCall.getNormalDest();
    printer << ",";
    printer << ' ';
    printer << tryCall.getUnwindDest();
  }

  cir::CallingConv callingConv = callLikeOp.getCallingConv();
  if (callingConv != cir::CallingConv::C)
    printer << " cc(" << stringifyCallingConv(callingConv) << ")";

  if (op->hasAttr(CIRDialect::getMustTailAttrName()))
    printer << " musttail";

  if (isNothrow)
    printer << " nothrow";

  if (op->hasAttr(CIRDialect::getNoUnwindAttrName()))
    printer << " nounwind";

  if (op->hasAttr(CIRDialect::getWillReturnAttrName()))
    printer << " willreturn";

  llvm::StringRef elidedAttrs[] = {
      CIRDialect::getCalleeAttrName(),
      CIRDialect::getCallingConvAttrName(),
      CIRDialect::getMustTailAttrName(),
      CIRDialect::getNoThrowAttrName(),
      CIRDialect::getNoUnwindAttrName(),
      CIRDialect::getWillReturnAttrName(),
      CIRDialect::getOperandSegmentSizesAttrName(),
      "res_attrs",
      "arg_attrs",
  };
  // TODO: Split inherent and discardable attribute printing instead of
  // materializing a single dictionary that mixes the two storage classes.
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  for (mlir::NamedAttribute attr : op->getDiscardableAttrs())
    if (!llvm::is_contained(elidedAttrs, attr.getName()))
      attrs.push_back(attr);
  op->getName().walkInherentAttrs(op, [&](llvm::StringRef name,
                                          mlir::Attribute &attr) {
    if (!llvm::is_contained(elidedAttrs, name))
      attrs.emplace_back(mlir::StringAttr::get(op->getContext(), name), attr);
  });
  llvm::sort(attrs, [](mlir::NamedAttribute lhs, mlir::NamedAttribute rhs) {
    return lhs.getName().strref() < rhs.getName().strref();
  });
  printer.printOptionalAttrDict(attrs);
  printer << " : ";
  if (calleeSym || !argAttrs) {
    call_interface_impl::printFunctionSignature(
        printer, op->getOperands().getTypes(), argAttrs,
        /*isVariadic=*/false, op->getResultTypes(), resAttrs);
  } else {
    // indirect function calls use an 'arg' type for the type of its indirect
    // argument.  However, we don't store a similar attribute collection.  In
    // order to make `printFunctionSignature` have the attributes line up, we
    // have to make a 'shimmed' copy of the attributes that have a blank set of
    // attributes for the indirect argument.
    llvm::SmallVector<Attribute> shimmedArgAttrs;
    shimmedArgAttrs.push_back(mlir::DictionaryAttr::get(op->getContext(), {}));
    shimmedArgAttrs.append(argAttrs.begin(), argAttrs.end());
    call_interface_impl::printFunctionSignature(
        printer, op->getOperands().getTypes(),
        mlir::ArrayAttr::get(op->getContext(), shimmedArgAttrs),
        /*isVariadic=*/false, op->getResultTypes(), resAttrs);
  }
}

mlir::ParseResult cir::CallOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  return parseCallCommon(parser, result);
}

void cir::CallOp::print(mlir::OpAsmPrinter &p) {
  mlir::Value indirectCallee = isIndirect() ? getIndirectCall() : nullptr;
  printCallCommon(*this, getCalleeAttr(), indirectCallee, p, getNothrow(),
                  getArgAttrsAttr(), getResAttrsAttr());
}

static LogicalResult
verifyCallCommInSymbolUses(mlir::Operation *op,
                           SymbolTableCollection &symbolTable) {
  auto callIf = cast<cir::CIRCallOpInterface>(op);

  // An indirect call is checked against the function type its callee pointer
  // points to, and a direct call against its callee's declaration.
  cir::FuncType fnType;
  bool hasPrototype = true;
  auto fnAttr =
      op->getAttrOfType<FlatSymbolRefAttr>(CIRDialect::getCalleeAttrName());
  if (!fnAttr) {
    if (op->getNumOperands() == 0)
      return op->emitOpError("indirect call requires a callee operand");
    mlir::Type calleeTy = callIf.getIndirectCall().getType();
    if (auto ptrTy = mlir::dyn_cast<cir::PointerType>(calleeTy))
      fnType = mlir::dyn_cast<cir::FuncType>(ptrTy.getPointee());
    if (!fnType)
      return op->emitOpError()
             << "indirect callee must be a pointer to a function, but has type "
             << calleeTy;
  } else {
    auto fn = symbolTable.lookupNearestSymbolFrom<cir::FuncOp>(op, fnAttr);
    if (!fn)
      return op->emitOpError() << "'" << fnAttr.getValue()
                               << "' does not reference a valid function";
    fnType = fn.getFunctionType();
    hasPrototype = !fn.getNoProto();
  }

  // Verify that the operand and result types match the callee.  Operands are
  // not checked for a direct call to a function declared no_proto.
  if (hasPrototype) {
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
               << callIf.getArgOperand(i).getType() << " for operand number "
               << i;
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
// TryCallOp
//===----------------------------------------------------------------------===//

mlir::OperandRange cir::TryCallOp::getArgOperands() {
  if (isIndirect())
    return getArgs().drop_front(1);
  return getArgs();
}

mlir::MutableOperandRange cir::TryCallOp::getArgOperandsMutable() {
  mlir::MutableOperandRange args = getArgsMutable();
  if (isIndirect())
    return args.slice(1, args.size() - 1);
  return args;
}

mlir::Value cir::TryCallOp::getIndirectCall() {
  assert(isIndirect());
  return getOperand(0);
}

/// Return the operand at index 'i'.
Value cir::TryCallOp::getArgOperand(unsigned i) {
  if (isIndirect())
    ++i;
  return getOperand(i);
}

/// Return the number of operands.
unsigned cir::TryCallOp::getNumArgOperands() {
  if (isIndirect())
    return this->getOperation()->getNumOperands() - 1;
  return this->getOperation()->getNumOperands();
}

LogicalResult
cir::TryCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyCallCommInSymbolUses(*this, symbolTable);
}

mlir::ParseResult cir::TryCallOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  return parseCallCommon(parser, result, /*hasDestinationBlocks=*/true);
}

void cir::TryCallOp::print(::mlir::OpAsmPrinter &p) {
  mlir::Value indirectCallee = isIndirect() ? getIndirectCall() : nullptr;
  printCallCommon(*this, getCalleeAttr(), indirectCallee, p, getNothrow(),
                  getArgAttrsAttr(), getResAttrsAttr(), getNormalDest(),
                  getUnwindDest());
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
  // A function is not an object, so it cannot be the type of a global.  A
  // global that holds a function's address carries a pointer type instead.
  if (mlir::isa<cir::FuncType>(getSymType()))
    return emitOpError("global type cannot be a function type");

  // Verify that the initial value, if present, is either a unit attribute or
  // an attribute CIR supports.
  if (getInitialValue().has_value()) {
    if (checkConstantTypes(getOperation(), getSymType(), *getInitialValue())
            .failed())
      return failure();
  }

  // The initial value can't be recomputed from the type once the ctor is
  // lowered (e.g. a null member pointer becomes -1 after CXXABILowering).
  if (!getCtorRegion().empty() && !getInitialValue().has_value())
    return emitOpError(
        "cannot have a constructor region without an initial value");

  std::optional<DynamicInitInfoAttr> info = getDynamicInitInfo();

  if ((getDynamicInitGuard().has_value()) && (!info || info->getLocal()) &&
      (!getCtorRegion().empty() || !getDtorRegion().empty()))
    return emitOpError(
        "Cannot have a static-local global-op with a constructor or "
        "destructor, they require in-function initialization via LocalInitOp");

  // CIRGen emits 'dynamic_init_guard' and 'dynamic_init_info' together and
  // they are only meaningful together: the guard drives lowering, which reads
  // the info. Require both or neither so malformed .cir can carry neither a
  // guard without info nor a dangling info nothing will read.
  if (getDynamicInitGuard().has_value() != getDynamicInitInfo().has_value())
    return emitOpError("'dynamic_init_guard' and 'dynamic_init_info' must be "
                       "present together");

  if (getTlsRefs()) {
    // 'Unordered' TLS globals (variable template instantiations) legitimately
    // carry both: they can't use the shared __tls_init guard, so CIRGen also
    // gives them their own dynamic-init guard, mangled the same as
    // 'tls_refs's own guard name. 'Ordered' TLS globals share __tls_init and
    // have no per-variable guard name, so a dynamic-init guard there would be
    // a mismatch.
    if (getDynamicInitGuard().has_value() && !getTlsRefs()->getGuardName())
      return emitOpError("cannot have a dynamic-init guard combined with "
                         "ordered tls references");
    if (!getTlsModel())
      return emitOpError("'tls_refs' only valid for tls");
  }

  if (getAliasee().has_value()) {
    if (getInitialValue().has_value() || !getCtorRegion().empty() ||
        !getDtorRegion().empty())
      return emitOpError("global alias shall not have an initializer or "
                         "constructor/destructor regions");
  }

  // TODO(CIR): Many other checks for properties that haven't been upstreamed
  // yet.

  return success();
}

void cir::GlobalOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, llvm::StringRef sym_name,
    mlir::Type sym_type, bool isConstant,
    mlir::ptr::MemorySpaceAttrInterface addrSpace,
    cir::GlobalLinkageKind linkage,
    function_ref<void(OpBuilder &, Location)> ctorBuilder,
    function_ref<void(OpBuilder &, Location)> dtorBuilder) {
  odsState.addAttribute(getSymNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getSymTypeAttrName(odsState.name),
                        mlir::TypeAttr::get(sym_type));
  auto &properties = odsState.getOrAddProperties<cir::GlobalOp::Properties>();
  properties.setConstant(isConstant);

  addrSpace = normalizeDefaultAddressSpace(addrSpace);
  if (addrSpace)
    odsState.addAttribute(getAddrSpaceAttrName(odsState.name), addrSpace);

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
    regions.emplace_back(getOperation());
    return;
  }

  // Don't consider the ctor region if it is empty.
  Region *ctorRegion = &this->getCtorRegion();
  if (ctorRegion->empty())
    ctorRegion = nullptr;

  // Don't consider the dtor region if it is empty.
  Region *dtorRegion = &this->getDtorRegion();
  if (dtorRegion->empty())
    dtorRegion = nullptr;

  // If the condition isn't constant, both regions may be executed.
  if (ctorRegion)
    regions.push_back(RegionSuccessor(ctorRegion));
  if (dtorRegion)
    regions.push_back(RegionSuccessor(dtorRegion));
}

static void printComdatName(OpAsmPrinter &p, StringAttr comdat) {
  if (!comdat)
    return;
  p << "comdat";
  if (!comdat.getValue().empty())
    p << "(\"" << comdat.getValue() << "\")";
}

void cir::printComdatName(OpAsmPrinter &p, cir::GlobalOp op,
                          StringAttr comdat) {
  ::printComdatName(p, comdat);
}

ParseResult cir::parseComdatName(OpAsmParser &parser, StringAttr &comdatAttr) {
  if (parser.parseOptionalKeyword("comdat").failed())
    return success();
  std::string comdatKey;
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseString(&comdatKey).failed() ||
        parser.parseRParen().failed())
      return failure();
  }
  comdatAttr = parser.getBuilder().getStringAttr(comdatKey);
  return success();
}

void cir::printGlobalOpTypeAndInitialValue(OpAsmPrinter &p, cir::GlobalOp op,
                                           TypeAttr type, Attribute initAttr,
                                           mlir::Region &ctorRegion,
                                           mlir::Region &dtorRegion) {
  auto printType = [&]() { p << ": " << type; };
  // Aliases are definitions but they have no initial value or ctor/dtor; the
  // assembly prints them like declarations (`: type`).
  if (op.isDeclaration() || op.getAliasee()) {
    printType();
    return;
  }

  p << "= ";
  if (initAttr)
    printConstant(p, initAttr);

  if (!ctorRegion.empty()) {
    p << " ctor ";
    p.printRegion(ctorRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }

  if (!dtorRegion.empty()) {
    p << " dtor ";
    p.printRegion(dtorRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }
}

ParseResult cir::parseGlobalOpTypeAndInitialValue(OpAsmParser &parser,
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
    // Parse constant with initializer, examples:
    //  cir.global @y = 3.400000e+00 : f32
    //  cir.global @rgb = #cir.const_array<[...] : !cir.array<!s8i x 3>>
    if (parseConstantValue(parser, initialValueAttr).failed())
      return failure();

    assert(mlir::isa<mlir::TypedAttr>(initialValueAttr) &&
           "Non-typed attrs shouldn't appear here.");
    opTy = mlir::cast<mlir::TypedAttr>(initialValueAttr).getType();

    // Parse constructor, example:
    //  cir.global @rgb = #cir.zero : type ctor { ... }
    if (!parser.parseOptionalKeyword("ctor")) {
      auto parseLoc = parser.getCurrentLocation();
      if (parser.parseRegion(ctorRegion, /*arguments=*/{}, /*argTypes=*/{}))
        return failure();
      if (ensureRegionTerm(parser, ctorRegion, parseLoc).failed())
        return failure();
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
  mlir::ptr::MemorySpaceAttrInterface symAddrSpaceAttr{};
  if (auto g = dyn_cast<GlobalOp>(op)) {
    symTy = g.getSymType();
    symAddrSpaceAttr = g.getAddrSpaceAttr();
    // Verify that for thread local global access, the global needs to
    // be marked with tls bits.
    if (getTls() && !g.getTlsModel())
      return emitOpError("access to global not marked thread local");

    // Verify that the static_local attribute on GetGlobalOp matches the
    // dynamic_init_guard attribute on GlobalOp. GetGlobalOp uses a UnitAttr,
    // GlobalOp uses DynamicInitGuardAttr. Both should be present, or neither.
    bool getGlobalIsStaticLocal = getStaticLocal();
    bool globalIsStaticLocal =
        g.getDynamicInitGuard().has_value() &&
        (!g.getDynamicInitInfo() || g.getDynamicInitInfo()->getLocal());

    if (getGlobalIsStaticLocal != globalIsStaticLocal &&
        !getOperation()->getParentOfType<cir::GlobalOp>())
      return emitOpError("static_local attribute mismatch");
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

  // A function address may also be taken in the program address space.
  auto targetAS =
      dyn_cast_if_present<TargetAddressSpaceAttr>(resultType.getAddrSpace());
  bool isProgramAS =
      isa<FuncOp>(op) && targetAS &&
      targetAS.getValue() ==
          CIRDataLayout(getOperation()->getParentOfType<ModuleOp>())
              .getProgramAddressSpace();
  if (!isProgramAS && symAddrSpaceAttr != resultType.getAddrSpace()) {
    return emitOpError()
           << "result type address space does not match the address "
              "space of the global @"
           << getName();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// Returns the name used for the linkage attribute. This *must* correspond to
/// the name of the attribute in ODS.
static llvm::StringRef getLinkageAttrNameString() { return "linkage"; }

/// Declared cir.func attributes with no dedicated syntax, which print in and
/// parse from the attribute dictionary instead.
static bool isFuncAttrInDict(llvm::StringRef name) {
  return name == CIRDialect::getMemoryEffectsAttrName() ||
         name == CIRDialect::getUwtableAttrName() ||
         name == CIRDialect::getFramePointerAttrName();
}

void cir::FuncOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, FuncType type,
                        GlobalLinkageKind linkage, CallingConv callingConv) {
  result.addRegion();
  result.addAttribute(getSymNameAttrName(result.name),
                      builder.getStringAttr(name));
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));
  result.addAttribute(
      getLinkageAttrNameString(),
      GlobalLinkageKindAttr::get(builder.getContext(), linkage));
  result.addAttribute(getCallingConvAttrName(result.name),
                      CallingConvAttr::get(builder.getContext(), callingConv));
  // A function starts out written in source-level types, and CallConvLowering
  // rewrites it to the target ABI later.
  result.addAttribute(getRequiresCallConvLoweringAttrName(result.name),
                      builder.getUnitAttr());
}

ParseResult cir::FuncOp::parse(OpAsmParser &parser, OperationState &state) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  mlir::Builder &builder = parser.getBuilder();

  mlir::StringAttr builtinNameAttr = getBuiltinAttrName(state.name);
  mlir::StringAttr coroutineKindNameAttr = getCoroutineKindAttrName(state.name);
  mlir::StringAttr inlineKindNameAttr = getInlineKindAttrName(state.name);
  mlir::StringAttr lambdaNameAttr = getLambdaAttrName(state.name);
  mlir::StringAttr noProtoNameAttr = getNoProtoAttrName(state.name);
  mlir::StringAttr requiresCallConvLoweringNameAttr =
      getRequiresCallConvLoweringAttrName(state.name);
  mlir::StringAttr comdatNameAttr = getComdatAttrName(state.name);
  mlir::StringAttr alignmentNameAttr = getAlignmentAttrName(state.name);
  mlir::StringAttr preferredAlignmentNameAttr =
      getPreferredAlignmentAttrName(state.name);
  mlir::StringAttr visNameAttr = getSymVisibilityAttrName(state.name);
  mlir::StringAttr dsoLocalNameAttr = getDsoLocalAttrName(state.name);
  mlir::StringAttr funcInfoNameAttr = getFuncInfoAttrName(state.name);

  if (::mlir::succeeded(parser.parseOptionalKeyword(builtinNameAttr.strref())))
    state.addAttribute(builtinNameAttr, parser.getBuilder().getUnitAttr());

  // Parse optional coroutine kind attribute
  {
    static constexpr llvm::StringRef keywords[] = {"coroutine",
                                                   "flattened_coroutine"};
    llvm::StringRef keyword;
    if (parser.parseOptionalKeyword(&keyword, keywords).succeeded())
      state.addAttribute(
          coroutineKindNameAttr,
          cir::CoroutineKindAttr::get(builder.getContext(),
                                      *cir::symbolizeCoroutineKind(keyword)));
  }

  // Parse optional inline kind attribute
  cir::InlineKindAttr inlineKindAttr;
  if (failed(parseInlineKindAttr(parser, inlineKindAttr)))
    return failure();
  if (inlineKindAttr)
    state.addAttribute(inlineKindNameAttr, inlineKindAttr);

  if (::mlir::succeeded(parser.parseOptionalKeyword(lambdaNameAttr.strref())))
    state.addAttribute(lambdaNameAttr, parser.getBuilder().getUnitAttr());
  if (parser.parseOptionalKeyword(noProtoNameAttr).succeeded())
    state.addAttribute(noProtoNameAttr, parser.getBuilder().getUnitAttr());

  {
    StringAttr comdatAttr;
    if (parseComdatName(parser, comdatAttr).failed())
      return failure();
    if (comdatAttr)
      state.addAttribute(comdatNameAttr, comdatAttr);
  }

  auto parseAlignmentBody = [&](int64_t &value) {
    if (parser.parseLParen().failed() || parser.parseInteger(value).failed() ||
        parser.parseRParen().failed())
      return failure();

    if (value <= 0)
      return static_cast<LogicalResult>(parser.emitError(
          loc, "function alignment must be a positive integer"));

    return success();
  };

  if (parser.parseOptionalKeyword(alignmentNameAttr).succeeded()) {
    int64_t value;
    if (parseAlignmentBody(value).failed())
      return failure();
    state.addAttribute(alignmentNameAttr, builder.getI64IntegerAttr(value));
  }

  if (parser.parseOptionalKeyword(preferredAlignmentNameAttr).succeeded()) {
    int64_t value;
    if (parseAlignmentBody(value).failed())
      return failure();
    state.addAttribute(preferredAlignmentNameAttr,
                       builder.getI64IntegerAttr(value));
  }

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

  state.getOrAddProperties<cir::FuncOp::Properties>().global_visibility =
      parseOptionalCIRKeyword(parser, cir::VisibilityKind::Default);

  if (parser.parseOptionalKeyword(dsoLocalNameAttr).succeeded())
    state.addAttribute(dsoLocalNameAttr, parser.getBuilder().getUnitAttr());

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, getSymNameAttrName(state.name),
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
  llvm::SmallVector<mlir::Attribute> argAttrs;
  bool argAttrsEmpty = true;
  for (OpAsmParser::Argument &arg : arguments) {
    argTypes.push_back(arg.type);
    // Add the 'empty' attribute anyway to make sure the arity matches, but we
    // only want to 'set' the attribute at the top level if there is SOME data
    // along the way.
    argAttrs.push_back(arg.attrs);
    if (arg.attrs)
      argAttrsEmpty = false;
  }

  // These should be in sync anyway, but test both of them anyway.
  if (resultTypes.size() > 1 || resultAttrs.size() > 1)
    return parser.emitError(
        loc, "functions with multiple return types are not supported");

  mlir::Type returnType =
      (resultTypes.empty() ? cir::VoidType::get(builder.getContext())
                           : resultTypes.front());

  cir::FuncType fnType =
      cir::FuncType::getChecked([&]() { return parser.emitError(loc); },
                                argTypes, returnType, isVariadic);
  if (!fnType)
    return failure();

  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(fnType));

  if (!resultAttrs.empty() && resultAttrs[0])
    state.addAttribute(
        getResAttrsAttrName(state.name),
        mlir::ArrayAttr::get(parser.getContext(), {resultAttrs[0]}));

  if (!argAttrsEmpty)
    state.addAttribute(getArgAttrsAttrName(state.name),
                       mlir::ArrayAttr::get(parser.getContext(), argAttrs));

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

  mlir::StringAttr personalityNameAttr = getPersonalityAttrName(state.name);
  if (parser.parseOptionalKeyword("personality").succeeded()) {
    if (parser.parseLParen().failed())
      return failure();
    mlir::StringAttr personalityAttr;
    if (parser.parseOptionalSymbolName(personalityAttr).failed())
      return failure();
    state.addAttribute(personalityNameAttr,
                       FlatSymbolRefAttr::get(personalityAttr));
    if (parser.parseRParen().failed())
      return failure();
  }

  // Default to C calling convention if no keyword is provided.
  mlir::StringAttr callConvNameAttr = getCallingConvAttrName(state.name);
  cir::CallingConv callConv = cir::CallingConv::C;
  if (parser.parseOptionalKeyword("cc").succeeded()) {
    if (parser.parseLParen().failed())
      return failure();
    if (parseCIRKeyword<cir::CallingConv>(parser, callConv).failed())
      return parser.emitError(loc) << "unknown calling convention";
    if (parser.parseRParen().failed())
      return failure();
  }
  state.addAttribute(callConvNameAttr,
                     cir::CallingConvAttr::get(parser.getContext(), callConv));

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

  // Parse the func_info attribute
  if (parser.parseOptionalKeyword("func_info").succeeded()) {
    if (parser.parseLess().failed())
      return failure();

    llvm::SMLoc attrLoc = parser.getCurrentLocation();
    mlir::Attribute attr;
    if (parser.parseAttribute(attr).failed())
      return failure();
    if (!mlir::isa<cir::CXXCtorAttr, cir::CXXDtorAttr, cir::CXXAssignAttr,
                   cir::FuncIdentityAttr>(attr))
      return parser.emitError(attrLoc,
                              "expected a function info attribute, got ")
             << attr;
    state.addAttribute(funcInfoNameAttr, attr);

    if (parser.parseGreater().failed())
      return failure();
  }

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

  // Parse optional annotations attribute (an ArrayAttr of AnnotationAttr).
  mlir::StringAttr annotationsNameAttr = getAnnotationsAttrName(state.name);
  mlir::ArrayAttr annotationsAttr;
  if (parser.parseOptionalAttribute(annotationsAttr).has_value() &&
      annotationsAttr)
    state.addAttribute(annotationsNameAttr, annotationsAttr);

  if (parser.parseOptionalKeyword(requiresCallConvLoweringNameAttr).succeeded())
    state.addAttribute(requiresCallConvLoweringNameAttr,
                       parser.getBuilder().getUnitAttr());

  // Parse the rest of the attributes.
  NamedAttrList parsedAttrs;
  llvm::SMLoc attrsLoc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttrs))
    return failure();

  // Every other declared attribute has dedicated syntax above.
  for (StringRef disallowed : cir::FuncOp::getAttributeNames()) {
    if (isFuncAttrInDict(disallowed))
      continue;
    if (parsedAttrs.get(disallowed))
      return parser.emitError(loc, "attribute '")
             << disallowed
             << "' should not be specified in the explicit attribute list";
  }

  if (checkEffectAttrKinds(parser, attrsLoc, parsedAttrs).failed())
    return failure();

  state.attributes.append(parsedAttrs);

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

bool cir::FuncOp::isCXXSpecialMemberFunction() {
  // The func_info union can grow forms that are not special members, so the
  // check names the concrete forms rather than testing for presence.
  mlir::Attribute attr = getFuncInfoAttr();
  return attr && mlir::isa<CXXCtorAttr, CXXDtorAttr, CXXAssignAttr>(attr);
}

bool cir::FuncOp::isCxxConstructor() {
  auto attr = getFuncInfoAttr();
  return attr && dyn_cast<CXXCtorAttr>(attr);
}

bool cir::FuncOp::isCxxDestructor() {
  auto attr = getFuncInfoAttr();
  return attr && dyn_cast<CXXDtorAttr>(attr);
}

bool cir::FuncOp::isCxxSpecialAssignment() {
  auto attr = getFuncInfoAttr();
  return attr && dyn_cast<CXXAssignAttr>(attr);
}

std::optional<CtorKind> cir::FuncOp::getCxxConstructorKind() {
  mlir::Attribute attr = getFuncInfoAttr();
  if (attr) {
    if (auto ctor = dyn_cast<CXXCtorAttr>(attr))
      return ctor.getCtorKind();
  }
  return std::nullopt;
}

std::optional<AssignKind> cir::FuncOp::getCxxSpecialAssignKind() {
  mlir::Attribute attr = getFuncInfoAttr();
  if (attr) {
    if (auto assign = dyn_cast<CXXAssignAttr>(attr))
      return assign.getAssignKind();
  }
  return std::nullopt;
}

bool cir::FuncOp::isCxxTrivialMemberFunction() {
  mlir::Attribute attr = getFuncInfoAttr();
  if (attr) {
    if (auto ctor = dyn_cast<CXXCtorAttr>(attr))
      return ctor.getIsTrivial();
    if (auto dtor = dyn_cast<CXXDtorAttr>(attr))
      return dtor.getIsTrivial();
    if (auto assign = dyn_cast<CXXAssignAttr>(attr))
      return assign.getIsTrivial();
  }
  return false;
}

mlir::Region *cir::FuncOp::getCallableRegion() {
  // Declarations and aliases have no body to analyze or inline. Returning the
  // empty region would make interprocedural analyses (e.g. SCCP) treat calls
  // to them as having no returns, leaving their results uninitialized.
  if (getBody().empty())
    return nullptr;
  return &getBody();
}

void cir::FuncOp::print(OpAsmPrinter &p) {
  if (getBuiltin())
    p << " builtin";

  if (std::optional<cir::CoroutineKind> coroKind = getCoroutineKind())
    p << ' ' << cir::stringifyCoroutineKind(*coroKind);

  printInlineKindAttr(p, getInlineKindAttr());

  if (getLambda())
    p << " lambda";

  if (getNoProto())
    p << " no_proto";

  if (getComdatAttr()) {
    p << ' ';
    ::printComdatName(p, getComdatAttr());
  }

  if (getAlignment())
    p << " alignment(" << *getAlignment() << ')';

  if (getPreferredAlignment())
    p << " preferred_alignment(" << *getPreferredAlignment() << ')';

  if (getLinkage() != GlobalLinkageKind::ExternalLinkage)
    p << ' ' << stringifyGlobalLinkageKind(getLinkage());

  mlir::SymbolTable::Visibility vis = getVisibility();
  if (vis != mlir::SymbolTable::Visibility::Public)
    p << ' ' << vis;

  if (getGlobalVisibility() != cir::VisibilityKind::Default)
    p << ' ' << stringifyVisibilityKind(getGlobalVisibility());

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

  if (getCallingConv() != cir::CallingConv::C) {
    p << " cc(";
    p << stringifyCallingConv(getCallingConv());
    p << ")";
  }

  if (std::optional<StringRef> personalityName = getPersonality()) {
    p << " personality(";
    p.printSymbolName(*personalityName);
    p << ")";
  }

  if (mlir::Attribute funcInfo = getFuncInfoAttr()) {
    p << " func_info<";
    p.printAttribute(funcInfo);
    p << '>';
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

  if (mlir::ArrayAttr annotations = getAnnotationsAttr()) {
    p << ' ';
    p.printAttribute(annotations);
  }

  if (getRequiresCallConvLowering())
    p << ' ' << getRequiresCallConvLoweringAttrName().getValue();

  // Every other declared attribute is printed by the syntax above.
  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  for (llvm::StringRef name : cir::FuncOp::getAttributeNames())
    if (!isFuncAttrInDict(name))
      elidedAttrs.push_back(name);
  function_interface_impl::printFunctionAttributes(p, *this, elidedAttrs);

  // Print the body if this is not an external function.
  Region &body = getOperation()->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

mlir::LogicalResult cir::FuncOp::verify() {

  if (!isDeclaration() && getCoroutineKind() == cir::CoroutineKind::Coroutine) {
    int coroutineOpCount = 0;
    this->walk([&](Operation *op) {
      if (isa<CoroutineOp>(op)) {
        coroutineOpCount++;
        if (coroutineOpCount > 1) {
          return mlir::WalkResult::interrupt();
        }
      }
      return mlir::WalkResult::advance();
    });
    if (coroutineOpCount != 1)
      return emitOpError()
             << "coroutine function must have exactly one cir.coroutine op";
  }

  llvm::SmallSet<llvm::StringRef, 16> labels;
  llvm::SmallSet<llvm::StringRef, 16> gotos;
  llvm::SmallSet<llvm::StringRef, 16> blockAddresses;
  bool invalidBlockAddress = false;
  getOperation()->walk([&](mlir::Operation *op) {
    if (auto lab = dyn_cast<cir::LabelOp>(op)) {
      labels.insert(lab.getLabel());
    } else if (auto goTo = dyn_cast<cir::GotoOp>(op)) {
      gotos.insert(goTo.getLabel());
    } else if (auto blkAdd = dyn_cast<cir::BlockAddressOp>(op)) {
      if (blkAdd.getBlockAddrInfoAttr().getFunc().getAttr() != getSymName()) {
        // Stop the walk early, no need to continue
        invalidBlockAddress = true;
        return mlir::WalkResult::interrupt();
      }
      blockAddresses.insert(blkAdd.getBlockAddrInfoAttr().getLabel());
    }
    return mlir::WalkResult::advance();
  });

  if (invalidBlockAddress)
    return emitOpError() << "blockaddress references a different function";

  llvm::SmallSet<llvm::StringRef, 16> mismatched;
  if (!labels.empty() || !gotos.empty()) {
    mismatched = llvm::set_difference(gotos, labels);

    if (!mismatched.empty())
      return emitOpError() << "goto/label mismatch";
  }

  mismatched.clear();

  if (!labels.empty() || !blockAddresses.empty()) {
    mismatched = llvm::set_difference(blockAddresses, labels);

    if (!mismatched.empty())
      return emitOpError()
             << "expects an existing label target in the referenced function";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsSymbols.cpp.inc"
