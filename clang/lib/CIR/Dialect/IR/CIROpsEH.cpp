//===- CIROpsEH.cpp - CIR ops in partition `EH` ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR ops in partition `EH`: the hand-written
// parts first, then the TableGen-generated definitions.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIROpsEH.h"
#include "clang/CIR/Dialect/IR/CIROpsBuiltins.h"
#include "clang/CIR/Dialect/IR/CIROpsControlFlow.h"
#include "clang/CIR/Dialect/IR/CIROpsSymbols.h"

#include "CIROpDefinitions.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// ThrowOp / TryThrowOp
//===----------------------------------------------------------------------===//

template <typename ThrowOpTy>
static mlir::LogicalResult verifyThrowOpImpl(ThrowOpTy op) {
  if (op.rethrows())
    return mlir::success();

  if (op.getNumOperands() != 0) {
    if (op.getTypeInfo())
      return mlir::success();
    return op.emitOpError() << "'type_info' symbol attribute missing";
  }

  return mlir::failure();
}

mlir::LogicalResult cir::ThrowOp::verify() { return verifyThrowOpImpl(*this); }

mlir::LogicalResult cir::TryThrowOp::verify() {
  return verifyThrowOpImpl(*this);
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

void cir::TryOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getTryRegion());
  } else {
    // A handler exit leaves the operation.
    mlir::Operation *term = point.getTerminatorPredecessorOrNull();
    if (term->getParentRegion() != &getTryRegion()) {
      regions.emplace_back(getOperation());
      return;
    }
    // A cir.resume ending the try region unwinds into a handler, so it leaves
    // the operation only when there is none. Every other exit can leave it.
    if (!isa<cir::ResumeOp>(term) || getHandlerRegions().empty())
      regions.emplace_back(getOperation());
  }

  // Most handlers are entered by an unwind out of a call in the try region,
  // which is not a branch point, so the handlers are successors of the parent
  // and of every exit from the try region to stand in for it.
  //
  // TODO(CIR): If we know a target function never throws a specific type, we
  // can remove the catch handler.
  regions.reserve(regions.size() + getHandlerRegions().size());
  for (mlir::Region &handlerRegion : getHandlerRegions())
    regions.emplace_back(&handlerRegion);
}

/// Verify that each cir.init_catch_param in \p handlerRegion, outside any
/// nested cir.try, agrees with the handler's leading cir.construct_catch_param
/// \p constructOp, which may be null.  A cir.init_catch_param of kind
/// reference_to_pointer or non_trivial_copy needs \p constructOp on its
/// parameter, and a cir.init_catch_param on the parameter of \p constructOp
/// must have the kind of \p constructOp.
static LogicalResult
verifyCatchParamPairing(mlir::Region &handlerRegion,
                        cir::ConstructCatchParamOp constructOp) {
  mlir::WalkResult result =
      handlerRegion.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
        if (mlir::isa<cir::TryOp>(op))
          return mlir::WalkResult::skip();
        auto initOp = mlir::dyn_cast<cir::InitCatchParamOp>(op);
        if (!initOp)
          return mlir::WalkResult::advance();

        cir::InitCatchKind kind = initOp.getKind();
        bool sameParam =
            constructOp && constructOp.getParamAddr() == initOp.getParamAddr();
        if (sameParam && constructOp.getKind() != kind) {
          initOp.emitOpError("kind '")
              << cir::stringifyInitCatchKind(kind)
              << "' does not match the kind '"
              << cir::stringifyInitCatchKind(constructOp.getKind())
              << "' of the preceding 'cir.construct_catch_param'";
          return mlir::WalkResult::interrupt();
        }
        if ((kind == cir::InitCatchKind::ReferenceToPointer ||
             kind == cir::InitCatchKind::NonTrivialCopy) &&
            !sameParam) {
          initOp.emitOpError("'")
              << cir::stringifyInitCatchKind(kind)
              << "' requires a preceding 'cir.construct_catch_param' of the "
                 "same kind on the same parameter";
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
  return failure(result.wasInterrupted());
}

LogicalResult cir::TryOp::verifyRegions() {
  mlir::ArrayAttr handlerTypes = getHandlerTypes();
  if (!handlerTypes) {
    if (!getHandlerRegions().empty())
      return emitOpError(
          "handler regions must be empty when no handler types are present");
    return success();
  }

  mlir::MutableArrayRef<mlir::Region> handlerRegions = getHandlerRegions();

  // The parser and builder won't allow this to happen, but the loop below
  // relies on the sizes being the same, so we check it here.
  if (handlerRegions.size() != handlerTypes.size())
    return emitOpError(
        "number of handler regions and handler types must match");

  // A filter handler and an unexpected handler together implement a dynamic
  // exception specification. The filter try operation wraps the whole function
  // body and exists only to check the specification, so the two must appear
  // together and must stand alone. A catch-all in the same list would make the
  // filter unreachable, and the filter handler already provides the
  // continue-unwinding path that an unwind handler would supply.
  if (llvm::any_of(handlerTypes, [](mlir::Attribute typeAttr) {
        return mlir::isa<cir::EhFilterAttr, cir::EhUnexpectedAttr>(typeAttr);
      })) {
    if (handlerTypes.size() != 2 ||
        !mlir::isa<cir::EhFilterAttr>(handlerTypes[0]) ||
        !mlir::isa<cir::EhUnexpectedAttr>(handlerTypes[1]))
      return emitOpError("a filter handler must be followed by an unexpected "
                         "handler, and the two must be the only handlers");
  }

  for (const auto &[typeAttr, handlerRegion] :
       llvm::zip(handlerTypes, handlerRegions)) {
    // Verify that handler regions have a !cir.eh_token block argument.
    mlir::Block &entryBlock = handlerRegion.front();
    if (entryBlock.getNumArguments() != 1 ||
        !mlir::isa<cir::EhTokenType>(entryBlock.getArgument(0).getType()))
      return emitOpError(
          "handler region must have a single '!cir.eh_token' argument");

    // The unwind, filter and unexpected regions do not require a
    // cir.begin_catch. None of them catches the exception.
    if (mlir::isa<cir::UnwindAttr, cir::EhFilterAttr, cir::EhUnexpectedAttr>(
            typeAttr))
      continue;

    if (entryBlock.empty())
      return emitOpError("catch handler region must not be empty");

    // A terminate scope, which wraps the body of a function that cannot throw,
    // is a catch-all handler that only terminates the program. The exception is
    // caught by the runtime helper that cir.eh.terminate lowers to, so the
    // handler region has no cir.begin_catch of its own.
    if (mlir::isa<cir::EhTerminateOp>(entryBlock.front())) {
      if (!mlir::isa<cir::CatchAllAttr>(typeAttr))
        return emitOpError("'cir.eh.terminate' is only allowed in a catch-all "
                           "handler region");
      continue;
    }

    // Nothing may run in a catch handler before cir.begin_catch, so it has to
    // be the handler region's first operation, with two exceptions.
    //
    // When lifetime markers are enabled, the catch parameter's storage can be
    // marked by a cir.lifetime.start. That parameter is the only variable
    // whose lifetime begins before the catch is entered, so there is at most
    // one such marker. Its lifetime-end cleanup has to run after the catch
    // handler is finished (or exited by an exception unwind), so if there is a
    // lifetime begin marker, it is followed by a cir.cleanup.scope that
    // encloses the the rest of the handler with a cir.lifetime.end in its
    // cleanup region.
    //
    // A cir.construct_catch_param may also precede cir.begin_catch, to
    // perform any pre-begin_catch initialization of the catch parameter.
    mlir::Operation *firstOp = &entryBlock.front();
    if (mlir::isa<cir::LifetimeStartOp>(firstOp)) {
      mlir::Operation *next = firstOp->getNextNode();
      auto lifetimeScope = mlir::dyn_cast_if_present<cir::CleanupScopeOp>(next);
      if (!lifetimeScope)
        return emitOpError("'cir.lifetime.start' in a catch handler region "
                           "must be followed by the 'cir.cleanup.scope' of "
                           "its lifetime-end cleanup");
      if (lifetimeScope.getBodyRegion().empty())
        return emitOpError(
            "'cir.lifetime.start' in a catch handler region must be "
            "followed by the 'cir.cleanup.scope' of its lifetime-end "
            "cleanup");
      mlir::Block &scopeBody = lifetimeScope.getBodyRegion().front();
      firstOp = scopeBody.empty() ? nullptr : &scopeBody.front();
    }

    auto constructOp =
        mlir::dyn_cast_if_present<cir::ConstructCatchParamOp>(firstOp);
    if (constructOp)
      firstOp = firstOp->getNextNode();
    if (!mlir::isa_and_present<cir::BeginCatchOp>(firstOp))
      return emitOpError(
          "catch handler region must start with 'cir.begin_catch'");

    if (failed(verifyCatchParamPairing(handlerRegion, constructOp)))
      return failure();
  }

  return success();
}

void cir::printTryHandlerRegions(
    mlir::OpAsmPrinter &printer, cir::TryOp op,
    mlir::MutableArrayRef<mlir::Region> handlerRegions,
    mlir::ArrayAttr handlerTypes) {
  if (!handlerTypes)
    return;

  for (const auto [typeIdx, typeAttr] : llvm::enumerate(handlerTypes)) {
    if (typeIdx)
      printer << " ";

    if (mlir::isa<cir::CatchAllAttr>(typeAttr)) {
      printer << "catch all ";
    } else if (mlir::isa<cir::UnwindAttr>(typeAttr)) {
      printer << "unwind ";
    } else if (auto filterAttr = mlir::dyn_cast<cir::EhFilterAttr>(typeAttr)) {
      printer << "filter [";
      llvm::interleaveComma(
          filterAttr.getPermittedTypes(), printer,
          [&](mlir::Attribute sym) { printer.printAttribute(sym); });
      printer << "] ";
    } else if (mlir::isa<cir::EhUnexpectedAttr>(typeAttr)) {
      printer << "unexpected ";
    } else {
      printer << "catch [type ";
      printer.printAttribute(typeAttr);
      printer << "] ";
    }

    // Print the handler region's !cir.eh_token block argument.
    mlir::Region &region = handlerRegions[typeIdx];
    if (!region.empty() && region.front().getNumArguments() > 0) {
      printer << "(";
      printer.printRegionArgument(region.front().getArgument(0));
      printer << ") ";
    }

    printer.printRegion(region,
                        /*printEntryBLockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
}

mlir::ParseResult cir::parseTryHandlerRegions(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<std::unique_ptr<mlir::Region>> &handlerRegions,
    mlir::ArrayAttr &handlerTypes) {

  auto parseCheckedCatcherRegion = [&]() -> mlir::ParseResult {
    handlerRegions.emplace_back(new mlir::Region);

    mlir::Region &currRegion = *handlerRegions.back();

    // Parse the required region argument: (%eh_token : !cir.eh_token)
    llvm::SmallVector<mlir::OpAsmParser::Argument> regionArgs;
    if (parser.parseLParen())
      return failure();
    mlir::OpAsmParser::Argument arg;
    if (parser.parseArgument(arg, /*allowType=*/true))
      return failure();
    regionArgs.push_back(arg);
    if (parser.parseRParen())
      return failure();

    mlir::SMLoc regionLoc = parser.getCurrentLocation();
    if (parser.parseRegion(currRegion, regionArgs)) {
      handlerRegions.clear();
      return failure();
    }

    if (currRegion.empty())
      return parser.emitError(regionLoc, "handler region shall not be empty");

    if (!(currRegion.back().mightHaveTerminator() &&
          currRegion.back().getTerminator()))
      return parser.emitError(
          regionLoc, "blocks are expected to be explicitly terminated");

    return success();
  };

  bool hasCatchAll = false;
  llvm::SmallVector<mlir::Attribute, 4> catcherAttrs;
  while (parser.parseOptionalKeyword("catch").succeeded()) {
    bool hasLSquare = parser.parseOptionalLSquare().succeeded();

    llvm::StringRef attrStr;
    if (parser.parseOptionalKeyword(&attrStr, {"all", "type"}).failed())
      return parser.emitError(parser.getCurrentLocation(),
                              "expected 'all' or 'type' keyword");

    bool isCatchAll = attrStr == "all";
    if (isCatchAll) {
      if (hasCatchAll)
        return parser.emitError(parser.getCurrentLocation(),
                                "can't have more than one catch all");
      hasCatchAll = true;
    }

    mlir::Attribute exceptionRTTIAttr;
    if (!isCatchAll && parser.parseAttribute(exceptionRTTIAttr).failed())
      return parser.emitError(parser.getCurrentLocation(),
                              "expected valid RTTI info attribute");

    catcherAttrs.push_back(isCatchAll
                               ? cir::CatchAllAttr::get(parser.getContext())
                               : exceptionRTTIAttr);

    if (hasLSquare && isCatchAll)
      return parser.emitError(parser.getCurrentLocation(),
                              "catch all dosen't need RTTI info attribute");

    if (hasLSquare && parser.parseRSquare().failed())
      return parser.emitError(parser.getCurrentLocation(),
                              "expected `]` after RTTI info attribute");

    if (parseCheckedCatcherRegion().failed())
      return mlir::failure();
  }

  // A filter handler carries the type info symbols permitted by the enclosing
  // function's dynamic exception specification. TryOp::verifyRegions enforces
  // that it is paired with an unexpected handler and that the two stand alone.
  if (parser.parseOptionalKeyword("filter").succeeded()) {
    mlir::SMLoc filterLoc = parser.getCurrentLocation();
    llvm::SmallVector<mlir::Attribute, 4> permittedTypes;
    auto parsePermittedType = [&]() -> mlir::ParseResult {
      mlir::SMLoc typeLoc = parser.getCurrentLocation();
      mlir::Attribute rtti;
      if (parser.parseAttribute(rtti).failed())
        return mlir::failure();
      if (!mlir::isa<cir::GlobalViewAttr>(rtti))
        return parser.emitError(typeLoc, "expected a type info symbol naming a "
                                         "permitted exception type");
      permittedTypes.push_back(rtti);
      return mlir::success();
    };
    if (parser
            .parseCommaSeparatedList(mlir::OpAsmParser::Delimiter::Square,
                                     parsePermittedType)
            .failed())
      return mlir::failure();

    auto filterAttr = cir::EhFilterAttr::getChecked(
        [&]() { return parser.emitError(filterLoc); }, parser.getContext(),
        parser.getBuilder().getArrayAttr(permittedTypes));
    if (!filterAttr)
      return mlir::failure();
    catcherAttrs.push_back(filterAttr);
    if (parseCheckedCatcherRegion().failed())
      return mlir::failure();
  }

  if (parser.parseOptionalKeyword("unexpected").succeeded()) {
    catcherAttrs.push_back(cir::EhUnexpectedAttr::get(parser.getContext()));
    if (parseCheckedCatcherRegion().failed())
      return mlir::failure();
  }

  if (parser.parseOptionalKeyword("unwind").succeeded()) {
    if (hasCatchAll)
      return parser.emitError(parser.getCurrentLocation(),
                              "unwind can't be used with catch all");

    catcherAttrs.push_back(cir::UnwindAttr::get(parser.getContext()));
    if (parseCheckedCatcherRegion().failed())
      return mlir::failure();
  }

  handlerTypes = parser.getBuilder().getArrayAttr(catcherAttrs);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// EhTypeIdOp
//===----------------------------------------------------------------------===//

LogicalResult
cir::EhTypeIdOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = symbolTable.lookupNearestSymbolFrom(*this, getTypeSymAttr());
  if (!isa_and_nonnull<GlobalOp>(op))
    return emitOpError("'")
           << getTypeSym() << "' does not reference a valid cir.global";
  return success();
}

//===----------------------------------------------------------------------===//
// InitCatchParamOp
//===----------------------------------------------------------------------===//

/// For the reference_to_pointer and reference_to_record_pointer kinds, emit an
/// error on \p op unless \p paramAddrType is the type of the address of a
/// reference to a pointer.  Any other \p kind is accepted.
static LogicalResult verifyCatchParamAddr(mlir::Operation *op,
                                          cir::InitCatchKind kind,
                                          cir::PointerType paramAddrType) {
  if (kind != cir::InitCatchKind::ReferenceToPointer &&
      kind != cir::InitCatchKind::ReferenceToRecordPointer)
    return success();

  auto refType = mlir::dyn_cast<cir::PointerType>(paramAddrType.getPointee());
  if (!refType || !mlir::isa<cir::PointerType>(refType.getPointee()))
    return op->emitOpError("'")
           << cir::stringifyInitCatchKind(kind)
           << "' requires 'param_addr' to be the address of a reference to a "
              "pointer";
  return success();
}

LogicalResult cir::InitCatchParamOp::verify() {
  return verifyCatchParamAddr(*this, getKind(), getParamAddr().getType());
}

//===----------------------------------------------------------------------===//
// ConstructCatchParamOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ConstructCatchParamOp::verify() {
  return verifyCatchParamAddr(*this, getKind(), getParamAddr().getType());
}

LogicalResult cir::ConstructCatchParamOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  auto copyFnAttr = getCopyFnAttr();
  if (!copyFnAttr)
    return success();
  auto fn =
      symbolTable.lookupNearestSymbolFrom<cir::FuncOp>(*this, getCopyFnAttr());
  if (!fn)
    return emitOpError("'")
           << *getCopyFn() << "' does not reference a valid cir.func";

  if (!fn->hasAttr(cir::CIRDialect::getCatchCopyThunkAttrName()))
    return emitOpError("catch-init copy_fn must be tagged with the ")
           << cir::CIRDialect::getCatchCopyThunkAttrName() << " attribute";

  cir::FuncType fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != 2 || !fnType.hasVoidReturn())
    return emitOpError("catch-init copy_fn must take two pointer arguments and "
                       "return void");

  if (fnType.getInput(0) != getParamAddr().getType())
    return emitOpError("first argument of catch-init copy_fn must match the "
                       "type of 'param_addr'");

  if (fnType.getInput(1) != getParamAddr().getType())
    return emitOpError(
        "second argument of catch-init copy_fn must be a pointer "
        "to the catch type");

  return success();
}

//===----------------------------------------------------------------------===//
// EhDispatchOp
//===----------------------------------------------------------------------===//

LogicalResult cir::EhDispatchOp::verify() {
  mlir::ArrayAttr handlerTypes = getCatchTypesAttr();
  if (!handlerTypes)
    return success();

  bool hasFilter = false;
  for (mlir::Attribute typeAttr : handlerTypes) {
    if (!mlir::isa<cir::EhFilterAttr>(typeAttr))
      continue;
    if (hasFilter)
      return emitOpError("can't have more than one 'filter' handler");
    hasFilter = true;
  }

  // Unlike the other handlers, a filter does not take the place of the default
  // destination. Its destination is taken when the exception is not permitted
  // by the specification, and the default 'unwind' destination is taken when
  // it is. A catch-all default would leave the filter unreachable.
  if (hasFilter && getDefaultIsCatchAll())
    return emitOpError(
        "'filter' handler requires an 'unwind' default destination");

  return success();
}

ParseResult cir::parseEhDispatchDestinations(
    OpAsmParser &parser, mlir::ArrayAttr &catchTypes,
    SmallVectorImpl<Block *> &catchDestinations, Block *&defaultDestination,
    mlir::UnitAttr &defaultIsCatchAll) {
  // Parse: [ ... ]
  if (parser.parseLSquare())
    return failure();

  SmallVector<Attribute> handlerTypes;
  bool hasCatchAll = false;
  bool hasUnwind = false;

  // Parse handler list.
  auto parseHandler = [&]() -> ParseResult {
    // Check for 'catch_all' or 'unwind' keywords.
    if (succeeded(parser.parseOptionalKeyword("catch_all"))) {
      if (hasCatchAll)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'catch_all' handler");
      if (hasUnwind)
        return parser.emitError(parser.getCurrentLocation(),
                                "cannot have both 'catch_all' and 'unwind'");
      hasCatchAll = true;

      if (parser.parseColon().failed())
        return failure();

      if (parser.parseSuccessor(defaultDestination).failed())
        return failure();

      return success();
    }

    if (succeeded(parser.parseOptionalKeyword("unwind"))) {
      if (hasUnwind)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'unwind' handler");
      if (hasCatchAll)
        return parser.emitError(parser.getCurrentLocation(),
                                "cannot have both 'catch_all' and 'unwind'");
      hasUnwind = true;

      if (parser.parseColon().failed())
        return failure();

      if (parser.parseSuccessor(defaultDestination).failed())
        return failure();
      return success();
    }

    // A filter handler carries the type info symbols permitted by the
    // enclosing function's dynamic exception specification. Its destination is
    // taken when the exception is *not* one of those types.
    if (succeeded(parser.parseOptionalKeyword("filter"))) {
      SMLoc filterLoc = parser.getCurrentLocation();
      SmallVector<Attribute, 4> permittedTypes;
      auto parsePermittedType = [&]() -> ParseResult {
        mlir::SMLoc typeLoc = parser.getCurrentLocation();
        mlir::Attribute rtti;
        if (parser.parseAttribute(rtti).failed())
          return failure();
        if (!mlir::isa<cir::GlobalViewAttr>(rtti))
          return parser.emitError(typeLoc,
                                  "expected a type info symbol naming a "
                                  "permitted exception type");
        permittedTypes.push_back(rtti);
        return success();
      };
      if (parser
              .parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                       parsePermittedType)
              .failed())
        return failure();

      auto filterAttr = cir::EhFilterAttr::getChecked(
          [&]() { return parser.emitError(filterLoc); }, parser.getContext(),
          parser.getBuilder().getArrayAttr(permittedTypes));
      if (!filterAttr)
        return failure();
      handlerTypes.push_back(filterAttr);

      if (parser.parseColon().failed())
        return failure();

      Block *dest;
      if (parser.parseSuccessor(dest).failed())
        return failure();
      catchDestinations.push_back(dest);
      return success();
    }

    // Otherwise, expect 'catch(<attr> : <type>) : ^block'.
    // The 'catch(...)' wrapper allows the attribute to include its type
    // without conflicting with the ':' used for the block destination.
    if (parser.parseKeyword("catch").failed())
      return failure();

    if (parser.parseLParen().failed())
      return failure();

    mlir::Attribute catchTypeAttr;
    if (parser.parseAttribute(catchTypeAttr).failed())
      return failure();
    handlerTypes.push_back(catchTypeAttr);

    if (parser.parseRParen().failed())
      return failure();

    if (parser.parseColon().failed())
      return failure();

    Block *dest;
    if (parser.parseSuccessor(dest).failed())
      return failure();
    catchDestinations.push_back(dest);
    return success();
  };

  if (parser.parseCommaSeparatedList(parseHandler).failed())
    return failure();

  if (parser.parseRSquare().failed())
    return failure();

  // Verify we have catch_all or unwind.
  if (!hasCatchAll && !hasUnwind)
    return parser.emitError(parser.getCurrentLocation(),
                            "must have either 'catch_all' or 'unwind' handler");

  // Add attributes and successors.
  if (!handlerTypes.empty())
    catchTypes = parser.getBuilder().getArrayAttr(handlerTypes);

  if (hasCatchAll)
    defaultIsCatchAll = parser.getBuilder().getUnitAttr();

  return success();
}

void cir::printEhDispatchDestinations(OpAsmPrinter &p, cir::EhDispatchOp op,
                                      mlir::ArrayAttr catchTypes,
                                      SuccessorRange catchDestinations,
                                      Block *defaultDestination,
                                      mlir::UnitAttr defaultIsCatchAll) {
  p << " [";
  p.printNewline();

  // If we have at least one catch type, print them.
  if (catchTypes) {
    // Print type handlers using 'catch(<attr>) : ^block' syntax.
    llvm::interleave(
        llvm::zip(catchTypes, catchDestinations),
        [&](auto i) {
          mlir::Attribute typeAttr = std::get<0>(i);
          if (auto filterAttr = mlir::dyn_cast<cir::EhFilterAttr>(typeAttr)) {
            p << "  filter(";
            llvm::interleaveComma(
                filterAttr.getPermittedTypes(), p,
                [&](mlir::Attribute sym) { p.printAttribute(sym); });
          } else {
            p << "  catch(";
            p.printAttribute(typeAttr);
          }
          p << ") : ";
          p.printSuccessor(std::get<1>(i));
        },
        [&] {
          p << ',';
          p.printNewline();
        });

    p << ", ";
    p.printNewline();
  }

  // Print catch_all or unwind handler.
  if (defaultIsCatchAll)
    p << "  catch_all : ";
  else
    p << "  unwind : ";
  p.printSuccessor(defaultDestination);
  p.printNewline();

  p << "]";
}

//===----------------------------------------------------------------------===//
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsEH.cpp.inc"
