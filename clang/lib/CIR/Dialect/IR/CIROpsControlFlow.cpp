//===- CIROpsControlFlow.cpp - CIR ops in partition `ControlFlow` ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR ops in partition `ControlFlow`: the hand-written
// parts first, then the TableGen-generated definitions.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIROpsControlFlow.h"
#include "clang/CIR/Dialect/IR/CIROpsBuiltins.h"
#include "clang/CIR/Dialect/IR/CIROpsClassMembers.h"
#include "clang/CIR/Dialect/IR/CIROpsCore.h"
#include "clang/CIR/Dialect/IR/CIROpsCoroutine.h"
#include "clang/CIR/Dialect/IR/CIROpsEH.h"
#include "clang/CIR/Dialect/IR/CIROpsSymbols.h"

#include "CIROpDefinitions.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace cir;

// Check if a region's termination omission is valid and, if so, creates and
// inserts the omitted terminator into the region.
LogicalResult cir::ensureRegionTerm(OpAsmParser &parser, Region &region,
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
  cir::YieldOp::create(builder, eLoc);
  return success();
}

// True if the region's terminator should be omitted.
bool cir::omitRegionTerm(mlir::Region &r) {
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

mlir::ParseResult cir::parseOmittedTerminatorRegion(mlir::OpAsmParser &parser,
                                                    mlir::Region &region) {
  auto regionLoc = parser.getCurrentLocation();
  if (parser.parseRegion(region))
    return failure();
  if (ensureRegionTerm(parser, region, regionLoc).failed())
    return failure();
  return success();
}

void cir::printOmittedTerminatorRegion(mlir::OpAsmPrinter &printer,
                                       cir::ScopeOp &op, mlir::Region &region) {
  printer.printRegion(region,
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/!omitRegionTerm(region));
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

  // Parent is a loop: condition may branch to the body, or on the false edge
  // to the per-iteration cleanup region if present, otherwise to the parent op.
  if (auto loopOp = dyn_cast<LoopOpInterface>(getOperation()->getParentOp())) {
    regions.emplace_back(&loopOp.getBody());
    if (mlir::Region *cleanup = loopOp.maybeGetCleanup())
      regions.emplace_back(cleanup);
    else
      regions.emplace_back(getOperation());
    return;
  }

  // Parent is an await: condition may branch to resume or suspend regions.
  auto await = cast<AwaitOp>(getOperation()->getParentOp());
  regions.emplace_back(&await.getResume());
  regions.emplace_back(&await.getSuspend());
}

MutableOperandRange
cir::ConditionOp::getMutableSuccessorOperands(RegionSuccessor point) {
  // No values are yielded to the successor region.
  return MutableOperandRange(getOperation(), 0, 0);
}

MutableOperandRange
cir::ResumeOp::getMutableSuccessorOperands(RegionSuccessor point) {
  // The eh_token operand is not modeled as forwarded to any successor.
  return MutableOperandRange(getOperation(), 0, 0);
}

LogicalResult cir::ConditionOp::verify() {
  if (!isa<LoopOpInterface, AwaitOp>(getOperation()->getParentOp()))
    return emitOpError("condition must be within a conditional region");
  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp & ForOp
//===----------------------------------------------------------------------===//

template <typename LoopOpTy>
static LogicalResult verifyLoopCleanup(LoopOpTy op) {
  std::optional<cir::CleanupKind> cleanupKind = op.getCleanupKind();

  // The cleanup kind attribute must be present exactly when a (non-empty)
  // cleanup region is present.
  if (cleanupKind.has_value() == op.getCleanup().empty())
    return op.emitOpError("cleanup kind must be present if and only if the "
                          "cleanup region is non-empty");

  // A loop's per-iteration cleanup runs on every normal exit edge (loop exit,
  // end of iteration, break/continue), so an EH-only cleanup is meaningless.
  // Only 'normal' (exceptions disabled) and 'all' (normal + EH unwind) apply.
  if (cleanupKind == cir::CleanupKind::EH)
    return op.emitOpError("loop cleanup kind must be 'normal' or 'all', "
                          "not 'eh'");

  return success();
}

LogicalResult cir::WhileOp::verify() { return verifyLoopCleanup(*this); }

LogicalResult cir::ForOp::verify() { return verifyLoopCleanup(*this); }

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

  p.printOptionalAttrDict(
      getOperation()->getDiscardableAttrDictionary().getValue());
}

/// Default callback for IfOp builders.
void cir::buildTerminatedBody(OpBuilder &builder, Location loc) {
  // add cir.yield to end of the block
  cir::YieldOp::create(builder, loc);
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
    regions.emplace_back(getOperation());
    return;
  }

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->getElseRegion();
  if (elseRegion->empty())
    elseRegion = nullptr;

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getThenRegion()));
  if (elseRegion)
    regions.push_back(RegionSuccessor(elseRegion));
  else
    regions.emplace_back(getOperation());
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

LogicalResult cir::ScopeOp::fold(FoldAdaptor /*adaptor*/,
                                 SmallVectorImpl<OpFoldResult> &results) {
  // Only fold "trivial" scopes: a single block containing only a `cir.yield`.
  if (!getRegion().hasOneBlock())
    return failure();
  Block &block = getRegion().front();
  if (block.getOperations().size() != 1)
    return failure();

  auto yield = dyn_cast<cir::YieldOp>(block.front());
  if (!yield)
    return failure();

  // Only fold when the scope produces a value.
  if (getNumResults() != 1 || yield.getNumOperands() != 1)
    return failure();

  results.push_back(yield.getOperand(0));
  return success();
}

//===----------------------------------------------------------------------===//
// CleanupScopeOp
//===----------------------------------------------------------------------===//

static bool isRedundantBeforeReturn(mlir::Region &cleanupRegion) {
  for (mlir::Block &block : cleanupRegion) {
    for (mlir::Operation &op : block) {
      if (isa<cir::YieldOp, cir::LifetimeEndOp, cir::StackRestoreOp>(op))
        continue;
      // The stack restore reloads the pointer saved before the VLA.
      auto loadOp = dyn_cast<cir::LoadOp>(op);
      if (loadOp && loadOp.getResult().hasOneUse() &&
          isa<cir::StackRestoreOp>(*loadOp.getResult().getUsers().begin()))
        continue;
      return false;
    }
  }
  return true;
}

LogicalResult cir::CleanupScopeOp::verify() {
  // If the cleanup contains a musttail call, it must be a cleanup that can be
  // skipped on return (such as a lifetime end or a stack restore). Other
  // cleanups must never contain musttail calls.
  cir::CallOp mustTailCall;
  getBodyRegion().walk([&](cir::CallOp callOp) {
    if (!callOp.getMusttail())
      return WalkResult::advance();
    mustTailCall = callOp;
    return WalkResult::interrupt();
  });
  if (!mustTailCall)
    return success();

  if (isRedundantBeforeReturn(getCleanupRegion()))
    return success();

  InFlightDiagnostic diag =
      emitOpError("cleanup is not redundant before a return, so it cannot be "
                  "skipped by a musttail call");
  diag.attachNote(mustTailCall.getLoc()) << "musttail call is here";
  return diag;
}

void cir::CleanupScopeOp::getSuccessorRegions(
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // The cleanup can also run from the middle of the body, on an unwind out of
  // a call or on a cir.return, cir.break, cir.continue or cir.goto. None of
  // these is a branch point, so the cleanup region is a successor of the
  // parent and of every body exit to stand in for them.
  if (point.isParent()) {
    regions.reserve(regions.size() + 2);
    regions.emplace_back(&getBodyRegion());
    regions.emplace_back(&getCleanupRegion());
    return;
  }

  mlir::Operation *term = point.getTerminatorPredecessorOrNull();
  if (term->getParentRegion() == &getCleanupRegion()) {
    regions.emplace_back(getOperation());
    return;
  }

  regions.emplace_back(&getCleanupRegion());

  // cir.resume unwinds out of the body, and every other terminator is a
  // normal exit. An exit that the cleanup kind does not cover can also leave
  // the operation directly.
  cir::CleanupKindAttr kind = getCleanupKindAttr();
  if (isa<cir::ResumeOp>(term) ? !kind.isEH() : !kind.isNormal())
    regions.emplace_back(getOperation());
}

LogicalResult cir::CleanupScopeOp::canonicalize(CleanupScopeOp op,
                                                PatternRewriter &rewriter) {
  auto isRegionTrivial = [](Region &region) {
    assert(!region.empty() && "CleanupScopeOp regions must not be empty");
    if (!region.hasOneBlock())
      return false;
    Block &block = llvm::getSingleElement(region);
    return llvm::hasSingleElement(block) &&
           isa<cir::YieldOp>(llvm::getSingleElement(block));
  };

  Region &body = op.getBodyRegion();
  Region &cleanup = op.getCleanupRegion();

  // An EH-only cleanup scope with an empty body can never trigger its cleanup
  // region — there are no operations in the body that could throw. Erase it.
  if (op.getCleanupKind() == CleanupKind::EH && isRegionTrivial(body)) {
    rewriter.eraseOp(op);
    return success();
  }

  // A cleanup scope with a trivial cleanup region has no cleanup to perform.
  // Inline the body into the parent block and erase the scope.
  if (!isRegionTrivial(cleanup) || !body.hasOneBlock())
    return failure();

  Block &bodyBlock = body.front();
  if (!isa<cir::YieldOp>(bodyBlock.getTerminator()))
    return failure();

  Operation *yield = bodyBlock.getTerminator();
  rewriter.inlineBlockBefore(&bodyBlock, op);
  rewriter.eraseOp(yield);
  rewriter.eraseOp(op);
  return success();
}

void cir::CleanupScopeOp::build(
    OpBuilder &builder, OperationState &result, CleanupKind cleanupKind,
    function_ref<void(OpBuilder &, Location)> bodyBuilder,
    function_ref<void(OpBuilder &, Location)> cleanupBuilder) {
  result.addAttribute(getCleanupKindAttrName(result.name),
                      CleanupKindAttr::get(builder.getContext(), cleanupKind));

  OpBuilder::InsertionGuard guard(builder);

  // Build body region.
  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion);
  if (bodyBuilder)
    bodyBuilder(builder, result.location);

  // Build cleanup region.
  Region *cleanupRegion = result.addRegion();
  builder.createBlock(cleanupRegion);
  if (cleanupBuilder)
    cleanupBuilder(builder, result.location);
}

//===----------------------------------------------------------------------===//
// BrOp
//===----------------------------------------------------------------------===//

/// Merges blocks connected by a unique unconditional branch.
///
///   ^bb0:              ^bb0:
///     ...                ...
///     cir.br ^bb1  =>    ...
///   ^bb1:                cir.return
///     ...
///     cir.return
LogicalResult cir::BrOp::canonicalize(BrOp op, PatternRewriter &rewriter) {
  Block *src = op->getBlock();
  Block *dst = op.getDest();

  // Do not fold self-loops.
  if (src == dst)
    return failure();

  // Only merge when this is the unique edge between the blocks.
  if (src->getNumSuccessors() != 1 || dst->getSinglePredecessor() != src)
    return failure();

  // Don't merge blocks that start with LabelOp or IndirectBrOp.
  // This is to avoid merging blocks that have an indirect predecessor.
  if (isa<cir::LabelOp, cir::IndirectBrOp>(dst->front()))
    return failure();

  // An operand that is an argument of the destination itself (possible in an
  // unreachable cycle) would be replaced by itself, leaving its other uses
  // dangling once the destination is erased.
  if (llvm::any_of(op.getDestOperands(), [&](Value operand) {
        auto arg = dyn_cast<BlockArgument>(operand);
        return arg && arg.getOwner() == dst;
      }))
    return failure();

  // Copy the operands out: erasing the branch frees its operand storage.
  SmallVector<Value> operands(op.getDestOperands());
  rewriter.eraseOp(op);
  rewriter.mergeBlocks(dst, src, operands);
  return success();
}

mlir::SuccessorOperands cir::BrOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return mlir::SuccessorOperands(getDestOperandsMutable());
}

Block *cir::BrOp::getSuccessorForOperands(ArrayRef<Attribute>) {
  return getDest();
}

//===----------------------------------------------------------------------===//
// IndirectBrCondOp
//===----------------------------------------------------------------------===//

mlir::SuccessorOperands
cir::IndirectBrOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return mlir::SuccessorOperands(getSuccOperandsMutable()[index]);
}

ParseResult cir::parseIndirectBrOpSucessors(
    OpAsmParser &parser, Type &flagType,
    SmallVectorImpl<Block *> &succOperandBlocks,
    SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &succOperands,
    SmallVectorImpl<SmallVector<Type>> &succOperandsTypes) {
  if (failed(parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Square,
          [&]() {
            Block *destination = nullptr;
            SmallVector<OpAsmParser::UnresolvedOperand> operands;
            SmallVector<Type> operandTypes;

            if (parser.parseSuccessor(destination).failed())
              return failure();

            if (succeeded(parser.parseOptionalLParen())) {
              if (failed(parser.parseOperandList(
                      operands, OpAsmParser::Delimiter::None)) ||
                  failed(parser.parseColonTypeList(operandTypes)) ||
                  failed(parser.parseRParen()))
                return failure();
            }
            succOperandBlocks.push_back(destination);
            succOperands.emplace_back(operands);
            succOperandsTypes.emplace_back(operandTypes);
            return success();
          },
          "successor blocks")))
    return failure();
  return success();
}

void cir::printIndirectBrOpSucessors(OpAsmPrinter &p, cir::IndirectBrOp op,
                                     Type flagType, SuccessorRange succs,
                                     OperandRangeRange succOperands,
                                     const TypeRangeRange &succOperandsTypes) {
  p << "[";
  llvm::interleave(
      llvm::zip(succs, succOperands),
      [&](auto i) {
        p.printNewline();
        p.printSuccessorAndUseList(std::get<0>(i), std::get<1>(i));
      },
      [&] { p << ','; });
  if (!succOperands.empty())
    p.printNewline();
  p << "]";
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

SuccessorOperands cir::SwitchFlatOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  if (index == 0)
    return SuccessorOperands(getDefaultOperandsMutable());
  return SuccessorOperands(getCaseOperandsMutable()[index - 1]);
}

Block *
cir::SwitchFlatOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  auto cond = dyn_cast_if_present<cir::IntAttr>(operands.front());
  if (!cond)
    return nullptr;
  for (auto [value, dest] : llvm::zip(getCaseValues(), getCaseDestinations())) {
    const APInt &caseValue = cast<cir::IntAttr>(value).getValue();
    if (caseValue == cond.getValue())
      return dest;
  }
  return getDefaultDestination();
}

/// <cases> ::= `[` (case (`,` case )* )? `]`
/// <case>  ::= integer `:` bb-id (`(` ssa-use-and-type-list `)`)?
ParseResult cir::parseSwitchFlatOpCases(
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

void cir::printSwitchFlatOpCases(OpAsmPrinter &p, cir::SwitchFlatOp op,
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
// TernaryOp
//===----------------------------------------------------------------------===//

void cir::TernaryOp::build(
    OpBuilder &builder, OperationState &result, Value cond,
    function_ref<void(OpBuilder &, Location)> trueBuilder,
    function_ref<void(OpBuilder &, Location)> falseBuilder) {
  result.addOperands(cond);
  OpBuilder::InsertionGuard guard(builder);
  Region *trueRegion = result.addRegion();
  builder.createBlock(trueRegion);
  trueBuilder(builder, result.location);
  Region *falseRegion = result.addRegion();
  builder.createBlock(falseRegion);
  falseBuilder(builder, result.location);

  // Get result type from whichever branch has a yield (the other may have
  // unreachable from a throw expression)
  cir::YieldOp yield;
  if (trueRegion->back().mightHaveTerminator())
    yield = dyn_cast_or_null<cir::YieldOp>(trueRegion->back().getTerminator());
  if (!yield && falseRegion->back().mightHaveTerminator())
    yield = dyn_cast_or_null<cir::YieldOp>(falseRegion->back().getTerminator());

  assert((!yield || yield.getNumOperands() <= 1) &&
         "expected zero or one result type");
  if (yield && yield.getNumOperands() == 1)
    result.addTypes(TypeRange{yield.getOperandTypes().front()});
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::SelectOp::fold(FoldAdaptor adaptor) {
  if (auto condition =
          mlir::dyn_cast_if_present<cir::BoolAttr>(adaptor.getCondition()))
    return condition.getValue() ? getTrueValue() : getFalseValue();

  // cir.select if %0 then x else x -> x
  mlir::Attribute trueValue = adaptor.getTrueValue();
  mlir::Attribute falseValue = adaptor.getFalseValue();
  if (trueValue == falseValue)
    return trueValue;
  if (getTrueValue() == getFalseValue())
    return getTrueValue();

  return {};
}

LogicalResult cir::SelectOp::verify() {
  // AllTypesMatch already guarantees trueVal and falseVal have matching types.
  auto condTy = dyn_cast<cir::VectorType>(getCondition().getType());

  // If condition is not a vector, no further checks are needed.
  if (!condTy)
    return success();

  // When condition is a vector, both other operands must also be vectors.
  if (!isa<cir::VectorType>(getTrueValue().getType()) ||
      !isa<cir::VectorType>(getFalseValue().getType())) {
    return emitOpError()
           << "expected both true and false operands to be vector types "
              "when the condition is a vector boolean type";
  }

  return success();
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
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsControlFlow.cpp.inc"
