#include "inter/Transforms/Passes.h"

#include "inter/Dialect/Inter/IR/XW.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include <optional>

namespace inter {
#define GEN_PASS_DEF_NARROWINTEGERRANGES
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;
using namespace mlir::dataflow;

namespace {

enum class NarrowKind { None, Signed, Unsigned, Both };

static NarrowKind getNarrowKind(const ConstantIntRanges &range,
                                unsigned width) {
  unsigned sourceWidth = range.smin().getBitWidth();
  if (sourceWidth <= width)
    return NarrowKind::None;
  unsigned removedWidth = sourceWidth - width;
  bool signedRange = range.smin().getNumSignBits() >= removedWidth + 1 &&
                     range.smax().getNumSignBits() >= removedWidth + 1;
  bool unsignedRange = range.umin().countLeadingZeros() >= removedWidth &&
                       range.umax().countLeadingZeros() >= removedWidth;
  if (signedRange && unsignedRange)
    return NarrowKind::Both;
  if (signedRange)
    return NarrowKind::Signed;
  if (unsignedRange)
    return NarrowKind::Unsigned;
  return NarrowKind::None;
}

static NarrowKind mergeKinds(NarrowKind lhs, NarrowKind rhs) {
  if (lhs == NarrowKind::None || rhs == NarrowKind::None)
    return NarrowKind::None;
  if (lhs == NarrowKind::Both)
    return rhs;
  if (rhs == NarrowKind::Both)
    return lhs;
  return lhs == rhs ? lhs : NarrowKind::None;
}

static std::optional<ConstantIntRanges> getRange(DataFlowSolver &solver,
                                                 Value value) {
  const IntegerValueRangeLattice *lattice =
      solver.lookupState<IntegerValueRangeLattice>(value);
  if (!lattice || lattice->getValue().isUninitialized())
    return std::nullopt;
  return lattice->getValue().getValue();
}

static NarrowKind getCommonNarrowKind(DataFlowSolver &solver,
                                      ValueRange values) {
  NarrowKind kind = NarrowKind::Both;
  for (Value value : values) {
    std::optional<ConstantIntRanges> range = getRange(solver, value);
    if (!range)
      return NarrowKind::None;
    kind = mergeKinds(kind, getNarrowKind(*range, 32));
  }
  return kind;
}

static NarrowKind getRequiredKind(xw::BinaryKind kind) {
  if (kind == xw::BinaryKind::DivSI || kind == xw::BinaryKind::RemSI ||
      kind == xw::BinaryKind::ShRSI)
    return NarrowKind::Signed;
  if (kind == xw::BinaryKind::DivUI || kind == xw::BinaryKind::RemUI ||
      kind == xw::BinaryKind::ShRUI)
    return NarrowKind::Unsigned;
  return NarrowKind::Both;
}

static bool isSupported(xw::BinaryKind kind) {
  return kind != xw::BinaryKind::MulHUI;
}

static bool hasNarrowShiftAmount(DataFlowSolver &solver,
                                 xw::BinaryOp operation) {
  if (operation.getKind() != xw::BinaryKind::ShLI &&
      operation.getKind() != xw::BinaryKind::ShRUI &&
      operation.getKind() != xw::BinaryKind::ShRSI)
    return true;
  std::optional<ConstantIntRanges> range = getRange(solver, operation.getRhs());
  return range && range->umax().ult(32);
}

static Value createNarrowValue(OpBuilder &builder, Location location,
                               Value value) {
  Type i32 = builder.getI32Type();
  if (value.getType() == i32)
    return value;
  if (xw::ConstantOp constant = value.getDefiningOp<xw::ConstantOp>()) {
    if (IntegerAttr attribute = dyn_cast<IntegerAttr>(constant.getValue()))
      return xw::ConstantOp::create(
                 builder, location, i32,
                 builder.getIntegerAttr(i32, attribute.getValue().trunc(32)))
          .getResult();
  }
  if (xw::CastOp cast = value.getDefiningOp<xw::CastOp>()) {
    if (cast.getKind() == xw::CastKind::IntConvert &&
        cast.getSource().getType() == i32)
      return cast.getSource();
  }
  return xw::CastOp::create(builder, location, i32, xw::CastKind::IntConvert,
                            value, DictionaryAttr())
      .getResult();
}

static Value createExtendedValue(OpBuilder &builder, Location location,
                                 Value value, Type type, NarrowKind kind) {
  NamedAttrList policy;
  xw::CastExtension extension = kind == NarrowKind::Signed
                                    ? xw::CastExtension::Sign
                                    : xw::CastExtension::Zero;
  policy.set("extension",
             xw::CastExtensionPolicyAttr::get(builder.getContext(), extension));
  return xw::CastOp::create(builder, location, type, xw::CastKind::IntConvert,
                            value, builder.getDictionaryAttr(policy))
      .getResult();
}

struct BinaryPlan {
  xw::BinaryOp operation;
  NarrowKind kind;
};

static void collectBinaryPlans(func::FuncOp function, DataFlowSolver &solver,
                               SmallVectorImpl<BinaryPlan> &plans) {
  function.walk([&](xw::BinaryOp operation) {
    if (!operation.getType().isInteger(64) ||
        !operation.getLhs().getType().isInteger(64) ||
        !operation.getRhs().getType().isInteger(64) ||
        !isSupported(operation.getKind()) ||
        !hasNarrowShiftAmount(solver, operation))
      return;
    SmallVector<Value, 3> values = {operation.getLhs(), operation.getRhs(),
                                    operation.getResult()};
    NarrowKind kind = getCommonNarrowKind(solver, values);
    kind = mergeKinds(kind, getRequiredKind(operation.getKind()));
    if (kind != NarrowKind::None)
      plans.push_back({operation, kind});
  });
}

static void applyBinaryPlans(ArrayRef<BinaryPlan> plans) {
  for (const BinaryPlan &plan : plans) {
    xw::BinaryOp operation = plan.operation;
    OpBuilder builder(operation);
    Value lhs =
        createNarrowValue(builder, operation.getLoc(), operation.getLhs());
    Value rhs =
        createNarrowValue(builder, operation.getLoc(), operation.getRhs());
    xw::BinaryOp narrowed =
        xw::BinaryOp::create(builder, operation.getLoc(), builder.getI32Type(),
                             operation.getKind(), lhs, rhs);
    narrowed.setOverflowFlags(operation.getOverflowFlags());
    Value extended =
        createExtendedValue(builder, operation.getLoc(), narrowed.getResult(),
                            operation.getType(), plan.kind);
    operation.getResult().replaceAllUsesWith(extended);
    operation.erase();
  }
}

struct LoopPlan {
  scf::ForOp operation;
  NarrowKind kind;
};

static void collectLoopPlans(func::FuncOp function, DataFlowSolver &solver,
                             SmallVectorImpl<LoopPlan> &plans) {
  function.walk([&](scf::ForOp operation) {
    if (!operation.getInductionVar().getType().isInteger(64))
      return;
    SmallVector<Value, 4> values = {
        operation.getLowerBound(), operation.getUpperBound(),
        operation.getStep(), operation.getInductionVar()};
    NarrowKind kind = getCommonNarrowKind(solver, values);
    if (kind != NarrowKind::Both)
      return;
    std::optional<ConstantIntRanges> inductionRange =
        getRange(solver, operation.getInductionVar());
    std::optional<ConstantIntRanges> stepRange =
        getRange(solver, operation.getStep());
    if (!inductionRange || !stepRange)
      return;
    ConstantIntRanges nextRange(
        inductionRange->smin().sadd_sat(stepRange->smin()),
        inductionRange->smax().sadd_sat(stepRange->smax()),
        inductionRange->umin().uadd_sat(stepRange->umin()),
        inductionRange->umax().uadd_sat(stepRange->umax()));
    if (getNarrowKind(nextRange, 32) == NarrowKind::Both)
      plans.push_back({operation, kind});
  });
}

static void applyLoopPlans(ArrayRef<LoopPlan> plans) {
  for (const LoopPlan &plan : plans) {
    scf::ForOp operation = plan.operation;
    OpBuilder builder(operation);
    Value lower = createNarrowValue(builder, operation.getLoc(),
                                    operation.getLowerBound());
    Value upper = createNarrowValue(builder, operation.getLoc(),
                                    operation.getUpperBound());
    Value step =
        createNarrowValue(builder, operation.getLoc(), operation.getStep());
    operation.getLowerBoundMutable().assign(lower);
    operation.getUpperBoundMutable().assign(upper);
    operation.getStepMutable().assign(step);

    BlockArgument induction = cast<BlockArgument>(operation.getInductionVar());
    Type originalType = induction.getType();
    induction.setType(builder.getI32Type());
    builder.setInsertionPointToStart(operation.getBody());
    Value extended = createExtendedValue(builder, operation.getLoc(), induction,
                                         originalType, plan.kind);
    induction.replaceAllUsesExcept(extended, extended.getDefiningOp());
  }
}

static LogicalResult narrowFunction(func::FuncOp function) {
  DataFlowSolver solver;
  loadBaselineAnalyses(solver);
  solver.load<IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(function)))
    return function.emitError("integer range analysis failed");

  SmallVector<BinaryPlan> binaryPlans;
  SmallVector<LoopPlan> loopPlans;
  collectBinaryPlans(function, solver, binaryPlans);
  collectLoopPlans(function, solver, loopPlans);
  applyBinaryPlans(binaryPlans);
  applyLoopPlans(loopPlans);
  return success();
}

struct NarrowIntegerRanges final
    : inter::impl::NarrowIntegerRangesBase<NarrowIntegerRanges> {
  void runOnOperation() override {
    if (getOperation()
            .walk([&](func::FuncOp function) {
              return failed(narrowFunction(function)) ? WalkResult::interrupt()
                                                      : WalkResult::advance();
            })
            .wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
