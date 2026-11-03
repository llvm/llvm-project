#include "inter/Analysis/DistributionAnalysis.h"
#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"

namespace inter {
#define GEN_PASS_DEF_REFINEDISTRIBUTION
#include "inter/Transforms/Passes.h.inc"
} // namespace inter.

using namespace mlir;

namespace {

static unsigned getCardinality(DataFlowSolver &solver, Value value,
                               unsigned width) {
  const inter::DistributionLattice *lattice =
      solver.lookupState<inter::DistributionLattice>(value);
  unsigned cardinality = lattice ? lattice->getValue().cardinality : width;
  return cardinality ? cardinality : width;
}

static Type getRefinedType(Type type, unsigned cardinality, bool wrapBare,
                           bool mask) {
  if (isa<xw::MemTokenType>(type))
    return type;
  if (xw::SimdType simd = dyn_cast<xw::SimdType>(type)) {
    if (cardinality == 1)
      return simd.getElementType();
    return xw::SimdType::get(type.getContext(), simd.getElementType(),
                             cardinality);
  }
  if (isa<xw::MaskType>(type)) {
    if (cardinality == 1)
      return IntegerType::get(type.getContext(), 1);
    return xw::MaskType::get(type.getContext(), cardinality);
  }
  if (wrapBare && cardinality > 1) {
    if (mask)
      return xw::MaskType::get(type.getContext(), cardinality);
    return xw::SimdType::get(type.getContext(), type, cardinality);
  }
  return type;
}

static bool isFunctionArgument(Value value) {
  if (isa<BlockArgument>(value)) {
    BlockArgument argument = cast<BlockArgument>(value);
    if (argument.getOwner()->isEntryBlock() &&
        isa<func::FuncOp>(argument.getOwner()->getParentOp()))
      return true;
  }
  return false;
}

static SmallVector<SmallVector<Value>>
getStructuralClasses(func::FuncOp function) {
  SmallVector<SmallVector<Value>> classes;
  function.walk([&](Operation *op) {
    if (scf::IfOp ifOp = dyn_cast<scf::IfOp>(op)) {
      if (!ifOp.getNumResults())
        return;
      scf::YieldOp thenYield = ifOp.thenYield();
      scf::YieldOp elseYield = ifOp.elseYield();
      for (unsigned index = 0; index < ifOp.getNumResults(); ++index)
        classes.push_back({ifOp.getResult(index), thenYield.getOperand(index),
                           elseYield.getOperand(index)});
      return;
    }
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op)) {
      scf::YieldOp yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (unsigned index = 0; index < forOp.getNumResults(); ++index)
        classes.push_back({forOp.getInitArgs()[index],
                           forOp.getRegionIterArgs()[index],
                           yield.getOperand(index), forOp.getResult(index)});
      return;
    }
    if (scf::WhileOp whileOp = dyn_cast<scf::WhileOp>(op)) {
      scf::ConditionOp condition =
          cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator());
      scf::YieldOp yield =
          cast<scf::YieldOp>(whileOp.getAfterBody()->getTerminator());
      for (unsigned index = 0; index < whileOp.getNumOperands(); ++index)
        classes.push_back({whileOp.getOperand(index),
                           whileOp.getBeforeArguments()[index],
                           yield.getOperand(index)});
      for (unsigned index = 0; index < whileOp.getNumResults(); ++index)
        classes.push_back({condition.getArgs()[index],
                           whileOp.getAfterArguments()[index],
                           whileOp.getResult(index)});
      return;
    }
    if (xw::WhereOp where = dyn_cast<xw::WhereOp>(op)) {
      if (!where.getNumResults())
        return;
      xw::YieldOp thenYield =
          cast<xw::YieldOp>(where.getThenRegion().front().getTerminator());
      if (where.getElseRegion().empty())
        return;
      xw::YieldOp elseYield =
          cast<xw::YieldOp>(where.getElseRegion().front().getTerminator());
      for (unsigned index = 0; index < where.getNumResults(); ++index)
        classes.push_back({where.getResult(index), thenYield.getOperand(index),
                           elseYield.getOperand(index)});
    }
  });
  return classes;
}

static LogicalResult refineTypes(DataFlowSolver &solver, func::FuncOp function,
                                 unsigned width) {
  DenseMap<Value, unsigned> cardinalities;
  DenseMap<Value, bool> structuralValues;
  DenseMap<Value, bool> maskValues;
  function.walk([&](Operation *op) {
    for (Value result : op->getResults())
      cardinalities[result] = getCardinality(solver, result, width);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          cardinalities[argument] = getCardinality(solver, argument, width);
  });

  SmallVector<SmallVector<Value>> classes = getStructuralClasses(function);
  for (ArrayRef<Value> values : classes) {
    bool mask = llvm::any_of(
        values, [](Value value) { return isa<xw::MaskType>(value.getType()); });
    for (Value value : values) {
      structuralValues[value] = true;
      maskValues[value] = mask;
    }
  }
  for (ArrayRef<Value> values : classes)
    for (Value value : values)
      if (isa<BlockArgument>(value) ||
          isa<scf::IfOp, scf::ForOp, scf::WhileOp, xw::WhereOp>(
              value.getDefiningOp()))
        cardinalities[value] = 0;
  bool changed;
  do {
    changed = false;
    for (ArrayRef<Value> values : classes) {
      unsigned cardinality = 0;
      for (Value value : values)
        cardinality = inter::Distribution::join({cardinality},
                                                {cardinalities.lookup(value)})
                          .cardinality;
      for (Value value : values)
        if (cardinalities[value] != cardinality) {
          cardinalities[value] = cardinality;
          changed = true;
        }
    }
  } while (changed);

  llvm::EquivalenceClasses<Value> equivalenceClasses;
  for (auto [value, cardinality] : cardinalities)
    equivalenceClasses.insert(value);
  for (ArrayRef<Value> values : classes)
    for (Value value : values.drop_front())
      equivalenceClasses.unionSets(values.front(), value);

  IRRewriter rewriter(function.getContext());
  for (auto iterator = equivalenceClasses.begin();
       iterator != equivalenceClasses.end(); ++iterator) {
    const auto *leader = *iterator;
    if (!leader->isLeader())
      continue;
    SmallVector<Value> values;
    SmallVector<Type> originalTypes;
    SmallVector<Operation *> affected;
    for (auto member = equivalenceClasses.member_begin(*leader);
         member != equivalenceClasses.member_end(); ++member) {
      Value value = *member;
      if (isFunctionArgument(value) ||
          isa_and_nonnull<func::CallOp>(value.getDefiningOp()) ||
          llvm::any_of(value.getUsers(), [](Operation *user) {
            return isa<func::ReturnOp>(user);
          })) {
        values.clear();
        break;
      }
      Operation *definition = value.getDefiningOp();
      bool xwResult =
          definition && definition->getDialect()->getNamespace() == "xw";
      bool mask =
          maskValues.lookup(value) ||
          isa_and_nonnull<xw::CmpIOp, xw::CmpFOp, xw::PtrCmpOp>(definition);
      Type refined =
          getRefinedType(value.getType(), cardinalities.lookup(value),
                         xwResult || structuralValues.lookup(value), mask);
      if (refined == value.getType())
        continue;
      values.push_back(value);
      originalTypes.push_back(value.getType());
      if (definition)
        affected.push_back(definition);
      for (Operation *user : value.getUsers())
        affected.push_back(user);
      value.setType(refined);
    }
    if (values.empty())
      continue;

    xw::SplatOp adapter;
    if (values.size() == 1) {
      if (xw::AtomicRMWOp atomic = dyn_cast_or_null<xw::AtomicRMWOp>(
              values.front().getDefiningOp())) {
        xw::SimdType type = dyn_cast<xw::SimdType>(values.front().getType());
        if (type && atomic.getValue().getType() == type.getElementType()) {
          rewriter.setInsertionPoint(atomic);
          adapter = xw::SplatOp::create(rewriter, atomic.getLoc(), type,
                                        atomic.getValue());
          atomic.getValueMutable().assign(adapter.getResult());
          affected.push_back(adapter);
        }
      }
    }

    llvm::sort(affected);
    affected.erase(llvm::unique(affected), affected.end());
    ScopedDiagnosticHandler suppress(function.getContext(),
                                     [](Diagnostic &) { return success(); });
    bool valid = llvm::all_of(affected, [](Operation *operation) {
      return succeeded(verify(operation, /*verifyRecursively=*/false));
    });
    if (valid)
      continue;
    if (adapter) {
      cast<xw::AtomicRMWOp>(values.front().getDefiningOp())
          .getValueMutable()
          .assign(adapter.getSource());
      adapter.erase();
    }
    for (auto [value, type] : llvm::zip(values, originalTypes))
      value.setType(type);
  }
  return success();
}

static void removeDistributionAttrs(func::FuncOp function) {
  function.walk([](Operation *op) {
    op->removeAttr("xw.distribution");
    op->removeAttr("xw.provisional_cardinality");
  });
  for (unsigned index = 0; index < function.getNumArguments(); ++index)
    function.removeArgAttr(index, "xw.distribution");
}

struct RefineDistribution final
    : inter::impl::RefineDistributionBase<RefineDistribution> {
  using RefineDistributionBase::RefineDistributionBase;

  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      unsigned width = simdWidth;
      if (IntegerAttr attr =
              function->getAttrOfType<IntegerAttr>("xw.simd_width"))
        width = attr.getInt();
      if (width != 8 && width != 16 && width != 32) {
        function.emitOpError("requires xw.simd_width 8, 16, or 32");
        return signalPassFailure();
      }

      DataFlowConfig config;
      config.setInterprocedural(false);
      DataFlowSolver solver(config);
      solver.load<dataflow::DeadCodeAnalysis>();
      solver.load<dataflow::SparseConstantPropagation>();
      inter::DistributionAnalysis *analysis =
          solver.load<inter::DistributionAnalysis>(width);
      if (failed(solver.initializeAndRun(function))) {
        function.emitOpError("distribution dataflow failed to converge");
        return signalPassFailure();
      }
      for (StringRef cause : analysis->getUnknownCauses())
        function.emitRemark()
            << "distribution refinement retained full width: " << cause;

      if (failed(refineTypes(solver, function, width)))
        return signalPassFailure();
      removeDistributionAttrs(function);
    }
  }
};

} // namespace.
