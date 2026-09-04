#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include <array>

namespace inter {
#define GEN_PASS_DEF_DISCOVERCACHECONTROLS
#include "inter/Transforms/Passes.h.inc"
} // namespace inter.

using namespace mlir;
using namespace mlir::dataflow;

namespace {

enum class CacheStateKind { Uninitialized, Unspecified, Exact, Conflict };
enum class AccessKind { Load, Store };

struct CacheProfile {
  CacheStateKind kind = CacheStateKind::Uninitialized;
  std::array<std::optional<xw::CachePolicyKind>, 3> levels;

  static CacheProfile unspecified() {
    return {CacheStateKind::Unspecified, {}};
  }
  static CacheProfile conflict() {
    return {CacheStateKind::Conflict, {}};
  }

  static CacheProfile join(const CacheProfile &lhs, const CacheProfile &rhs) {
    if (lhs.kind == CacheStateKind::Uninitialized)
      return rhs;
    if (rhs.kind == CacheStateKind::Uninitialized)
      return lhs;
    if (lhs == rhs)
      return lhs;
    return conflict();
  }

  bool operator==(const CacheProfile &rhs) const {
    return kind == rhs.kind && levels == rhs.levels;
  }
};

struct PointerCacheState {
  CacheProfile load;
  CacheProfile store;

  static PointerCacheState unspecified() {
    return {CacheProfile::unspecified(), CacheProfile::unspecified()};
  }
  static PointerCacheState join(const PointerCacheState &lhs,
                                const PointerCacheState &rhs) {
    return {CacheProfile::join(lhs.load, rhs.load),
            CacheProfile::join(lhs.store, rhs.store)};
  }
  bool operator==(const PointerCacheState &rhs) const {
    return load == rhs.load && store == rhs.store;
  }
  void print(llvm::raw_ostream &os) const { os << "pointer-cache-state"; }
};

struct CacheDecoration {
  AccessKind access;
  unsigned level;
  xw::CachePolicyKind policy;
};

using PointerCacheLattice = Lattice<PointerCacheState>;
using DecorationMap = DenseMap<Operation *, CacheDecoration>;

static bool isPointer(Type type) { return isa<LLVM::LLVMPointerType>(type); }

static bool isPointerAnnotation(Operation *op) {
  if (isa<LLVM::PtrAnnotation>(op))
    return true;
  auto intrinsic = dyn_cast<LLVM::CallIntrinsicOp>(op);
  return intrinsic &&
         intrinsic.getIntrin().starts_with("llvm.ptr.annotation") &&
         intrinsic.getArgs().size() == 5 && intrinsic.getNumResults() == 1;
}

static Value getAnnotationOperand(Operation *op, unsigned index) {
  if (auto annotation = dyn_cast<LLVM::PtrAnnotation>(op))
    return annotation->getOperand(index);
  return cast<LLVM::CallIntrinsicOp>(op).getArgs()[index];
}

class CacheControlAnalysis final
    : public SparseForwardDataFlowAnalysis<PointerCacheLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CacheControlAnalysis)

  CacheControlAnalysis(DataFlowSolver &solver, const DecorationMap &decorations)
      : SparseForwardDataFlowAnalysis(solver), decorations(decorations) {}

  LogicalResult
  visitOperation(Operation *op, ArrayRef<const PointerCacheLattice *> operands,
                 ArrayRef<PointerCacheLattice *> results) override {
    if (results.empty())
      return success();

    PointerCacheState state = PointerCacheState::unspecified();
    if (isPointerAnnotation(op)) {
      state = operands.front()->getValue();
      const CacheDecoration &decoration = decorations.lookup(op);
      CacheProfile &profile =
          decoration.access == AccessKind::Load ? state.load : state.store;
      if (profile.kind == CacheStateKind::Unspecified) {
        profile.kind = CacheStateKind::Exact;
      } else if (profile.kind != CacheStateKind::Exact) {
        profile = CacheProfile::conflict();
      }
      if (profile.kind == CacheStateKind::Exact) {
        std::optional<xw::CachePolicyKind> &level =
            profile.levels[decoration.level];
        if (level && *level != decoration.policy)
          profile = CacheProfile::conflict();
        else
          level = decoration.policy;
      }
    } else if (isa<LLVM::GEPOp, LLVM::BitcastOp, LLVM::AddrSpaceCastOp,
                   LLVM::FreezeOp>(op)) {
      state = operands.front()->getValue();
    } else if (auto select = dyn_cast<LLVM::SelectOp>(op)) {
      state = PointerCacheState::join(operands[1]->getValue(),
                                      operands[2]->getValue());
    }

    for (auto [result, value] : llvm::zip(results, op->getResults())) {
      PointerCacheState resultState =
          isPointer(value.getType()) ? state : PointerCacheState::unspecified();
      propagateIfChanged(result, result->join(resultState));
    }
    return success();
  }

  void setToEntryState(PointerCacheLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(PointerCacheState::unspecified()));
  }

private:
  const DecorationMap &decorations;
};

static FailureOr<CacheDecoration> parseDecoration(Operation *op) {
  auto address = getAnnotationOperand(op, 1).getDefiningOp<LLVM::AddressOfOp>();
  if (!address)
    return op->emitOpError("cache annotation must reference a metadata global"),
           failure();
  LLVM::GlobalOp global = SymbolTable::lookupNearestSymbolFrom<LLVM::GlobalOp>(
      address, address.getGlobalNameAttr());
  auto value = global ? dyn_cast_or_null<StringAttr>(global.getValueOrNull())
                      : StringAttr();
  if (!value)
    return op->emitOpError("cache annotation global must contain a string"),
           failure();

  StringRef payload = value.getValue();
  if (!payload.empty() && payload.back() == '\0')
    payload = payload.drop_back();
  if (!payload.consume_front("{") || !payload.consume_back("}"))
    return op->emitOpError("malformed cache annotation payload"), failure();
  auto [tokenText, values] = payload.split(':');
  if (!values.consume_front("\"") || !values.consume_back("\""))
    return op->emitOpError("malformed cache annotation values"), failure();
  auto [levelText, policyText] = values.split(',');
  uint64_t token = 0;
  uint64_t level = 0;
  uint64_t policy = 0;
  if (tokenText.getAsInteger(10, token) || levelText.getAsInteger(10, level) ||
      policyText.getAsInteger(10, policy) || level > 2)
    return op->emitOpError("invalid cache annotation integers"), failure();

  AccessKind access;
  xw::CachePolicyKind semanticPolicy;
  if (token == 6442) {
    access = AccessKind::Load;
    switch (policy) {
    case 0:
      semanticPolicy = xw::CachePolicyKind::Uncached;
      break;
    case 1:
      semanticPolicy = xw::CachePolicyKind::Cached;
      break;
    case 2:
      semanticPolicy = xw::CachePolicyKind::Streaming;
      break;
    case 3:
      semanticPolicy = xw::CachePolicyKind::ReadInvalidate;
      break;
    default:
      return op->emitOpError("unsupported load cache policy"), failure();
    }
  } else if (token == 6443) {
    access = AccessKind::Store;
    switch (policy) {
    case 0:
      semanticPolicy = xw::CachePolicyKind::Uncached;
      break;
    case 1:
      semanticPolicy = xw::CachePolicyKind::WriteThrough;
      break;
    case 2:
      semanticPolicy = xw::CachePolicyKind::WriteBack;
      break;
    case 3:
      semanticPolicy = xw::CachePolicyKind::Streaming;
      break;
    default:
      return op->emitOpError("unsupported store cache policy"), failure();
    }
  } else {
    return op->emitOpError("unsupported pointer annotation decoration"),
           failure();
  }
  return CacheDecoration{access, static_cast<unsigned>(level), semanticPolicy};
}

static std::optional<AccessKind> getConsumerAccess(Operation *op,
                                                   Value &pointer) {
  if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
    pointer = load.getAddr();
    return AccessKind::Load;
  }
  if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
    pointer = store.getAddr();
    return AccessKind::Store;
  }
  auto call = dyn_cast<LLVM::CallOp>(op);
  if (!call || call.getArgOperands().empty() || !call.getCallee())
    return std::nullopt;
  StringRef callee = *call.getCallee();
  pointer = call.getArgOperands().front();
  if (callee.contains("intel_sub_group_2d_block_write"))
    return AccessKind::Store;
  if (callee.contains("intel_sub_group_2d_block_read") ||
      callee.contains("intel_sub_group_2d_block_prefetch"))
    return AccessKind::Load;
  return std::nullopt;
}

static DictionaryAttr getCacheControlAttr(MLIRContext *context,
                                          const CacheProfile &profile) {
  NamedAttrList controls;
  constexpr std::array<StringLiteral, 3> twoLevelNames = {"l1", "l3", ""};
  constexpr std::array<StringLiteral, 3> threeLevelNames = {"l1", "l2", "l3"};
  const std::array<StringLiteral, 3> &names =
      profile.levels[2] ? threeLevelNames : twoLevelNames;
  for (auto [name, policy] : llvm::zip(names, profile.levels))
    if (policy)
      controls.set(name, xw::CachePolicyAttr::get(context, *policy));
  return controls.getDictionary(context);
}

static void eraseResolvedAnnotations(ModuleOp moduleOp,
                                     const DecorationMap &decorations) {
  DenseSet<FlatSymbolRefAttr> metadataGlobals;
  for (Operation *operation : llvm::make_first_range(decorations)) {
    for (unsigned index : {1U, 2U, 4U}) {
      Value operand = getAnnotationOperand(operation, index);
      if (auto address = operand.getDefiningOp<LLVM::AddressOfOp>())
        metadataGlobals.insert(address.getGlobalNameAttr());
    }
    operation->getResult(0).replaceAllUsesWith(
        getAnnotationOperand(operation, 0));
    operation->erase();
  }

  SmallVector<LLVM::AddressOfOp> deadAddresses;
  moduleOp.walk([&](LLVM::AddressOfOp address) {
    if (metadataGlobals.contains(address.getGlobalNameAttr()) &&
        address->use_empty())
      deadAddresses.push_back(address);
  });
  for (LLVM::AddressOfOp address : deadAddresses)
    address.erase();

  for (LLVM::GlobalOp global :
       llvm::make_early_inc_range(moduleOp.getOps<LLVM::GlobalOp>())) {
    FlatSymbolRefAttr symbol =
        FlatSymbolRefAttr::get(moduleOp.getContext(), global.getSymName());
    if (!metadataGlobals.contains(symbol))
      continue;
    bool referenced = false;
    moduleOp.walk([&](LLVM::AddressOfOp address) {
      referenced |= address.getGlobalNameAttr() == symbol;
    });
    if (!referenced)
      global.erase();
  }

  for (LLVM::LLVMFuncOp function :
       llvm::make_early_inc_range(moduleOp.getOps<LLVM::LLVMFuncOp>()))
    if (function.isExternal() &&
        function.getName().starts_with("llvm.ptr.annotation"))
      function.erase();
}

struct DiscoverCacheControls final
    : inter::impl::DiscoverCacheControlsBase<DiscoverCacheControls> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    DecorationMap decorations;
    WalkResult parsed = moduleOp.walk([&](Operation *op) {
      if (!isPointerAnnotation(op))
        return WalkResult::advance();
      FailureOr<CacheDecoration> decoration = parseDecoration(op);
      if (failed(decoration))
        return WalkResult::interrupt();
      decorations.insert({op, *decoration});
      return WalkResult::advance();
    });
    if (parsed.wasInterrupted())
      return signalPassFailure();
    if (decorations.empty())
      return;

    DataFlowConfig config;
    config.setInterprocedural(false);
    DataFlowSolver solver(config);
    loadBaselineAnalyses(solver);
    solver.load<CacheControlAnalysis>(decorations);
    if (failed(solver.initializeAndRun(moduleOp))) {
      moduleOp.emitError("cache-control dataflow failed to converge");
      return signalPassFailure();
    }

    bool failedConsumer = false;
    moduleOp.walk([&](Operation *op) {
      Value pointer;
      std::optional<AccessKind> access = getConsumerAccess(op, pointer);
      if (!access)
        return;
      const PointerCacheLattice *lattice =
          solver.lookupState<PointerCacheLattice>(pointer);
      if (!lattice)
        return;
      const CacheProfile &profile = *access == AccessKind::Load
                                        ? lattice->getValue().load
                                        : lattice->getValue().store;
      if (profile.kind == CacheStateKind::Conflict) {
        op->emitOpError(*access == AccessKind::Load
                            ? "conflicting load cache controls"
                            : "conflicting store cache controls");
        failedConsumer = true;
      } else if (profile.kind == CacheStateKind::Exact) {
        op->setAttr("xw.cache_control",
                    getCacheControlAttr(&getContext(), profile));
      }
    });
    if (failedConsumer)
      return signalPassFailure();
    eraseResolvedAnnotations(moduleOp, decorations);
  }
};

} // namespace.
