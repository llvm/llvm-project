// inter-select-to-machine: lower an llvm-dialect kernel to xemachine ops.
//
// M1 scope: straight-line kernels, i32 lane values, A64 stateless global
// memory, one work-item id builtin. Ordinary values remain virtual for the
// register allocator; only architectural payload/prologue values are pinned.

#include "inter/Analysis/UniformityAnalysis.h"
#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Support/Builtins.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

namespace inter {
#define GEN_PASS_DEF_SELECTTOMACHINE
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;
using namespace inter::xemachine;

namespace {

// EU register payload layout (see inter/docs/PayloadContract.md).
constexpr int kInlineMirrorSize = 32;
constexpr int kLocalIdLoadOffset = 0x20;
constexpr int kPerThreadPayloadSize = 192;

FailureOr<uint64_t> getTypeSize(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return llvm::divideCeil(integer.getWidth(), 8u);
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type)) {
    FailureOr<uint64_t> elementSize = getTypeSize(array.getElementType());
    if (failed(elementSize))
      return failure();
    return array.getNumElements() * *elementSize;
  }
  return failure();
}

FailureOr<uint64_t> getSlmSize(ModuleOp moduleOp) {
  uint64_t size = 0;
  unsigned globals = 0;
  for (LLVM::GlobalOp global : moduleOp.getOps<LLVM::GlobalOp>()) {
    if (global.getAddrSpace() != 3)
      continue;
    if (++globals > 1)
      return global.emitOpError("multiple SLM globals are not supported"),
             failure();
    FailureOr<uint64_t> globalSize = getTypeSize(global.getGlobalType());
    if (failed(globalSize))
      return global.emitOpError("unsupported SLM global type"), failure();
    uint64_t alignment = global.getAlignment().value_or(1);
    size = llvm::alignTo(size, alignment) + *globalSize;
  }
  return size;
}

struct SelectToMachine
    : public inter::impl::SelectToMachineBase<SelectToMachine> {
  void runOnOperation() override {
    SmallVector<func::FuncOp> kernels;
    getOperation().walk([&](func::FuncOp func) {
      if (func->hasAttr("xemachine.kernel"))
        kernels.push_back(func);
    });
    if (kernels.empty()) {
      getOperation().emitError("no kernel function found");
      return signalPassFailure();
    }
    for (func::FuncOp kernel : kernels) {
      DataFlowSolver solver;
      dataflow::loadBaselineAnalyses(solver);
      solver.load<inter::UniformityAnalysis>();
      if (failed(solver.initializeAndRun(kernel)))
        return signalPassFailure();
      uniformity = &solver;
      if (failed(lowerKernel(kernel)))
        return signalPassFailure();
    }
  }

  MLIRContext *ctx = nullptr;
  DataFlowSolver *uniformity = nullptr;
  std::optional<Location> loc;
  std::optional<OpBuilder> b;
  DenseMap<Value, Value> vmap;
  ArrayAttr kernelArgs;
  Value gidValue;
  Value localXValue;
  Value tailValue;
  bool prologueEmitted = false;
  bool kernelUsesThreadIds = false;

  struct WideValue {
    Value low;
    Value high;
  };
  DenseMap<Value, WideValue> globalPointers;
  DenseMap<Value, WideValue> wideOffsets;
  DenseMap<Value, Value> slmPointers;

  Type grf(int dwords) { return RegType::get(ctx, dwords, -1); }
  Type i32() { return IntegerType::get(ctx, 32); }
  Type i16() { return IntegerType::get(ctx, 16); }
  Type i64() { return IntegerType::get(ctx, 64); }

  RegionAttr rcanon() { return RegionAttr::get(ctx, 1, 1, 0); }
  RegionAttr runiform() { return RegionAttr::get(ctx, 0, 1, 0); }
  DstRegionAttr dcanon() { return DstRegionAttr::get(ctx, 1); }
  TypeAttr ty(Type t_) { return TypeAttr::get(t_); }

  Value imm(int64_t v, Type elemTy) {
    return ImmOp::create(*b, *loc, ImmType::get(ctx), v, elemTy).getResult();
  }

  Value archreg(int index) {
    return ArchRegOp::create(*b, *loc, RegType::get(ctx, 16, index),
                             b->getI32IntegerAttr(index))
        .getResult();
  }

  // Memory ops carry the AA-decided dependency token; each machine op's
  // result token maps back to the frontend op's token result. memToken is
  // scratch for the barrier's internal chaining.
  Value memToken;

  Value emitLoadA64(Type dstTy, Value addrPayload, Value depTok) {
    auto op = LoadA64Op::create(*b, *loc, dstTy, MemTokenType::get(ctx),
                                addrPayload, depTok, 32);
    memToken = op.getToken();
    return op.getDst();
  }

  void emitStoreA64(Value addrPayload, Value dataPayload, Value depTok) {
    auto op = StoreA64Op::create(*b, *loc, MemTokenType::get(ctx), addrPayload,
                                 dataPayload, depTok, 32);
    memToken = op.getToken();
  }

  Value emitLoadBlock(Type dstTy, Value addrPayload, int words) {
    auto op = LoadBlockA32Op::create(*b, *loc, dstTy, MemTokenType::get(ctx),
                                     addrPayload, Value(), words);
    memToken = op.getToken();
    return op.getDst();
  }

  void emitSync(SyncKind kind) {
    auto op = SyncOp::create(*b, *loc, MemTokenType::get(ctx),
                             SyncKindAttr::get(ctx, kind), memToken);
    memToken = op.getToken();
  }

  void emitLocalIdLoadEntry() {
    Value r0 = archreg(0);
    Value r1 = archreg(1);

    // Hardware local-ID generation places IDs in r1-r3 and inline data in r4.
    // The software entry mirrors that layout before converging at byte 192.
    MovOp::create(*b, *loc, RegType::get(ctx, 16, 4), i32(), /*execSize=*/8,
                  dcanon(), rcanon(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                  /*noMask=*/true, /*maskOffset=*/0, r1);
    Value base =
        AndOp::create(*b, *loc, RegType::get(ctx, 16, 5), i32(),
                      /*execSize=*/1, dcanon(), runiform(), RegionAttr(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), /*noMask=*/true, /*maskOffset=*/0, r0,
                      imm(0xFFFFFFC0, i32()))
            .getResult();
    Value perThreadBase =
        AddOp::create(*b, *loc, RegType::get(ctx, 16, 6), i32(),
                      /*execSize=*/1, dcanon(), runiform(), RegionAttr(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), /*noMask=*/true, /*maskOffset=*/0, base,
                      imm(kLocalIdLoadOffset, i32()))
            .getResult();
    Value threadSlot =
        AndOp::create(*b, *loc, RegType::get(ctx, 16, 7), i32(),
                      /*execSize=*/1, dcanon(), runiform(), RegionAttr(),
                      IntegerAttr(), b->getI32IntegerAttr(4), IntegerAttr(),
                      TypeAttr(), TypeAttr(), /*noMask=*/true,
                      /*maskOffset=*/0, r0, imm(0xff, i32()))
            .getResult();
    Value threadOffsetAcc =
        MulOp::create(*b, *loc, ARFType::get(ctx, ARFFile::acc, 16, 0), i32(),
                      /*execSize=*/1, dcanon(), runiform(), RegionAttr(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(),
                      /*noMask=*/true, /*maskOffset=*/0, threadSlot,
                      imm(kPerThreadPayloadSize, i32()))
            .getResult();
    Value threadOffset =
        MovOp::create(*b, *loc, RegType::get(ctx, 16, 8), i32(),
                      /*execSize=*/1, dcanon(), runiform(), IntegerAttr(),
                      IntegerAttr(), TypeAttr(), /*noMask=*/true,
                      /*maskOffset=*/0, threadOffsetAcc)
            .getResult();
    Value perThreadAddr =
        AddOp::create(*b, *loc, RegType::get(ctx, 16, 9), i32(),
                      /*execSize=*/1, dcanon(), runiform(), runiform(),
                      IntegerAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                      TypeAttr(), /*noMask=*/true, /*maskOffset=*/0,
                      perThreadBase, threadOffset)
            .getResult();
    auto load = LoadBlockA32Op::create(*b, *loc, RegType::get(ctx, 16, 1),
                                       MemTokenType::get(ctx), perThreadAddr,
                                       Value(), 16);
    load->setAttr(kAllowFixedOverlapAttrName, b->getUnitAttr());
    memToken = load.getToken();

    // Keep the hardware-generated-local-ID entry at the zeinfo offset.
    emitSync(SyncKind::nop);
    emitSync(SyncKind::nop);
    emitSync(SyncKind::nop);
    emitSync(SyncKind::nop);
  }

  void emitArgumentLoadEntry() {
    MovOp::create(*b, *loc, RegType::get(ctx, 16, 4), i32(), /*execSize=*/8,
                  dcanon(), rcanon(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                  /*noMask=*/true, /*maskOffset=*/0, archreg(1));
    for (unsigned unused : llvm::seq<unsigned>(11)) {
      (void)unused;
      emitSync(SyncKind::nop);
    }
  }

  LogicalResult lowerKernel(func::FuncOp kernel) {
    ctx = kernel.getContext();
    loc = kernel.getLoc();
    vmap.clear();
    memToken = nullptr;
    gidValue = nullptr;
    tailValue = nullptr;
    prologueEmitted = false;
    globalPointers.clear();
    wideOffsets.clear();
    slmPointers.clear();
    kernelArgs = kernel->getAttrOfType<ArrayAttr>(kKernelArgsAttrName);
    if (!kernelArgs && kernel.getNumArguments() == 0)
      kernelArgs = ArrayAttr::get(ctx, {});
    if (failed(verifyKernelArgLayout(kernel.getFunctionType(), kernelArgs,
                                     kernel.getOperation())))
      return failure();

    OpBuilder moduleBuilder(kernel);
    auto func = func::FuncOp::create(moduleBuilder, kernel.getLoc(),
                                     (kernel.getName() + "_xm").str(),
                                     moduleBuilder.getFunctionType({}, {}));
    func->setAttr(kTargetAttrName,
                  TargetAttr::get(ctx, moduleBuilder.getStringAttr("bmg")));
    b = OpBuilder::atBlockBegin(func.addEntryBlock());

    kernelUsesThreadIds = false;
    kernel.walk([&](Operation *operation) {
      kernelUsesThreadIds |= isa<xw::GlobalIdOp, xw::LocalIdOp>(operation);
    });
    FailureOr<uint64_t> slmSize = getSlmSize(getOperation());
    if (failed(slmSize))
      return failure();
    func->setAttr(kKernelTypeAttrName, TypeAttr::get(kernel.getFunctionType()));
    func->setAttr(kKernelArgsAttrName, kernelArgs);
    func->setAttr(kGrfCountAttrName, moduleBuilder.getI32IntegerAttr(128));
    func->setAttr(kReservedGrfCountAttrName,
                  moduleBuilder.getI32IntegerAttr(5));
    func->setAttr(kSimdSizeAttrName, moduleBuilder.getI32IntegerAttr(32));
    bool needsPayload = kernelUsesThreadIds || kernel.getNumArguments() != 0;
    if (kernelUsesThreadIds)
      func->setAttr(kUsesThreadIdsAttrName, moduleBuilder.getUnitAttr());
    if (needsPayload) {
      func->setAttr(kInlineDataPayloadSizeAttrName,
                    moduleBuilder.getI32IntegerAttr(kInlineMirrorSize));
      func->setAttr(kPayloadEntryOffsetAttrName,
                    moduleBuilder.getI32IntegerAttr(kPerThreadPayloadSize));
    }
    if (kernelUsesThreadIds)
      func->setAttr(kPerThreadPayloadSizeAttrName,
                    moduleBuilder.getI32IntegerAttr(kPerThreadPayloadSize));
    if (*slmSize != 0)
      func->setAttr(kSlmSizeAttrName,
                    moduleBuilder.getI64IntegerAttr(*slmSize));
    if (needsPayload && failed(emitPrologueAndGid()))
      return failure();
    for (BlockArgument argument : kernel.getArguments()) {
      auto pointer = dyn_cast<LLVM::LLVMPointerType>(argument.getType());
      if (pointer && pointer.getAddressSpace() == 1 &&
          failed(materializeGlobalPointer(argument)))
        return failure();
    }
    if (failed(lowerBlock(kernel.getBody().front())))
      return failure();
    func::ReturnOp::create(*b, *loc);
    std::string name = kernel.getName().str();
    kernel.erase();
    func.setName(StringAttr::get(ctx, name));
    return success();
  }

  // One dispatch step per op; regions recurse.
  FailureOr<Value> mapDependency(Operation *operation, Value dependency) {
    if (!dependency)
      return Value();
    Value mapped = vmap.lookup(dependency);
    if (!mapped)
      return operation->emitOpError("memory dependency not lowered"), failure();
    return mapped;
  }

  bool isOffsetArithmetic(Operation *operation) {
    return isa<LLVM::ConstantOp, LLVM::AddOp, LLVM::SubOp, LLVM::MulOp,
               LLVM::ShlOp, LLVM::TruncOp, LLVM::ZExtOp, LLVM::SExtOp>(
        operation);
  }

  bool isPointerOffsetOnly(Value value, llvm::SmallPtrSetImpl<Value> &active) {
    if (!active.insert(value).second)
      return false;
    bool hasUse = false;
    for (Operation *user : value.getUsers()) {
      hasUse = true;
      if (auto ptrAdd = dyn_cast<xw::PtrAddOp>(user)) {
        if (ptrAdd.getOffset() != value) {
          active.erase(value);
          return false;
        }
        continue;
      }
      if (auto extend = dyn_cast<xw::WideExtendOp>(user)) {
        if (extend.getInput() != value) {
          active.erase(value);
          return false;
        }
        continue;
      }
      if (!isOffsetArithmetic(user) || user->getNumResults() != 1 ||
          !isPointerOffsetOnly(user->getResult(0), active)) {
        active.erase(value);
        return false;
      }
    }
    active.erase(value);
    return hasUse;
  }

  bool isPointerOffsetOnly(Operation *operation) {
    if (!isOffsetArithmetic(operation) || operation->getNumResults() != 1)
      return false;
    llvm::SmallPtrSet<Value, 8> active;
    return isPointerOffsetOnly(operation->getResult(0), active);
  }

  LogicalResult lowerBlock(Block &blk) {
    for (Operation &op : blk) {
      if (isPointerOffsetOnly(&op)) {
        bool directlyUsed =
            llvm::any_of(op.getResult(0).getUsers(), [](Operation *user) {
              return isa<xw::PtrAddOp>(user);
            });
        if (directlyUsed) {
          FailureOr<Value> lowered = lowerPacked(op.getResult(0));
          if (failed(lowered))
            return failure();
          vmap[op.getResult(0)] = *lowered;
        }
        continue;
      }
      if (auto gid = dyn_cast<xw::GlobalIdOp>(&op)) {
        if (failed(emitPrologueAndGid()))
          return failure();
        vmap[op.getResult(0)] = gidValue;
      } else if (isa<xw::LocalIdOp>(&op)) {
        if (failed(emitPrologueAndGid()))
          return failure();
        // Widen the packed u16 lane ids to dwords once.
        Value lid =
            MovOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32, dcanon(),
                          rcanon(), IntegerAttr(), IntegerAttr(), ty(i16()),
                          /*noMask=*/false, /*maskOffset=*/0, localXValue)
                .getResult();
        vmap[op.getResult(0)] = lid;
      } else if (isa<xw::WideConstantOp, xw::WideExtendOp, xw::WideAddOp,
                     xw::WideSubOp, xw::WideShlOp>(&op)) {
        FailureOr<WideValue> lowered = lowerWide(op.getResult(0));
        if (failed(lowered))
          return failure();
        wideOffsets[op.getResult(0)] = *lowered;
      } else if (auto tok = dyn_cast<xw::TokenOp>(&op)) {
        memToken = TokenOp::create(*b, *loc, MemTokenType::get(ctx)).getToken();
        vmap[tok.getToken()] = memToken;
      } else if (auto join = dyn_cast<xw::TokenJoinOp>(&op)) {
        SmallVector<Value> deps;
        for (Value dependency : join.getDependencies()) {
          FailureOr<Value> mapped = mapDependency(&op, dependency);
          if (failed(mapped))
            return failure();
          deps.push_back(*mapped);
        }
        memToken = TokenJoinOp::create(*b, *loc, MemTokenType::get(ctx), deps)
                       .getToken();
        vmap[join.getToken()] = memToken;
      } else if (auto barrier = dyn_cast<xw::BarrierOp>(&op)) {
        FailureOr<Value> dependency =
            mapDependency(&op, barrier.getDependency());
        if (failed(dependency))
          return failure();
        emitBarrier(*dependency);
        vmap[barrier.getToken()] = memToken;
      } else if (auto atomic = dyn_cast<xw::AtomicAddOp>(&op)) {
        if (failed(emitAtomicAdd(atomic)))
          return failure();
      } else if (auto ptrAdd = dyn_cast<xw::PtrAddOp>(&op)) {
        if (failed(materializePointer(ptrAdd.getResult())))
          return failure();
      } else if (isa<LLVM::AndOp, LLVM::TruncOp, LLVM::ZExtOp>(&op)) {
        // 64->32 id truncations: forward the mapped source value.
        vmap[op.getResult(0)] = vmap.lookup(op.getOperand(0));
      } else if (auto addressOf = dyn_cast<LLVM::AddressOfOp>(&op)) {
        if (failed(materializePointer(addressOf.getResult())))
          return failure();
      } else if (isa<LLVM::GEPOp>(&op)) {
        return op.emitOpError("GEP was not normalized to xw.ptradd"), failure();
      } else if (isa<LLVM::ConstantOp, arith::ConstantOp>(&op)) {
        auto intAttr = dyn_cast<IntegerAttr>(op.getAttr("value"));
        if (!intAttr)
          return emitError(op.getLoc(), "non-integer constant"), failure();
        vmap[op.getResult(0)] = imm(intAttr.getValue().getSExtValue(), i32());
      } else if (auto icmp = dyn_cast<LLVM::ICmpOp>(&op)) {
        if (failed(emitCmp(icmp)))
          return failure();
      } else if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        if (failed(emitIf(ifOp)))
          return failure();
      } else if (isa<scf::YieldOp>(&op)) {
        continue; // structural; merge movs are injected by emitIf
      } else if (auto load = dyn_cast<xw::LoadOp>(&op)) {
        FailureOr<Value> dependency = mapDependency(&op, load.getDependency());
        if (failed(dependency))
          return failure();
        if (failed(emitLoad(load.getAddress(), load.getValue(), *dependency)))
          return failure();
        vmap[load.getToken()] = memToken;
      } else if (isa<LLVM::AddOp>(&op)) {
        if (failed(emitSum(&op)))
          return failure();
      } else if (auto sub = dyn_cast<LLVM::SubOp>(&op)) {
        Value lhs = vmap.lookup(sub.getLhs());
        Value rhs = vmap.lookup(sub.getRhs());
        if (!lhs || !rhs)
          return emitError(op.getLoc(), "sub operand not lowered"), failure();
        if (lhs.getDefiningOp<ImmOp>()) {
          // imm - x: negate via sub with imm in src1 position? For M3 the
          // only shape is const - laneValue; handle both orders.
        }
        // sub a,b lowers to add(-b, a): operands are (b, a).
        Value diff =
            SubOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32, dcanon(),
                          rcanon(), rcanon(), IntegerAttr(), IntegerAttr(),
                          IntegerAttr(), TypeAttr(), TypeAttr(),
                          /*noMask=*/false, /*maskOffset=*/0, rhs, lhs)
                .getResult();
        vmap[op.getResult(0)] = diff;
      } else if (auto store = dyn_cast<xw::StoreOp>(&op)) {
        FailureOr<Value> dependency = mapDependency(&op, store.getDependency());
        if (failed(dependency))
          return failure();
        if (!store.getValue().getType().isInteger(32))
          return store.emitOpError("only i32 stores are selected"), failure();
        FailureOr<Value> data =
            materializeDwordValue(store.getValue(), store.getOperation());
        if (failed(data) ||
            failed(emitStore(store.getAddress(), *data, *dependency)))
          return failure();
        vmap[store.getToken()] = memToken;
      } else if (isa<LLVM::ReturnOp, func::ReturnOp>(&op)) {
        emitEot();
      } else {
        return op.emitOpError("unsupported operation during Inter machine "
                              "selection"),
               failure();
      }
    }
    return success();
  }

  // icmp predicate -> EU condition modifier; signedness remains on the op.
  std::optional<CondModifier> mapPredicate(LLVM::ICmpPredicate pred) {
    switch (pred) {
    case LLVM::ICmpPredicate::eq:
      return CondModifier::eq;
    case LLVM::ICmpPredicate::ne:
      return CondModifier::ne;
    case LLVM::ICmpPredicate::ugt:
    case LLVM::ICmpPredicate::sgt:
      return CondModifier::gt;
    case LLVM::ICmpPredicate::uge:
    case LLVM::ICmpPredicate::sge:
      return CondModifier::ge;
    case LLVM::ICmpPredicate::ult:
    case LLVM::ICmpPredicate::slt:
      return CondModifier::lt;
    case LLVM::ICmpPredicate::ule:
    case LLVM::ICmpPredicate::sle:
      return CondModifier::le;
    default:
      return std::nullopt;
    }
  }

  // Resolve a cmp operand: kernel scalar args read the tail load, constants
  // become immediates, everything else comes from the value map.
  struct CmpOperand {
    Value v;
    int sub = 0;
    RegionAttr region;
  };
  FailureOr<CmpOperand> cmpOperand(Value v, Operation *user) {
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      FailureOr<std::pair<Value, int>> location =
          getPayloadLocation(barg, KernelArgKind::by_value, 4, user);
      if (failed(location))
        return failure();
      return CmpOperand{location->first, location->second, runiform()};
    }
    Value mapped = vmap.lookup(v);
    if (!mapped)
      return CmpOperand{nullptr, 0, runiform()};
    if (mapped.getDefiningOp<ImmOp>())
      return CmpOperand{mapped, 0, RegionAttr()};
    return CmpOperand{mapped, 0, rcanon()};
  }

  LogicalResult emitCmp(LLVM::ICmpOp icmp) {
    FailureOr<CmpOperand> lhs = cmpOperand(icmp.getLhs(), icmp);
    FailureOr<CmpOperand> rhs = cmpOperand(icmp.getRhs(), icmp);
    if (failed(lhs) || failed(rhs))
      return failure();
    if (!lhs->v || !rhs->v)
      return emitError(icmp.getLoc(), "icmp operand not lowered"), failure();
    auto cond = mapPredicate(icmp.getPredicate());
    if (!cond)
      return emitError(icmp.getLoc(), "unsupported predicate"), failure();
    CmpOp compare = CmpOp::create(
        *b, *loc, ARFType::get(ctx, ARFFile::f, 2, 0),
        CondModifierAttr::get(ctx, *cond), ty(i32()), b->getI32IntegerAttr(32),
        lhs->region, rhs->region, b->getI32IntegerAttr(lhs->sub),
        b->getI32IntegerAttr(rhs->sub), TypeAttr(), TypeAttr(), lhs->v, rhs->v);
    if (icmp.getPredicate() == LLVM::ICmpPredicate::sgt ||
        icmp.getPredicate() == LLVM::ICmpPredicate::sge ||
        icmp.getPredicate() == LLVM::ICmpPredicate::slt ||
        icmp.getPredicate() == LLVM::ICmpPredicate::sle)
      compare->setAttr("signed", b->getUnitAttr());
    Value flag = compare.getResult();
    vmap[icmp.getResult()] = flag;
    return success();
  }

  // Sparse dataflow uniformity analysis (design doc section 7): the
  // condition is branch-uniform if the lattice says Const or Uniform.
  bool isBranchVarying(Value cond) {
    const auto *lat = uniformity->lookupState<inter::UniformityLattice>(cond);
    if (!lat)
      return true;
    return !lat->getUniformity().isAtMost(inter::UniformityKind::Uniform);
  }

  // scf.if -> exec_if/uniform_if. Results are pre-allocated; each region
  // ends with movs merging yielded values into them (PHI lowering).
  LogicalResult emitIf(scf::IfOp ifOp) {
    Value cond = vmap.lookup(ifOp.getCondition());
    if (!cond)
      return emitError(ifOp.getLoc(), "if condition not lowered"), failure();
    bool varying = isBranchVarying(ifOp.getCondition());

    SmallVector<Type> resultTypes;
    for (Value result : ifOp.getResults()) {
      if (isa<MemTokenType>(result.getType()))
        resultTypes.push_back(MemTokenType::get(ctx));
      else
        resultTypes.push_back(grf(32));
    }

    Operation *ifm;
    if (varying)
      ifm = ExecIfOp::create(*b, *loc, resultTypes, cond);
    else
      ifm = UniformIfOp::create(*b, *loc, resultTypes, cond);
    for (auto [i, r] : llvm::enumerate(ifOp.getResults()))
      vmap[r] = ifm->getResult(i);

    if (failed(emitIfRegions(ifOp, ifm, varying)))
      return failure();
    for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
      if (isa<MemTokenType>(result.getType()))
        memToken = ifm->getResult(index);
    }
    return success();
  }

  LogicalResult emitIfRegions(scf::IfOp ifOp, Operation *ifm, bool varying) {
    Region *thenR = varying ? &cast<ExecIfOp>(ifm).getThenRegion()
                            : &cast<UniformIfOp>(ifm).getThenRegion();
    Region *elseR = varying ? &cast<ExecIfOp>(ifm).getElseRegion()
                            : &cast<UniformIfOp>(ifm).getElseRegion();
    SmallVector<std::pair<Region *, Region *>, 2> regions = {
        {&ifOp.getThenRegion(), thenR}};
    if (!ifOp.getElseRegion().empty())
      regions.emplace_back(&ifOp.getElseRegion(), elseR);
    Value entryToken = memToken;
    for (auto [scfRegion, machineRegion] : regions) {
      memToken = entryToken; // each region starts from the if-entry token
      b->setInsertionPointToStart(&machineRegion->emplaceBlock());
      if (failed(lowerBlock(scfRegion->front())))
        return failure();
      // Merge movs into the pre-allocated results; the yield carries the
      // values for the region-branch verifier (the emitter prints no yield).
      auto yield = cast<scf::YieldOp>(scfRegion->front().getTerminator());
      SmallVector<Value> yieldVals;
      for (auto [i, yielded] : llvm::enumerate(yield.getOperands())) {
        Value v = vmap.lookup(yielded);
        if (!v)
          return emitError(yield.getLoc(), "yielded value not lowered"),
                 failure();
        if (isa<MemTokenType>(yielded.getType())) {
          yieldVals.push_back(v);
          continue;
        }
        // The mov result aliases the exec_if result register; yielding it
        // keeps types consistent along the region-branch edges.
        auto merge =
            MovOp::create(*b, *loc, ifm->getResult(i).getType(), i32(),
                          /*execSize=*/32, dcanon(), rcanon(), IntegerAttr(),
                          IntegerAttr(), TypeAttr(), /*noMask=*/false,
                          /*maskOffset=*/0, v);
        yieldVals.push_back(merge.getResult());
      }
      YieldOp::create(*b, *loc, yieldVals);
    }
    b->setInsertionPointAfter(ifm);
    return success();
  }

  // r1 contains hardware-generated or software-loaded local IDs; r4 contains
  // inline data. The cross-thread tail starts at blob+0.
  LogicalResult emitPrologueAndGid() {
    if (prologueEmitted)
      return success();
    prologueEmitted = true;
    if (kernelUsesThreadIds)
      emitLocalIdLoadEntry();
    else
      emitArgumentLoadEntry();
    emitSync(SyncKind::allwr);

    Value r0 = archreg(0);

    Value base =
        AndOp::create(*b, *loc, grf(16), i32(), /*execSize=*/1, dcanon(),
                      runiform(), RegionAttr(), IntegerAttr(), IntegerAttr(),
                      IntegerAttr(), TypeAttr(), TypeAttr(), /*noMask=*/true,
                      /*maskOffset=*/0, r0, imm(0xFFFFFFC0, i32()))
            .getResult();
    // Cross-thread tail: d32x8t at blob+0.
    tailValue = emitLoadBlock(grf(16), base, 8);
    if (!kernelUsesThreadIds)
      return success();

    Value localX = archreg(1);
    Value inlineData = archreg(4);
    localXValue = localX;
    // gid base: groupX * enq_local_size.x, via the accumulator.
    Value acc = MulOp::create(*b, *loc, ARFType::get(ctx, ARFFile::acc, 16, 0),
                              i32(), /*execSize=*/1, dcanon(), runiform(),
                              runiform(), IntegerAttr(),
                              b->getI32IntegerAttr(1), b->getI32IntegerAttr(3),
                              /*noMask=*/true, /*maskOffset=*/0, r0, inlineData)
                    .getResult();
    Value gidBase = MovOp::create(*b, *loc, grf(16), i32(), /*execSize=*/1,
                                  dcanon(), runiform(), IntegerAttr(),
                                  IntegerAttr(), TypeAttr(), /*noMask=*/true,
                                  /*maskOffset=*/0, acc)
                        .getResult();
    gidValue = Add3Op::create(*b, *loc, grf(32), i32(), /*execSize=*/32,
                              dcanon(), runiform(), rcanon(), runiform(),
                              IntegerAttr(), IntegerAttr(), IntegerAttr(),
                              IntegerAttr(), TypeAttr(), ty(i16()), TypeAttr(),
                              /*noMask=*/false, /*maskOffset=*/0, gidBase,
                              localX, inlineData)
                   .getResult();
    return success();
  }

  FailureOr<KernelArgAttr> getKernelArg(BlockArgument argument,
                                        KernelArgKind expectedKind,
                                        uint64_t expectedSize,
                                        Operation *user) {
    unsigned index = argument.getArgNumber();
    if (index >= kernelArgs.size())
      return user->emitOpError("kernel argument index is out of range"),
             failure();
    auto descriptor = dyn_cast<KernelArgAttr>(kernelArgs[index]);
    if (!descriptor || descriptor.getKind() != expectedKind ||
        descriptor.getSize() != expectedSize)
      return user->emitOpError("kernel argument descriptor does not match use"),
             failure();
    return descriptor;
  }

  FailureOr<std::pair<Value, int>> getPayloadLocation(BlockArgument argument,
                                                      KernelArgKind kind,
                                                      uint64_t size,
                                                      Operation *user) {
    FailureOr<KernelArgAttr> descriptor =
        getKernelArg(argument, kind, size, user);
    if (failed(descriptor))
      return failure();
    uint64_t offset = descriptor->getOffset();
    if (offset % size != 0)
      return user->emitOpError("kernel argument payload is misaligned"),
             failure();
    if (offset < kInlineMirrorSize)
      return std::pair<Value, int>{archreg(4), static_cast<int>(offset / size)};
    if (!tailValue || offset + size > 64)
      return user->emitOpError("kernel argument is outside the loaded payload"),
             failure();
    return std::pair<Value, int>{tailValue,
                                 static_cast<int>((offset - 32) / size)};
  }

  std::optional<int64_t> getImmediate(Value value) {
    if (auto immediate = value.getDefiningOp<ImmOp>())
      return immediate.getValue();
    return std::nullopt;
  }

  WideValue getWideConstant(uint64_t value) {
    Value constant = imm(static_cast<int64_t>(value), i64());
    return {constant, constant};
  }

  WideValue widenPacked(Value value, bool isSigned) {
    if (std::optional<int64_t> immediate = getImmediate(value)) {
      uint64_t extended = isSigned ? static_cast<uint64_t>(static_cast<int64_t>(
                                         static_cast<int32_t>(*immediate)))
                                   : static_cast<uint32_t>(*immediate);
      return getWideConstant(extended);
    }
    auto widenHalf = [&](int firstLane) {
      ShlOp move = ShlOp::create(
          *b, *loc, grf(32), i64(), /*execSize=*/16, dcanon(), rcanon(),
          RegionAttr(), IntegerAttr(), b->getI32IntegerAttr(firstLane),
          IntegerAttr(), ty(i32()), TypeAttr(), /*noMask=*/false,
          /*maskOffset=*/firstLane, value, imm(0, i16()));
      if (isSigned)
        move->setAttr("signedSource", b->getUnitAttr());
      return move.getResult();
    };
    return {widenHalf(0), widenHalf(16)};
  }

  WideValue materializeWide(WideValue value) {
    auto materialize = [&](Value half, int maskOffset) {
      if (!half.getDefiningOp<ImmOp>())
        return half;
      return MovOp::create(*b, *loc, grf(32), i64(), /*execSize=*/16, dcanon(),
                           RegionAttr(), IntegerAttr(), IntegerAttr(),
                           TypeAttr(), /*noMask=*/false, maskOffset, half)
          .getResult();
    };
    return {materialize(value.low, 0), materialize(value.high, 16)};
  }

  WideValue addWide(WideValue lhs, WideValue rhs) {
    auto addHalf = [&](Value left, Value right, int maskOffset) {
      if (getImmediate(right) && !getImmediate(left))
        std::swap(left, right);
      if (std::optional<int64_t> leftImm = getImmediate(left))
        if (std::optional<int64_t> rightImm = getImmediate(right))
          return imm(static_cast<int64_t>(static_cast<uint64_t>(*leftImm) +
                                          static_cast<uint64_t>(*rightImm)),
                     i64());
      return AddOp::create(*b, *loc, grf(32), i64(), /*execSize=*/16, dcanon(),
                           rcanon(), rcanon(), IntegerAttr(), IntegerAttr(),
                           IntegerAttr(), TypeAttr(), TypeAttr(),
                           /*noMask=*/false, maskOffset, left, right)
          .getResult();
    };
    return {addHalf(lhs.low, rhs.low, 0), addHalf(lhs.high, rhs.high, 16)};
  }

  WideValue subWide(WideValue lhs, WideValue rhs) {
    if (std::optional<int64_t> lhsImm = getImmediate(lhs.low))
      if (std::optional<int64_t> rhsImm = getImmediate(rhs.low))
        return getWideConstant(static_cast<uint64_t>(*lhsImm) -
                               static_cast<uint64_t>(*rhsImm));
    lhs = materializeWide(lhs);
    auto subHalf = [&](Value left, Value right, int maskOffset) {
      return SubOp::create(*b, *loc, grf(32), i64(), /*execSize=*/16, dcanon(),
                           rcanon(), rcanon(), IntegerAttr(), IntegerAttr(),
                           IntegerAttr(), TypeAttr(), TypeAttr(),
                           /*noMask=*/false, maskOffset, right, left)
          .getResult();
    };
    return {subHalf(lhs.low, rhs.low, 0), subHalf(lhs.high, rhs.high, 16)};
  }

  WideValue shiftWide(WideValue value, unsigned amount) {
    if (std::optional<int64_t> immediate = getImmediate(value.low))
      return getWideConstant(static_cast<uint64_t>(*immediate) << amount);
    auto shiftHalf = [&](Value half, int maskOffset) {
      return ShlOp::create(*b, *loc, grf(32), i64(), /*execSize=*/16, dcanon(),
                           rcanon(), RegionAttr(), IntegerAttr(), IntegerAttr(),
                           IntegerAttr(), TypeAttr(), TypeAttr(),
                           /*noMask=*/false, maskOffset, half,
                           imm(amount, i16()))
          .getResult();
    };
    return {shiftHalf(value.low, 0), shiftHalf(value.high, 16)};
  }

  WideValue multiplyWide(WideValue value, uint64_t multiplier) {
    if (multiplier == 0)
      return getWideConstant(0);
    WideValue result;
    for (unsigned bit : llvm::seq<unsigned>(64)) {
      if (!(multiplier & (uint64_t(1) << bit)))
        continue;
      WideValue term = bit == 0 ? value : shiftWide(value, bit);
      result = result.low ? addWide(result, term) : term;
    }
    return result;
  }

  FailureOr<Value> lowerPacked(Value value) {
    if (Value mapped = vmap.lookup(value))
      return mapped;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation *user = *argument.getUsers().begin();
      FailureOr<std::pair<Value, int>> location =
          getPayloadLocation(argument, KernelArgKind::by_value, 4, user);
      if (failed(location))
        return failure();
      return MovOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32, dcanon(),
                           runiform(), IntegerAttr(),
                           b->getI32IntegerAttr(location->second), TypeAttr(),
                           /*noMask=*/false, /*maskOffset=*/0, location->first)
          .getResult();
    }
    if (std::optional<int64_t> constant = getConstantIntValue(value))
      return imm(static_cast<int32_t>(*constant), i32());
    if (auto castOp = value.getDefiningOp<LLVM::ZExtOp>())
      return lowerPacked(castOp.getArg());
    if (auto castOp = value.getDefiningOp<LLVM::TruncOp>())
      return lowerPacked(castOp.getArg());
    if (auto add = value.getDefiningOp<LLVM::AddOp>()) {
      FailureOr<Value> lhs = lowerPacked(add.getLhs());
      FailureOr<Value> rhs = lowerPacked(add.getRhs());
      if (failed(lhs) || failed(rhs))
        return failure();
      return AddOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32, dcanon(),
                           rcanon(), rcanon(), IntegerAttr(), IntegerAttr(),
                           IntegerAttr(), TypeAttr(), TypeAttr(),
                           /*noMask=*/false, /*maskOffset=*/0, *lhs, *rhs)
          .getResult();
    }
    if (auto sub = value.getDefiningOp<LLVM::SubOp>()) {
      FailureOr<Value> lhs = lowerPacked(sub.getLhs());
      FailureOr<Value> rhs = lowerPacked(sub.getRhs());
      if (failed(lhs) || failed(rhs))
        return failure();
      return SubOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32, dcanon(),
                           rcanon(), rcanon(), IntegerAttr(), IntegerAttr(),
                           IntegerAttr(), TypeAttr(), TypeAttr(),
                           /*noMask=*/false, /*maskOffset=*/0, *rhs, *lhs)
          .getResult();
    }
    if (auto mul = value.getDefiningOp<LLVM::MulOp>()) {
      Value varying = mul.getLhs();
      std::optional<int64_t> constant = getConstantIntValue(mul.getRhs());
      if (!constant) {
        varying = mul.getRhs();
        constant = getConstantIntValue(mul.getLhs());
      }
      if (!constant)
        return mul.emitOpError(
                   "dynamic pointer-offset multiplication is not supported"),
               failure();
      FailureOr<Value> lowered = lowerPacked(varying);
      if (failed(lowered))
        return failure();
      uint32_t multiplier = static_cast<uint32_t>(*constant);
      Value result;
      for (unsigned bit : llvm::seq<unsigned>(32)) {
        if (!(multiplier & (uint32_t(1) << bit)))
          continue;
        Value term =
            bit == 0
                ? *lowered
                : ShlOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32,
                                dcanon(), rcanon(), RegionAttr(), IntegerAttr(),
                                IntegerAttr(), IntegerAttr(), TypeAttr(),
                                TypeAttr(), /*noMask=*/false, /*maskOffset=*/0,
                                *lowered, imm(bit, i16()))
                      .getResult();
        result =
            result ? AddOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32,
                                   dcanon(), rcanon(), rcanon(), IntegerAttr(),
                                   IntegerAttr(), IntegerAttr(), TypeAttr(),
                                   TypeAttr(), /*noMask=*/false,
                                   /*maskOffset=*/0, result, term)
                         .getResult()
                   : term;
      }
      return result ? result : imm(0, i32());
    }
    return value.getDefiningOp()->emitOpError(
               "unsupported 32-bit pointer offset expression"),
           failure();
  }

  FailureOr<WideValue> lowerWide(Value value) {
    if (auto found = wideOffsets.find(value); found != wideOffsets.end())
      return found->second;
    if (auto constant = value.getDefiningOp<xw::WideConstantOp>())
      return getWideConstant(static_cast<uint64_t>(constant.getValue()));
    if (auto extend = value.getDefiningOp<xw::WideExtendOp>()) {
      FailureOr<Value> packed = lowerPacked(extend.getInput());
      if (failed(packed))
        return failure();
      return widenPacked(*packed, extend.getSigned_());
    }
    if (auto add = value.getDefiningOp<xw::WideAddOp>()) {
      FailureOr<WideValue> lhs = lowerWide(add.getLhs());
      FailureOr<WideValue> rhs = lowerWide(add.getRhs());
      if (failed(lhs) || failed(rhs))
        return failure();
      return addWide(*lhs, *rhs);
    }
    if (auto sub = value.getDefiningOp<xw::WideSubOp>()) {
      FailureOr<WideValue> lhs = lowerWide(sub.getLhs());
      FailureOr<WideValue> rhs = lowerWide(sub.getRhs());
      if (failed(lhs) || failed(rhs))
        return failure();
      return subWide(*lhs, *rhs);
    }
    if (auto shl = value.getDefiningOp<xw::WideShlOp>()) {
      FailureOr<WideValue> lowered = lowerWide(shl.getInput());
      if (failed(lowered))
        return failure();
      return shiftWide(*lowered, static_cast<unsigned>(shl.getAmount()));
    }
    return value.getDefiningOp()->emitOpError(
               "expected decomposed wide offset"),
           failure();
  }

  FailureOr<WideValue> materializeGlobalPointer(Value pointer) {
    if (auto found = globalPointers.find(pointer);
        found != globalPointers.end())
      return found->second;
    if (auto argument = dyn_cast<BlockArgument>(pointer)) {
      FailureOr<std::pair<Value, int>> location =
          getPayloadLocation(argument, KernelArgKind::by_pointer, 8,
                             argument.getOwner()->getParentOp());
      if (failed(location))
        return failure();
      auto broadcast = [&](int maskOffset) {
        return MovOp::create(*b, *loc, grf(32), i64(), /*execSize=*/16,
                             dcanon(), runiform(), IntegerAttr(),
                             b->getI32IntegerAttr(location->second), TypeAttr(),
                             /*noMask=*/false, maskOffset, location->first)
            .getResult();
      };
      return WideValue{broadcast(0), broadcast(16)};
    }
    auto ptrAdd = pointer.getDefiningOp<xw::PtrAddOp>();
    if (!ptrAdd)
      return pointer.getDefiningOp()->emitOpError(
                 "unsupported global pointer producer"),
             failure();
    FailureOr<WideValue> base = materializeGlobalPointer(ptrAdd.getBase());
    FailureOr<WideValue> offset = lowerWide(ptrAdd.getOffset());
    if (failed(base) || failed(offset))
      return failure();
    WideValue result = addWide(*base, *offset);
    globalPointers[pointer] = result;
    return result;
  }

  FailureOr<Value> materializeSlmPointer(Value pointer) {
    if (Value found = slmPointers.lookup(pointer))
      return found;
    if (pointer.getDefiningOp<LLVM::AddressOfOp>()) {
      Value zero = MovOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32,
                                 DstRegionAttr(), RegionAttr(), IntegerAttr(),
                                 IntegerAttr(), TypeAttr(), /*noMask=*/false,
                                 /*maskOffset=*/0, imm(0, i32()))
                       .getResult();
      slmPointers[pointer] = zero;
      return zero;
    }
    if (isa<BlockArgument>(pointer))
      return emitError(pointer.getLoc(),
                       "SLM pointer block arguments are not supported"),
             failure();
    auto ptrAdd = pointer.getDefiningOp<xw::PtrAddOp>();
    if (!ptrAdd)
      return pointer.getDefiningOp()->emitOpError(
                 "unsupported SLM pointer producer"),
             failure();
    FailureOr<Value> base = materializeSlmPointer(ptrAdd.getBase());
    FailureOr<Value> offset = lowerPacked(ptrAdd.getOffset());
    if (failed(base) || failed(offset))
      return failure();
    Value result =
        AddOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32, dcanon(),
                      rcanon(), rcanon(), IntegerAttr(), IntegerAttr(),
                      IntegerAttr(), TypeAttr(), TypeAttr(), /*noMask=*/false,
                      /*maskOffset=*/0, *base, *offset)
            .getResult();
    slmPointers[pointer] = result;
    return result;
  }

  LogicalResult materializePointer(Value pointer) {
    auto pointerType = dyn_cast<LLVM::LLVMPointerType>(pointer.getType());
    if (!pointerType)
      return emitError(pointer.getLoc(), "expected an opaque LLVM pointer"),
             failure();
    if (pointerType.getAddressSpace() == 1) {
      if (failed(materializeGlobalPointer(pointer)))
        return failure();
      return success();
    }
    if (pointerType.getAddressSpace() == 3) {
      if (failed(materializeSlmPointer(pointer)))
        return failure();
      return success();
    }
    return emitError(pointer.getLoc(), "unsupported pointer address space"),
           failure();
  }

  FailureOr<Value> getGlobalAddressPayload(Value pointer) {
    FailureOr<WideValue> address = materializeGlobalPointer(pointer);
    if (failed(address))
      return failure();
    return TupleFromElementsOp::create(*b, *loc, grf(64),
                                       ValueRange{address->low, address->high})
        .getTuple();
  }

  bool isSlmAddress(Value pointer) {
    auto type = dyn_cast<LLVM::LLVMPointerType>(pointer.getType());
    return type && type.getAddressSpace() == 3;
  }

  LogicalResult emitLoad(Value address, Value result, Value depTok) {
    if (!result.getType().isInteger(32))
      return emitError(result.getLoc(), "only i32 loads are selected"),
             failure();
    if (failed(materializePointer(address)))
      return failure();
    if (isSlmAddress(address)) {
      FailureOr<Value> materialized = materializeSlmPointer(address);
      if (failed(materialized))
        return failure();
      auto op = LoadSLMOp::create(*b, *loc, grf(32), MemTokenType::get(ctx),
                                  *materialized, depTok, 32);
      memToken = op.getToken();
      vmap[result] = op.getDst();
      return success();
    }
    FailureOr<Value> payload = getGlobalAddressPayload(address);
    if (failed(payload))
      return failure();
    Value v = emitLoadA64(grf(32), *payload, depTok);
    vmap[result] = v;
    return success();
  }

  LogicalResult emitSum(Operation *op) {
    Value lhs = vmap.lookup(op->getOperand(0));
    Value rhs = vmap.lookup(op->getOperand(1));
    if (!lhs || !rhs)
      return emitError(op->getLoc(), "add operand not lowered"), failure();
    Value sum =
        AddOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32, dcanon(),
                      rcanon(), rcanon(), IntegerAttr(), IntegerAttr(),
                      IntegerAttr(), TypeAttr(), TypeAttr(), /*noMask=*/false,
                      /*maskOffset=*/0, lhs, rhs)
            .getResult();
    vmap[op->getResult(0)] = sum;
    return success();
  }

  LogicalResult emitStore(Value address, Value data, Value depTok) {
    if (!data)
      return emitError(address.getLoc(), "store value was not lowered"),
             failure();
    if (failed(materializePointer(address)))
      return failure();
    if (isSlmAddress(address)) {
      FailureOr<Value> materialized = materializeSlmPointer(address);
      if (failed(materialized))
        return failure();
      auto op = StoreSLMOp::create(*b, *loc, MemTokenType::get(ctx),
                                   *materialized, data, depTok, 32);
      memToken = op.getToken();
      return success();
    }
    FailureOr<Value> payload = getGlobalAddressPayload(address);
    if (failed(payload))
      return failure();
    emitStoreA64(*payload, data, depTok);
    return success();
  }

  void emitEot() {
    Value scratch = MovOp::create(*b, *loc, grf(16), i32(), /*execSize=*/16,
                                  dcanon(), rcanon(), IntegerAttr(),
                                  IntegerAttr(), TypeAttr(), /*noMask=*/true,
                                  /*maskOffset=*/0, archreg(0))
                        .getResult();
    EotOp::create(*b, *loc, scratch, memToken);
  }

  // barrier(CLK_*_MEM_FENCE): fence.slm -> drain -> signal -> sync.bar, all
  // token-chained (design doc: every part explicit, ordered by tokens).
  void emitBarrier(Value entryDep) {
    Value r0 = archreg(0);
    auto fence = FenceSLMOp::create(*b, *loc, grf(16), MemTokenType::get(ctx),
                                    r0, entryDep);
    memToken = fence.getToken();
    auto awaitOp = FenceAwaitOp::create(*b, *loc, MemTokenType::get(ctx),
                                        fence.getReadback(), memToken);
    memToken = awaitOp.getToken();

    // Barrier payload: zero GRF with dword 2 = 0x100 and byte 10 = r0.11.
    Value payload = MovOp::create(*b, *loc, grf(16), i32(), /*execSize=*/16,
                                  DstRegionAttr(), RegionAttr(), IntegerAttr(),
                                  IntegerAttr(), TypeAttr(), /*noMask=*/true,
                                  /*maskOffset=*/0, imm(0, i32()))
                        .getResult();
    Value control =
        MovOp::create(*b, *loc, grf(16), i32(), /*execSize=*/1, dcanon(),
                      RegionAttr(), b->getI32IntegerAttr(2), IntegerAttr(),
                      TypeAttr(), /*noMask=*/true, /*maskOffset=*/0,
                      imm(0x100, i32()))
            .getResult();
    payload =
        UpdateTupleOp::create(*b, *loc, grf(16), payload, ValueRange{control},
                              b->getArrayAttr({b->getI64IntegerAttr(0)}))
            .getResult();
    Value header =
        MovOp::create(*b, *loc, grf(16), IntegerType::get(ctx, 8),
                      /*execSize=*/2, dcanon(), runiform(),
                      b->getI32IntegerAttr(10), b->getI32IntegerAttr(11),
                      TypeAttr(), /*noMask=*/true, /*maskOffset=*/0, r0)
            .getResult();
    payload =
        UpdateTupleOp::create(*b, *loc, grf(16), payload, ValueRange{header},
                              b->getArrayAttr({b->getI64IntegerAttr(0)}))
            .getResult();
    auto sig = BarrierSignalOp::create(*b, *loc, MemTokenType::get(ctx),
                                       payload, memToken);
    memToken = sig.getToken();
    emitSync(SyncKind::bar);
  }

  FailureOr<Value> materializeDwordValue(Value value, Operation *user) {
    Value mapped = vmap.lookup(value);
    if (!mapped) {
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        FailureOr<std::pair<Value, int>> location =
            getPayloadLocation(argument, KernelArgKind::by_value, 4, user);
        if (failed(location))
          return failure();
        mapped =
            MovOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32, dcanon(),
                          runiform(), IntegerAttr(),
                          b->getI32IntegerAttr(location->second), TypeAttr(),
                          /*noMask=*/false, /*maskOffset=*/0, location->first)
                .getResult();
      } else {
        return user->emitOpError("value was not lowered"), failure();
      }
    }
    if (!mapped.getDefiningOp<ImmOp>())
      return mapped;
    return MovOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32,
                         DstRegionAttr(), RegionAttr(), IntegerAttr(),
                         IntegerAttr(), TypeAttr(), /*noMask=*/false,
                         /*maskOffset=*/0, mapped)
        .getResult();
  }

  // atomic_add: naive per-lane vector form; IGC's prefix-sum folding
  // is an optimization we do not replicate.
  LogicalResult emitAtomicAdd(xw::AtomicAddOp call) {
    if (!call.getValue().getType().isInteger(32) ||
        !call.getOld().getType().isInteger(32))
      return call.emitOpError("only i32 atomic operations are selected"),
             failure();
    auto pointerType =
        dyn_cast<LLVM::LLVMPointerType>(call.getAddress().getType());
    if (!pointerType || pointerType.getAddressSpace() != 1)
      return call.emitOpError("atomic address must be a global pointer"),
             failure();
    if (failed(materializePointer(call.getAddress())))
      return failure();
    FailureOr<Value> dependency =
        mapDependency(call.getOperation(), call.getDependency());
    if (failed(dependency))
      return failure();
    FailureOr<Value> address = getGlobalAddressPayload(call.getAddress());
    FailureOr<Value> data =
        materializeDwordValue(call.getValue(), call.getOperation());
    if (failed(address) || failed(data))
      return failure();
    auto op = AtomicIAddA64Op::create(*b, *loc, grf(32), MemTokenType::get(ctx),
                                      *address, *data, *dependency, 32);
    memToken = op.getToken();
    vmap[call.getToken()] = memToken;
    vmap[call.getOld()] = op.getDst();
    return success();
  }
};

} // namespace
