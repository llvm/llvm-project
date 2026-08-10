// inter-select-to-machine: lower an llvm-dialect kernel to xemachine ops.
//
// M1 scope: straight-line kernels, i32 lane values, A64 stateless global
// memory, one work-item id builtin. Physical GRFs are assigned by
// construction (bump allocation in emission order); the real regalloc
// transform loop replaces this in M4. Address-payload contiguity for
// scattered sends is guaranteed by allocation order.

#include "inter/Analysis/UniformityAnalysis.h"
#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Support/Builtins.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

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
constexpr int kFirstArgOffset = 24;
constexpr int kLocalIdLoadOffset = 0x20;
constexpr int kPerThreadPayloadSize = 192;

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
  int nextGRF = 4; // The dual-entry payload prologue reserves r0-r9.
  DenseMap<Value, Value> vmap;
  Value gidValue;
  Value localXValue;
  Value tailValue;
  Value byteOffLo, byteOffHi;
  bool prologueEmitted = false;

  Type grf(int dwords) {
    int base = nextGRF;
    nextGRF += (dwords + 15) / 16;
    return RegType::get(ctx, dwords, base);
  }
  Type i32() { return IntegerType::get(ctx, 32); }
  Type i16() { return IntegerType::get(ctx, 16); }
  Type i64() { return IntegerType::get(ctx, 64); }

  RegionAttr rcanon() { return RegionAttr::get(ctx, 1, 1, 0); }
  RegionAttr runiform() { return RegionAttr::get(ctx, 0, 1, 0); }
  RegionAttr rstride2() { return RegionAttr::get(ctx, 2, 1, 0); }
  DstRegionAttr dcanon() { return DstRegionAttr::get(ctx, 1); }
  DstRegionAttr dstride2() { return DstRegionAttr::get(ctx, 2); }
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
    memToken = load.getToken();

    // Keep the hardware-generated-local-ID entry at the zeinfo offset.
    emitSync(SyncKind::nop);
    emitSync(SyncKind::nop);
    emitSync(SyncKind::nop);
    emitSync(SyncKind::nop);
    nextGRF = 10;
  }

  LogicalResult lowerKernel(func::FuncOp kernel) {
    ctx = kernel.getContext();
    loc = kernel.getLoc();
    nextGRF = 4; // The dual-entry payload prologue reserves r0-r9.
    vmap.clear();
    memToken = nullptr;
    gidValue = nullptr;
    tailValue = nullptr;
    byteOffLo = byteOffHi = nullptr;
    prologueEmitted = false;

    OpBuilder moduleBuilder(kernel);
    auto func = func::FuncOp::create(moduleBuilder, kernel.getLoc(),
                                     (kernel.getName() + "_xm").str(),
                                     moduleBuilder.getFunctionType({}, {}));
    func->setAttr("xemachine.target",
                  TargetAttr::get(ctx, moduleBuilder.getStringAttr("bmg")));
    b = OpBuilder::atBlockBegin(func.addEntryBlock());

    bool usesThreadIds = false;
    kernel.walk([&](Operation *operation) {
      usesThreadIds |= isa<xw::GlobalIdOp, xw::LocalIdOp>(operation);
    });
    if (usesThreadIds && failed(emitPrologueAndGid()))
      return failure();
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

  LogicalResult lowerBlock(Block &blk) {
    for (Operation &op : blk) {
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
      } else if (isa<LLVM::AndOp, LLVM::TruncOp, LLVM::ZExtOp>(&op)) {
        // 64->32 id truncations: forward the mapped source value.
        vmap[op.getResult(0)] = vmap.lookup(op.getOperand(0));
      } else if (auto c = dyn_cast<LLVM::ConstantOp>(&op)) {
        auto intAttr = dyn_cast<IntegerAttr>(c.getValue());
        if (!intAttr)
          return emitError(op.getLoc(), "non-integer constant"), failure();
        vmap[op.getResult(0)] = imm(intAttr.getValue().getSExtValue(), i32());
      } else if (isa<LLVM::GEPOp>(&op)) {
        continue; // lowered lazily at the memory op
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
        if (failed(
                emitLoad(cast<LLVM::GEPOp>(load.getAddress().getDefiningOp()),
                         load.getValue(), *dependency)))
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
        if (failed(
                emitStore(cast<LLVM::GEPOp>(store.getAddress().getDefiningOp()),
                          vmap.lookup(store.getValue()), *dependency)))
          return failure();
        vmap[store.getToken()] = memToken;
      } else if (isa<LLVM::ReturnOp, func::ReturnOp>(&op)) {
        emitEot();
      }
    }
    return success();
  }

  // icmp predicate -> EU condition modifier; sign rides on the operand types.
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

  // Scalar (4-byte) kernel arg from the cross-thread tail load.
  std::pair<Value, int> scalarArg(int argIndex) {
    int offset = kFirstArgOffset + argIndex * 8; // 8-byte arg slots
    return {tailValue, (offset - kInlineMirrorSize) / 4};
  }

  // Resolve a cmp operand: kernel scalar args read the tail load, constants
  // become immediates, everything else comes from the value map.
  struct CmpOperand {
    Value v;
    int sub = 0;
    RegionAttr region;
  };
  CmpOperand cmpOperand(Value v) {
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      auto [tv, sub] = scalarArg(barg.getArgNumber());
      return {tv, sub, runiform()};
    }
    Value mapped = vmap.lookup(v);
    if (!mapped)
      return {nullptr, 0, runiform()};
    if (mapped.getDefiningOp<ImmOp>())
      return {mapped, 0, RegionAttr()};
    return {mapped, 0, rcanon()};
  }

  LogicalResult emitCmp(LLVM::ICmpOp icmp) {
    CmpOperand lhs = cmpOperand(icmp.getLhs());
    CmpOperand rhs = cmpOperand(icmp.getRhs());
    if (!lhs.v || !rhs.v)
      return emitError(icmp.getLoc(), "icmp operand not lowered"), failure();
    auto cond = mapPredicate(icmp.getPredicate());
    if (!cond)
      return emitError(icmp.getLoc(), "unsupported predicate"), failure();
    Value flag = CmpOp::create(*b, *loc, ARFType::get(ctx, ARFFile::f, 2, 0),
                               CondModifierAttr::get(ctx, *cond), ty(i32()),
                               b->getI32IntegerAttr(32), lhs.region, rhs.region,
                               b->getI32IntegerAttr(lhs.sub),
                               b->getI32IntegerAttr(rhs.sub), TypeAttr(),
                               TypeAttr(), lhs.v, rhs.v)
                     .getResult();
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
    emitLocalIdLoadEntry();
    emitSync(SyncKind::allwr);

    Value r0 = archreg(0);
    Value localX = archreg(1);
    Value inlineData = archreg(4);

    Value base =
        AndOp::create(*b, *loc, grf(16), i32(), /*execSize=*/1, dcanon(),
                      runiform(), RegionAttr(), IntegerAttr(), IntegerAttr(),
                      IntegerAttr(), TypeAttr(), TypeAttr(), /*noMask=*/true,
                      /*maskOffset=*/0, r0, imm(0xFFFFFFC0, i32()))
            .getResult();
    localXValue = localX;
    // Cross-thread tail: d32x8t at blob+0.
    tailValue = emitLoadBlock(grf(16), base, 8);
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
    emitByteOffsets(); // keep offset chains at top level: regions reference
                       // them
    return success();
  }

  // 32-bit gid -> byte offsets (gid * 4) as qwords, in two 16-lane halves.
  void emitByteOffsets() {
    if (byteOffLo)
      return;
    Value zeroLo = MovOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32,
                                 DstRegionAttr(), RegionAttr(), IntegerAttr(),
                                 IntegerAttr(), TypeAttr(), /*noMask=*/false,
                                 /*maskOffset=*/0, imm(0, i32()))
                       .getResult();
    Value zeroHi = MovOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32,
                                 DstRegionAttr(), RegionAttr(), IntegerAttr(),
                                 IntegerAttr(), TypeAttr(), /*noMask=*/false,
                                 /*maskOffset=*/0, imm(0, i32()))
                       .getResult();
    Value w0 = MovOp::create(*b, *loc, zeroLo.getType(), i32(),
                             /*execSize=*/16, dstride2(), rcanon(),
                             IntegerAttr(), IntegerAttr(), TypeAttr(),
                             /*noMask=*/false, /*maskOffset=*/0, gidValue)
                   .getResult();
    Value w1 =
        MovOp::create(*b, *loc, zeroHi.getType(), i32(),
                      /*execSize=*/16, dstride2(), rcanon(), IntegerAttr(),
                      b->getI32IntegerAttr(16), TypeAttr(), /*noMask=*/false,
                      /*maskOffset=*/16, gidValue)
            .getResult();
    byteOffLo =
        ShlOp::create(*b, *loc, grf(32), i64(), /*execSize=*/16, dcanon(),
                      rstride2(), RegionAttr(), IntegerAttr(), IntegerAttr(),
                      IntegerAttr(), ty(i32()), TypeAttr(), /*noMask=*/false,
                      /*maskOffset=*/0, w0, imm(2, i16()))
            .getResult();
    byteOffHi =
        ShlOp::create(*b, *loc, grf(32), i64(), /*execSize=*/16, dcanon(),
                      rstride2(), RegionAttr(), IntegerAttr(), IntegerAttr(),
                      IntegerAttr(), ty(i32()), TypeAttr(), /*noMask=*/false,
                      /*maskOffset=*/16, w1, imm(2, i16()))
            .getResult();
  }

  // Pointer arg: offset < 32 reads inline r1 (qword sub), else the tail load.
  std::pair<Value, int> pointerArg(int argIndex) {
    int offset = kFirstArgOffset + argIndex * 8;
    if (offset < kInlineMirrorSize)
      return {archreg(4), offset / 8};
    return {tailValue, (offset - kInlineMirrorSize) / 8};
  }

  // addr = byteOff + ptr per 16-lane half. The two results are adjacent GRF
  // pairs by allocation order; the send payload uses the first.
  std::pair<Value, Value> emitAddress(Value lo, Value hi, int argIndex) {
    auto [ptr, sub] = pointerArg(argIndex);
    Value a0 = AddOp::create(*b, *loc, grf(32), i64(), /*execSize=*/16,
                             dcanon(), rcanon(), runiform(), IntegerAttr(),
                             IntegerAttr(), b->getI32IntegerAttr(sub),
                             TypeAttr(), TypeAttr(), /*noMask=*/false,
                             /*maskOffset=*/0, lo, ptr)
                   .getResult();
    Value a1 = AddOp::create(*b, *loc, grf(32), i64(), /*execSize=*/16,
                             dcanon(), rcanon(), runiform(), IntegerAttr(),
                             IntegerAttr(), b->getI32IntegerAttr(sub),
                             TypeAttr(), TypeAttr(), /*noMask=*/false,
                             /*maskOffset=*/16, hi, ptr)
                   .getResult();
    return {a0, a1};
  }

  bool isSlmAddress(Value ptr) {
    auto pt = dyn_cast<LLVM::LLVMPointerType>(ptr.getType());
    return pt && pt.getAddressSpace() == 3;
  }

  // SLM byte address for a gep: index * 4. The index is the last gep index.
  Value emitSlmAddress(LLVM::GEPOp gep) {
    // GEP indices are PointerUnion<IntegerAttr, Value>; take the last one.
    Value idx;
    for (auto ix : gep.getIndices()) {
      if (auto cst = dyn_cast<IntegerAttr>(ix))
        idx = imm(cst.getInt(), i32());
      else
        idx = vmap.lookup(cast<Value>(ix));
    }
    if (!idx) {
      emitError(gep.getLoc(), "slm index not lowered");
      return nullptr;
    }
    // local-id-derived values are u16 lanes; everything else is dwords.
    TypeAttr srcTy = idx == localXValue ? ty(i16()) : ty(i32());
    return ShlOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32, dcanon(),
                         rcanon(), RegionAttr(), IntegerAttr(), IntegerAttr(),
                         IntegerAttr(), srcTy, TypeAttr(), /*noMask=*/false,
                         /*maskOffset=*/0, idx, imm(2, i16()))
        .getResult();
  }

  LogicalResult emitLoad(LLVM::GEPOp gep, Value result, Value depTok) {
    if (isSlmAddress(gep.getResult())) {
      Value addr = emitSlmAddress(gep);
      if (!addr)
        return failure();
      auto op = LoadSLMOp::create(*b, *loc, grf(32), MemTokenType::get(ctx),
                                  addr, depTok, 32);
      memToken = op.getToken();
      vmap[result] = op.getDst();
      return success();
    }
    if (!gidValue)
      return emitError(gep.getLoc(), "memory op before global id"), failure();
    int argIndex = cast<BlockArgument>(gep.getBase()).getArgNumber();
    emitByteOffsets();
    auto [a0, a1] = emitAddress(byteOffLo, byteOffHi, argIndex);
    Value v = emitLoadA64(grf(32), a0, depTok);
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

  LogicalResult emitStore(LLVM::GEPOp gep, Value data, Value depTok) {
    if (isSlmAddress(gep.getResult())) {
      Value addr = emitSlmAddress(gep);
      if (!addr)
        return failure();
      auto op = StoreSLMOp::create(*b, *loc, MemTokenType::get(ctx), addr, data,
                                   depTok, 32);
      memToken = op.getToken();
      return success();
    }
    int argIndex = cast<BlockArgument>(gep.getBase()).getArgNumber();
    auto [a0, a1] = emitAddress(byteOffLo, byteOffHi, argIndex);
    emitStoreA64(a0, data, depTok);
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
    payload = MovOp::create(*b, *loc, payload.getType(), i32(), /*execSize=*/1,
                            dcanon(), RegionAttr(), b->getI32IntegerAttr(2),
                            IntegerAttr(), TypeAttr(), /*noMask=*/true,
                            /*maskOffset=*/0, imm(0x100, i32()))
                  .getResult();
    payload = MovOp::create(*b, *loc, payload.getType(),
                            IntegerType::get(ctx, 8), /*execSize=*/2, dcanon(),
                            runiform(), b->getI32IntegerAttr(10),
                            b->getI32IntegerAttr(11), TypeAttr(),
                            /*noMask=*/true, /*maskOffset=*/0, r0)
                  .getResult();
    auto sig = BarrierSignalOp::create(*b, *loc, MemTokenType::get(ctx),
                                       payload, memToken);
    memToken = sig.getToken();
    emitSync(SyncKind::bar);
  }

  // atomic_add(ptr, 1): naive per-lane vector form; IGC's prefix-sum folding
  // is an optimization we do not replicate.
  LogicalResult emitAtomicAdd(xw::AtomicAddOp call) {
    auto barg = dyn_cast<BlockArgument>(call.getAddress());
    if (!barg)
      return emitError(call.getLoc(), "atomic arg must be a kernel pointer"),
             failure();
    FailureOr<Value> dependency =
        mapDependency(call.getOperation(), call.getDependency());
    if (failed(dependency))
      return failure();
    auto [ptr, sub] = pointerArg(barg.getArgNumber());
    // Per-lane address payload: the uniform pointer broadcast into qwords.
    Type i64t = i64();
    Value a0 = MovOp::create(*b, *loc, grf(32), i64t, /*execSize=*/16, dcanon(),
                             runiform(), IntegerAttr(),
                             b->getI32IntegerAttr(sub), TypeAttr(),
                             /*noMask=*/false, /*maskOffset=*/0, ptr)
                   .getResult();
    MovOp::create(*b, *loc, grf(32), i64t, /*execSize=*/16, dcanon(),
                  runiform(), IntegerAttr(), b->getI32IntegerAttr(sub),
                  TypeAttr(), /*noMask=*/false, /*maskOffset=*/16, ptr);
    Value ones = MovOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32,
                               DstRegionAttr(), RegionAttr(), IntegerAttr(),
                               IntegerAttr(), TypeAttr(), /*noMask=*/false,
                               /*maskOffset=*/0, imm(1, i16()))
                     .getResult();
    auto op = AtomicIAddA64Op::create(*b, *loc, grf(32), MemTokenType::get(ctx),
                                      a0, ones, *dependency, 32);
    memToken = op.getToken();
    vmap[call.getToken()] = memToken;
    vmap[call.getOld()] = op.getDst();
    return success();
  }
};

} // namespace
