// inter-select-to-machine: lower an llvm-dialect kernel to xemachine ops.
//
// M1 scope: straight-line kernels, i32 lane values, A64 stateless global
// memory, one work-item id builtin. Physical GRFs are assigned by
// construction (bump allocation in emission order); the real regalloc
// transform loop replaces this in M4. Address-payload contiguity for
// scattered sends is guaranteed by allocation order.

#include "inter/Analysis/UniformityAnalysis.h"
#include "inter/Support/Builtins.h"
#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

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

struct SelectToMachine
    : public inter::impl::SelectToMachineBase<SelectToMachine> {
  void runOnOperation() override {
    func::FuncOp kernel;
    getOperation().walk([&](func::FuncOp f) {
      if (f->hasAttr("xemachine.kernel") && !kernel)
        kernel = f;
    });
    if (!kernel) {
      getOperation().emitError("no kernel function found");
      return signalPassFailure();
    }
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<inter::UniformityAnalysis>();
    // Run on the kernel: sparse liveness seeds from the top op's region.
    if (failed(solver.initializeAndRun(kernel)))
      return signalPassFailure();
    uniformity = &solver;
    if (failed(lowerKernel(kernel)))
      return signalPassFailure();
  }

  MLIRContext *ctx = nullptr;
  DataFlowSolver *uniformity = nullptr;
  std::optional<Location> loc;
  std::optional<OpBuilder> b;
  int nextGRF = 4; // r0-r3 reserved: r0 header, r1 inline, r2/r3 scratch
  DenseMap<Value, Value> vmap;
  Value gidValue;
  Value tailValue;
  Value byteOffLo, byteOffHi;
  bool prologueEmitted = false;
  bool loadsPending = false;

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

  Value emitSend(Type dstTy, Value addrPayload, Value dataPayload, SendFn fn,
                 int desc, int exdesc, int execSize, bool noMask, bool eot) {
    return SendOp::create(
               *b, *loc, dstTy, MemTokenType::get(ctx),
               SendFnAttr::get(ctx, fn), b->getI32IntegerAttr(0),
               b->getI32IntegerAttr(desc), b->getI32IntegerAttr(exdesc),
               b->getI32IntegerAttr(execSize),
               noMask ? b->getUnitAttr() : UnitAttr(),
               eot ? b->getUnitAttr() : UnitAttr(), addrPayload, dataPayload,
               /*dependency=*/Value(), /*swsb=*/IntegerAttr())
        .getDst();
  }

  LogicalResult lowerKernel(func::FuncOp kernel) {
    ctx = kernel.getContext();
    loc = kernel.getLoc();
    OpBuilder moduleBuilder(kernel);
    auto func = moduleBuilder.create<func::FuncOp>(
        kernel.getLoc(), (kernel.getName() + "_xm").str(),
        moduleBuilder.getFunctionType({}, {}));
    func->setAttr("xemachine.target",
                  TargetAttr::get(ctx, moduleBuilder.getStringAttr("bmg")));
    b = OpBuilder::atBlockBegin(func.addEntryBlock());

    if (failed(lowerBlock(kernel.getBody().front())))
      return failure();
    func::ReturnOp::create(*b, *loc);
    std::string name = kernel.getName().str();
    kernel.erase();
    func.setName(StringAttr::get(ctx, name));
    return success();
  }

  // One dispatch step per op; regions recurse.
  LogicalResult lowerBlock(Block &blk) {
    for (Operation &op : blk) {
      if (auto call = dyn_cast<LLVM::CallOp>(&op)) {
        auto callee = call.getCallee();
        if (!callee || !callee->starts_with(inter::builtins::kGetGlobalId))
          return emitError(op.getLoc(), "unsupported call"), failure();
        if (failed(emitPrologueAndGid()))
          return failure();
      } else if (isa<LLVM::AndOp>(&op)) {
        vmap[op.getResult(0)] = gidValue;
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
      } else if (auto load = dyn_cast<LLVM::LoadOp>(&op)) {
        if (failed(emitLoad(cast<LLVM::GEPOp>(load.getAddr().getDefiningOp()),
                            load.getResult())))
          return failure();
      } else if (isa<LLVM::AddOp>(&op)) {
        if (failed(emitSum(&op)))
          return failure();
      } else if (auto store = dyn_cast<LLVM::StoreOp>(&op)) {
        if (failed(emitStore(cast<LLVM::GEPOp>(store.getAddr().getDefiningOp()),
                             vmap.lookup(store.getValue()))))
          return failure();
      } else if (isa<LLVM::ReturnOp, func::ReturnOp>(&op)) {
        emitEot();
      }
    }
    return success();
  }

  // icmp predicate -> EU condition modifier; sign rides on the operand types.
  std::optional<CondModifier> mapPredicate(LLVM::ICmpPredicate pred) {
    switch (pred) {
    case LLVM::ICmpPredicate::eq: return CondModifier::eq;
    case LLVM::ICmpPredicate::ne: return CondModifier::ne;
    case LLVM::ICmpPredicate::ugt:
    case LLVM::ICmpPredicate::sgt: return CondModifier::gt;
    case LLVM::ICmpPredicate::uge:
    case LLVM::ICmpPredicate::sge: return CondModifier::ge;
    case LLVM::ICmpPredicate::ult:
    case LLVM::ICmpPredicate::slt: return CondModifier::lt;
    case LLVM::ICmpPredicate::ule:
    case LLVM::ICmpPredicate::sle: return CondModifier::le;
    default: return std::nullopt;
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
    maybeSync({lhs.v, rhs.v});
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
    for (Value r : ifOp.getResults())
      resultTypes.push_back(grf(32));

    Operation *ifm;
    if (varying)
      ifm = ExecIfOp::create(*b, *loc, resultTypes, cond);
    else
      ifm = UniformIfOp::create(*b, *loc, resultTypes, cond);
    for (auto [i, r] : llvm::enumerate(ifOp.getResults()))
      vmap[r] = ifm->getResult(i);

    return emitIfRegions(ifOp, ifm, varying);
  }

  LogicalResult emitIfRegions(scf::IfOp ifOp, Operation *ifm, bool varying) {
    Region *thenR = varying ? &cast<ExecIfOp>(ifm).getThenRegion()
                            : &cast<UniformIfOp>(ifm).getThenRegion();
    Region *elseR = varying ? &cast<ExecIfOp>(ifm).getElseRegion()
                            : &cast<UniformIfOp>(ifm).getElseRegion();
    std::array<std::pair<Region *, Region *>, 2> regions = {
        {{&ifOp.getThenRegion(), thenR}, {&ifOp.getElseRegion(), elseR}}};
    for (auto [scfRegion, machineRegion] : regions) {
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
        // The mov result aliases the exec_if result register; yielding it
        // keeps types consistent along the region-branch edges.
        maybeSync({v});
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

  // r0.0 & ~0x3F = blob base; per-thread local X at blob+0x20; cross-thread
  // tail at blob+0. gid = r0.1 * enq_local.x + localX + gid_off.x.
  LogicalResult emitPrologueAndGid() {
    if (prologueEmitted)
      return success();
    prologueEmitted = true;
    Value r0 = archreg(0);
    Value r1 = archreg(1);

    Value base = AndOp::create(*b, *loc, grf(16), i32(), /*execSize=*/1,
                               dcanon(), runiform(), RegionAttr(),
                               IntegerAttr(), IntegerAttr(), IntegerAttr(),
                               TypeAttr(), TypeAttr(), /*noMask=*/true,
                               /*maskOffset=*/0, r0, imm(0xFFFFFFC0, i32()))
                     .getResult();
    Value perThreadAddr =
        AddOp::create(*b, *loc, grf(16), i32(), /*execSize=*/1, dcanon(),
                      runiform(), RegionAttr(), IntegerAttr(), IntegerAttr(),
                      IntegerAttr(), TypeAttr(), TypeAttr(), /*noMask=*/true,
                      /*maskOffset=*/0, base, imm(kLocalIdLoadOffset, i32()))
            .getResult();

    // Local X ids: d32x16t at blob+0x20 (one GRF: 32 lanes of u16).
    Value localX = emitSend(grf(16), perThreadAddr, Value(), SendFn::ugm,
                            0x6219D500, 0xFF000000, 1, true, false);
    // Cross-thread tail: d32x8t at blob+0.
    tailValue = emitSend(grf(16), base, Value(), SendFn::ugm, 0x6219C500,
                         0xFF000000, 1, true, false);
    SyncOp::create(*b, *loc, SyncKindAttr::get(ctx, SyncKind::allrd));

    // gid base: groupX * enq_local_size.x, via the accumulator.
    Value acc = MulOp::create(*b, *loc, ARFType::get(ctx, ARFFile::acc, 16, 0),
                              i32(), /*execSize=*/1, dcanon(), runiform(),
                              runiform(), IntegerAttr(),
                              b->getI32IntegerAttr(1), b->getI32IntegerAttr(3),
                              /*noMask=*/true, /*maskOffset=*/0, r0, r1)
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
                              localX, r1)
                   .getResult();
    emitByteOffsets(); // keep offset chains at top level: regions reference them
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
    Value w1 = MovOp::create(*b, *loc, zeroHi.getType(), i32(),
                             /*execSize=*/16, dstride2(), rcanon(),
                             IntegerAttr(), b->getI32IntegerAttr(16),
                             TypeAttr(), /*noMask=*/false, /*maskOffset=*/16,
                             gidValue)
                   .getResult();
    byteOffLo = ShlOp::create(*b, *loc, grf(32), i64(), /*execSize=*/16,
                              dcanon(), rstride2(), RegionAttr(), IntegerAttr(),
                              IntegerAttr(), IntegerAttr(), ty(i32()),
                              TypeAttr(), /*noMask=*/false, /*maskOffset=*/0,
                              w0, imm(2, i16()))
                    .getResult();
    byteOffHi = ShlOp::create(*b, *loc, grf(32), i64(), /*execSize=*/16,
                              dcanon(), rstride2(), RegionAttr(), IntegerAttr(),
                              IntegerAttr(), IntegerAttr(), ty(i32()),
                              TypeAttr(), /*noMask=*/false, /*maskOffset=*/16,
                              w1, imm(2, i16()))
                    .getResult();
  }

  // Pointer arg: offset < 32 reads inline r1 (qword sub), else the tail load.
  std::pair<Value, int> pointerArg(int argIndex) {
    int offset = kFirstArgOffset + argIndex * 8;
    if (offset < kInlineMirrorSize)
      return {archreg(1), offset / 8};
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

  LogicalResult emitLoad(LLVM::GEPOp gep, Value result) {
    if (!gidValue)
      return emitError(gep.getLoc(), "memory op before global id"), failure();
    int argIndex = cast<BlockArgument>(gep.getBase()).getArgNumber();
    emitByteOffsets();
    auto [a0, a1] = emitAddress(byteOffLo, byteOffHi, argIndex);
    Value v = emitSend(grf(32), a0, Value(), SendFn::ugm, 0x08200580, 0, 32,
                       false, false);
    vmap[result] = v;
    loadsPending = true;
    return success();
  }

  // sync.allrd between a load and its first consumer; tokens/trackers are
  // the SWSB milestone's business.
  void maybeSync(ValueRange vs) {
    if (!loadsPending)
      return;
    for (Value v : vs) {
      if (v && v.getDefiningOp<SendOp>()) {
        SyncOp::create(*b, *loc, SyncKindAttr::get(ctx, SyncKind::allrd));
        loadsPending = false;
        return;
      }
    }
  }

  LogicalResult emitSum(Operation *op) {
    maybeSync({vmap.lookup(op->getOperand(0)), vmap.lookup(op->getOperand(1))});
    Value lhs = vmap.lookup(op->getOperand(0));
    Value rhs = vmap.lookup(op->getOperand(1));
    if (!lhs || !rhs)
      return emitError(op->getLoc(), "add operand not lowered"), failure();
    Value sum = AddOp::create(*b, *loc, grf(32), i32(), /*execSize=*/32,
                              dcanon(), rcanon(), rcanon(), IntegerAttr(),
                              IntegerAttr(), IntegerAttr(), TypeAttr(),
                              TypeAttr(), /*noMask=*/false, /*maskOffset=*/0,
                              lhs, rhs)
                      .getResult();
    vmap[op->getResult(0)] = sum;
    return success();
  }

  LogicalResult emitStore(LLVM::GEPOp gep, Value data) {
    int argIndex = cast<BlockArgument>(gep.getBase()).getArgNumber();
    auto [a0, a1] = emitAddress(byteOffLo, byteOffHi, argIndex);
    emitSend(grf(0), a0, data, SendFn::ugm, 0x08000584, 0, 32, false, false);
    return success();
  }

  void emitEot() {
    Value scratch = MovOp::create(*b, *loc, grf(16), i32(), /*execSize=*/16,
                                  dcanon(), rcanon(), IntegerAttr(),
                                  IntegerAttr(), TypeAttr(), /*noMask=*/true,
                                  /*maskOffset=*/0, archreg(0))
                        .getResult();
    emitSend(grf(0), scratch, Value(), SendFn::gtwy, 0x02000010, 0, 1, true,
             true);
  }
};

} // namespace
