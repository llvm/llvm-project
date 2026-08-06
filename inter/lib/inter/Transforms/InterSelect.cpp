// inter-select-to-machine: lower an llvm-dialect kernel to xemachine ops.
//
// M1 scope: straight-line kernels, i32 lane values, A64 stateless global
// memory, one work-item id builtin. Physical GRFs are assigned by
// construction (bump allocation in emission order); the real regalloc
// transform loop replaces this in M4. Address-payload contiguity for
// scattered sends is guaranteed by allocation order.

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

struct SelectToMachine
    : public inter::impl::SelectToMachineBase<SelectToMachine> {
  void runOnOperation() override {
    LLVM::LLVMFuncOp kernel;
    getOperation().walk([&](LLVM::LLVMFuncOp f) {
      if (!f.isDeclaration() && !kernel)
        kernel = f;
    });
    if (!kernel) {
      getOperation().emitError("no kernel function found");
      return signalPassFailure();
    }
    if (failed(lowerKernel(kernel)))
      return signalPassFailure();
  }

  MLIRContext *ctx = nullptr;
  std::optional<Location> loc;
  std::optional<OpBuilder> b;
  int nextGRF = 4; // r0-r3 reserved: r0 header, r1 inline, r2/r3 scratch
  DenseMap<Value, Value> vmap;
  Value gidValue;
  Value tailValue;
  Value byteOffLo, byteOffHi;
  bool prologueEmitted = false;
  bool dataSyncEmitted = false;

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

  LogicalResult lowerKernel(LLVM::LLVMFuncOp kernel) {
    ctx = kernel.getContext();
    loc = kernel.getLoc();
    OpBuilder moduleBuilder(kernel);
    auto func = moduleBuilder.create<func::FuncOp>(
        kernel.getLoc(), kernel.getName(),
        moduleBuilder.getFunctionType({}, {}));
    func->setAttr("xemachine.target",
                  TargetAttr::get(ctx, moduleBuilder.getStringAttr("bmg")));
    b = OpBuilder::atBlockBegin(func.addEntryBlock());

    for (Operation &op : kernel.getBody().front()) {
      if (auto call = dyn_cast<LLVM::CallOp>(&op)) {
        auto callee = call.getCallee();
        if (!callee || !callee->starts_with("_Z13get_global_idj"))
          return emitError(op.getLoc(), "unsupported call"), failure();
        if (failed(emitPrologueAndGid()))
          return failure();
      } else if (isa<LLVM::AndOp>(&op)) {
        vmap[op.getResult(0)] = gidValue;
      } else if (isa<LLVM::GEPOp>(&op)) {
        continue; // lowered lazily at the memory op
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
      } else if (isa<LLVM::ReturnOp>(&op)) {
        emitEot();
      }
    }
    func::ReturnOp::create(*b, *loc);
    kernel.erase();
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
    return success();
  }

  LogicalResult emitSum(Operation *op) {
    if (!dataSyncEmitted) {
      SyncOp::create(*b, *loc, SyncKindAttr::get(ctx, SyncKind::allrd));
      dataSyncEmitted = true;
    }
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
