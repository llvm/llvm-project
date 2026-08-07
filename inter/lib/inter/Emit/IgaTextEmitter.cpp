// IGA text emitter for xemachine: func.func of machine ops -> IGA assembly.
//
// SWSB annotations are computed here from SSA def-use: A@d/I@d distance
// dependencies on the youngest producer (in-order pipe retirement covers the
// rest), token assignment on sends, and sync.allrd as the load-consumer
// barrier. Conservative by design; the real SWSB pass replaces this in M4.
//
// Control flow: exec_if/uniform_if lower to the predicated-goto + join
// pattern IGC emits (see the reference disasms in inter/docs). Labels are
// L1, L2, ... in emission order.

#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace inter::xemachine;

namespace {

struct Emitter {
  MLIRContext *ctx;
  llvm::raw_ostream &os;
  DenseMap<Value, Operation *> defOp;
  DenseMap<Operation *, int> insnIndex; // emitted instructions only
  int nextInsn = 0;
  int nextToken = 0;
  int nextLabel = 1;

  StringRef typeSuffix(Type ty) {
    if (ty.isInteger(8))
      return "ub";
    if (ty.isInteger(16))
      return "uw";
    if (ty.isInteger(64))
      return "q";
    if (ty.isF32())
      return "f";
    return "ud";
  }

  // Register reference: r{base+advance}.{subInTypeUnits}.
  std::string regRef(Type regTy, int sub, Type elemTy) {
    int unitBytes = elemTy.isInteger(64) ? 8 : elemTy.isInteger(16) ? 2 : 4;
    int base = 0;
    if (auto r = dyn_cast<RegType>(regTy))
      base = r.getBaseGRF();
    int advance = sub * unitBytes / 64;
    int rem = (sub * unitBytes) % 64 / unitBytes;
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "r" << (base + advance) << "." << rem;
    return ss.str();
  }

  std::string arfRef(ARFType t, int sub) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << stringifyARFFile(t.getFile()) << t.getIndex() << "." << sub;
    return ss.str();
  }

  std::string operandRef(Value v, int sub, Type elemTy) {
    if (auto imm = v.getDefiningOp<ImmOp>())
      return (llvm::Twine("0x") + llvm::Twine::utohexstr(imm.getValue()) + ":" +
              typeSuffix(imm.getElemType()))
          .str();
    if (isa<ARFType>(v.getType()))
      return arfRef(cast<ARFType>(v.getType()), sub);
    return regRef(v.getType(), sub, elemTy);
  }

  int getSub(Operation *op, StringRef name) {
    if (auto a = op->getAttrOfType<IntegerAttr>(name))
      return a.getInt();
    return 0;
  }

  Type getSrcType(Operation *op, StringRef name, Type fallback) {
    if (auto a = op->getAttrOfType<TypeAttr>(name))
      return a.getValue();
    return fallback;
  }

  // Distance to the youngest in-pipe producer among operands, if any.
  int youngestProducerDist(Operation *op, ValueRange operands) {
    int youngest = -1;
    for (Value v : operands) {
      if (isa<MemTokenType>(v.getType()))
        continue;
      Operation *def = defOp.lookup(v);
      if (!def || isa<SendOp>(def))
        continue;
      auto it = insnIndex.find(def);
      if (it == insnIndex.end())
        continue; // defined later in text order; should not happen pre-RA
      int d = insnIndex[op] - it->second;
      if (d >= 1 && d <= 15)
        youngest = youngest < 0 ? d : std::min(youngest, d);
    }
    return youngest;
  }

  void emitFunc(func::FuncOp func) {
    os << "L0:\n";
    func.walk([&](Operation *op) {
      if (isa<ImmOp, ArchRegOp, ArfRegOp>(op))
        return;
      for (Value r : op->getResults())
        defOp[r] = op;
    });
    emitBlock(func.getBody().front());
  }

  void emitBlock(Block &blk) {
    for (Operation &op : blk) {
      if (isa<ImmOp, ArchRegOp, ArfRegOp>(op))
        continue;
      if (isa<YieldOp>(op))
        continue; // structural; merge movs carry the data
      insnIndex[&op] = nextInsn++;
      if (auto send = dyn_cast<SendOp>(&op))
        emitSend(send);
      else if (auto sync = dyn_cast<SyncOp>(&op))
        emitSync(sync);
      else if (isa<LoadA64Op, StoreA64Op, LoadSLMOp, StoreSLMOp,
                   AtomicIAddA64Op, LoadBlockA32Op, FenceSLMOp, BarrierSignalOp,
                   EotOp>(op))
        emitMessage(&op);
      else if (auto await = dyn_cast<FenceAwaitOp>(&op))
        emitFenceAwait(await);
      else if (auto cmp = dyn_cast<CmpOp>(&op))
        emitCmp(cmp);
      else if (auto ifOp = dyn_cast<ExecIfOp>(&op))
        emitIf(ifOp.getOperation());
      else if (auto ifOp = dyn_cast<UniformIfOp>(&op))
        emitIf(ifOp.getOperation());
      else if (isa<func::ReturnOp>(&op))
        continue;
      else
        emitAlu(&op);
    }
  }

  std::string label(int n) { return "L" + std::to_string(n); }

  void emitGoto(StringRef pred, int jip, int uip) {
    insnIndex[nullptr] = nextInsn; // control instructions occupy slots
    os << "        ";
    if (!pred.empty())
      os << "(" << pred << ") ";
    os << "goto (32|M0)  " << label(jip) << "  " << label(uip) << "\n";
    ++nextInsn;
  }

  void emitJoin(int self, int uip) {
    os << label(self) << ":\n";
    os << "        join (32|M0)  " << label(uip) << "\n";
    ++nextInsn;
  }

  // exec_if/uniform_if -> the predicated-goto + join diamond. Matches the
  // IGC-emitted pattern:
  //   (~f) goto L1 L1 ; then ; goto L1 L2 ; L1: join L2 ; else ; L2: join L3
  void emitIf(Operation *ifOp) {
    Value cond = ifOp->getOperand(0);
    std::string flag = "~" + arfRef(cast<ARFType>(cond.getType()), 0);

    int l1 = nextLabel++, l2 = nextLabel++, l3 = nextLabel++;
    emitGoto(flag, l1, l1);
    if (auto ifo = dyn_cast<ExecIfOp>(ifOp))
      emitBlock(ifo.getThenRegion().front());
    else
      emitBlock(cast<UniformIfOp>(ifOp).getThenRegion().front());
    emitGoto("", l1, l2);
    emitJoin(l1, l2);
    if (auto ifo = dyn_cast<ExecIfOp>(ifOp)) {
      if (!ifo.getElseRegion().empty())
        emitBlock(ifo.getElseRegion().front());
    } else {
      auto uifo = cast<UniformIfOp>(ifOp);
      if (!uifo.getElseRegion().empty())
        emitBlock(uifo.getElseRegion().front());
    }
    emitJoin(l2, l3);
    os << label(l3) << ":\n";
  }

  void emitSync(SyncOp sync) {
    os << "        sync." << stringifySyncKind(sync.getKind());
    if (sync.getKind() == SyncKind::bar)
      os << " 0x0\n";
    else
      os << " null\n";
  }

  // Message descriptor table for the alias ops.
  struct MsgForm {
    StringRef fn;
    uint32_t desc;
    uint32_t exdesc;
    bool writesDst;
    bool isStore;
  };

  void emitMessage(Operation *op) {
    MsgForm m;
    if (isa<LoadA64Op>(op))
      m = {"ugm", 0x08200580, 0x0, true, false};
    else if (isa<StoreA64Op>(op))
      m = {"ugm", 0x08000584, 0x0, false, true};
    else if (isa<LoadSLMOp>(op))
      m = {"slm", 0x04200500, 0x0, true, false};
    else if (isa<StoreSLMOp>(op))
      m = {"slm", 0x04000504, 0x0, false, true};
    else if (isa<AtomicIAddA64Op>(op))
      m = {"ugm", 0x0410058C, 0x0, true, true};
    else if (isa<FenceSLMOp>(op))
      m = {"slm", 0x0210001F, 0x0, true, false};
    else if (isa<BarrierSignalOp>(op))
      m = {"gtwy", 0x02000004, 0x0, false, false};
    else if (isa<EotOp>(op))
      m = {"gtwy", 0x02000010, 0x0, false, false};
    else if (auto blk = dyn_cast<LoadBlockA32Op>(op)) {
      uint32_t desc = blk.getWords() == 32   ? 0x6229E500
                      : blk.getWords() == 16 ? 0x6219D500
                                             : 0x6219C500;
      m = {"ugm", desc, 0xFF000000, true, false};
    } else {
      op->emitError("unknown message op");
      return;
    }

    int token = nextToken++;
    int d = youngestProducerDist(op, op->getOperands());
    bool eot = isa<EotOp>(op);
    std::string annot;
    if (eot)
      annot = d >= 0 ? "{EOT,I@" + std::to_string(d) + ",$" +
                           std::to_string(token) + "}"
                     : "{EOT,$" + std::to_string(token) + "}";
    else if (d >= 0)
      annot = "{A@" + std::to_string(d) + ",$" + std::to_string(token) + "}";
    else
      annot = "{$" + std::to_string(token) + "}";

    int execSize = 1;
    if (auto a = op->getAttrOfType<IntegerAttr>("execSize"))
      execSize = a.getInt();
    bool noMask = op->hasAttr("noMask") || execSize == 1;

    Value addr = op->getOperand(0);
    Value data;
    if (auto st = dyn_cast<StoreA64Op>(op)) data = st.getDataPayload();
    else if (auto st = dyn_cast<StoreSLMOp>(op)) data = st.getDataPayload();
    else if (auto at = dyn_cast<AtomicIAddA64Op>(op)) data = at.getDataPayload();

    os << (noMask ? "(W)     " : "        ");
    os << "send." << m.fn << " (" << execSize << "|M0)  ";
    if (m.writesDst)
      os << regRef(cast<RegType>(op->getResult(0).getType()), 0, i32Ty());
    else
      os << "null";
    os << "  " << regRef(cast<RegType>(addr.getType()), 0, i32Ty()) << "  ";
    if (data) {
      auto dt = cast<RegType>(data.getType());
      std::string ref = regRef(dt, 0, i32Ty());
      if (dt.getWidthDwords() > 16)
        ref += ":" + std::to_string(dt.getWidthDwords() / 16);
      os << ref;
    } else {
      os << "null:0";
    }
    os << "  0x" << llvm::Twine::utohexstr(m.exdesc) << "  0x"
       << llvm::Twine::utohexstr(m.desc) << "           " << annot << "\n";
  }

  // Fence drain: a null-dst mov reading the fence readback register.
  void emitFenceAwait(FenceAwaitOp op) {
    int d = youngestProducerDist(op, op.getOperands());
    os << "(W)     mov (8|M0)  null<1>:ud  "
       << regRef(cast<RegType>(op.getReadback().getType()), 0, i32Ty())
       << "<1;1,0>:ud  ";
    if (d >= 0)
      os << "{I@" << d << "}";
    os << "\n";
  }

  void emitSend(SendOp send) {
    int token = nextToken++;
    int d = youngestProducerDist(send, send.getOperands());

    std::string annot;
    if (send.getEot())
      annot = d >= 0 ? "{EOT,I@" + std::to_string(d) + ",$" +
                           std::to_string(token) + "}"
                     : "{EOT,$" + std::to_string(token) + "}";
    else if (d >= 0)
      annot = "{A@" + std::to_string(d) + ",$" + std::to_string(token) + "}";
    else
      annot = "{$" + std::to_string(token) + "}";

    os << (send.getNoMask() ? "(W)     " : "        ");
    os << "send." << stringifySendFn(send.getFn()) << " (" << send.getExecSize()
       << "|M0)  ";
    auto dstTy = cast<RegType>(send.getDst().getType());
    os << (dstTy.getWidthDwords() == 0 ? "null"
                                       : regRef(dstTy, 0, i32Ty()));
    os << "  " << regRef(cast<RegType>(send.getAddrPayload().getType()), 0,
                          i32Ty())
       << "  ";
    if (Value data = send.getDataPayload()) {
      auto dt = cast<RegType>(data.getType());
      std::string ref = regRef(dt, 0, i32Ty());
      if (dt.getWidthDwords() > 16)
        ref += ":" + std::to_string(dt.getWidthDwords() / 16);
      os << ref;
    } else {
      os << "null:0";
    }
    os << "  0x" << llvm::Twine::utohexstr(send.getExdesc()) << "  0x"
       << llvm::Twine::utohexstr(send.getDesc()) << "           " << annot
       << "\n";
  }

  void emitCmp(CmpOp cmp) {
    std::string annot;
    int d = youngestProducerDist(cmp, cmp.getOperands());
    if (d >= 0)
      annot = "{I@" + std::to_string(d) + "}";
    os << "        cmp (" << cmp.getExecSize() << "|M0)  ("
       << stringifyCondModifier(cmp.getCond()) << ")"
       << arfRef(cast<ARFType>(cmp.getFlag().getType()), 0) << "   null<1>:"
       << typeSuffix(cmp.getElemType()) << "  ";
    for (auto [i, v] : llvm::enumerate(cmp.getOperands())) {
      std::string reg = "<1;1,0>";
      StringRef rn = i == 0 ? "src0Region" : "src1Region";
      if (auto r = cmp->getAttrOfType<RegionAttr>(rn))
        reg = "<" + std::to_string(r.getVstride()) + ";" +
              std::to_string(r.getWidth()) + "," +
              std::to_string(r.getHstride()) + ">";
      Type st = getSrcType(cmp, i == 0 ? "src0Type" : "src1Type",
                           cmp.getElemType());
      os << operandRef(v, getSub(cmp, i == 0 ? "src0Sub" : "src1Sub"), st);
      if (!isa<ImmOp>(v.getDefiningOp()))
        os << reg << ":" << typeSuffix(st);
      os << "  ";
    }
    os << annot << "\n";
  }

  Type i32Ty() { return IntegerType::get(ctx, 32); }

  void emitAlu(Operation *op) {
    bool isSub = isa<SubOp>(op);
    StringRef name = op->getName().getStringRef().split('.').second;
    int execSize = 16;
    if (auto a = op->getAttrOfType<IntegerAttr>("execSize"))
      execSize = a.getInt();
    int maskOffset = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("maskOffset"))
      maskOffset = a.getInt();
    bool noMask = op->hasAttr("noMask");
    Type elemTy = cast<TypeAttr>(op->getAttr("elemType")).getValue();

    os << (noMask ? "(W)     " : "        ");
    os << (isSub ? "add" : name) << " (" << execSize << "|M" << maskOffset
       << ")  ";

    Value dst = op->getResult(0);
    Type dstTy = dst.getType();
    std::string dstRef = isa<ARFType>(dstTy)
                             ? arfRef(cast<ARFType>(dstTy), 0)
                             : regRef(dstTy, getSub(op, "dstSub"), elemTy);
    std::string dstReg = "<1>";
    if (auto r = op->getAttrOfType<DstRegionAttr>("dstRegion"))
      dstReg = "<" + std::to_string(r.getHstride()) + ">";
    os << dstRef << dstReg << ":" << typeSuffix(elemTy) << "  ";

    for (auto [i, v] : llvm::enumerate(op->getOperands())) {
      std::string rn = "src" + std::to_string(i) + "Region";
      std::string sn = "src" + std::to_string(i) + "Sub";
      std::string tn = "src" + std::to_string(i) + "Type";
      std::string reg = "<1;1,0>";
      if (auto r = op->getAttrOfType<RegionAttr>(rn))
        reg = "<" + std::to_string(r.getVstride()) + ";" +
              std::to_string(r.getWidth()) + "," +
              std::to_string(r.getHstride()) + ">";
      Type st = getSrcType(op, tn, elemTy);
      if (isSub && i == 0)
        os << "-";
      os << operandRef(v, getSub(op, sn), st);
      if (!isa<ImmOp>(v.getDefiningOp()))
        os << reg << ":" << typeSuffix(st);
      os << "  ";
    }
    int d = youngestProducerDist(op, op->getOperands());
    if (d >= 0)
      os << "{I@" << d << "}";
    os << "\n";
  }
};

} // namespace

namespace inter {

mlir::LogicalResult emitIgaAsm(ModuleOp mod, llvm::raw_ostream &os) {
  func::FuncOp kernel;
  mod.walk([&](func::FuncOp f) {
    if (!kernel)
      kernel = f;
  });
  if (!kernel)
    return mod.emitError("no func.func kernel found"), failure();
  Emitter e{mod.getContext(), os};
  e.emitFunc(kernel);
  return success();
}

} // namespace inter
