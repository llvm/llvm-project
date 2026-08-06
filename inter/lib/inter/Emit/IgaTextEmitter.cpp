// IGA text emitter for xemachine: func.func of machine ops -> IGA assembly.
//
// SWSB annotations are computed here from SSA def-use: A@d/I@d distance
// dependencies on the youngest producer (in-order pipe retirement covers the
// rest), token assignment on sends, and sync.allrd as the load-consumer
// barrier. Conservative by design; the real SWSB pass replaces this in M4.

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
  DenseMap<Value, int> immValue;
  int nextInsn = 0;
  int nextToken = 0;

  StringRef typeSuffix(Type ty) {
    if (ty.isInteger(16))
      return "uw";
    if (ty.isInteger(64))
      return "q";
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

  std::string operandRef(Value v, int sub, Type elemTy) {
    if (auto imm = v.getDefiningOp<ImmOp>())
      return (llvm::Twine("0x") + llvm::Twine::utohexstr(imm.getValue()) + ":" +
              typeSuffix(imm.getElemType()))
          .str();
    if (auto ar = v.getDefiningOp<ArchRegOp>())
      return regRef(v.getType(), sub, elemTy);
    if (auto af = v.getDefiningOp<ArfRegOp>()) {
      auto t = cast<ARFType>(v.getType());
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << stringifyARFFile(t.getFile()) << t.getIndex() << "." << sub;
      return ss.str();
    }
    if (auto mul = v.getDefiningOp<MulOp>()) {
      auto t = cast<ARFType>(v.getType());
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << stringifyARFFile(t.getFile()) << t.getIndex() << "." << sub;
      return ss.str();
    }
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

  Attribute getAttrOrNull(Operation *op, StringRef name) {
    return op->getAttr(name);
  }

  std::string swsbForAlu(Operation *op, ValueRange operands) {
    int youngest = -1;
    for (Value v : operands) {
      Operation *def = defOp.lookup(v);
      if (!def)
        continue;
      if (isa<SendOp>(def))
        continue; // covered by sync.allrd placement
      auto it = insnIndex.find(def);
      if (it == insnIndex.end())
        continue;
      int d = insnIndex[op] - it->second;
      if (d >= 1 && d <= 15)
        youngest = youngest < 0 ? d : std::min(youngest, d);
    }
    if (youngest < 0)
      return "";
    return (llvm::Twine("{I@") + llvm::Twine(youngest) + "}").str();
  }

  void emit() {}

  void emitFunc(func::FuncOp func) {
    os << "L0:\n";
    Block &blk = func.getBody().front();
    // First pass: def indices for distance computation.
    for (Operation &op : blk) {
      if (isa<ImmOp, ArchRegOp, ArfRegOp>(op))
        continue;
      for (Value r : op.getResults())
        defOp[r] = &op;
    }
    for (Operation &op : blk) {
      if (isa<ImmOp, ArchRegOp, ArfRegOp>(op))
        continue;
      insnIndex[&op] = nextInsn++;
      emitOp(&op);
    }
  }

  void emitOp(Operation *op) {
    if (auto sync = dyn_cast<SyncOp>(op)) {
      os << "        sync." << stringifySyncKind(sync.getKind()) << " null\n";
      return;
    }
    if (auto send = dyn_cast<SendOp>(op))
      return emitSend(send);
    if (auto ret = dyn_cast<func::ReturnOp>(op))
      return;
    emitAlu(op);
  }

  void emitSend(SendOp send) {
    int token = nextToken++;
    std::string annot;
    if (send.getEot())
      annot = "{EOT,$" + std::to_string(token) + "}";
    else
      annot = "{$" + std::to_string(token) + "}";

    // Distance dependency for ALU-produced payloads (address/data).
    int youngest = -1;
    for (Value v : send.getOperands()) {
      Operation *def = defOp.lookup(v);
      if (!def || isa<SendOp>(def))
        continue;
      auto it = insnIndex.find(def);
      if (it != insnIndex.end()) {
        int d = insnIndex[send] - it->second;
        if (d >= 1 && d <= 15)
          youngest = youngest < 0 ? d : std::min(youngest, d);
      }
    }
    if (youngest >= 0) {
      std::string dep = (send.getEot() ? "I@" : "A@") + std::to_string(youngest);
      annot = "{" + dep + ",$" + std::to_string(token) + "}";
      if (send.getEot())
        annot = "{EOT," + dep + ",$" + std::to_string(token) + "}";
    }

    os << (send.getNoMask() ? "(W)     " : "        ");
    os << "send." << stringifySendFn(send.getFn()) << " (" << send.getExecSize()
       << "|M0) ";
    // dst
    auto dstTy = cast<RegType>(send.getDst().getType());
    os << (dstTy.getWidthDwords() == 0 ? "null" : regRef(dstTy, 0, i32Ty()));
    os << "  ";
    os << regRef(cast<RegType>(send.getAddrPayload().getType()), 0, i32Ty());
    os << "  ";
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

  Type i32Ty() { return IntegerType::get(ctx, 32); }

  void emitAlu(Operation *op) {
    StringRef name = op->getName().getStringRef();
    // strip dialect prefix
    name = name.split('.').second;
    int execSize = 16;
    if (auto a = op->getAttrOfType<IntegerAttr>("execSize"))
      execSize = a.getInt();
    int maskOffset = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("maskOffset"))
      maskOffset = a.getInt();
    bool noMask = op->hasAttr("noMask");
    Type elemTy = cast<TypeAttr>(op->getAttr("elemType")).getValue();

    os << (noMask ? "(W)     " : "        ");
    os << name << " (" << execSize << "|M" << maskOffset << ")  ";

    // dst
    Value dst = op->getResult(0);
    Type dstTy = dst.getType();
    std::string dstRef;
    if (isa<ARFType>(dstTy)) {
      auto t = cast<ARFType>(dstTy);
      dstRef = (llvm::Twine(stringifyARFFile(t.getFile())) +
                llvm::Twine(t.getIndex()) + "." + llvm::Twine(0))
                   .str();
    } else {
      dstRef = regRef(dstTy, getSub(op, "dstSub"), elemTy);
    }
    std::string dstReg = "<1>";
    if (auto r = op->getAttrOfType<DstRegionAttr>("dstRegion"))
      dstReg = "<" + std::to_string(r.getHstride()) + ">";
    os << dstRef << dstReg << ":" << typeSuffix(elemTy) << "  ";

    // srcs
    std::string annot = swsbForAlu(op, op->getOperands());
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
      os << operandRef(v, getSub(op, sn), st);
      if (!isa<ImmOp>(v.getDefiningOp()))
        os << reg << ":" << typeSuffix(st);
      os << "  ";
    }
    os << annot << "\n";
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
