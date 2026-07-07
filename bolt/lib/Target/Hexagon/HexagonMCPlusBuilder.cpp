//===- bolt/Target/Hexagon/HexagonMCPlusBuilder.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Hexagon-specific MCPlus builder.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/HexagonMCExpr.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "mcplus"

using namespace llvm;
using namespace bolt;

namespace {

class HexagonBundleEmissionState : public MCPlusBuilder::BundleEmissionState {
  const MCPlusBuilder &MIB;
  MCContext &Ctx;
  SmallVector<MCInst, 4> Packet;

  void doFlush(function_ref<void(const MCInst &)> EmitBundle) {
    if (Packet.empty())
      return;
    LLVM_DEBUG({
      dbgs() << "BOLT-DEBUG: flushing packet (" << Packet.size()
             << " instrs)\n";
    });
    // Detect hardware loop end markers. In the original encoding,
    // INST_PARSE_LOOP_END on instruction index 0 means endloop0
    // (inner loop), on index 1 means endloop1 (outer loop).
    bool InnerLoop = false;
    bool OuterLoop = false;
    for (unsigned I = 0, E = Packet.size(); I < E; ++I) {
      if (MIB.hasAnnotation(Packet[I], "HexLoopEnd")) {
        if (I == 0)
          InnerLoop = true;
        else if (I == 1)
          OuterLoop = true;
      }
    }
    MCInst Bundle = MIB.createBundle(Ctx, Packet, InnerLoop, OuterLoop);
    EmitBundle(Bundle);
    Packet.clear();
  }

public:
  HexagonBundleEmissionState(const MCPlusBuilder &MIB, MCContext &Ctx)
      : MIB(MIB), Ctx(Ctx) {}

  bool
  onBeginBasicBlock(const BinaryBasicBlock &BB, const BinaryBasicBlock *PrevBB,
                    function_ref<void(const MCInst &)> EmitBundle) override {
    if (Packet.empty())
      return true;
    // Carry the pending packet across the BB boundary when the BB is
    // a pure fallthrough: either unreachable (pred_size == 0) or the
    // only predecessor is the immediately preceding layout block.
    // Entry points do not force a flush on Hexagon: the processor
    // executes entire packets atomically, so a jump to a mid-packet
    // label still runs all instructions from the packet start.
    bool CanCarryAcross =
        BB.pred_size() == 0 ||
        (BB.pred_size() == 1 && PrevBB && *BB.pred_begin() == PrevBB);
    if (!CanCarryAcross || Packet.size() >= 4) {
      // Before flushing, check whether the next BB starts with a
      // .new value consumer (compare-and-jump or store). If so,
      // keep the packet open: the producer in the current packet
      // must stay in the same emitted packet as the consumer.
      bool NextIsNewValue = false;
      if (!BB.empty()) {
        auto First = BB.begin();
        while (First != BB.end() && MIB.isPseudo(*First))
          ++First;
        if (First != BB.end())
          NextIsNewValue = MIB.isNewValueConsumer(*First);
      }
      if (!NextIsNewValue)
        doFlush(EmitBundle);
    }
    return Packet.empty();
  }

  bool canEmitInstructionLabel() const override { return Packet.empty(); }

  bool
  processInstruction(MCInst &Inst,
                     function_ref<void(const MCInst &)> EmitBundle) override {
    bool IsPacketEnd = MIB.hasAnnotation(Inst, "HexPacketEnd");
    // Skip MC pseudo instructions (e.g. A2_nop) that the Hexagon code
    // emitter cannot encode. If the pseudo is at a packet end, flush
    // the packet without it.
    if (MIB.isPseudo(Inst)) {
      if (IsPacketEnd)
        doFlush(EmitBundle);
      return true;
    }
    // Safety: if the packet already has 4 instructions (Hexagon max),
    // flush before adding more.
    if (Packet.size() >= 4)
      doFlush(EmitBundle);
    Packet.push_back(Inst);
    if (IsPacketEnd)
      doFlush(EmitBundle);
    return true;
  }

  void flush(function_ref<void(const MCInst &)> EmitBundle) override {
    doFlush(EmitBundle);
  }
};

class HexagonMCPlusBuilder : public MCPlusBuilder {
  /// Mark an instruction as the end of a Hexagon packet. This helper
  /// casts away const because addAnnotation is non-const (it lazily
  /// registers annotation indices), but these create* methods are const
  /// in the base class.
  void setHexPacketEnd(MCInst &Inst) const {
    const_cast<HexagonMCPlusBuilder *>(this)->addAnnotation(
        Inst, "HexPacketEnd", true);
  }

  /// Create a symbol reference expression wrapped in HexagonMCExpr.
  /// The Hexagon MC layer requires all expressions be wrapped this way.
  const MCExpr *createHexExpr(const MCSymbol *Sym, MCContext &Ctx) const {
    return HexagonMCExpr::create(MCSymbolRefExpr::create(Sym, Ctx), Ctx);
  }

public:
  using MCPlusBuilder::MCPlusBuilder;

  bool equals(const MCSpecifierExpr &A, const MCSpecifierExpr &B,
              CompFuncTy Comp) const override {
    // Hexagon uses MCTargetExpr (HexagonMCExpr) rather than MCSpecifierExpr
    // for most expressions. Delegate to base class for any MCSpecifierExpr
    // that may appear.
    return MCPlusBuilder::equals(A, B, Comp);
  }

  void getCalleeSavedRegs(BitVector &Regs) const override {
    // Hexagon ABI: R16-R27 are callee-saved.
    // R29 (SP), R30 (FP), R31 (LR) are also callee-saved
    // (saved/restored by allocframe/deallocframe).
    Regs |= getAliases(Hexagon::R16);
    Regs |= getAliases(Hexagon::R17);
    Regs |= getAliases(Hexagon::R18);
    Regs |= getAliases(Hexagon::R19);
    Regs |= getAliases(Hexagon::R20);
    Regs |= getAliases(Hexagon::R21);
    Regs |= getAliases(Hexagon::R22);
    Regs |= getAliases(Hexagon::R23);
    Regs |= getAliases(Hexagon::R24);
    Regs |= getAliases(Hexagon::R25);
    Regs |= getAliases(Hexagon::R26);
    Regs |= getAliases(Hexagon::R27);
    Regs |= getAliases(Hexagon::R29);
    Regs |= getAliases(Hexagon::R30);
    Regs |= getAliases(Hexagon::R31);
  }

  bool shouldRecordCodeRelocation(uint32_t RelType) const override {
    switch (RelType) {
    case ELF::R_HEX_B22_PCREL:
    case ELF::R_HEX_B15_PCREL:
    case ELF::R_HEX_B7_PCREL:
    case ELF::R_HEX_B13_PCREL:
    case ELF::R_HEX_B9_PCREL:
    case ELF::R_HEX_B32_PCREL_X:
    case ELF::R_HEX_B22_PCREL_X:
    case ELF::R_HEX_B15_PCREL_X:
    case ELF::R_HEX_B13_PCREL_X:
    case ELF::R_HEX_B9_PCREL_X:
    case ELF::R_HEX_B7_PCREL_X:
    case ELF::R_HEX_32_PCREL:
    case ELF::R_HEX_6_PCREL_X:
    case ELF::R_HEX_32_6_X:
    case ELF::R_HEX_PLT_B22_PCREL:
    case ELF::R_HEX_GOT_32_6_X:
    case ELF::R_HEX_GOT_16_X:
    case ELF::R_HEX_GOT_11_X:
    case ELF::R_HEX_GOTREL_32_6_X:
    case ELF::R_HEX_GOTREL_16_X:
    case ELF::R_HEX_GOTREL_11_X:
    case ELF::R_HEX_TPREL_32_6_X:
    case ELF::R_HEX_TPREL_16_X:
    case ELF::R_HEX_TPREL_11_X:
    case ELF::R_HEX_LO16:
    case ELF::R_HEX_HI16:
    case ELF::R_HEX_16_X:
    case ELF::R_HEX_12_X:
    case ELF::R_HEX_11_X:
    case ELF::R_HEX_10_X:
    case ELF::R_HEX_9_X:
    case ELF::R_HEX_8_X:
    case ELF::R_HEX_7_X:
    case ELF::R_HEX_6_X:
    case ELF::R_HEX_32:
      return true;
    default:
      llvm_unreachable("Unexpected Hexagon relocation type in code");
    }
  }

  bool isNoop(const MCInst &Inst) const override {
    return Inst.getOpcode() == Hexagon::A2_nop;
  }

  bool isPseudo(const MCInst &Inst) const override {
    // V65 map-to-raw pseudos: the disassembler produces these for
    // allocframe, deallocframe, and dealloc_return. They are marked
    // Pseudo in tablegen but represent real instructions. Treat them
    // as non-pseudo so BOLT tracks them; they get lowered back to raw
    // forms during encoding (in lowerToRaw).
    switch (Inst.getOpcode()) {
    case Hexagon::S6_allocframe_to_raw:
    case Hexagon::L6_deallocframe_map_to_raw:
    case Hexagon::L6_return_map_to_raw:
    case Hexagon::L4_return_map_to_raw_t:
    case Hexagon::L4_return_map_to_raw_f:
    case Hexagon::L4_return_map_to_raw_tnew_pt:
    case Hexagon::L4_return_map_to_raw_fnew_pt:
    case Hexagon::L4_return_map_to_raw_tnew_pnt:
    case Hexagon::L4_return_map_to_raw_fnew_pnt:
      return false;
    default:
      return MCPlusBuilder::isPseudo(Inst);
    }
  }

  bool isIndirectCall(const MCInst &Inst) const override {
    if (!isCall(Inst))
      return false;

    switch (Inst.getOpcode()) {
    default:
      return false;
    case Hexagon::J2_callr:
    case Hexagon::J2_callrt:
    case Hexagon::J2_callrf:
      return true;
    }
  }

  bool hasPCRelOperand(const MCInst &Inst) const override {
    // Any direct branch or call has a PC-relative target operand.
    if ((isBranch(Inst) || isCall(Inst)) && !isIndirectBranch(Inst) &&
        !isIndirectCall(Inst))
      return HexagonMCInstrInfo::isExtendable(*Info, Inst);
    return false;
  }

  unsigned getInvertedBranchOpcode(unsigned Opcode) const {
    switch (Opcode) {
    default:
      llvm_unreachable("Failed to invert branch opcode");
      return Opcode;
    case Hexagon::J2_jumpt:
      return Hexagon::J2_jumpf;
    case Hexagon::J2_jumpf:
      return Hexagon::J2_jumpt;
    case Hexagon::J2_jumptnew:
      return Hexagon::J2_jumpfnew;
    case Hexagon::J2_jumpfnew:
      return Hexagon::J2_jumptnew;
    case Hexagon::J2_jumptnewpt:
      return Hexagon::J2_jumpfnewpt;
    case Hexagon::J2_jumpfnewpt:
      return Hexagon::J2_jumptnewpt;
    }
  }

  bool isReversibleBranch(const MCInst &Inst) const override {
    // Only simple J2 predicate-based conditional branches can be inverted.
    // J4 compound compare-and-jump instructions cannot be trivially inverted
    // because the comparison is fused into the opcode.
    switch (Inst.getOpcode()) {
    case Hexagon::J2_jumpt:
    case Hexagon::J2_jumpf:
    case Hexagon::J2_jumptnew:
    case Hexagon::J2_jumpfnew:
    case Hexagon::J2_jumptnewpt:
    case Hexagon::J2_jumpfnewpt:
      return true;
    default:
      return false;
    }
  }

  void reverseBranchCondition(MCInst &Inst, const MCSymbol *TBB,
                              MCContext *Ctx) const override {
    auto Opcode = getInvertedBranchOpcode(Inst.getOpcode());
    Inst.setOpcode(Opcode);
    replaceBranchTarget(Inst, TBB, Ctx);
  }

  void replaceBranchTarget(MCInst &Inst, const MCSymbol *TBB,
                           MCContext *Ctx) const override {
    unsigned SymOpIndex;
    bool Result = getSymbolRefOperandNum(Inst, SymOpIndex);
    assert(Result && "unimplemented branch");
    (void)Result;

    Inst.getOperand(SymOpIndex) =
        MCOperand::createExpr(createHexExpr(TBB, *Ctx));
  }

  IndirectBranchType analyzeIndirectBranch(
      MCInst &Instruction, InstructionIterator Begin, InstructionIterator End,
      const unsigned PtrSize, MCInst *&MemLocInstr, unsigned &BaseRegNum,
      unsigned &IndexRegNum, int64_t &DispValue, const MCExpr *&DispExpr,
      MCInst *&PCRelBaseOut, MCInst *&FixedEntryLoadInst) const override {
    MemLocInstr = nullptr;
    BaseRegNum = 0;
    IndexRegNum = 0;
    DispValue = 0;
    DispExpr = nullptr;
    PCRelBaseOut = nullptr;
    FixedEntryLoadInst = nullptr;

    // J2_jumpr R31 is a return, not an indirect branch.
    if (Instruction.getOpcode() == Hexagon::J2_jumpr &&
        Instruction.getOperand(0).getReg() == Hexagon::R31)
      return IndirectBranchType::UNKNOWN;

    // Other J2_jumpr instructions could be tail calls.
    if (Instruction.getOpcode() == Hexagon::J2_jumpr)
      return IndirectBranchType::POSSIBLE_TAIL_CALL;

    return IndirectBranchType::UNKNOWN;
  }

  bool convertJmpToTailCall(MCInst &Inst) override {
    if (isTailCall(Inst))
      return false;

    // Only convert unconditional jumps. Return false for conditional
    // branches so the caller can handle them as conditional tail calls.
    if (isConditionalBranch(Inst))
      return false;

    setTailCall(Inst);
    return true;
  }

  bool isEpilogue(const BinaryBasicBlock &BB) const override {
    // A BB ending with jumpr r31 or dealloc_return is an epilogue.
    for (auto I = BB.rbegin(), E = BB.rend(); I != E; ++I) {
      const MCInst &Inst = *I;
      if (isNoop(Inst))
        continue;
      // jumpr r31 is a return.
      if (Inst.getOpcode() == Hexagon::J2_jumpr && Inst.getNumOperands() > 0 &&
          Inst.getOperand(0).isReg() &&
          Inst.getOperand(0).getReg() == Hexagon::R31)
        return true;
      // dealloc_return variants.
      switch (Inst.getOpcode()) {
      case Hexagon::L4_return:
      case Hexagon::L6_return_map_to_raw:
      case Hexagon::L4_return_t:
      case Hexagon::L4_return_f:
      case Hexagon::L4_return_tnew_pnt:
      case Hexagon::L4_return_tnew_pt:
      case Hexagon::L4_return_fnew_pnt:
      case Hexagon::L4_return_fnew_pt:
      case Hexagon::L4_return_map_to_raw_t:
      case Hexagon::L4_return_map_to_raw_f:
      case Hexagon::L4_return_map_to_raw_tnew_pnt:
      case Hexagon::L4_return_map_to_raw_tnew_pt:
      case Hexagon::L4_return_map_to_raw_fnew_pnt:
      case Hexagon::L4_return_map_to_raw_fnew_pt:
        return true;
      default:
        return false;
      }
    }
    return false;
  }

  int getPCRelEncodingSize(const MCInst &Inst) const override {
    // ExtentBits already includes the alignment bits.  For example,
    // J2_jump (B22_PCREL) has ExtentBits=24, ExtentAlign=2; the raw
    // field width is 24-2=22 bits, and the byte-addressable range is
    // 2^(24-1) = +-8MB.  This matches the AArch64 convention where
    // getPCRelEncodingSize returns the total byte-addressable bits.
    return HexagonMCInstrInfo::getExtentBits(*Info, Inst);
  }

  int getUncondBranchEncodingSize() const override { return 24; }

  void createReturn(MCInst &Inst) const override {
    // jumpr R31
    Inst.setOpcode(Hexagon::J2_jumpr);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(Hexagon::R31));
    setHexPacketEnd(Inst);
  }

  void createUncondBranch(MCInst &Inst, const MCSymbol *TBB,
                          MCContext *Ctx) const override {
    Inst.setOpcode(Hexagon::J2_jump);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(createHexExpr(TBB, *Ctx)));
    setHexPacketEnd(Inst);
  }

  StringRef getTrapFillValue() const override {
    // trap0(#0) = 0x5400c000 (little-endian), with packet-end parse bits.
    static const char Trap[] = "\x00\xc0\x00\x54";
    return StringRef(Trap, 4);
  }

  void createCall(MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) override {
    Inst.setOpcode(Hexagon::J2_call);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(createHexExpr(Target, *Ctx)));
    setHexPacketEnd(Inst);
  }

  void createTailCall(MCInst &Inst, const MCSymbol *Target,
                      MCContext *Ctx) override {
    Inst.setOpcode(Hexagon::J2_jump);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(createHexExpr(Target, *Ctx)));
    setTailCall(Inst);
    setHexPacketEnd(Inst);
  }

  void createLongTailCall(InstructionListType &Seq, const MCSymbol *Target,
                          MCContext *Ctx) override {
    // Hexagon J2_jump with a constant extender covers the full 32-bit
    // address space, so a "long" tail call is the same as a regular one.
    MCInst Inst;
    createTailCall(Inst, Target, Ctx);
    Seq.push_back(Inst);
  }

  bool analyzeBranch(InstructionIterator Begin, InstructionIterator End,
                     const MCSymbol *&TBB, const MCSymbol *&FBB,
                     MCInst *&CondBranch,
                     MCInst *&UncondBranch) const override {
    auto I = End;

    while (I != Begin) {
      --I;

      // Ignore nops and CFIs
      if (isPseudo(*I) || isNoop(*I))
        continue;

      // Stop when we find the first non-terminator
      if (!isTerminator(*I) || isTailCall(*I) || !isBranch(*I))
        break;

      // Handle unconditional branches.
      if (isUnconditionalBranch(*I)) {
        // If any code was seen after this unconditional branch, we've seen
        // unreachable code. Ignore them.
        CondBranch = nullptr;
        UncondBranch = &*I;
        const MCSymbol *Sym = getTargetSymbol(*I);
        assert(Sym != nullptr &&
               "Couldn't extract BB symbol from jump operand");
        TBB = Sym;
        continue;
      }

      // Handle conditional branches and ignore indirect branches
      if (isIndirectBranch(*I))
        return false;

      if (CondBranch == nullptr) {
        const MCSymbol *TargetBB = getTargetSymbol(*I);
        if (TargetBB == nullptr) {
          // Unrecognized branch target
          return false;
        }
        FBB = TBB;
        TBB = TargetBB;
        CondBranch = &*I;
        continue;
      }

      llvm_unreachable("multiple conditional branches in one BB");
    }

    return true;
  }

  bool getSymbolRefOperandNum(const MCInst &Inst, unsigned &OpNum) const {
    // For direct branches and calls, the branch target is the extendable
    // operand. Use TSFlags to find it generically, covering J2 and J4
    // compound compare-and-jump instructions.
    if (HexagonMCInstrInfo::isExtendable(*Info, Inst)) {
      OpNum = HexagonMCInstrInfo::getExtendableOp(*Info, Inst);
      return true;
    }
    return false;
  }

  const MCSymbol *getTargetSymbol(const MCExpr *Expr) const override {
    if (auto *HExpr = dyn_cast<HexagonMCExpr>(Expr))
      return getTargetSymbol(HExpr->getExpr());

    return MCPlusBuilder::getTargetSymbol(Expr);
  }

  const MCSymbol *getTargetSymbol(const MCInst &Inst,
                                  unsigned OpNum = 0) const override {
    if (!OpNum && !getSymbolRefOperandNum(Inst, OpNum))
      return nullptr;

    const MCOperand &Op = Inst.getOperand(OpNum);
    if (!Op.isExpr())
      return nullptr;

    return getTargetSymbol(Op.getExpr());
  }

  bool lowerTailCall(MCInst &Inst) override {
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    if (getConditionalTailCall(Inst))
      unsetConditionalTailCall(Inst);
    return true;
  }

  void createNoop(MCInst &Inst) const override {
    Inst.clear();
    Inst.setOpcode(Hexagon::A2_nop);
    setHexPacketEnd(Inst);
  }

  void createTrap(MCInst &Inst) const override {
    Inst.clear();
    Inst.setOpcode(Hexagon::J2_trap0);
    Inst.addOperand(MCOperand::createImm(0));
    setHexPacketEnd(Inst);
  }

  bool isTrap(const MCInst &Inst) const override {
    return Inst.getOpcode() == Hexagon::J2_trap0 ||
           Inst.getOpcode() == Hexagon::J2_trap1;
  }

  MCPhysReg getStackPointer() const override { return Hexagon::R29; }

  MCPhysReg getFramePointer() const override { return Hexagon::R30; }

  MCPhysReg getFlagsReg() const override {
    // Hexagon has no monolithic flags register; predicate registers P0-P3
    // are used for conditional execution. Return 0 (no flags register).
    return 0;
  }

  std::optional<uint32_t>
  getInstructionSize(const MCInst &Inst) const override {
    // All Hexagon instructions are 4 bytes. This avoids the MCCodeEmitter
    // which asserts on non-BUNDLE instructions.
    return 4;
  }

  std::unique_ptr<BundleEmissionState>
  createBundleEmissionState(MCContext &Ctx) const override {
    return std::make_unique<HexagonBundleEmissionState>(*this, Ctx);
  }

  bool isNewValueConsumer(const MCInst &Inst) const override {
    return HexagonMCInstrInfo::isNewValue(*Info, Inst);
  }

  MCInst createBundle(MCContext &Ctx, ArrayRef<MCInst> Instrs,
                      bool InnerLoop = false,
                      bool OuterLoop = false) const override {
    MCInst Bundle;
    Bundle.setOpcode(Hexagon::BUNDLE);
    // Immediate operand encoding hardware loop end bits:
    // bit 0 = inner loop, bit 1 = outer loop.
    unsigned LoopBits = 0;
    if (InnerLoop)
      LoopBits |= 1;
    if (OuterLoop)
      LoopBits |= 2;
    Bundle.addOperand(MCOperand::createImm(LoopBits));

    unsigned RealInstrCount = 0;
    for (const MCInst &Inst : Instrs) {
      // Constant extender (immext / A4_ext) instructions must be
      // included in the bundle. The MC code emitter requires them
      // to set State.Extended, which tells it to encode only the
      // lower bits of the next instruction's immediate (the upper
      // bits come from the immext word). Do not count immext toward
      // RealInstrCount because it does not occupy a "real" slot for
      // purposes of hardware-loop NOP padding.
      if (Inst.getOpcode() == Hexagon::A4_ext) {
        MCInst *Copy = Ctx.createMCInst();
        *Copy = Inst;
        stripAnnotations(*Copy);
        Bundle.addOperand(MCOperand::createInst(Copy));
        continue;
      }
      MCInst *Copy = Ctx.createMCInst();
      *Copy = Inst;
      // Strip BOLT annotations before adding to bundle -- the streamer
      // and code emitter do not expect them.
      stripAnnotations(*Copy);
      // Lower map-to-raw pseudos back to raw forms for encoding.
      lowerToRaw(*Copy);
      // The Hexagon MC code emitter expects all immediate operands to be
      // wrapped in HexagonMCExpr. BOLT passes (e.g. peephole tail-call
      // traps) may create instructions with raw MCOperand::createImm()
      // immediates. Wrap them here so encoding succeeds.
      for (unsigned OpI = 0, OpE = Copy->getNumOperands(); OpI < OpE; ++OpI) {
        MCOperand &Op = Copy->getOperand(OpI);
        if (Op.isImm()) {
          const MCExpr *CE = MCConstantExpr::create(Op.getImm(), Ctx);
          Op = MCOperand::createExpr(HexagonMCExpr::create(CE, Ctx));
        }
      }
      assert((Copy->getNumOperands() == 0 || !Copy->getOperand(0).isImm()) &&
             "All operands must be wrapped in HexagonMCExpr");
      Bundle.addOperand(MCOperand::createInst(Copy));
      ++RealInstrCount;
    }

    // Hardware loop end packets require at least 2 instructions because
    // the loop end is encoded in the parse bits of slot 0 (inner) or
    // slot 1 (outer), while the packet end is encoded on the last slot.
    // These parse bits cannot be on the same instruction word. Pad with
    // a NOP if needed. This can happen when BOLT removes padding NOPs
    // from the end of a hardware loop body.
    unsigned MinSize = InnerLoop ? 2 : (OuterLoop ? 3 : 0);
    while (RealInstrCount < MinSize) {
      MCInst *Nop = Ctx.createMCInst();
      Nop->setOpcode(Hexagon::A2_nop);
      Bundle.addOperand(MCOperand::createInst(Nop));
      ++RealInstrCount;
    }

    return Bundle;
  }

  /// Lower map-to-raw pseudo instructions back to their real (raw)
  /// forms. The Hexagon disassembler simplifies certain instructions
  /// by removing implicit operands and using pseudo opcodes; we must
  /// reverse this before encoding.
  static void lowerToRaw(MCInst &Inst) {
    switch (Inst.getOpcode()) {
    default:
      break;
    case Hexagon::S6_allocframe_to_raw: {
      // (#Ii) -> S2_allocframe (R29, R29, #Ii)
      MCOperand Imm = Inst.getOperand(0);
      Inst.clear();
      Inst.setOpcode(Hexagon::S2_allocframe);
      Inst.addOperand(MCOperand::createReg(Hexagon::R29));
      Inst.addOperand(MCOperand::createReg(Hexagon::R29));
      Inst.addOperand(Imm);
      break;
    }
    case Hexagon::L6_return_map_to_raw:
      // () -> L4_return (D15, R30)
      Inst.clear();
      Inst.setOpcode(Hexagon::L4_return);
      Inst.addOperand(MCOperand::createReg(Hexagon::D15));
      Inst.addOperand(MCOperand::createReg(Hexagon::R30));
      break;
    case Hexagon::L6_deallocframe_map_to_raw:
      // () -> L2_deallocframe (D15, R30)
      Inst.clear();
      Inst.setOpcode(Hexagon::L2_deallocframe);
      Inst.addOperand(MCOperand::createReg(Hexagon::D15));
      Inst.addOperand(MCOperand::createReg(Hexagon::R30));
      break;
    // Conditional return variants: (Pv4) -> L4_return_* (D15, Pv4, R30)
    // The disassembler erased D15 (index 0) and R30 (index 2), leaving (Pv4).
    // Restore to (D15, Pv4, R30).
    case Hexagon::L4_return_map_to_raw_t:
    case Hexagon::L4_return_map_to_raw_f:
    case Hexagon::L4_return_map_to_raw_tnew_pt:
    case Hexagon::L4_return_map_to_raw_fnew_pt:
    case Hexagon::L4_return_map_to_raw_tnew_pnt:
    case Hexagon::L4_return_map_to_raw_fnew_pnt: {
      static const unsigned CondReturnOpcodes[][2] = {
          {Hexagon::L4_return_map_to_raw_t, Hexagon::L4_return_t},
          {Hexagon::L4_return_map_to_raw_f, Hexagon::L4_return_f},
          {Hexagon::L4_return_map_to_raw_tnew_pt, Hexagon::L4_return_tnew_pt},
          {Hexagon::L4_return_map_to_raw_fnew_pt, Hexagon::L4_return_fnew_pt},
          {Hexagon::L4_return_map_to_raw_tnew_pnt, Hexagon::L4_return_tnew_pnt},
          {Hexagon::L4_return_map_to_raw_fnew_pnt, Hexagon::L4_return_fnew_pnt},
      };
      unsigned NewOpc = 0;
      for (const auto &Pair : CondReturnOpcodes) {
        if (Pair[0] == Inst.getOpcode()) {
          NewOpc = Pair[1];
          break;
        }
      }
      assert(NewOpc && "Conditional return opcode not found in mapping");
      MCOperand Pred = Inst.getOperand(0);
      Inst.clear();
      Inst.setOpcode(NewOpc);
      Inst.addOperand(MCOperand::createReg(Hexagon::D15));
      Inst.addOperand(Pred);
      Inst.addOperand(MCOperand::createReg(Hexagon::R30));
      break;
    }
    }
  }

  bool evaluateMemOperandTarget(const MCInst &Inst, uint64_t &Target,
                                uint64_t Address,
                                uint64_t Size) const override {
    return false;
  }

  uint16_t getMinFunctionAlignment() const override {
    // Hexagon packets are 4-byte aligned.
    return 4;
  }

  void createStackPointerIncrement(MCInst &Inst, int Imm,
                                   bool NoFlagsClobber = false) const override {
    // R29 = add(R29, #-Imm)
    Inst = MCInstBuilder(Hexagon::A2_addi)
               .addReg(Hexagon::R29)
               .addReg(Hexagon::R29)
               .addImm(-Imm);
    setHexPacketEnd(Inst);
  }

  void createStackPointerDecrement(MCInst &Inst, int Imm,
                                   bool NoFlagsClobber = false) const override {
    // R29 = add(R29, #Imm)
    Inst = MCInstBuilder(Hexagon::A2_addi)
               .addReg(Hexagon::R29)
               .addReg(Hexagon::R29)
               .addImm(Imm);
    setHexPacketEnd(Inst);
  }
};

} // end anonymous namespace

namespace llvm {
namespace bolt {

MCPlusBuilder *createHexagonMCPlusBuilder(const MCInstrAnalysis *Analysis,
                                          const MCInstrInfo *Info,
                                          const MCRegisterInfo *RegInfo,
                                          const MCSubtargetInfo *STI) {
  return new HexagonMCPlusBuilder(Analysis, Info, RegInfo, STI);
}

} // namespace bolt
} // namespace llvm
