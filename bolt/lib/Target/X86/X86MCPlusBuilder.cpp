//===- bolt/Target/X86/X86MCPlusBuilder.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides X86-specific MCPlus builder.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86EncodingOptimization.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "X86MCSymbolizer.h"
#include "bolt/Core/MCPlus.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include <set>

#define DEBUG_TYPE "mcplus"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

static cl::opt<bool> X86StripRedundantAddressSize(
    "x86-strip-redundant-address-size",
    cl::desc("Remove redundant Address-Size override prefix"), cl::init(true),
    cl::cat(BoltOptCategory));

} // namespace opts

namespace {

bool isMOVSX64rm32(const MCInst &Inst) {
  return Inst.getOpcode() == X86::MOVSX64rm32;
}

bool isADD64rr(const MCInst &Inst) { return Inst.getOpcode() == X86::ADD64rr; }

bool isADDri(const MCInst &Inst) {
  return Inst.getOpcode() == X86::ADD64ri32 ||
         Inst.getOpcode() == X86::ADD64ri8;
}

// Create instruction to increment contents of target by 1
static InstructionListType createIncMemory(const MCSymbol *Target,
                                           MCContext *Ctx) {
  InstructionListType Insts;
  Insts.emplace_back();
  Insts.back().setOpcode(X86::LOCK_INC64m);
  Insts.back().clear();
  Insts.back().addOperand(MCOperand::createReg(X86::RIP));        // BaseReg
  Insts.back().addOperand(MCOperand::createImm(1));               // ScaleAmt
  Insts.back().addOperand(MCOperand::createReg(X86::NoRegister)); // IndexReg

  Insts.back().addOperand(MCOperand::createExpr(
      MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None,
                              *Ctx))); // Displacement
  Insts.back().addOperand(
      MCOperand::createReg(X86::NoRegister)); // AddrSegmentReg
  return Insts;
}

#define GET_INSTRINFO_OPERAND_TYPES_ENUM
#define GET_INSTRINFO_OPERAND_TYPE
#define GET_INSTRINFO_MEM_OPERAND_SIZE
#include "X86GenInstrInfo.inc"

class X86MCPlusBuilder : public MCPlusBuilder {
public:
  using MCPlusBuilder::MCPlusBuilder;

  std::unique_ptr<MCSymbolizer>
  createTargetSymbolizer(BinaryFunction &Function,
                         bool CreateNewSymbols) const override {
    return std::make_unique<X86MCSymbolizer>(Function, CreateNewSymbols);
  }

  bool isBranch(const MCInst &Inst) const override {
    return Analysis->isBranch(Inst) && !isTailCall(Inst);
  }

  bool isNoop(const MCInst &Inst) const override {
    return X86::isNOP(Inst.getOpcode());
  }

  unsigned getCondCode(const MCInst &Inst) const override {
    unsigned Opcode = Inst.getOpcode();
    if (X86::isJCC(Opcode))
      return Inst.getOperand(Info->get(Opcode).NumOperands - 1).getImm();
    return X86::COND_INVALID;
  }

  unsigned getInvertedCondCode(unsigned CC) const override {
    switch (CC) {
    default: return X86::COND_INVALID;
    case X86::COND_E:  return X86::COND_NE;
    case X86::COND_NE: return X86::COND_E;
    case X86::COND_L:  return X86::COND_GE;
    case X86::COND_LE: return X86::COND_G;
    case X86::COND_G:  return X86::COND_LE;
    case X86::COND_GE: return X86::COND_L;
    case X86::COND_B:  return X86::COND_AE;
    case X86::COND_BE: return X86::COND_A;
    case X86::COND_A:  return X86::COND_BE;
    case X86::COND_AE: return X86::COND_B;
    case X86::COND_S:  return X86::COND_NS;
    case X86::COND_NS: return X86::COND_S;
    case X86::COND_P:  return X86::COND_NP;
    case X86::COND_NP: return X86::COND_P;
    case X86::COND_O:  return X86::COND_NO;
    case X86::COND_NO: return X86::COND_O;
    }
  }

  unsigned getCondCodesLogicalOr(unsigned CC1, unsigned CC2) const override {
    enum DecodedCondCode : uint8_t {
      DCC_EQUAL = 0x1,
      DCC_GREATER = 0x2,
      DCC_LESSER = 0x4,
      DCC_GREATER_OR_LESSER = 0x6,
      DCC_UNSIGNED = 0x8,
      DCC_SIGNED = 0x10,
      DCC_INVALID = 0x20,
    };

    auto decodeCondCode = [&](unsigned CC) -> uint8_t {
      switch (CC) {
      default: return DCC_INVALID;
      case X86::COND_E: return DCC_EQUAL;
      case X86::COND_NE: return DCC_GREATER | DCC_LESSER;
      case X86::COND_L: return DCC_LESSER | DCC_SIGNED;
      case X86::COND_LE: return DCC_EQUAL | DCC_LESSER | DCC_SIGNED;
      case X86::COND_G: return DCC_GREATER | DCC_SIGNED;
      case X86::COND_GE: return DCC_GREATER | DCC_EQUAL | DCC_SIGNED;
      case X86::COND_B: return DCC_LESSER | DCC_UNSIGNED;
      case X86::COND_BE: return DCC_EQUAL | DCC_LESSER | DCC_UNSIGNED;
      case X86::COND_A: return DCC_GREATER | DCC_UNSIGNED;
      case X86::COND_AE: return DCC_GREATER | DCC_EQUAL | DCC_UNSIGNED;
      }
    };

    uint8_t DCC = decodeCondCode(CC1) | decodeCondCode(CC2);

    if (DCC & DCC_INVALID)
      return X86::COND_INVALID;

    if (DCC & DCC_SIGNED && DCC & DCC_UNSIGNED)
      return X86::COND_INVALID;

    switch (DCC) {
    default: return X86::COND_INVALID;
    case DCC_EQUAL | DCC_LESSER | DCC_SIGNED: return X86::COND_LE;
    case DCC_EQUAL | DCC_LESSER | DCC_UNSIGNED: return X86::COND_BE;
    case DCC_EQUAL | DCC_GREATER | DCC_SIGNED: return X86::COND_GE;
    case DCC_EQUAL | DCC_GREATER | DCC_UNSIGNED: return X86::COND_AE;
    case DCC_GREATER | DCC_LESSER | DCC_SIGNED: return X86::COND_NE;
    case DCC_GREATER | DCC_LESSER | DCC_UNSIGNED: return X86::COND_NE;
    case DCC_GREATER | DCC_LESSER: return X86::COND_NE;
    case DCC_EQUAL | DCC_SIGNED: return X86::COND_E;
    case DCC_EQUAL | DCC_UNSIGNED: return X86::COND_E;
    case DCC_EQUAL: return X86::COND_E;
    case DCC_LESSER | DCC_SIGNED: return X86::COND_L;
    case DCC_LESSER | DCC_UNSIGNED: return X86::COND_B;
    case DCC_GREATER | DCC_SIGNED: return X86::COND_G;
    case DCC_GREATER | DCC_UNSIGNED: return X86::COND_A;
    }
  }

  bool isValidCondCode(unsigned CC) const override {
    return (CC != X86::COND_INVALID);
  }

  bool isBreakpoint(const MCInst &Inst) const override {
    return Inst.getOpcode() == X86::INT3;
  }

  bool isPrefix(const MCInst &Inst) const override {
    const MCInstrDesc &Desc = Info->get(Inst.getOpcode());
    return X86II::isPrefix(Desc.TSFlags);
  }

  bool isRep(const MCInst &Inst) const override {
    return Inst.getFlags() == X86::IP_HAS_REPEAT;
  }

  bool deleteREPPrefix(MCInst &Inst) const override {
    if (Inst.getFlags() == X86::IP_HAS_REPEAT) {
      Inst.setFlags(0);
      return true;
    }
    return false;
  }

  bool isIndirectCall(const MCInst &Inst) const override {
    return isCall(Inst) &&
           ((getMemoryOperandNo(Inst) != -1) || Inst.getOperand(0).isReg());
  }

  bool isPop(const MCInst &Inst) const override {
    return getPopSize(Inst) == 0 ? false : true;
  }

  bool isTerminateBranch(const MCInst &Inst) const override {
    return Inst.getOpcode() == X86::ENDBR32 || Inst.getOpcode() == X86::ENDBR64;
  }

  int getPopSize(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    case X86::POP16r:
    case X86::POP16rmm:
    case X86::POP16rmr:
    case X86::POPF16:
    case X86::POPA16:
    case X86::POPDS16:
    case X86::POPES16:
    case X86::POPFS16:
    case X86::POPGS16:
    case X86::POPSS16:
      return 2;
    case X86::POP32r:
    case X86::POP32rmm:
    case X86::POP32rmr:
    case X86::POPA32:
    case X86::POPDS32:
    case X86::POPES32:
    case X86::POPF32:
    case X86::POPFS32:
    case X86::POPGS32:
    case X86::POPSS32:
      return 4;
    case X86::POP64r:
    case X86::POP64rmm:
    case X86::POP64rmr:
    case X86::POPF64:
    case X86::POPFS64:
    case X86::POPGS64:
      return 8;
    }
    return 0;
  }

  bool isPush(const MCInst &Inst) const override {
    return getPushSize(Inst) == 0 ? false : true;
  }

  int getPushSize(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    case X86::PUSH16i8:
    case X86::PUSH16r:
    case X86::PUSH16rmm:
    case X86::PUSH16rmr:
    case X86::PUSHA16:
    case X86::PUSHCS16:
    case X86::PUSHDS16:
    case X86::PUSHES16:
    case X86::PUSHF16:
    case X86::PUSHFS16:
    case X86::PUSHGS16:
    case X86::PUSHSS16:
    case X86::PUSH16i:
      return 2;
    case X86::PUSH32i8:
    case X86::PUSH32r:
    case X86::PUSH32rmm:
    case X86::PUSH32rmr:
    case X86::PUSHA32:
    case X86::PUSHCS32:
    case X86::PUSHDS32:
    case X86::PUSHES32:
    case X86::PUSHF32:
    case X86::PUSHFS32:
    case X86::PUSHGS32:
    case X86::PUSHSS32:
    case X86::PUSH32i:
      return 4;
    case X86::PUSH64i32:
    case X86::PUSH64i8:
    case X86::PUSH64r:
    case X86::PUSH64rmm:
    case X86::PUSH64rmr:
    case X86::PUSHF64:
    case X86::PUSHFS64:
    case X86::PUSHGS64:
      return 8;
    }
    return 0;
  }

  bool isSUB(const MCInst &Inst) const override {
    return X86::isSUB(Inst.getOpcode());
  }

  bool isLEA64r(const MCInst &Inst) const override {
    return Inst.getOpcode() == X86::LEA64r;
  }

  bool isLeave(const MCInst &Inst) const override {
    return Inst.getOpcode() == X86::LEAVE || Inst.getOpcode() == X86::LEAVE64;
  }

  bool isMoveMem2Reg(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    case X86::MOV16rm:
    case X86::MOV32rm:
    case X86::MOV64rm:
      return true;
    }
    return false;
  }

  bool isUnsupportedInstruction(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return false;

    case X86::LOOP:
    case X86::LOOPE:
    case X86::LOOPNE:
    case X86::JECXZ:
    case X86::JRCXZ:
      // These have a short displacement, and therefore (often) break after
      // basic block relayout.
      return true;
    }
  }

  bool mayLoad(const MCInst &Inst) const override {
    if (isPop(Inst))
      return true;

    int MemOpNo = getMemoryOperandNo(Inst);
    const MCInstrDesc &MCII = Info->get(Inst.getOpcode());

    if (MemOpNo == -1)
      return false;

    return MCII.mayLoad();
  }

  bool mayStore(const MCInst &Inst) const override {
    if (isPush(Inst))
      return true;

    int MemOpNo = getMemoryOperandNo(Inst);
    const MCInstrDesc &MCII = Info->get(Inst.getOpcode());

    if (MemOpNo == -1)
      return false;

    return MCII.mayStore();
  }

  bool isCleanRegXOR(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    case X86::XOR16rr:
    case X86::XOR32rr:
    case X86::XOR64rr:
      break;
    default:
      return false;
    }
    return (Inst.getOperand(0).getReg() == Inst.getOperand(2).getReg());
  }

  bool isPacked(const MCInst &Inst) const override {
    const MCInstrDesc &Desc = Info->get(Inst.getOpcode());
    return (Desc.TSFlags & X86II::OpPrefixMask) == X86II::PD;
  }

  bool shouldRecordCodeRelocation(uint64_t RelType) const override {
    switch (RelType) {
    case ELF::R_X86_64_8:
    case ELF::R_X86_64_16:
    case ELF::R_X86_64_32:
    case ELF::R_X86_64_32S:
    case ELF::R_X86_64_64:
    case ELF::R_X86_64_PC8:
    case ELF::R_X86_64_PC32:
    case ELF::R_X86_64_PC64:
    case ELF::R_X86_64_GOTPC64:
    case ELF::R_X86_64_GOTPCRELX:
    case ELF::R_X86_64_REX_GOTPCRELX:
      return true;
    case ELF::R_X86_64_PLT32:
    case ELF::R_X86_64_GOTPCREL:
    case ELF::R_X86_64_TPOFF32:
    case ELF::R_X86_64_GOTTPOFF:
      return false;
    default:
      llvm_unreachable("Unexpected x86 relocation type in code");
    }
  }

  StringRef getTrapFillValue() const override { return StringRef("\314", 1); }

  struct IndJmpMatcherFrag1 : MCInstMatcher {
    std::unique_ptr<MCInstMatcher> Base;
    std::unique_ptr<MCInstMatcher> Scale;
    std::unique_ptr<MCInstMatcher> Index;
    std::unique_ptr<MCInstMatcher> Offset;

    IndJmpMatcherFrag1(std::unique_ptr<MCInstMatcher> Base,
                       std::unique_ptr<MCInstMatcher> Scale,
                       std::unique_ptr<MCInstMatcher> Index,
                       std::unique_ptr<MCInstMatcher> Offset)
        : Base(std::move(Base)), Scale(std::move(Scale)),
          Index(std::move(Index)), Offset(std::move(Offset)) {}

    bool match(const MCRegisterInfo &MRI, MCPlusBuilder &MIB,
               MutableArrayRef<MCInst> InInstrWindow, int OpNum) override {
      if (!MCInstMatcher::match(MRI, MIB, InInstrWindow, OpNum))
        return false;

      if (CurInst->getOpcode() != X86::JMP64m)
        return false;

      int MemOpNo = MIB.getMemoryOperandNo(*CurInst);
      if (MemOpNo == -1)
        return false;

      if (!Base->match(MRI, MIB, this->InstrWindow, MemOpNo + X86::AddrBaseReg))
        return false;
      if (!Scale->match(MRI, MIB, this->InstrWindow,
                        MemOpNo + X86::AddrScaleAmt))
        return false;
      if (!Index->match(MRI, MIB, this->InstrWindow,
                        MemOpNo + X86::AddrIndexReg))
        return false;
      if (!Offset->match(MRI, MIB, this->InstrWindow, MemOpNo + X86::AddrDisp))
        return false;
      return true;
    }

    void annotate(MCPlusBuilder &MIB, StringRef Annotation) override {
      MIB.addAnnotation(*CurInst, Annotation, true);
      Base->annotate(MIB, Annotation);
      Scale->annotate(MIB, Annotation);
      Index->annotate(MIB, Annotation);
      Offset->annotate(MIB, Annotation);
    }
  };

  std::unique_ptr<MCInstMatcher>
  matchIndJmp(std::unique_ptr<MCInstMatcher> Base,
              std::unique_ptr<MCInstMatcher> Scale,
              std::unique_ptr<MCInstMatcher> Index,
              std::unique_ptr<MCInstMatcher> Offset) const override {
    return std::unique_ptr<MCInstMatcher>(
        new IndJmpMatcherFrag1(std::move(Base), std::move(Scale),
                               std::move(Index), std::move(Offset)));
  }

  struct IndJmpMatcherFrag2 : MCInstMatcher {
    std::unique_ptr<MCInstMatcher> Reg;

    IndJmpMatcherFrag2(std::unique_ptr<MCInstMatcher> Reg)
        : Reg(std::move(Reg)) {}

    bool match(const MCRegisterInfo &MRI, MCPlusBuilder &MIB,
               MutableArrayRef<MCInst> InInstrWindow, int OpNum) override {
      if (!MCInstMatcher::match(MRI, MIB, InInstrWindow, OpNum))
        return false;

      if (CurInst->getOpcode() != X86::JMP64r)
        return false;

      return Reg->match(MRI, MIB, this->InstrWindow, 0);
    }

    void annotate(MCPlusBuilder &MIB, StringRef Annotation) override {
      MIB.addAnnotation(*CurInst, Annotation, true);
      Reg->annotate(MIB, Annotation);
    }
  };

  std::unique_ptr<MCInstMatcher>
  matchIndJmp(std::unique_ptr<MCInstMatcher> Target) const override {
    return std::unique_ptr<MCInstMatcher>(
        new IndJmpMatcherFrag2(std::move(Target)));
  }

  struct LoadMatcherFrag1 : MCInstMatcher {
    std::unique_ptr<MCInstMatcher> Base;
    std::unique_ptr<MCInstMatcher> Scale;
    std::unique_ptr<MCInstMatcher> Index;
    std::unique_ptr<MCInstMatcher> Offset;

    LoadMatcherFrag1(std::unique_ptr<MCInstMatcher> Base,
                     std::unique_ptr<MCInstMatcher> Scale,
                     std::unique_ptr<MCInstMatcher> Index,
                     std::unique_ptr<MCInstMatcher> Offset)
        : Base(std::move(Base)), Scale(std::move(Scale)),
          Index(std::move(Index)), Offset(std::move(Offset)) {}

    bool match(const MCRegisterInfo &MRI, MCPlusBuilder &MIB,
               MutableArrayRef<MCInst> InInstrWindow, int OpNum) override {
      if (!MCInstMatcher::match(MRI, MIB, InInstrWindow, OpNum))
        return false;

      if (CurInst->getOpcode() != X86::MOV64rm &&
          CurInst->getOpcode() != X86::MOVSX64rm32)
        return false;

      int MemOpNo = MIB.getMemoryOperandNo(*CurInst);
      if (MemOpNo == -1)
        return false;

      if (!Base->match(MRI, MIB, this->InstrWindow, MemOpNo + X86::AddrBaseReg))
        return false;
      if (!Scale->match(MRI, MIB, this->InstrWindow,
                        MemOpNo + X86::AddrScaleAmt))
        return false;
      if (!Index->match(MRI, MIB, this->InstrWindow,
                        MemOpNo + X86::AddrIndexReg))
        return false;
      if (!Offset->match(MRI, MIB, this->InstrWindow, MemOpNo + X86::AddrDisp))
        return false;
      return true;
    }

    void annotate(MCPlusBuilder &MIB, StringRef Annotation) override {
      MIB.addAnnotation(*CurInst, Annotation, true);
      Base->annotate(MIB, Annotation);
      Scale->annotate(MIB, Annotation);
      Index->annotate(MIB, Annotation);
      Offset->annotate(MIB, Annotation);
    }
  };

  std::unique_ptr<MCInstMatcher>
  matchLoad(std::unique_ptr<MCInstMatcher> Base,
            std::unique_ptr<MCInstMatcher> Scale,
            std::unique_ptr<MCInstMatcher> Index,
            std::unique_ptr<MCInstMatcher> Offset) const override {
    return std::unique_ptr<MCInstMatcher>(
        new LoadMatcherFrag1(std::move(Base), std::move(Scale),
                             std::move(Index), std::move(Offset)));
  }

  struct AddMatcher : MCInstMatcher {
    std::unique_ptr<MCInstMatcher> A;
    std::unique_ptr<MCInstMatcher> B;

    AddMatcher(std::unique_ptr<MCInstMatcher> A,
               std::unique_ptr<MCInstMatcher> B)
        : A(std::move(A)), B(std::move(B)) {}

    bool match(const MCRegisterInfo &MRI, MCPlusBuilder &MIB,
               MutableArrayRef<MCInst> InInstrWindow, int OpNum) override {
      if (!MCInstMatcher::match(MRI, MIB, InInstrWindow, OpNum))
        return false;

      if (CurInst->getOpcode() == X86::ADD64rr ||
          CurInst->getOpcode() == X86::ADD64rr_DB ||
          CurInst->getOpcode() == X86::ADD64rr_REV) {
        if (!A->match(MRI, MIB, this->InstrWindow, 1)) {
          if (!B->match(MRI, MIB, this->InstrWindow, 1))
            return false;
          return A->match(MRI, MIB, this->InstrWindow, 2);
        }

        if (B->match(MRI, MIB, this->InstrWindow, 2))
          return true;

        if (!B->match(MRI, MIB, this->InstrWindow, 1))
          return false;
        return A->match(MRI, MIB, this->InstrWindow, 2);
      }

      return false;
    }

    void annotate(MCPlusBuilder &MIB, StringRef Annotation) override {
      MIB.addAnnotation(*CurInst, Annotation, true);
      A->annotate(MIB, Annotation);
      B->annotate(MIB, Annotation);
    }
  };

  std::unique_ptr<MCInstMatcher>
  matchAdd(std::unique_ptr<MCInstMatcher> A,
           std::unique_ptr<MCInstMatcher> B) const override {
    return std::unique_ptr<MCInstMatcher>(
        new AddMatcher(std::move(A), std::move(B)));
  }

  struct LEAMatcher : MCInstMatcher {
    std::unique_ptr<MCInstMatcher> Target;

    LEAMatcher(std::unique_ptr<MCInstMatcher> Target)
        : Target(std::move(Target)) {}

    bool match(const MCRegisterInfo &MRI, MCPlusBuilder &MIB,
               MutableArrayRef<MCInst> InInstrWindow, int OpNum) override {
      if (!MCInstMatcher::match(MRI, MIB, InInstrWindow, OpNum))
        return false;

      if (CurInst->getOpcode() != X86::LEA64r)
        return false;

      if (CurInst->getOperand(1 + X86::AddrScaleAmt).getImm() != 1 ||
          CurInst->getOperand(1 + X86::AddrIndexReg).getReg() !=
              X86::NoRegister ||
          (CurInst->getOperand(1 + X86::AddrBaseReg).getReg() !=
               X86::NoRegister &&
           CurInst->getOperand(1 + X86::AddrBaseReg).getReg() != X86::RIP))
        return false;

      return Target->match(MRI, MIB, this->InstrWindow, 1 + X86::AddrDisp);
    }

    void annotate(MCPlusBuilder &MIB, StringRef Annotation) override {
      MIB.addAnnotation(*CurInst, Annotation, true);
      Target->annotate(MIB, Annotation);
    }
  };

  std::unique_ptr<MCInstMatcher>
  matchLoadAddr(std::unique_ptr<MCInstMatcher> Target) const override {
    return std::unique_ptr<MCInstMatcher>(new LEAMatcher(std::move(Target)));
  }

  bool hasPCRelOperand(const MCInst &Inst) const override {
    for (const MCOperand &Operand : Inst)
      if (Operand.isReg() && Operand.getReg() == X86::RIP)
        return true;
    return false;
  }

  int getMemoryOperandNo(const MCInst &Inst) const override {
    unsigned Opcode = Inst.getOpcode();
    const MCInstrDesc &Desc = Info->get(Opcode);
    int MemOpNo = X86II::getMemoryOperandNo(Desc.TSFlags);
    if (MemOpNo >= 0)
      MemOpNo += X86II::getOperandBias(Desc);
    return MemOpNo;
  }

  bool hasEVEXEncoding(const MCInst &Inst) const override {
    const MCInstrDesc &Desc = Info->get(Inst.getOpcode());
    return (Desc.TSFlags & X86II::EncodingMask) == X86II::EVEX;
  }

  std::optional<X86MemOperand>
  evaluateX86MemoryOperand(const MCInst &Inst) const override {
    int MemOpNo = getMemoryOperandNo(Inst);
    if (MemOpNo < 0)
      return std::nullopt;
    unsigned MemOpOffset = static_cast<unsigned>(MemOpNo);

    if (MemOpOffset + X86::AddrSegmentReg >= MCPlus::getNumPrimeOperands(Inst))
      return std::nullopt;

    const MCOperand &Base = Inst.getOperand(MemOpOffset + X86::AddrBaseReg);
    const MCOperand &Scale = Inst.getOperand(MemOpOffset + X86::AddrScaleAmt);
    const MCOperand &Index = Inst.getOperand(MemOpOffset + X86::AddrIndexReg);
    const MCOperand &Disp = Inst.getOperand(MemOpOffset + X86::AddrDisp);
    const MCOperand &Segment =
        Inst.getOperand(MemOpOffset + X86::AddrSegmentReg);

    // Make sure it is a well-formed memory operand.
    if (!Base.isReg() || !Scale.isImm() || !Index.isReg() ||
        (!Disp.isImm() && !Disp.isExpr()) || !Segment.isReg())
      return std::nullopt;

    X86MemOperand MO;
    MO.BaseRegNum = Base.getReg();
    MO.ScaleImm = Scale.getImm();
    MO.IndexRegNum = Index.getReg();
    MO.DispImm = Disp.isImm() ? Disp.getImm() : 0;
    MO.DispExpr = Disp.isExpr() ? Disp.getExpr() : nullptr;
    MO.SegRegNum = Segment.getReg();
    return MO;
  }

  bool evaluateMemOperandTarget(const MCInst &Inst, uint64_t &Target,
                                uint64_t Address,
                                uint64_t Size) const override {
    std::optional<X86MemOperand> MO = evaluateX86MemoryOperand(Inst);
    if (!MO)
      return false;

    // Make sure it's a well-formed addressing we can statically evaluate.
    if ((MO->BaseRegNum != X86::RIP && MO->BaseRegNum != X86::NoRegister) ||
        MO->IndexRegNum != X86::NoRegister ||
        MO->SegRegNum != X86::NoRegister || MO->DispExpr)
      return false;

    Target = MO->DispImm;
    if (MO->BaseRegNum == X86::RIP) {
      assert(Size != 0 && "instruction size required in order to statically "
                          "evaluate RIP-relative address");
      Target += Address + Size;
    }
    return true;
  }

  MCInst::iterator getMemOperandDisp(MCInst &Inst) const override {
    int MemOpNo = getMemoryOperandNo(Inst);
    if (MemOpNo < 0)
      return Inst.end();
    return Inst.begin() + (MemOpNo + X86::AddrDisp);
  }

  bool replaceMemOperandDisp(MCInst &Inst, MCOperand Operand) const override {
    MCOperand *OI = getMemOperandDisp(Inst);
    if (OI == Inst.end())
      return false;
    *OI = Operand;
    return true;
  }

  /// Get the registers used as function parameters.
  /// This function is specific to the x86_64 abi on Linux.
  BitVector getRegsUsedAsParams() const override {
    BitVector Regs = BitVector(RegInfo->getNumRegs(), false);
    Regs |= getAliases(X86::RSI);
    Regs |= getAliases(X86::RDI);
    Regs |= getAliases(X86::RDX);
    Regs |= getAliases(X86::RCX);
    Regs |= getAliases(X86::R8);
    Regs |= getAliases(X86::R9);
    return Regs;
  }

  void getCalleeSavedRegs(BitVector &Regs) const override {
    Regs |= getAliases(X86::RBX);
    Regs |= getAliases(X86::RBP);
    Regs |= getAliases(X86::R12);
    Regs |= getAliases(X86::R13);
    Regs |= getAliases(X86::R14);
    Regs |= getAliases(X86::R15);
  }

  void getDefaultDefIn(BitVector &Regs) const override {
    assert(Regs.size() >= RegInfo->getNumRegs() &&
           "The size of BitVector is less than RegInfo->getNumRegs().");
    Regs.set(X86::RAX);
    Regs.set(X86::RCX);
    Regs.set(X86::RDX);
    Regs.set(X86::RSI);
    Regs.set(X86::RDI);
    Regs.set(X86::R8);
    Regs.set(X86::R9);
    Regs.set(X86::XMM0);
    Regs.set(X86::XMM1);
    Regs.set(X86::XMM2);
    Regs.set(X86::XMM3);
    Regs.set(X86::XMM4);
    Regs.set(X86::XMM5);
    Regs.set(X86::XMM6);
    Regs.set(X86::XMM7);
  }

  void getDefaultLiveOut(BitVector &Regs) const override {
    assert(Regs.size() >= RegInfo->getNumRegs() &&
           "The size of BitVector is less than RegInfo->getNumRegs().");
    Regs |= getAliases(X86::RAX);
    Regs |= getAliases(X86::RDX);
    Regs |= getAliases(X86::RCX);
    Regs |= getAliases(X86::XMM0);
    Regs |= getAliases(X86::XMM1);
  }

  void getGPRegs(BitVector &Regs, bool IncludeAlias) const override {
    if (IncludeAlias) {
      Regs |= getAliases(X86::RAX);
      Regs |= getAliases(X86::RBX);
      Regs |= getAliases(X86::RBP);
      Regs |= getAliases(X86::RSI);
      Regs |= getAliases(X86::RDI);
      Regs |= getAliases(X86::RDX);
      Regs |= getAliases(X86::RCX);
      Regs |= getAliases(X86::R8);
      Regs |= getAliases(X86::R9);
      Regs |= getAliases(X86::R10);
      Regs |= getAliases(X86::R11);
      Regs |= getAliases(X86::R12);
      Regs |= getAliases(X86::R13);
      Regs |= getAliases(X86::R14);
      Regs |= getAliases(X86::R15);
      return;
    }
    Regs.set(X86::RAX);
    Regs.set(X86::RBX);
    Regs.set(X86::RBP);
    Regs.set(X86::RSI);
    Regs.set(X86::RDI);
    Regs.set(X86::RDX);
    Regs.set(X86::RCX);
    Regs.set(X86::R8);
    Regs.set(X86::R9);
    Regs.set(X86::R10);
    Regs.set(X86::R11);
    Regs.set(X86::R12);
    Regs.set(X86::R13);
    Regs.set(X86::R14);
    Regs.set(X86::R15);
  }

  void getClassicGPRegs(BitVector &Regs) const override {
    Regs |= getAliases(X86::RAX);
    Regs |= getAliases(X86::RBX);
    Regs |= getAliases(X86::RBP);
    Regs |= getAliases(X86::RSI);
    Regs |= getAliases(X86::RDI);
    Regs |= getAliases(X86::RDX);
    Regs |= getAliases(X86::RCX);
  }

  void getRepRegs(BitVector &Regs) const override {
    Regs |= getAliases(X86::RCX);
  }

  MCPhysReg getAliasSized(MCPhysReg Reg, uint8_t Size) const override {
    Reg = getX86SubSuperRegister(Reg, Size * 8);
    assert((Reg != X86::NoRegister) && "Invalid register");
    return Reg;
  }

  bool isUpper8BitReg(MCPhysReg Reg) const override {
    switch (Reg) {
    case X86::AH:
    case X86::BH:
    case X86::CH:
    case X86::DH:
      return true;
    default:
      return false;
    }
  }

  bool cannotUseREX(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    case X86::MOV8mr_NOREX:
    case X86::MOV8rm_NOREX:
    case X86::MOV8rr_NOREX:
    case X86::MOVSX32rm8_NOREX:
    case X86::MOVSX32rr8_NOREX:
    case X86::MOVZX32rm8_NOREX:
    case X86::MOVZX32rr8_NOREX:
    case X86::MOV8mr:
    case X86::MOV8rm:
    case X86::MOV8rr:
    case X86::MOVSX32rm8:
    case X86::MOVSX32rr8:
    case X86::MOVZX32rm8:
    case X86::MOVZX32rr8:
    case X86::TEST8ri:
      for (const MCOperand &Operand : MCPlus::primeOperands(Inst)) {
        if (!Operand.isReg())
          continue;
        if (isUpper8BitReg(Operand.getReg()))
          return true;
      }
      [[fallthrough]];
    default:
      return false;
    }
  }

  static uint8_t getMemDataSize(const MCInst &Inst, int MemOpNo) {
    using namespace llvm::X86;
    int OpType = getOperandType(Inst.getOpcode(), MemOpNo);
    return getMemOperandSize(OpType) / 8;
  }

  /// Classifying a stack access as *not* "SIMPLE" here means we don't know how
  /// to change this instruction memory access. It will disable any changes to
  /// the stack layout, so we can't do the most aggressive form of shrink
  /// wrapping. We must do so in a way that keeps the original stack layout.
  /// Otherwise you need to adjust the offset of all instructions accessing the
  /// stack: we can't do that anymore because there is one instruction that is
  /// not simple. There are other implications as well. We have heuristics to
  /// detect when a register is callee-saved and thus eligible for shrink
  /// wrapping. If you are restoring a register using a non-simple stack access,
  /// then it is classified as NOT callee-saved, and it disables shrink wrapping
  /// for *that* register (but not for others).
  ///
  /// Classifying a stack access as "size 0" or detecting an indexed memory
  /// access (to address a vector, for example) here means we know there is a
  /// stack access, but we can't quite understand how wide is the access in
  /// bytes. This is very serious because we can't understand how memory
  /// accesses alias with each other for this function. This will essentially
  /// disable not only shrink wrapping but all frame analysis, it will fail it
  /// as "we don't understand this function and we give up on it".
  bool isStackAccess(const MCInst &Inst, bool &IsLoad, bool &IsStore,
                     bool &IsStoreFromReg, MCPhysReg &Reg, int32_t &SrcImm,
                     uint16_t &StackPtrReg, int64_t &StackOffset, uint8_t &Size,
                     bool &IsSimple, bool &IsIndexed) const override {
    // Detect simple push/pop cases first
    if (int Sz = getPushSize(Inst)) {
      IsLoad = false;
      IsStore = true;
      IsStoreFromReg = true;
      StackPtrReg = X86::RSP;
      StackOffset = -Sz;
      Size = Sz;
      IsSimple = true;
      if (Inst.getOperand(0).isImm())
        SrcImm = Inst.getOperand(0).getImm();
      else if (Inst.getOperand(0).isReg())
        Reg = Inst.getOperand(0).getReg();
      else
        IsSimple = false;

      return true;
    }
    if (int Sz = getPopSize(Inst)) {
      IsLoad = true;
      IsStore = false;
      if (Inst.getNumOperands() == 0 || !Inst.getOperand(0).isReg()) {
        IsSimple = false;
      } else {
        Reg = Inst.getOperand(0).getReg();
        IsSimple = true;
      }
      StackPtrReg = X86::RSP;
      StackOffset = 0;
      Size = Sz;
      return true;
    }

    struct InstInfo {
      // Size in bytes that Inst loads from memory.
      uint8_t DataSize;
      bool IsLoad;
      bool IsStore;
      bool StoreFromReg;
      bool Simple;
    };

    InstInfo I;
    int MemOpNo = getMemoryOperandNo(Inst);
    const MCInstrDesc &MCII = Info->get(Inst.getOpcode());
    // If it is not dealing with a memory operand, we discard it
    if (MemOpNo == -1 || MCII.isCall())
      return false;

    switch (Inst.getOpcode()) {
    default: {
      bool IsLoad = MCII.mayLoad();
      bool IsStore = MCII.mayStore();
      // Is it LEA? (deals with memory but is not loading nor storing)
      if (!IsLoad && !IsStore) {
        I = {0, IsLoad, IsStore, false, false};
        break;
      }
      uint8_t Sz = getMemDataSize(Inst, MemOpNo);
      I = {Sz, IsLoad, IsStore, false, false};
      break;
    }
    // Report simple stack accesses
    case X86::MOV8rm: I = {1, true, false, false, true}; break;
    case X86::MOV16rm: I = {2, true, false, false, true}; break;
    case X86::MOV32rm: I = {4, true, false, false, true}; break;
    case X86::MOV64rm: I = {8, true, false, false, true}; break;
    case X86::MOV8mr: I = {1, false, true, true, true};  break;
    case X86::MOV16mr: I = {2, false, true, true, true};  break;
    case X86::MOV32mr: I = {4, false, true, true, true};  break;
    case X86::MOV64mr: I = {8, false, true, true, true};  break;
    case X86::MOV8mi: I = {1, false, true, false, true}; break;
    case X86::MOV16mi: I = {2, false, true, false, true}; break;
    case X86::MOV32mi: I = {4, false, true, false, true}; break;
    } // end switch (Inst.getOpcode())

    std::optional<X86MemOperand> MO = evaluateX86MemoryOperand(Inst);
    if (!MO) {
      LLVM_DEBUG(dbgs() << "Evaluate failed on ");
      LLVM_DEBUG(Inst.dump());
      return false;
    }

    // Make sure it's a stack access
    if (MO->BaseRegNum != X86::RBP && MO->BaseRegNum != X86::RSP)
      return false;

    IsLoad = I.IsLoad;
    IsStore = I.IsStore;
    IsStoreFromReg = I.StoreFromReg;
    Size = I.DataSize;
    IsSimple = I.Simple;
    StackPtrReg = MO->BaseRegNum;
    StackOffset = MO->DispImm;
    IsIndexed =
        MO->IndexRegNum != X86::NoRegister || MO->SegRegNum != X86::NoRegister;

    if (!I.Simple)
      return true;

    // Retrieve related register in simple MOV from/to stack operations.
    unsigned MemOpOffset = static_cast<unsigned>(MemOpNo);
    if (I.IsLoad) {
      MCOperand RegOpnd = Inst.getOperand(0);
      assert(RegOpnd.isReg() && "unexpected destination operand");
      Reg = RegOpnd.getReg();
    } else if (I.IsStore) {
      MCOperand SrcOpnd =
          Inst.getOperand(MemOpOffset + X86::AddrSegmentReg + 1);
      if (I.StoreFromReg) {
        assert(SrcOpnd.isReg() && "unexpected source operand");
        Reg = SrcOpnd.getReg();
      } else {
        assert(SrcOpnd.isImm() && "unexpected source operand");
        SrcImm = SrcOpnd.getImm();
      }
    }

    return true;
  }

  void changeToPushOrPop(MCInst &Inst) const override {
    assert(!isPush(Inst) && !isPop(Inst));

    struct InstInfo {
      // Size in bytes that Inst loads from memory.
      uint8_t DataSize;
      bool IsLoad;
      bool StoreFromReg;
    };

    InstInfo I;
    switch (Inst.getOpcode()) {
    default: {
      llvm_unreachable("Unhandled opcode");
      return;
    }
    case X86::MOV16rm: I = {2, true, false}; break;
    case X86::MOV32rm: I = {4, true, false}; break;
    case X86::MOV64rm: I = {8, true, false}; break;
    case X86::MOV16mr: I = {2, false, true};  break;
    case X86::MOV32mr: I = {4, false, true};  break;
    case X86::MOV64mr: I = {8, false, true};  break;
    case X86::MOV16mi: I = {2, false, false}; break;
    case X86::MOV32mi: I = {4, false, false}; break;
    } // end switch (Inst.getOpcode())

    std::optional<X86MemOperand> MO = evaluateX86MemoryOperand(Inst);
    if (!MO) {
      llvm_unreachable("Evaluate failed");
      return;
    }
    // Make sure it's a stack access
    if (MO->BaseRegNum != X86::RBP && MO->BaseRegNum != X86::RSP) {
      llvm_unreachable("Not a stack access");
      return;
    }

    unsigned MemOpOffset = getMemoryOperandNo(Inst);
    unsigned NewOpcode = 0;
    if (I.IsLoad) {
      switch (I.DataSize) {
      case 2: NewOpcode = X86::POP16r; break;
      case 4: NewOpcode = X86::POP32r; break;
      case 8: NewOpcode = X86::POP64r; break;
      default:
        llvm_unreachable("Unexpected size");
      }
      unsigned RegOpndNum = Inst.getOperand(0).getReg();
      Inst.clear();
      Inst.setOpcode(NewOpcode);
      Inst.addOperand(MCOperand::createReg(RegOpndNum));
    } else {
      MCOperand SrcOpnd =
          Inst.getOperand(MemOpOffset + X86::AddrSegmentReg + 1);
      if (I.StoreFromReg) {
        switch (I.DataSize) {
        case 2: NewOpcode = X86::PUSH16r; break;
        case 4: NewOpcode = X86::PUSH32r; break;
        case 8: NewOpcode = X86::PUSH64r; break;
        default:
          llvm_unreachable("Unexpected size");
        }
        assert(SrcOpnd.isReg() && "Unexpected source operand");
        unsigned RegOpndNum = SrcOpnd.getReg();
        Inst.clear();
        Inst.setOpcode(NewOpcode);
        Inst.addOperand(MCOperand::createReg(RegOpndNum));
      } else {
        switch (I.DataSize) {
        case 2: NewOpcode = X86::PUSH16i8; break;
        case 4: NewOpcode = X86::PUSH32i8; break;
        case 8: NewOpcode = X86::PUSH64i32; break;
        default:
          llvm_unreachable("Unexpected size");
        }
        assert(SrcOpnd.isImm() && "Unexpected source operand");
        int64_t SrcImm = SrcOpnd.getImm();
        Inst.clear();
        Inst.setOpcode(NewOpcode);
        Inst.addOperand(MCOperand::createImm(SrcImm));
      }
    }
  }

  bool isStackAdjustment(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return false;
    case X86::SUB64ri32:
    case X86::SUB64ri8:
    case X86::ADD64ri32:
    case X86::ADD64ri8:
    case X86::LEA64r:
      break;
    }

    return any_of(defOperands(Inst), [](const MCOperand &Op) {
      return Op.isReg() && Op.getReg() == X86::RSP;
    });
  }

  bool
  evaluateStackOffsetExpr(const MCInst &Inst, int64_t &Output,
                          std::pair<MCPhysReg, int64_t> Input1,
                          std::pair<MCPhysReg, int64_t> Input2) const override {

    auto getOperandVal = [&](MCPhysReg Reg) -> ErrorOr<int64_t> {
      if (Reg == Input1.first)
        return Input1.second;
      if (Reg == Input2.first)
        return Input2.second;
      return make_error_code(errc::result_out_of_range);
    };

    switch (Inst.getOpcode()) {
    default:
      return false;

    case X86::SUB64ri32:
    case X86::SUB64ri8:
      if (!Inst.getOperand(2).isImm())
        return false;
      if (ErrorOr<int64_t> InputVal =
              getOperandVal(Inst.getOperand(1).getReg()))
        Output = *InputVal - Inst.getOperand(2).getImm();
      else
        return false;
      break;
    case X86::ADD64ri32:
    case X86::ADD64ri8:
      if (!Inst.getOperand(2).isImm())
        return false;
      if (ErrorOr<int64_t> InputVal =
              getOperandVal(Inst.getOperand(1).getReg()))
        Output = *InputVal + Inst.getOperand(2).getImm();
      else
        return false;
      break;
    case X86::ADD64i32:
      if (!Inst.getOperand(0).isImm())
        return false;
      if (ErrorOr<int64_t> InputVal = getOperandVal(X86::RAX))
        Output = *InputVal + Inst.getOperand(0).getImm();
      else
        return false;
      break;

    case X86::LEA64r: {
      std::optional<X86MemOperand> MO = evaluateX86MemoryOperand(Inst);
      if (!MO)
        return false;

      if (MO->BaseRegNum == X86::NoRegister ||
          MO->IndexRegNum != X86::NoRegister ||
          MO->SegRegNum != X86::NoRegister || MO->DispExpr)
        return false;

      if (ErrorOr<int64_t> InputVal = getOperandVal(MO->BaseRegNum))
        Output = *InputVal + MO->DispImm;
      else
        return false;

      break;
    }
    }
    return true;
  }

  bool isRegToRegMove(const MCInst &Inst, MCPhysReg &From,
                      MCPhysReg &To) const override {
    switch (Inst.getOpcode()) {
    default:
      return false;
    case X86::LEAVE:
    case X86::LEAVE64:
      To = getStackPointer();
      From = getFramePointer();
      return true;
    case X86::MOV64rr:
      To = Inst.getOperand(0).getReg();
      From = Inst.getOperand(1).getReg();
      return true;
    }
  }

  MCPhysReg getStackPointer() const override { return X86::RSP; }
  MCPhysReg getFramePointer() const override { return X86::RBP; }
  MCPhysReg getFlagsReg() const override { return X86::EFLAGS; }

  bool escapesVariable(const MCInst &Inst,
                       bool HasFramePointer) const override {
    int MemOpNo = getMemoryOperandNo(Inst);
    const MCInstrDesc &MCII = Info->get(Inst.getOpcode());
    const unsigned NumDefs = MCII.getNumDefs();
    static BitVector SPBPAliases(BitVector(getAliases(X86::RSP)) |=
                                 getAliases(X86::RBP));
    static BitVector SPAliases(getAliases(X86::RSP));

    // FIXME: PUSH can be technically a leak, but let's ignore this for now
    // because a lot of harmless prologue code will spill SP to the stack.
    // Unless push is clearly pushing an object address to the stack as
    // demonstrated by having a MemOp.
    bool IsPush = isPush(Inst);
    if (IsPush && MemOpNo == -1)
      return false;

    // We use this to detect LEA (has memop but does not access mem)
    bool AccessMem = MCII.mayLoad() || MCII.mayStore();
    bool DoesLeak = false;
    for (int I = 0, E = MCPlus::getNumPrimeOperands(Inst); I != E; ++I) {
      // Ignore if SP/BP is used to dereference memory -- that's fine
      if (MemOpNo != -1 && !IsPush && AccessMem && I >= MemOpNo &&
          I <= MemOpNo + 5)
        continue;
      // Ignore if someone is writing to SP/BP
      if (I < static_cast<int>(NumDefs))
        continue;

      const MCOperand &Operand = Inst.getOperand(I);
      if (HasFramePointer && Operand.isReg() && SPBPAliases[Operand.getReg()]) {
        DoesLeak = true;
        break;
      }
      if (!HasFramePointer && Operand.isReg() && SPAliases[Operand.getReg()]) {
        DoesLeak = true;
        break;
      }
    }

    // If potential leak, check if it is not just writing to itself/sp/bp
    if (DoesLeak) {
      DoesLeak = !any_of(defOperands(Inst), [&](const MCOperand &Operand) {
        assert(Operand.isReg());
        MCPhysReg Reg = Operand.getReg();
        return HasFramePointer ? SPBPAliases[Reg] : SPAliases[Reg];
      });
    }
    return DoesLeak;
  }

  bool addToImm(MCInst &Inst, int64_t &Amt, MCContext *Ctx) const override {
    unsigned ImmOpNo = -1U;
    int MemOpNo = getMemoryOperandNo(Inst);
    if (MemOpNo != -1)
      ImmOpNo = MemOpNo + X86::AddrDisp;
    else
      for (unsigned Index = 0; Index < MCPlus::getNumPrimeOperands(Inst);
           ++Index)
        if (Inst.getOperand(Index).isImm())
          ImmOpNo = Index;
    if (ImmOpNo == -1U)
      return false;

    MCOperand &Operand = Inst.getOperand(ImmOpNo);
    Amt += Operand.getImm();
    Operand.setImm(Amt);
    // Check for the need for relaxation
    if (int64_t(Amt) == int64_t(int8_t(Amt)))
      return true;

    // Relax instruction
    switch (Inst.getOpcode()) {
    case X86::SUB64ri8:
      Inst.setOpcode(X86::SUB64ri32);
      break;
    case X86::ADD64ri8:
      Inst.setOpcode(X86::ADD64ri32);
      break;
    default:
      // No need for relaxation
      break;
    }
    return true;
  }

  /// TODO: this implementation currently works for the most common opcodes that
  /// load from memory. It can be extended to work with memory store opcodes as
  /// well as more memory load opcodes.
  bool replaceMemOperandWithImm(MCInst &Inst, StringRef ConstantData,
                                uint64_t Offset) const override {
    enum CheckSignExt : uint8_t {
      NOCHECK = 0,
      CHECK8,
      CHECK32,
    };

    using CheckList = std::vector<std::pair<CheckSignExt, unsigned>>;
    struct InstInfo {
      // Size in bytes that Inst loads from memory.
      uint8_t DataSize;

      // True when the target operand has to be duplicated because the opcode
      // expects a LHS operand.
      bool HasLHS;

      // List of checks and corresponding opcodes to be used. We try to use the
      // smallest possible immediate value when various sizes are available,
      // hence we may need to check whether a larger constant fits in a smaller
      // immediate.
      CheckList Checks;
    };

    InstInfo I;

    switch (Inst.getOpcode()) {
    default: {
      switch (getPopSize(Inst)) {
      case 2:            I = {2, false, {{NOCHECK, X86::MOV16ri}}};  break;
      case 4:            I = {4, false, {{NOCHECK, X86::MOV32ri}}};  break;
      case 8:            I = {8, false, {{CHECK32, X86::MOV64ri32},
                                         {NOCHECK, X86::MOV64rm}}};  break;
      default:           return false;
      }
      break;
    }

    // MOV
    case X86::MOV8rm:      I = {1, false, {{NOCHECK, X86::MOV8ri}}};   break;
    case X86::MOV16rm:     I = {2, false, {{NOCHECK, X86::MOV16ri}}};  break;
    case X86::MOV32rm:     I = {4, false, {{NOCHECK, X86::MOV32ri}}};  break;
    case X86::MOV64rm:     I = {8, false, {{CHECK32, X86::MOV64ri32},
                                           {NOCHECK, X86::MOV64rm}}};  break;

    // MOVZX
    case X86::MOVZX16rm8:  I = {1, false, {{NOCHECK, X86::MOV16ri}}};  break;
    case X86::MOVZX32rm8:  I = {1, false, {{NOCHECK, X86::MOV32ri}}};  break;
    case X86::MOVZX32rm16: I = {2, false, {{NOCHECK, X86::MOV32ri}}};  break;

    // CMP
    case X86::CMP8rm:      I = {1, false, {{NOCHECK, X86::CMP8ri}}};   break;
    case X86::CMP16rm:     I = {2, false, {{CHECK8,  X86::CMP16ri8},
                                           {NOCHECK, X86::CMP16ri}}};  break;
    case X86::CMP32rm:     I = {4, false, {{CHECK8,  X86::CMP32ri8},
                                           {NOCHECK, X86::CMP32ri}}};  break;
    case X86::CMP64rm:     I = {8, false, {{CHECK8,  X86::CMP64ri8},
                                           {CHECK32, X86::CMP64ri32},
                                           {NOCHECK, X86::CMP64rm}}};  break;

    // TEST
    case X86::TEST8mr:     I = {1, false, {{NOCHECK, X86::TEST8ri}}};  break;
    case X86::TEST16mr:    I = {2, false, {{NOCHECK, X86::TEST16ri}}}; break;
    case X86::TEST32mr:    I = {4, false, {{NOCHECK, X86::TEST32ri}}}; break;
    case X86::TEST64mr:    I = {8, false, {{CHECK32, X86::TEST64ri32},
                                           {NOCHECK, X86::TEST64mr}}}; break;

    // ADD
    case X86::ADD8rm:      I = {1, true,  {{NOCHECK, X86::ADD8ri}}};   break;
    case X86::ADD16rm:     I = {2, true,  {{CHECK8,  X86::ADD16ri8},
                                           {NOCHECK, X86::ADD16ri}}};  break;
    case X86::ADD32rm:     I = {4, true,  {{CHECK8,  X86::ADD32ri8},
                                           {NOCHECK, X86::ADD32ri}}};  break;
    case X86::ADD64rm:     I = {8, true,  {{CHECK8,  X86::ADD64ri8},
                                           {CHECK32, X86::ADD64ri32},
                                           {NOCHECK, X86::ADD64rm}}};  break;

    // SUB
    case X86::SUB8rm:      I = {1, true,  {{NOCHECK, X86::SUB8ri}}};   break;
    case X86::SUB16rm:     I = {2, true,  {{CHECK8,  X86::SUB16ri8},
                                           {NOCHECK, X86::SUB16ri}}};  break;
    case X86::SUB32rm:     I = {4, true,  {{CHECK8,  X86::SUB32ri8},
                                           {NOCHECK, X86::SUB32ri}}};  break;
    case X86::SUB64rm:     I = {8, true,  {{CHECK8,  X86::SUB64ri8},
                                           {CHECK32, X86::SUB64ri32},
                                           {NOCHECK, X86::SUB64rm}}};  break;

    // AND
    case X86::AND8rm:      I = {1, true,  {{NOCHECK, X86::AND8ri}}};   break;
    case X86::AND16rm:     I = {2, true,  {{CHECK8,  X86::AND16ri8},
                                           {NOCHECK, X86::AND16ri}}};  break;
    case X86::AND32rm:     I = {4, true,  {{CHECK8,  X86::AND32ri8},
                                           {NOCHECK, X86::AND32ri}}};  break;
    case X86::AND64rm:     I = {8, true,  {{CHECK8,  X86::AND64ri8},
                                           {CHECK32, X86::AND64ri32},
                                           {NOCHECK, X86::AND64rm}}};  break;

    // OR
    case X86::OR8rm:       I = {1, true,  {{NOCHECK, X86::OR8ri}}};    break;
    case X86::OR16rm:      I = {2, true,  {{CHECK8,  X86::OR16ri8},
                                           {NOCHECK, X86::OR16ri}}};   break;
    case X86::OR32rm:      I = {4, true,  {{CHECK8,  X86::OR32ri8},
                                           {NOCHECK, X86::OR32ri}}};   break;
    case X86::OR64rm:      I = {8, true,  {{CHECK8,  X86::OR64ri8},
                                           {CHECK32, X86::OR64ri32},
                                           {NOCHECK, X86::OR64rm}}};   break;

    // XOR
    case X86::XOR8rm:      I = {1, true,  {{NOCHECK, X86::XOR8ri}}};   break;
    case X86::XOR16rm:     I = {2, true,  {{CHECK8,  X86::XOR16ri8},
                                           {NOCHECK, X86::XOR16ri}}};  break;
    case X86::XOR32rm:     I = {4, true,  {{CHECK8,  X86::XOR32ri8},
                                           {NOCHECK, X86::XOR32ri}}};  break;
    case X86::XOR64rm:     I = {8, true,  {{CHECK8,  X86::XOR64ri8},
                                           {CHECK32, X86::XOR64ri32},
                                           {NOCHECK, X86::XOR64rm}}};  break;
    }

    // Compute the immediate value.
    assert(Offset + I.DataSize <= ConstantData.size() &&
           "invalid offset for given constant data");
    int64_t ImmVal =
        DataExtractor(ConstantData, true, 8).getSigned(&Offset, I.DataSize);

    // Compute the new opcode.
    unsigned NewOpcode = 0;
    for (const std::pair<CheckSignExt, unsigned> &Check : I.Checks) {
      NewOpcode = Check.second;
      if (Check.first == NOCHECK)
        break;
      if (Check.first == CHECK8 && isInt<8>(ImmVal))
        break;
      if (Check.first == CHECK32 && isInt<32>(ImmVal))
        break;
    }
    if (NewOpcode == Inst.getOpcode())
      return false;

    // Modify the instruction.
    MCOperand ImmOp = MCOperand::createImm(ImmVal);
    uint32_t TargetOpNum = 0;
    // Test instruction does not follow the regular pattern of putting the
    // memory reference of a load (5 MCOperands) last in the list of operands.
    // Since it is not modifying the register operand, it is not treated as
    // a destination operand and it is not the first operand as it is in the
    // other instructions we treat here.
    if (NewOpcode == X86::TEST8ri || NewOpcode == X86::TEST16ri ||
        NewOpcode == X86::TEST32ri || NewOpcode == X86::TEST64ri32)
      TargetOpNum = getMemoryOperandNo(Inst) + X86::AddrNumOperands;

    MCOperand TargetOp = Inst.getOperand(TargetOpNum);
    Inst.clear();
    Inst.setOpcode(NewOpcode);
    Inst.addOperand(TargetOp);
    if (I.HasLHS)
      Inst.addOperand(TargetOp);
    Inst.addOperand(ImmOp);

    return true;
  }

  /// TODO: this implementation currently works for the most common opcodes that
  /// load from memory. It can be extended to work with memory store opcodes as
  /// well as more memory load opcodes.
  bool replaceMemOperandWithReg(MCInst &Inst, MCPhysReg RegNum) const override {
    unsigned NewOpcode;

    switch (Inst.getOpcode()) {
    default: {
      switch (getPopSize(Inst)) {
      case 2:            NewOpcode = X86::MOV16rr; break;
      case 4:            NewOpcode = X86::MOV32rr; break;
      case 8:            NewOpcode = X86::MOV64rr; break;
      default:           return false;
      }
      break;
    }

    // MOV
    case X86::MOV8rm:      NewOpcode = X86::MOV8rr;   break;
    case X86::MOV16rm:     NewOpcode = X86::MOV16rr;  break;
    case X86::MOV32rm:     NewOpcode = X86::MOV32rr;  break;
    case X86::MOV64rm:     NewOpcode = X86::MOV64rr;  break;
    }

    // Modify the instruction.
    MCOperand RegOp = MCOperand::createReg(RegNum);
    MCOperand TargetOp = Inst.getOperand(0);
    Inst.clear();
    Inst.setOpcode(NewOpcode);
    Inst.addOperand(TargetOp);
    Inst.addOperand(RegOp);

    return true;
  }

  bool isRedundantMove(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return false;

    // MOV
    case X86::MOV8rr:
    case X86::MOV16rr:
    case X86::MOV32rr:
    case X86::MOV64rr:
      break;
    }

    assert(Inst.getOperand(0).isReg() && Inst.getOperand(1).isReg());
    return Inst.getOperand(0).getReg() == Inst.getOperand(1).getReg();
  }

  bool requiresAlignedAddress(const MCInst &Inst) const override {
    const MCInstrDesc &Desc = Info->get(Inst.getOpcode());
    for (unsigned int I = 0; I < Desc.getNumOperands(); ++I) {
      const MCOperandInfo &Op = Desc.operands()[I];
      if (Op.OperandType != MCOI::OPERAND_REGISTER)
        continue;
      if (Op.RegClass == X86::VR128RegClassID)
        return true;
    }
    return false;
  }

  bool convertJmpToTailCall(MCInst &Inst) override {
    if (isTailCall(Inst))
      return false;

    int NewOpcode;
    switch (Inst.getOpcode()) {
    default:
      return false;
    case X86::JMP_1:
    case X86::JMP_2:
    case X86::JMP_4:
      NewOpcode = X86::JMP_4;
      break;
    case X86::JMP16m:
    case X86::JMP32m:
    case X86::JMP64m:
      NewOpcode = X86::JMP32m;
      break;
    case X86::JMP16r:
    case X86::JMP32r:
    case X86::JMP64r:
      NewOpcode = X86::JMP32r;
      break;
    }

    Inst.setOpcode(NewOpcode);
    setTailCall(Inst);
    return true;
  }

  bool convertTailCallToJmp(MCInst &Inst) override {
    int NewOpcode;
    switch (Inst.getOpcode()) {
    default:
      return false;
    case X86::JMP_4:
      NewOpcode = X86::JMP_1;
      break;
    case X86::JMP32m:
      NewOpcode = X86::JMP64m;
      break;
    case X86::JMP32r:
      NewOpcode = X86::JMP64r;
      break;
    }

    Inst.setOpcode(NewOpcode);
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    clearOffset(Inst);
    return true;
  }

  bool convertTailCallToCall(MCInst &Inst) override {
    int NewOpcode;
    switch (Inst.getOpcode()) {
    default:
      return false;
    case X86::JMP_4:
      NewOpcode = X86::CALL64pcrel32;
      break;
    case X86::JMP32m:
      NewOpcode = X86::CALL64m;
      break;
    case X86::JMP32r:
      NewOpcode = X86::CALL64r;
      break;
    }

    Inst.setOpcode(NewOpcode);
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    return true;
  }

  InstructionListType createIndirectPLTCall(MCInst &&DirectCall,
                                            const MCSymbol *TargetLocation,
                                            MCContext *Ctx) override {
    assert((DirectCall.getOpcode() == X86::CALL64pcrel32 ||
            (DirectCall.getOpcode() == X86::JMP_4 && isTailCall(DirectCall))) &&
           "64-bit direct (tail) call instruction expected");

    InstructionListType Code;
    // Create a new indirect call by converting the previous direct call.
    MCInst Inst = DirectCall;
    const auto NewOpcode =
        (Inst.getOpcode() == X86::CALL64pcrel32) ? X86::CALL64m : X86::JMP32m;
    Inst.setOpcode(NewOpcode);

    // Replace the first operand and preserve auxiliary operands of
    // the instruction.
    Inst.erase(Inst.begin());
    Inst.insert(Inst.begin(),
                MCOperand::createReg(X86::NoRegister)); // AddrSegmentReg
    Inst.insert(Inst.begin(),
                MCOperand::createExpr(                  // Displacement
                    MCSymbolRefExpr::create(TargetLocation,
                                            MCSymbolRefExpr::VK_None, *Ctx)));
    Inst.insert(Inst.begin(),
                MCOperand::createReg(X86::NoRegister)); // IndexReg
    Inst.insert(Inst.begin(),
                MCOperand::createImm(1));               // ScaleAmt
    Inst.insert(Inst.begin(),
                MCOperand::createReg(X86::RIP));        // BaseReg

    Code.emplace_back(Inst);
    return Code;
  }

  void convertIndirectCallToLoad(MCInst &Inst, MCPhysReg Reg) override {
    bool IsTailCall = isTailCall(Inst);
    if (IsTailCall)
      removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    if (Inst.getOpcode() == X86::CALL64m ||
        (Inst.getOpcode() == X86::JMP32m && IsTailCall)) {
      Inst.setOpcode(X86::MOV64rm);
      Inst.insert(Inst.begin(), MCOperand::createReg(Reg));
      return;
    }
    if (Inst.getOpcode() == X86::CALL64r ||
        (Inst.getOpcode() == X86::JMP32r && IsTailCall)) {
      Inst.setOpcode(X86::MOV64rr);
      Inst.insert(Inst.begin(), MCOperand::createReg(Reg));
      return;
    }
    LLVM_DEBUG(Inst.dump());
    llvm_unreachable("not implemented");
  }

  bool shortenInstruction(MCInst &Inst,
                          const MCSubtargetInfo &STI) const override {
    unsigned OldOpcode = Inst.getOpcode();
    unsigned NewOpcode = OldOpcode;

    int MemOpNo = getMemoryOperandNo(Inst);

    // Check and remove redundant Address-Size override prefix.
    if (opts::X86StripRedundantAddressSize) {
      uint64_t TSFlags = Info->get(OldOpcode).TSFlags;
      unsigned Flags = Inst.getFlags();

      if (!X86_MC::needsAddressSizeOverride(Inst, STI, MemOpNo, TSFlags) &&
          Flags & X86::IP_HAS_AD_SIZE)
        Inst.setFlags(Flags ^ X86::IP_HAS_AD_SIZE);
    }

    // Check and remove EIZ/RIZ. These cases represent ambiguous cases where
    // SIB byte is present, but no index is used and modrm alone should have
    // been enough. Converting to NoRegister effectively removes the SIB byte.
    if (MemOpNo >= 0) {
      MCOperand &IndexOp =
          Inst.getOperand(static_cast<unsigned>(MemOpNo) + X86::AddrIndexReg);
      if (IndexOp.getReg() == X86::EIZ || IndexOp.getReg() == X86::RIZ)
        IndexOp = MCOperand::createReg(X86::NoRegister);
    }

    if (isBranch(Inst)) {
      NewOpcode = getShortBranchOpcode(OldOpcode);
    } else if (OldOpcode == X86::MOV64ri) {
      if (Inst.getOperand(MCPlus::getNumPrimeOperands(Inst) - 1).isImm()) {
        const int64_t Imm =
            Inst.getOperand(MCPlus::getNumPrimeOperands(Inst) - 1).getImm();
        if (int64_t(Imm) == int64_t(int32_t(Imm)))
          NewOpcode = X86::MOV64ri32;
      }
    } else {
      // If it's arithmetic instruction check if signed operand fits in 1 byte.
      const unsigned ShortOpcode = X86::getOpcodeForShortImmediateForm(OldOpcode);
      if (ShortOpcode != OldOpcode &&
          Inst.getOperand(MCPlus::getNumPrimeOperands(Inst) - 1).isImm()) {
        int64_t Imm =
            Inst.getOperand(MCPlus::getNumPrimeOperands(Inst) - 1).getImm();
        if (int64_t(Imm) == int64_t(int8_t(Imm)))
          NewOpcode = ShortOpcode;
      }
    }

    if (NewOpcode == OldOpcode)
      return false;

    Inst.setOpcode(NewOpcode);
    return true;
  }

  bool
  convertMoveToConditionalMove(MCInst &Inst, unsigned CC, bool AllowStackMemOp,
                               bool AllowBasePtrStackMemOp) const override {
    // - Register-register moves are OK
    // - Stores are filtered out by opcode (no store CMOV)
    // - Non-stack loads are prohibited (generally unsafe)
    // - Stack loads are OK if AllowStackMemOp is true
    // - Stack loads with RBP are OK if AllowBasePtrStackMemOp is true
    if (mayLoad(Inst)) {
      // If stack memory operands are not allowed, no loads are allowed
      if (!AllowStackMemOp)
        return false;

      // If stack memory operands are allowed, check if it's a load from stack
      bool IsLoad, IsStore, IsStoreFromReg, IsSimple, IsIndexed;
      MCPhysReg Reg;
      int32_t SrcImm;
      uint16_t StackPtrReg;
      int64_t StackOffset;
      uint8_t Size;
      bool IsStackAccess =
          isStackAccess(Inst, IsLoad, IsStore, IsStoreFromReg, Reg, SrcImm,
                        StackPtrReg, StackOffset, Size, IsSimple, IsIndexed);
      // Prohibit non-stack-based loads
      if (!IsStackAccess)
        return false;
      // If stack memory operands are allowed, check if it's RBP-based
      if (!AllowBasePtrStackMemOp &&
          RegInfo->isSubRegisterEq(X86::RBP, StackPtrReg))
        return false;
    }

    unsigned NewOpcode = 0;
    switch (Inst.getOpcode()) {
    case X86::MOV16rr:
      NewOpcode = X86::CMOV16rr;
      break;
    case X86::MOV16rm:
      NewOpcode = X86::CMOV16rm;
      break;
    case X86::MOV32rr:
      NewOpcode = X86::CMOV32rr;
      break;
    case X86::MOV32rm:
      NewOpcode = X86::CMOV32rm;
      break;
    case X86::MOV64rr:
      NewOpcode = X86::CMOV64rr;
      break;
    case X86::MOV64rm:
      NewOpcode = X86::CMOV64rm;
      break;
    default:
      return false;
    }
    Inst.setOpcode(NewOpcode);
    // Insert CC at the end of prime operands, before annotations
    Inst.insert(Inst.begin() + MCPlus::getNumPrimeOperands(Inst),
                MCOperand::createImm(CC));
    // CMOV is a 3-operand MCInst, so duplicate the destination as src1
    Inst.insert(Inst.begin(), Inst.getOperand(0));
    return true;
  }

  bool lowerTailCall(MCInst &Inst) override {
    if (Inst.getOpcode() == X86::JMP_4 && isTailCall(Inst)) {
      Inst.setOpcode(X86::JMP_1);
      removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
      return true;
    }
    return false;
  }

  const MCSymbol *getTargetSymbol(const MCInst &Inst,
                                  unsigned OpNum = 0) const override {
    if (OpNum >= MCPlus::getNumPrimeOperands(Inst))
      return nullptr;

    const MCOperand &Op = Inst.getOperand(OpNum);
    if (!Op.isExpr())
      return nullptr;

    auto *SymExpr = dyn_cast<MCSymbolRefExpr>(Op.getExpr());
    if (!SymExpr || SymExpr->getKind() != MCSymbolRefExpr::VK_None)
      return nullptr;

    return &SymExpr->getSymbol();
  }

  // This is the same as the base class, but since we are overriding one of
  // getTargetSymbol's signatures above, we need to override all of them.
  const MCSymbol *getTargetSymbol(const MCExpr *Expr) const override {
    return &cast<const MCSymbolRefExpr>(Expr)->getSymbol();
  }

  bool analyzeBranch(InstructionIterator Begin, InstructionIterator End,
                     const MCSymbol *&TBB, const MCSymbol *&FBB,
                     MCInst *&CondBranch,
                     MCInst *&UncondBranch) const override {
    auto I = End;

    // Bottom-up analysis
    while (I != Begin) {
      --I;

      // Ignore nops and CFIs
      if (isPseudo(*I))
        continue;

      // Stop when we find the first non-terminator
      if (!isTerminator(*I))
        break;

      if (!isBranch(*I))
        break;

      // Handle unconditional branches.
      if ((I->getOpcode() == X86::JMP_1 || I->getOpcode() == X86::JMP_2 ||
           I->getOpcode() == X86::JMP_4) &&
          !isTailCall(*I)) {
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

      // Ignore indirect branches
      if (getCondCode(*I) == X86::COND_INVALID)
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

  /// Analyzes PIC-style jump table code template and return identified
  /// IndirectBranchType, MemLocInstr (all cases) and FixedEntryLoadInstr
  /// (POSSIBLE_PIC_FIXED_BRANCH case).
  template <typename Itr>
  std::tuple<IndirectBranchType, MCInst *, MCInst *>
  analyzePICJumpTable(Itr II, Itr IE, MCPhysReg R1, MCPhysReg R2) const {
    // Analyze PIC-style jump table code template:
    //
    //    lea PIC_JUMP_TABLE(%rip), {%r1|%r2}     <- MemLocInstr
    //    mov ({%r1|%r2}, %index, 4), {%r2|%r1}
    //    add %r2, %r1
    //    jmp *%r1
    //
    // or a fixed indirect jump template:
    //
    //    movslq En(%rip), {%r2|%r1}              <- FixedEntryLoadInstr
    //    lea PIC_JUMP_TABLE(%rip), {%r1|%r2}     <- MemLocInstr
    //    add %r2, %r1
    //    jmp *%r1
    //
    // (with any irrelevant instructions in-between)
    //
    // When we call this helper we've already determined %r1 and %r2, and
    // reverse instruction iterator \p II is pointing to the ADD instruction.
    //
    // PIC jump table looks like following:
    //
    //   JT:  ----------
    //    E1:| L1 - JT  |
    //       |----------|
    //    E2:| L2 - JT  |
    //       |----------|
    //       |          |
    //          ......
    //    En:| Ln - JT  |
    //        ----------
    //
    // Where L1, L2, ..., Ln represent labels in the function.
    //
    // The actual relocations in the table will be of the form:
    //
    //   Ln - JT
    //    = (Ln - En) + (En - JT)
    //    = R_X86_64_PC32(Ln) + En - JT
    //    = R_X86_64_PC32(Ln + offsetof(En))
    //
    auto isRIPRel = [&](X86MemOperand &MO) {
      // NB: DispExpr should be set
      return MO.DispExpr != nullptr &&
             MO.BaseRegNum == RegInfo->getProgramCounter() &&
             MO.IndexRegNum == X86::NoRegister &&
             MO.SegRegNum == X86::NoRegister;
    };
    auto isIndexed = [](X86MemOperand &MO, MCPhysReg R) {
      // NB: IndexRegNum should be set.
      return MO.IndexRegNum != X86::NoRegister && MO.BaseRegNum == R &&
             MO.ScaleImm == 4 && MO.DispImm == 0 &&
             MO.SegRegNum == X86::NoRegister;
    };
    LLVM_DEBUG(dbgs() << "Checking for PIC jump table\n");
    MCInst *FirstInstr = nullptr;
    MCInst *SecondInstr = nullptr;
    enum {
      NOMATCH = 0,
      MATCH_JUMP_TABLE,
      MATCH_FIXED_BRANCH,
    } MatchingState = NOMATCH;
    while (++II != IE) {
      MCInst &Instr = *II;
      const MCInstrDesc &InstrDesc = Info->get(Instr.getOpcode());
      if (!InstrDesc.hasDefOfPhysReg(Instr, R1, *RegInfo) &&
          !InstrDesc.hasDefOfPhysReg(Instr, R2, *RegInfo)) {
        // Ignore instructions that don't affect R1, R2 registers.
        continue;
      }
      const bool IsMOVSXInstr = isMOVSX64rm32(Instr);
      const bool IsLEAInstr = isLEA64r(Instr);
      if (MatchingState == NOMATCH) {
        if (IsMOVSXInstr)
          MatchingState = MATCH_JUMP_TABLE;
        else if (IsLEAInstr)
          MatchingState = MATCH_FIXED_BRANCH;
        else
          break;

        // Check if the first instruction is setting %r1 or %r2. In canonical
        // form lea sets %r1 and mov sets %r2. If it's the opposite - rename so
        // we have to only check a single form.
        unsigned DestReg = Instr.getOperand(0).getReg();
        MCPhysReg &ExpectReg = MatchingState == MATCH_JUMP_TABLE ? R2 : R1;
        if (DestReg != ExpectReg)
          std::swap(R1, R2);
        if (DestReg != ExpectReg)
          break;

        // Verify operands
        std::optional<X86MemOperand> MO = evaluateX86MemoryOperand(Instr);
        if (!MO)
          break;
        if ((MatchingState == MATCH_JUMP_TABLE && isIndexed(*MO, R1)) ||
            (MatchingState == MATCH_FIXED_BRANCH && isRIPRel(*MO)))
          FirstInstr = &Instr;
        else
          break;
      } else {
        unsigned ExpectReg = MatchingState == MATCH_JUMP_TABLE ? R1 : R2;
        if (!InstrDesc.hasDefOfPhysReg(Instr, ExpectReg, *RegInfo))
          continue;
        if ((MatchingState == MATCH_JUMP_TABLE && !IsLEAInstr) ||
            (MatchingState == MATCH_FIXED_BRANCH && !IsMOVSXInstr))
          break;
        if (Instr.getOperand(0).getReg() != ExpectReg)
          break;

        // Verify operands.
        std::optional<X86MemOperand> MO = evaluateX86MemoryOperand(Instr);
        if (!MO)
          break;
        if (!isRIPRel(*MO))
          break;
        SecondInstr = &Instr;
        break;
      }
    }

    if (!SecondInstr)
      return std::make_tuple(IndirectBranchType::UNKNOWN, nullptr, nullptr);

    if (MatchingState == MATCH_FIXED_BRANCH) {
      LLVM_DEBUG(dbgs() << "checking potential fixed indirect branch\n");
      return std::make_tuple(IndirectBranchType::POSSIBLE_PIC_FIXED_BRANCH,
                             FirstInstr, SecondInstr);
    }
    LLVM_DEBUG(dbgs() << "checking potential PIC jump table\n");
    return std::make_tuple(IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE,
                           SecondInstr, nullptr);
  }

  IndirectBranchType
  analyzeIndirectBranch(MCInst &Instruction, InstructionIterator Begin,
                        InstructionIterator End, const unsigned PtrSize,
                        MCInst *&MemLocInstrOut, unsigned &BaseRegNumOut,
                        unsigned &IndexRegNumOut, int64_t &DispValueOut,
                        const MCExpr *&DispExprOut, MCInst *&PCRelBaseOut,
                        MCInst *&FixedEntryLoadInst) const override {
    // Try to find a (base) memory location from where the address for
    // the indirect branch is loaded. For X86-64 the memory will be specified
    // in the following format:
    //
    //   {%rip}/{%basereg} + Imm + IndexReg * Scale
    //
    // We are interested in the cases where Scale == sizeof(uintptr_t) and
    // the contents of the memory are presumably an array of pointers to code.
    //
    // Normal jump table:
    //
    //    jmp *(JUMP_TABLE, %index, Scale)        <- MemLocInstr
    //
    //    or
    //
    //    mov (JUMP_TABLE, %index, Scale), %r1    <- MemLocInstr
    //    ...
    //    jmp %r1
    //
    // We handle PIC-style jump tables separately.
    //
    MemLocInstrOut = nullptr;
    BaseRegNumOut = X86::NoRegister;
    IndexRegNumOut = X86::NoRegister;
    DispValueOut = 0;
    DispExprOut = nullptr;
    FixedEntryLoadInst = nullptr;

    std::reverse_iterator<InstructionIterator> II(End);
    std::reverse_iterator<InstructionIterator> IE(Begin);

    IndirectBranchType Type = IndirectBranchType::UNKNOWN;

    // An instruction referencing memory used by jump instruction (directly or
    // via register). This location could be an array of function pointers
    // in case of indirect tail call, or a jump table.
    MCInst *MemLocInstr = nullptr;

    if (MCPlus::getNumPrimeOperands(Instruction) == 1) {
      // If the indirect jump is on register - try to detect if the
      // register value is loaded from a memory location.
      assert(Instruction.getOperand(0).isReg() && "register operand expected");
      const unsigned R1 = Instruction.getOperand(0).getReg();
      // Check if one of the previous instructions defines the jump-on register.
      for (auto PrevII = II; PrevII != IE; ++PrevII) {
        MCInst &PrevInstr = *PrevII;
        const MCInstrDesc &PrevInstrDesc = Info->get(PrevInstr.getOpcode());

        if (!PrevInstrDesc.hasDefOfPhysReg(PrevInstr, R1, *RegInfo))
          continue;

        if (isMoveMem2Reg(PrevInstr)) {
          MemLocInstr = &PrevInstr;
          break;
        }
        if (isADD64rr(PrevInstr)) {
          unsigned R2 = PrevInstr.getOperand(2).getReg();
          if (R1 == R2)
            return IndirectBranchType::UNKNOWN;
          std::tie(Type, MemLocInstr, FixedEntryLoadInst) =
              analyzePICJumpTable(PrevII, IE, R1, R2);
          break;
        }
        return IndirectBranchType::UNKNOWN;
      }
      if (!MemLocInstr) {
        // No definition seen for the register in this function so far. Could be
        // an input parameter - which means it is an external code reference.
        // It also could be that the definition happens to be in the code that
        // we haven't processed yet. Since we have to be conservative, return
        // as UNKNOWN case.
        return IndirectBranchType::UNKNOWN;
      }
    } else {
      MemLocInstr = &Instruction;
    }

    const MCRegister RIPRegister = RegInfo->getProgramCounter();

    // Analyze the memory location.
    std::optional<X86MemOperand> MO = evaluateX86MemoryOperand(*MemLocInstr);
    if (!MO)
      return IndirectBranchType::UNKNOWN;

    BaseRegNumOut = MO->BaseRegNum;
    IndexRegNumOut = MO->IndexRegNum;
    DispValueOut = MO->DispImm;
    DispExprOut = MO->DispExpr;

    if ((MO->BaseRegNum != X86::NoRegister && MO->BaseRegNum != RIPRegister) ||
        MO->SegRegNum != X86::NoRegister)
      return IndirectBranchType::UNKNOWN;

    if (MemLocInstr == &Instruction &&
        (!MO->ScaleImm || MO->IndexRegNum == X86::NoRegister)) {
      MemLocInstrOut = MemLocInstr;
      return IndirectBranchType::POSSIBLE_FIXED_BRANCH;
    }

    switch (Type) {
    case IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE:
      if (MO->ScaleImm != 1 || MO->BaseRegNum != RIPRegister)
        return IndirectBranchType::UNKNOWN;
      break;
    case IndirectBranchType::POSSIBLE_PIC_FIXED_BRANCH:
      break;
    default:
      if (MO->ScaleImm != PtrSize)
        return IndirectBranchType::UNKNOWN;
    }

    MemLocInstrOut = MemLocInstr;

    return Type;
  }

  /// Analyze a callsite to see if it could be a virtual method call.  This only
  /// checks to see if the overall pattern is satisfied, it does not guarantee
  /// that the callsite is a true virtual method call.
  /// The format of virtual method calls that are recognized is one of the
  /// following:
  ///
  ///  Form 1: (found in debug code)
  ///    add METHOD_OFFSET, %VtableReg
  ///    mov (%VtableReg), %MethodReg
  ///    ...
  ///    call or jmp *%MethodReg
  ///
  ///  Form 2:
  ///    mov METHOD_OFFSET(%VtableReg), %MethodReg
  ///    ...
  ///    call or jmp *%MethodReg
  ///
  ///  Form 3:
  ///    ...
  ///    call or jmp *METHOD_OFFSET(%VtableReg)
  ///
  bool analyzeVirtualMethodCall(InstructionIterator ForwardBegin,
                                InstructionIterator ForwardEnd,
                                std::vector<MCInst *> &MethodFetchInsns,
                                unsigned &VtableRegNum, unsigned &MethodRegNum,
                                uint64_t &MethodOffset) const override {
    VtableRegNum = X86::NoRegister;
    MethodRegNum = X86::NoRegister;
    MethodOffset = 0;

    std::reverse_iterator<InstructionIterator> Itr(ForwardEnd);
    std::reverse_iterator<InstructionIterator> End(ForwardBegin);

    MCInst &CallInst = *Itr++;
    assert(isIndirectBranch(CallInst) || isCall(CallInst));

    // The call can just be jmp offset(reg)
    if (std::optional<X86MemOperand> MO = evaluateX86MemoryOperand(CallInst)) {
      if (!MO->DispExpr && MO->BaseRegNum != X86::RIP &&
          MO->BaseRegNum != X86::RBP && MO->BaseRegNum != X86::NoRegister) {
        MethodRegNum = MO->BaseRegNum;
        if (MO->ScaleImm == 1 && MO->IndexRegNum == X86::NoRegister &&
            MO->SegRegNum == X86::NoRegister) {
          VtableRegNum = MethodRegNum;
          MethodOffset = MO->DispImm;
          MethodFetchInsns.push_back(&CallInst);
          return true;
        }
      }
      return false;
    }
    if (CallInst.getOperand(0).isReg())
      MethodRegNum = CallInst.getOperand(0).getReg();
    else
      return false;

    if (MethodRegNum == X86::RIP || MethodRegNum == X86::RBP) {
      VtableRegNum = X86::NoRegister;
      MethodRegNum = X86::NoRegister;
      return false;
    }

    // find load from vtable, this may or may not include the method offset
    while (Itr != End) {
      MCInst &CurInst = *Itr++;
      const MCInstrDesc &Desc = Info->get(CurInst.getOpcode());
      if (Desc.hasDefOfPhysReg(CurInst, MethodRegNum, *RegInfo)) {
        if (!mayLoad(CurInst))
          return false;
        if (std::optional<X86MemOperand> MO =
                evaluateX86MemoryOperand(CurInst)) {
          if (!MO->DispExpr && MO->ScaleImm == 1 &&
              MO->BaseRegNum != X86::RIP && MO->BaseRegNum != X86::RBP &&
              MO->BaseRegNum != X86::NoRegister &&
              MO->IndexRegNum == X86::NoRegister &&
              MO->SegRegNum == X86::NoRegister) {
            VtableRegNum = MO->BaseRegNum;
            MethodOffset = MO->DispImm;
            MethodFetchInsns.push_back(&CurInst);
            if (MethodOffset != 0)
              return true;
            break;
          }
        }
        return false;
      }
    }

    if (!VtableRegNum)
      return false;

    // look for any adds affecting the method register.
    while (Itr != End) {
      MCInst &CurInst = *Itr++;
      const MCInstrDesc &Desc = Info->get(CurInst.getOpcode());
      if (Desc.hasDefOfPhysReg(CurInst, VtableRegNum, *RegInfo)) {
        if (isADDri(CurInst)) {
          assert(!MethodOffset);
          MethodOffset = CurInst.getOperand(2).getImm();
          MethodFetchInsns.insert(MethodFetchInsns.begin(), &CurInst);
          break;
        }
      }
    }

    return true;
  }

  void createStackPointerIncrement(MCInst &Inst, int Size,
                                   bool NoFlagsClobber) const override {
    if (NoFlagsClobber) {
      Inst.setOpcode(X86::LEA64r);
      Inst.clear();
      Inst.addOperand(MCOperand::createReg(X86::RSP));
      Inst.addOperand(MCOperand::createReg(X86::RSP));        // BaseReg
      Inst.addOperand(MCOperand::createImm(1));               // ScaleAmt
      Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // IndexReg
      Inst.addOperand(MCOperand::createImm(-Size));           // Displacement
      Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // AddrSegmentReg
      return;
    }
    Inst.setOpcode(X86::SUB64ri8);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(X86::RSP));
    Inst.addOperand(MCOperand::createReg(X86::RSP));
    Inst.addOperand(MCOperand::createImm(Size));
  }

  void createStackPointerDecrement(MCInst &Inst, int Size,
                                   bool NoFlagsClobber) const override {
    if (NoFlagsClobber) {
      Inst.setOpcode(X86::LEA64r);
      Inst.clear();
      Inst.addOperand(MCOperand::createReg(X86::RSP));
      Inst.addOperand(MCOperand::createReg(X86::RSP));        // BaseReg
      Inst.addOperand(MCOperand::createImm(1));               // ScaleAmt
      Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // IndexReg
      Inst.addOperand(MCOperand::createImm(Size));            // Displacement
      Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // AddrSegmentReg
      return;
    }
    Inst.setOpcode(X86::ADD64ri8);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(X86::RSP));
    Inst.addOperand(MCOperand::createReg(X86::RSP));
    Inst.addOperand(MCOperand::createImm(Size));
  }

  void createSaveToStack(MCInst &Inst, const MCPhysReg &StackReg, int Offset,
                         const MCPhysReg &SrcReg, int Size) const override {
    unsigned NewOpcode;
    switch (Size) {
    default:
      llvm_unreachable("Invalid operand size");
      return;
    case 2:      NewOpcode = X86::MOV16mr; break;
    case 4:      NewOpcode = X86::MOV32mr; break;
    case 8:      NewOpcode = X86::MOV64mr; break;
    }
    Inst.setOpcode(NewOpcode);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(StackReg));        // BaseReg
    Inst.addOperand(MCOperand::createImm(1));               // ScaleAmt
    Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // IndexReg
    Inst.addOperand(MCOperand::createImm(Offset));          // Displacement
    Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // AddrSegmentReg
    Inst.addOperand(MCOperand::createReg(SrcReg));
  }

  void createRestoreFromStack(MCInst &Inst, const MCPhysReg &StackReg,
                              int Offset, const MCPhysReg &DstReg,
                              int Size) const override {
    return createLoad(Inst, StackReg, /*Scale=*/1, /*IndexReg=*/X86::NoRegister,
                      Offset, nullptr, /*AddrSegmentReg=*/X86::NoRegister,
                      DstReg, Size);
  }

  void createLoad(MCInst &Inst, const MCPhysReg &BaseReg, int64_t Scale,
                  const MCPhysReg &IndexReg, int64_t Offset,
                  const MCExpr *OffsetExpr, const MCPhysReg &AddrSegmentReg,
                  const MCPhysReg &DstReg, int Size) const override {
    unsigned NewOpcode;
    switch (Size) {
    default:
      llvm_unreachable("Invalid operand size");
      return;
    case 2:      NewOpcode = X86::MOV16rm; break;
    case 4:      NewOpcode = X86::MOV32rm; break;
    case 8:      NewOpcode = X86::MOV64rm; break;
    }
    Inst.setOpcode(NewOpcode);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(DstReg));
    Inst.addOperand(MCOperand::createReg(BaseReg));
    Inst.addOperand(MCOperand::createImm(Scale));
    Inst.addOperand(MCOperand::createReg(IndexReg));
    if (OffsetExpr)
      Inst.addOperand(MCOperand::createExpr(OffsetExpr)); // Displacement
    else
      Inst.addOperand(MCOperand::createImm(Offset)); // Displacement
    Inst.addOperand(MCOperand::createReg(AddrSegmentReg)); // AddrSegmentReg
  }

  InstructionListType createLoadImmediate(const MCPhysReg Dest,
                                          uint64_t Imm) const override {
    InstructionListType Insts;
    Insts.emplace_back();
    Insts.back().setOpcode(X86::MOV64ri32);
    Insts.back().clear();
    Insts.back().addOperand(MCOperand::createReg(Dest));
    Insts.back().addOperand(MCOperand::createImm(Imm));
    return Insts;
  }

  void createIJmp32Frag(SmallVectorImpl<MCInst> &Insts,
                        const MCOperand &BaseReg, const MCOperand &Scale,
                        const MCOperand &IndexReg, const MCOperand &Offset,
                        const MCOperand &TmpReg) const override {
    // The code fragment we emit here is:
    //
    //  mov32 (%base, %index, scale), %tmpreg
    //  ijmp *(%tmpreg)
    //
    MCInst IJmp;
    IJmp.setOpcode(X86::JMP64r);
    IJmp.addOperand(TmpReg);

    MCInst Load;
    Load.setOpcode(X86::MOV32rm);
    Load.addOperand(TmpReg);
    Load.addOperand(BaseReg);
    Load.addOperand(Scale);
    Load.addOperand(IndexReg);
    Load.addOperand(Offset);
    Load.addOperand(MCOperand::createReg(X86::NoRegister));

    Insts.push_back(Load);
    Insts.push_back(IJmp);
  }

  void createNoop(MCInst &Inst) const override {
    Inst.setOpcode(X86::NOOP);
    Inst.clear();
  }

  void createReturn(MCInst &Inst) const override {
    Inst.setOpcode(X86::RET64);
    Inst.clear();
  }

  InstructionListType createInlineMemcpy(bool ReturnEnd) const override {
    InstructionListType Code;
    if (ReturnEnd)
      Code.emplace_back(MCInstBuilder(X86::LEA64r)
                            .addReg(X86::RAX)
                            .addReg(X86::RDI)
                            .addImm(1)
                            .addReg(X86::RDX)
                            .addImm(0)
                            .addReg(X86::NoRegister));
    else
      Code.emplace_back(MCInstBuilder(X86::MOV64rr)
                            .addReg(X86::RAX)
                            .addReg(X86::RDI));

    Code.emplace_back(MCInstBuilder(X86::MOV32rr)
                          .addReg(X86::ECX)
                          .addReg(X86::EDX));
    Code.emplace_back(MCInstBuilder(X86::REP_MOVSB_64));

    return Code;
  }

  InstructionListType createOneByteMemcpy() const override {
    InstructionListType Code;
    Code.emplace_back(MCInstBuilder(X86::MOV8rm)
                          .addReg(X86::CL)
                          .addReg(X86::RSI)
                          .addImm(0)
                          .addReg(X86::NoRegister)
                          .addImm(0)
                          .addReg(X86::NoRegister));
    Code.emplace_back(MCInstBuilder(X86::MOV8mr)
                          .addReg(X86::RDI)
                          .addImm(0)
                          .addReg(X86::NoRegister)
                          .addImm(0)
                          .addReg(X86::NoRegister)
                          .addReg(X86::CL));
    Code.emplace_back(MCInstBuilder(X86::MOV64rr)
                          .addReg(X86::RAX)
                          .addReg(X86::RDI));
    return Code;
  }

  InstructionListType createCmpJE(MCPhysReg RegNo, int64_t Imm,
                                  const MCSymbol *Target,
                                  MCContext *Ctx) const override {
    InstructionListType Code;
    Code.emplace_back(MCInstBuilder(X86::CMP64ri8)
                          .addReg(RegNo)
                          .addImm(Imm));
    Code.emplace_back(MCInstBuilder(X86::JCC_1)
                          .addExpr(MCSymbolRefExpr::create(
                              Target, MCSymbolRefExpr::VK_None, *Ctx))
                          .addImm(X86::COND_E));
    return Code;
  }

  std::optional<Relocation>
  createRelocation(const MCFixup &Fixup,
                   const MCAsmBackend &MAB) const override {
    const MCFixupKindInfo &FKI = MAB.getFixupKindInfo(Fixup.getKind());

    assert(FKI.TargetOffset == 0 && "0-bit relocation offset expected");
    const uint64_t RelOffset = Fixup.getOffset();

    uint64_t RelType;
    if (FKI.Flags & MCFixupKindInfo::FKF_IsPCRel) {
      switch (FKI.TargetSize) {
      default:
        return std::nullopt;
      case  8: RelType = ELF::R_X86_64_PC8; break;
      case 16: RelType = ELF::R_X86_64_PC16; break;
      case 32: RelType = ELF::R_X86_64_PC32; break;
      case 64: RelType = ELF::R_X86_64_PC64; break;
      }
    } else {
      switch (FKI.TargetSize) {
      default:
        return std::nullopt;
      case  8: RelType = ELF::R_X86_64_8; break;
      case 16: RelType = ELF::R_X86_64_16; break;
      case 32: RelType = ELF::R_X86_64_32; break;
      case 64: RelType = ELF::R_X86_64_64; break;
      }
    }

    auto [RelSymbol, RelAddend] = extractFixupExpr(Fixup);

    return Relocation({RelOffset, RelSymbol, RelType, RelAddend, 0});
  }

  bool replaceImmWithSymbolRef(MCInst &Inst, const MCSymbol *Symbol,
                               int64_t Addend, MCContext *Ctx, int64_t &Value,
                               uint64_t RelType) const override {
    unsigned ImmOpNo = -1U;

    for (unsigned Index = 0; Index < MCPlus::getNumPrimeOperands(Inst);
         ++Index) {
      if (Inst.getOperand(Index).isImm()) {
        ImmOpNo = Index;
        // TODO: this is a bit hacky.  It finds the correct operand by
        // searching for a specific immediate value.  If no value is
        // provided it defaults to the last immediate operand found.
        // This could lead to unexpected results if the instruction
        // has more than one immediate with the same value.
        if (Inst.getOperand(ImmOpNo).getImm() == Value)
          break;
      }
    }

    if (ImmOpNo == -1U)
      return false;

    Value = Inst.getOperand(ImmOpNo).getImm();

    setOperandToSymbolRef(Inst, ImmOpNo, Symbol, Addend, Ctx, RelType);

    return true;
  }

  bool replaceRegWithImm(MCInst &Inst, unsigned Register,
                         int64_t Imm) const override {

    enum CheckSignExt : uint8_t {
      NOCHECK = 0,
      CHECK8,
      CHECK32,
    };

    using CheckList = std::vector<std::pair<CheckSignExt, unsigned>>;
    struct InstInfo {
      // Size in bytes that Inst loads from memory.
      uint8_t DataSize;

      // True when the target operand has to be duplicated because the opcode
      // expects a LHS operand.
      bool HasLHS;

      // List of checks and corresponding opcodes to be used. We try to use the
      // smallest possible immediate value when various sizes are available,
      // hence we may need to check whether a larger constant fits in a smaller
      // immediate.
      CheckList Checks;
    };

    InstInfo I;

    switch (Inst.getOpcode()) {
    default: {
      switch (getPushSize(Inst)) {

      case 2: I = {2, false, {{CHECK8, X86::PUSH16i8}, {NOCHECK, X86::PUSH16i}}}; break;
      case 4: I = {4, false, {{CHECK8, X86::PUSH32i8}, {NOCHECK, X86::PUSH32i}}}; break;
      case 8: I = {8, false, {{CHECK8, X86::PUSH64i8},
                              {CHECK32, X86::PUSH64i32},
                              {NOCHECK, Inst.getOpcode()}}}; break;
      default: return false;
      }
      break;
    }

    // MOV
    case X86::MOV8rr:       I = {1, false, {{NOCHECK, X86::MOV8ri}}}; break;
    case X86::MOV16rr:      I = {2, false, {{NOCHECK, X86::MOV16ri}}}; break;
    case X86::MOV32rr:      I = {4, false, {{NOCHECK, X86::MOV32ri}}}; break;
    case X86::MOV64rr:      I = {8, false, {{CHECK32, X86::MOV64ri32},
                                            {NOCHECK, X86::MOV64ri}}}; break;

    case X86::MOV8mr:       I = {1, false, {{NOCHECK, X86::MOV8mi}}}; break;
    case X86::MOV16mr:      I = {2, false, {{NOCHECK, X86::MOV16mi}}}; break;
    case X86::MOV32mr:      I = {4, false, {{NOCHECK, X86::MOV32mi}}}; break;
    case X86::MOV64mr:      I = {8, false, {{CHECK32, X86::MOV64mi32},
                                            {NOCHECK, X86::MOV64mr}}}; break;

    // MOVZX
    case X86::MOVZX16rr8:   I = {1, false, {{NOCHECK, X86::MOV16ri}}}; break;
    case X86::MOVZX32rr8:   I = {1, false, {{NOCHECK, X86::MOV32ri}}}; break;
    case X86::MOVZX32rr16:  I = {2, false, {{NOCHECK, X86::MOV32ri}}}; break;

    // CMP
    case X86::CMP8rr:       I = {1, false, {{NOCHECK, X86::CMP8ri}}}; break;
    case X86::CMP16rr:      I = {2, false, {{CHECK8, X86::CMP16ri8},
                                            {NOCHECK, X86::CMP16ri}}}; break;
    case X86::CMP32rr:      I = {4, false, {{CHECK8, X86::CMP32ri8},
                                            {NOCHECK, X86::CMP32ri}}}; break;
    case X86::CMP64rr:      I = {8, false, {{CHECK8, X86::CMP64ri8},
                                            {CHECK32, X86::CMP64ri32},
                                            {NOCHECK, X86::CMP64rr}}}; break;

    // TEST
    case X86::TEST8rr:      I = {1, false, {{NOCHECK, X86::TEST8ri}}}; break;
    case X86::TEST16rr:     I = {2, false, {{NOCHECK, X86::TEST16ri}}}; break;
    case X86::TEST32rr:     I = {4, false, {{NOCHECK, X86::TEST32ri}}}; break;
    case X86::TEST64rr:     I = {8, false, {{CHECK32, X86::TEST64ri32},
                                            {NOCHECK, X86::TEST64rr}}}; break;

    // ADD
    case X86::ADD8rr:       I = {1, true, {{NOCHECK, X86::ADD8ri}}}; break;
    case X86::ADD16rr:      I = {2, true, {{CHECK8, X86::ADD16ri8},
                                           {NOCHECK, X86::ADD16ri}}}; break;
    case X86::ADD32rr:      I = {4, true, {{CHECK8, X86::ADD32ri8},
                                           {NOCHECK, X86::ADD32ri}}}; break;
    case X86::ADD64rr:      I = {8, true, {{CHECK8, X86::ADD64ri8},
                                           {CHECK32, X86::ADD64ri32},
                                           {NOCHECK, X86::ADD64rr}}}; break;

    // SUB
    case X86::SUB8rr:       I = {1, true, {{NOCHECK, X86::SUB8ri}}}; break;
    case X86::SUB16rr:      I = {2, true, {{CHECK8, X86::SUB16ri8},
                                           {NOCHECK, X86::SUB16ri}}}; break;
    case X86::SUB32rr:      I = {4, true, {{CHECK8, X86::SUB32ri8},
                                           {NOCHECK, X86::SUB32ri}}}; break;
    case X86::SUB64rr:      I = {8, true, {{CHECK8, X86::SUB64ri8},
                                           {CHECK32, X86::SUB64ri32},
                                           {NOCHECK, X86::SUB64rr}}}; break;

    // AND
    case X86::AND8rr:       I = {1, true, {{NOCHECK, X86::AND8ri}}}; break;
    case X86::AND16rr:      I = {2, true, {{CHECK8, X86::AND16ri8},
                                           {NOCHECK, X86::AND16ri}}}; break;
    case X86::AND32rr:      I = {4, true, {{CHECK8, X86::AND32ri8},
                                           {NOCHECK, X86::AND32ri}}}; break;
    case X86::AND64rr:      I = {8, true, {{CHECK8, X86::AND64ri8},
                                           {CHECK32, X86::AND64ri32},
                                           {NOCHECK, X86::AND64rr}}}; break;

    // OR
    case X86::OR8rr:        I = {1, true, {{NOCHECK, X86::OR8ri}}}; break;
    case X86::OR16rr:       I = {2, true, {{CHECK8, X86::OR16ri8},
                                           {NOCHECK, X86::OR16ri}}}; break;
    case X86::OR32rr:       I = {4, true, {{CHECK8, X86::OR32ri8},
                                           {NOCHECK, X86::OR32ri}}}; break;
    case X86::OR64rr:       I = {8, true, {{CHECK8, X86::OR64ri8},
                                           {CHECK32, X86::OR64ri32},
                                           {NOCHECK, X86::OR64rr}}}; break;

    // XOR
    case X86::XOR8rr:       I = {1, true, {{NOCHECK, X86::XOR8ri}}}; break;
    case X86::XOR16rr:      I = {2, true, {{CHECK8, X86::XOR16ri8},
                                           {NOCHECK, X86::XOR16ri}}}; break;
    case X86::XOR32rr:      I = {4, true, {{CHECK8, X86::XOR32ri8},
                                           {NOCHECK, X86::XOR32ri}}}; break;
    case X86::XOR64rr:      I = {8, true, {{CHECK8, X86::XOR64ri8},
                                           {CHECK32, X86::XOR64ri32},
                                           {NOCHECK, X86::XOR64rr}}}; break;
    }

    // Compute the new opcode.
    unsigned NewOpcode = 0;
    for (const std::pair<CheckSignExt, unsigned> &Check : I.Checks) {
      NewOpcode = Check.second;
      if (Check.first == NOCHECK)
        break;
      if (Check.first == CHECK8 && isInt<8>(Imm))
        break;
      if (Check.first == CHECK32 && isInt<32>(Imm))
        break;
    }
    if (NewOpcode == Inst.getOpcode())
      return false;

    const MCInstrDesc &InstDesc = Info->get(Inst.getOpcode());

    unsigned NumFound = 0;
    for (unsigned Index = InstDesc.getNumDefs() + (I.HasLHS ? 1 : 0),
                  E = InstDesc.getNumOperands();
         Index != E; ++Index)
      if (Inst.getOperand(Index).isReg() &&
          Inst.getOperand(Index).getReg() == Register)
        NumFound++;

    if (NumFound != 1)
      return false;

    MCOperand TargetOp = Inst.getOperand(0);
    Inst.clear();
    Inst.setOpcode(NewOpcode);
    Inst.addOperand(TargetOp);
    if (I.HasLHS)
      Inst.addOperand(TargetOp);
    Inst.addOperand(MCOperand::createImm(Imm));

    return true;
  }

  bool replaceRegWithReg(MCInst &Inst, unsigned ToReplace,
                         unsigned ReplaceWith) const override {

    // Get the HasLHS value so that iteration can be done
    bool HasLHS;
    if (X86::isAND(Inst.getOpcode()) || X86::isADD(Inst.getOpcode()) ||
        X86::isSUB(Inst.getOpcode())) {
      HasLHS = true;
    } else if (isPop(Inst) || isPush(Inst) || X86::isCMP(Inst.getOpcode()) ||
               X86::isTEST(Inst.getOpcode())) {
      HasLHS = false;
    } else {
      switch (Inst.getOpcode()) {
      case X86::MOV8rr:
      case X86::MOV8rm:
      case X86::MOV8mr:
      case X86::MOV8ri:
      case X86::MOV16rr:
      case X86::MOV16rm:
      case X86::MOV16mr:
      case X86::MOV16ri:
      case X86::MOV32rr:
      case X86::MOV32rm:
      case X86::MOV32mr:
      case X86::MOV32ri:
      case X86::MOV64rr:
      case X86::MOV64rm:
      case X86::MOV64mr:
      case X86::MOV64ri:
      case X86::MOVZX16rr8:
      case X86::MOVZX32rr8:
      case X86::MOVZX32rr16:
      case X86::MOVSX32rm8:
      case X86::MOVSX32rr8:
      case X86::MOVSX64rm32:
      case X86::LEA64r:
        HasLHS = false;
        break;
      default:
        return false;
      }
    }

    const MCInstrDesc &InstDesc = Info->get(Inst.getOpcode());

    bool FoundOne = false;

    // Iterate only through src operands that arent also dest operands
    for (unsigned Index = InstDesc.getNumDefs() + (HasLHS ? 1 : 0),
                  E = InstDesc.getNumOperands();
         Index != E; ++Index) {
      BitVector RegAliases = getAliases(ToReplace, true);
      if (!Inst.getOperand(Index).isReg() ||
          !RegAliases.test(Inst.getOperand(Index).getReg()))
        continue;
      // Resize register if needed
      unsigned SizedReplaceWith = getAliasSized(
          ReplaceWith, getRegSize(Inst.getOperand(Index).getReg()));
      MCOperand NewOperand = MCOperand::createReg(SizedReplaceWith);
      Inst.getOperand(Index) = NewOperand;
      FoundOne = true;
    }

    // Return true if at least one operand was replaced
    return FoundOne;
  }

  void createUncondBranch(MCInst &Inst, const MCSymbol *TBB,
                          MCContext *Ctx) const override {
    Inst.clear();
    Inst.setOpcode(X86::JMP_1);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx)));
  }

  void createLongUncondBranch(MCInst &Inst, const MCSymbol *Target,
                              MCContext *Ctx) const override {
    Inst.setOpcode(X86::JMP_4);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx)));
  }

  void createCall(MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) override {
    Inst.setOpcode(X86::CALL64pcrel32);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx)));
  }

  void createTailCall(MCInst &Inst, const MCSymbol *Target,
                      MCContext *Ctx) override {
    return createDirectCall(Inst, Target, Ctx, /*IsTailCall*/ true);
  }

  void createLongTailCall(InstructionListType &Seq, const MCSymbol *Target,
                          MCContext *Ctx) override {
    Seq.clear();
    Seq.emplace_back();
    createDirectCall(Seq.back(), Target, Ctx, /*IsTailCall*/ true);
  }

  void createTrap(MCInst &Inst) const override {
    Inst.clear();
    Inst.setOpcode(X86::TRAP);
  }

  void createCondBranch(MCInst &Inst, const MCSymbol *Target, unsigned CC,
                        MCContext *Ctx) const override {
    Inst.setOpcode(X86::JCC_1);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx)));
    Inst.addOperand(MCOperand::createImm(CC));
  }

  void createLongCondBranch(MCInst &Inst, const MCSymbol *Target, unsigned CC,
                            MCContext *Ctx) const override {
    Inst.setOpcode(X86::JCC_4);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx)));
    Inst.addOperand(MCOperand::createImm(CC));
  }

  void reverseBranchCondition(MCInst &Inst, const MCSymbol *TBB,
                              MCContext *Ctx) const override {
    unsigned InvCC = getInvertedCondCode(getCondCode(Inst));
    assert(InvCC != X86::COND_INVALID && "invalid branch instruction");
    Inst.getOperand(Info->get(Inst.getOpcode()).NumOperands - 1).setImm(InvCC);
    Inst.getOperand(0) = MCOperand::createExpr(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx));
  }

  bool replaceBranchCondition(MCInst &Inst, const MCSymbol *TBB, MCContext *Ctx,
                              unsigned CC) const override {
    if (CC == X86::COND_INVALID)
      return false;
    Inst.getOperand(Info->get(Inst.getOpcode()).NumOperands - 1).setImm(CC);
    Inst.getOperand(0) = MCOperand::createExpr(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx));
    return true;
  }

  unsigned getCanonicalBranchCondCode(unsigned CC) const override {
    switch (CC) {
    default:           return X86::COND_INVALID;

    case X86::COND_E:  return X86::COND_E;
    case X86::COND_NE: return X86::COND_E;

    case X86::COND_L:  return X86::COND_L;
    case X86::COND_GE: return X86::COND_L;

    case X86::COND_LE: return X86::COND_G;
    case X86::COND_G:  return X86::COND_G;

    case X86::COND_B:  return X86::COND_B;
    case X86::COND_AE: return X86::COND_B;

    case X86::COND_BE: return X86::COND_A;
    case X86::COND_A:  return X86::COND_A;

    case X86::COND_S:  return X86::COND_S;
    case X86::COND_NS: return X86::COND_S;

    case X86::COND_P:  return X86::COND_P;
    case X86::COND_NP: return X86::COND_P;

    case X86::COND_O:  return X86::COND_O;
    case X86::COND_NO: return X86::COND_O;
    }
  }

  void replaceBranchTarget(MCInst &Inst, const MCSymbol *TBB,
                           MCContext *Ctx) const override {
    assert((isCall(Inst) || isBranch(Inst)) && !isIndirectBranch(Inst) &&
           "Invalid instruction");
    Inst.getOperand(0) = MCOperand::createExpr(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx));
  }

  MCPhysReg getX86R11() const override { return X86::R11; }

  unsigned getShortBranchOpcode(unsigned Opcode) const override {
    switch (Opcode) {
    default:
      return Opcode;
    case X86::JMP_2:
      return X86::JMP_1;
    case X86::JMP_4:
      return X86::JMP_1;
    case X86::JCC_2:
      return X86::JCC_1;
    case X86::JCC_4:
      return X86::JCC_1;
    }
  }

  MCPhysReg getIntArgRegister(unsigned ArgNo) const override {
    // FIXME: this should depend on the calling convention.
    switch (ArgNo) {
    case 0:   return X86::RDI;
    case 1:   return X86::RSI;
    case 2:   return X86::RDX;
    case 3:   return X86::RCX;
    case 4:   return X86::R8;
    case 5:   return X86::R9;
    default:  return getNoRegister();
    }
  }

  void createPause(MCInst &Inst) const override {
    Inst.clear();
    Inst.setOpcode(X86::PAUSE);
  }

  void createLfence(MCInst &Inst) const override {
    Inst.clear();
    Inst.setOpcode(X86::LFENCE);
  }

  void createDirectCall(MCInst &Inst, const MCSymbol *Target, MCContext *Ctx,
                        bool IsTailCall) override {
    Inst.clear();
    Inst.setOpcode(IsTailCall ? X86::JMP_4 : X86::CALL64pcrel32);
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx)));
    if (IsTailCall)
      setTailCall(Inst);
  }

  void createShortJmp(InstructionListType &Seq, const MCSymbol *Target,
                      MCContext *Ctx, bool IsTailCall) override {
    Seq.clear();
    MCInst Inst;
    Inst.setOpcode(X86::JMP_1);
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx)));
    if (IsTailCall)
      setTailCall(Inst);
    Seq.emplace_back(Inst);
  }

  bool isConditionalMove(const MCInst &Inst) const override {
    unsigned OpCode = Inst.getOpcode();
    return (OpCode == X86::CMOV16rr || OpCode == X86::CMOV32rr ||
            OpCode == X86::CMOV64rr);
  }

  bool isBranchOnMem(const MCInst &Inst) const override {
    unsigned OpCode = Inst.getOpcode();
    if (OpCode == X86::CALL64m || (OpCode == X86::JMP32m && isTailCall(Inst)) ||
        OpCode == X86::JMP64m)
      return true;

    return false;
  }

  bool isBranchOnReg(const MCInst &Inst) const override {
    unsigned OpCode = Inst.getOpcode();
    if (OpCode == X86::CALL64r || (OpCode == X86::JMP32r && isTailCall(Inst)) ||
        OpCode == X86::JMP64r)
      return true;

    return false;
  }

  void createPushRegister(MCInst &Inst, MCPhysReg Reg,
                          unsigned Size) const override {
    Inst.clear();
    unsigned NewOpcode = 0;
    if (Reg == X86::EFLAGS) {
      switch (Size) {
      case 2: NewOpcode = X86::PUSHF16;  break;
      case 4: NewOpcode = X86::PUSHF32;  break;
      case 8: NewOpcode = X86::PUSHF64;  break;
      default:
        llvm_unreachable("Unexpected size");
      }
      Inst.setOpcode(NewOpcode);
      return;
    }
    switch (Size) {
    case 2: NewOpcode = X86::PUSH16r;  break;
    case 4: NewOpcode = X86::PUSH32r;  break;
    case 8: NewOpcode = X86::PUSH64r;  break;
    default:
      llvm_unreachable("Unexpected size");
    }
    Inst.setOpcode(NewOpcode);
    Inst.addOperand(MCOperand::createReg(Reg));
  }

  void createPopRegister(MCInst &Inst, MCPhysReg Reg,
                         unsigned Size) const override {
    Inst.clear();
    unsigned NewOpcode = 0;
    if (Reg == X86::EFLAGS) {
      switch (Size) {
      case 2: NewOpcode = X86::POPF16;  break;
      case 4: NewOpcode = X86::POPF32;  break;
      case 8: NewOpcode = X86::POPF64;  break;
      default:
        llvm_unreachable("Unexpected size");
      }
      Inst.setOpcode(NewOpcode);
      return;
    }
    switch (Size) {
    case 2: NewOpcode = X86::POP16r;  break;
    case 4: NewOpcode = X86::POP32r;  break;
    case 8: NewOpcode = X86::POP64r;  break;
    default:
      llvm_unreachable("Unexpected size");
    }
    Inst.setOpcode(NewOpcode);
    Inst.addOperand(MCOperand::createReg(Reg));
  }

  void createPushFlags(MCInst &Inst, unsigned Size) const override {
    return createPushRegister(Inst, X86::EFLAGS, Size);
  }

  void createPopFlags(MCInst &Inst, unsigned Size) const override {
    return createPopRegister(Inst, X86::EFLAGS, Size);
  }

  void createAddRegImm(MCInst &Inst, MCPhysReg Reg, int64_t Value,
                       unsigned Size) const {
    unsigned int Opcode;
    switch (Size) {
    case 1: Opcode = X86::ADD8ri; break;
    case 2: Opcode = X86::ADD16ri; break;
    case 4: Opcode = X86::ADD32ri; break;
    default:
      llvm_unreachable("Unexpected size");
    }
    Inst.setOpcode(Opcode);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(Reg));
    Inst.addOperand(MCOperand::createReg(Reg));
    Inst.addOperand(MCOperand::createImm(Value));
  }

  void createClearRegWithNoEFlagsUpdate(MCInst &Inst, MCPhysReg Reg,
                                        unsigned Size) const {
    unsigned int Opcode;
    switch (Size) {
    case 1: Opcode = X86::MOV8ri; break;
    case 2: Opcode = X86::MOV16ri; break;
    case 4: Opcode = X86::MOV32ri; break;
    // Writing to a 32-bit register always zeros the upper 32 bits of the
    // full-width register
    case 8:
      Opcode = X86::MOV32ri;
      Reg = getAliasSized(Reg, 4);
      break;
    default:
      llvm_unreachable("Unexpected size");
    }
    Inst.setOpcode(Opcode);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(Reg));
    Inst.addOperand(MCOperand::createImm(0));
  }

  void createX86SaveOVFlagToRegister(MCInst &Inst, MCPhysReg Reg) const {
    Inst.setOpcode(X86::SETCCr);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(Reg));
    Inst.addOperand(MCOperand::createImm(X86::COND_O));
  }

  void createX86Lahf(MCInst &Inst) const {
    Inst.setOpcode(X86::LAHF);
    Inst.clear();
  }

  void createX86Sahf(MCInst &Inst) const {
    Inst.setOpcode(X86::SAHF);
    Inst.clear();
  }

  InstructionListType
  createInstrIncMemory(const MCSymbol *Target, MCContext *Ctx, bool IsLeaf,
                       unsigned CodePointerSize) const override {
    InstructionListType Instrs(IsLeaf ? 13 : 11);
    unsigned int I = 0;

    // Don't clobber application red zone (ABI dependent)
    if (IsLeaf)
      createStackPointerIncrement(Instrs[I++], 128,
                                  /*NoFlagsClobber=*/true);

    // Performance improvements based on the optimization discussed at
    // https://reviews.llvm.org/D6629
    // LAHF/SAHF are used instead of PUSHF/POPF
    // PUSHF
    createPushRegister(Instrs[I++], X86::RAX, 8);
    createClearRegWithNoEFlagsUpdate(Instrs[I++], X86::RAX, 8);
    createX86Lahf(Instrs[I++]);
    createPushRegister(Instrs[I++], X86::RAX, 8);
    createClearRegWithNoEFlagsUpdate(Instrs[I++], X86::RAX, 8);
    createX86SaveOVFlagToRegister(Instrs[I++], X86::AL);
    // LOCK INC
    InstructionListType IncMem = createIncMemory(Target, Ctx);
    assert(IncMem.size() == 1 && "Invalid IncMem size");
    std::copy(IncMem.begin(), IncMem.end(), Instrs.begin() + I);
    I += IncMem.size();
    // POPF
    createAddRegImm(Instrs[I++], X86::AL, 127, 1);
    createPopRegister(Instrs[I++], X86::RAX, 8);
    createX86Sahf(Instrs[I++]);
    createPopRegister(Instrs[I++], X86::RAX, 8);

    if (IsLeaf)
      createStackPointerDecrement(Instrs[I], 128,
                                  /*NoFlagsClobber=*/true);
    return Instrs;
  }

  void createSwap(MCInst &Inst, MCPhysReg Source, MCPhysReg MemBaseReg,
                  int64_t Disp) const {
    Inst.setOpcode(X86::XCHG64rm);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(Source));
    Inst.addOperand(MCOperand::createReg(Source));
    Inst.addOperand(MCOperand::createReg(MemBaseReg));      // BaseReg
    Inst.addOperand(MCOperand::createImm(1));               // ScaleAmt
    Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // IndexReg
    Inst.addOperand(MCOperand::createImm(Disp));            // Displacement
    Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // AddrSegmentReg
  }

  void createIndirectBranch(MCInst &Inst, MCPhysReg MemBaseReg,
                            int64_t Disp) const {
    Inst.setOpcode(X86::JMP64m);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(MemBaseReg));      // BaseReg
    Inst.addOperand(MCOperand::createImm(1));               // ScaleAmt
    Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // IndexReg
    Inst.addOperand(MCOperand::createImm(Disp));            // Displacement
    Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // AddrSegmentReg
  }

  InstructionListType createInstrumentedIndirectCall(MCInst &&CallInst,
                                                     MCSymbol *HandlerFuncAddr,
                                                     int CallSiteID,
                                                     MCContext *Ctx) override {
    // Check if the target address expression used in the original indirect call
    // uses the stack pointer, which we are going to clobber.
    static BitVector SPAliases(getAliases(X86::RSP));
    bool UsesSP = any_of(useOperands(CallInst), [&](const MCOperand &Op) {
      return Op.isReg() && SPAliases[Op.getReg()];
    });

    InstructionListType Insts;
    MCPhysReg TempReg = getIntArgRegister(0);
    // Code sequence used to enter indirect call instrumentation helper:
    //   push %rdi
    //   add $8, %rsp       ;; $rsp may be used in target, so fix it to prev val
    //   movq target, %rdi  ;; via convertIndirectCallTargetToLoad
    //   sub $8, %rsp       ;; restore correct stack value
    //   push %rdi
    //   movq $CallSiteID, %rdi
    //   push %rdi
    //   callq/jmp HandlerFuncAddr
    Insts.emplace_back();
    createPushRegister(Insts.back(), TempReg, 8);
    if (UsesSP) { // Only adjust SP if we really need to
      Insts.emplace_back();
      createStackPointerDecrement(Insts.back(), 8, /*NoFlagsClobber=*/false);
    }
    Insts.emplace_back(CallInst);
    // Insts.back() and CallInst now share the same annotation instruction.
    // Strip it from Insts.back(), only preserving tail call annotation.
    stripAnnotations(Insts.back(), /*KeepTC=*/true);
    convertIndirectCallToLoad(Insts.back(), TempReg);
    if (UsesSP) {
      Insts.emplace_back();
      createStackPointerIncrement(Insts.back(), 8, /*NoFlagsClobber=*/false);
    }
    Insts.emplace_back();
    createPushRegister(Insts.back(), TempReg, 8);
    InstructionListType LoadImm = createLoadImmediate(TempReg, CallSiteID);
    Insts.insert(Insts.end(), LoadImm.begin(), LoadImm.end());
    Insts.emplace_back();
    createPushRegister(Insts.back(), TempReg, 8);

    MCInst &NewCallInst = Insts.emplace_back();
    createDirectCall(NewCallInst, HandlerFuncAddr, Ctx, isTailCall(CallInst));

    // Carry over metadata including tail call marker if present.
    stripAnnotations(NewCallInst);
    moveAnnotations(std::move(CallInst), NewCallInst);

    return Insts;
  }

  InstructionListType createInstrumentedIndCallHandlerExitBB() const override {
    const MCPhysReg TempReg = getIntArgRegister(0);
    // We just need to undo the sequence created for every ind call in
    // instrumentIndirectTarget(), which can be accomplished minimally with:
    //   popfq
    //   pop %rdi
    //   add $16, %rsp
    //   xchg (%rsp), %rdi
    //   jmp *-8(%rsp)
    InstructionListType Insts(5);
    createPopFlags(Insts[0], 8);
    createPopRegister(Insts[1], TempReg, 8);
    createStackPointerDecrement(Insts[2], 16, /*NoFlagsClobber=*/false);
    createSwap(Insts[3], TempReg, X86::RSP, 0);
    createIndirectBranch(Insts[4], X86::RSP, -8);
    return Insts;
  }

  InstructionListType
  createInstrumentedIndTailCallHandlerExitBB() const override {
    const MCPhysReg TempReg = getIntArgRegister(0);
    // Same thing as above, but for tail calls
    //   popfq
    //   add $16, %rsp
    //   pop %rdi
    //   jmp *-16(%rsp)
    InstructionListType Insts(4);
    createPopFlags(Insts[0], 8);
    createStackPointerDecrement(Insts[1], 16, /*NoFlagsClobber=*/false);
    createPopRegister(Insts[2], TempReg, 8);
    createIndirectBranch(Insts[3], X86::RSP, -16);
    return Insts;
  }

  InstructionListType
  createInstrumentedIndCallHandlerEntryBB(const MCSymbol *InstrTrampoline,
                                          const MCSymbol *IndCallHandler,
                                          MCContext *Ctx) override {
    const MCPhysReg TempReg = getIntArgRegister(0);
    // Code sequence used to check whether InstrTampoline was initialized
    // and call it if so, returns via IndCallHandler.
    //   pushfq
    //   mov    InstrTrampoline,%rdi
    //   cmp    $0x0,%rdi
    //   je     IndCallHandler
    //   callq  *%rdi
    //   jmpq   IndCallHandler
    InstructionListType Insts;
    Insts.emplace_back();
    createPushFlags(Insts.back(), 8);
    Insts.emplace_back();
    createMove(Insts.back(), InstrTrampoline, TempReg, Ctx);
    InstructionListType cmpJmp = createCmpJE(TempReg, 0, IndCallHandler, Ctx);
    Insts.insert(Insts.end(), cmpJmp.begin(), cmpJmp.end());
    Insts.emplace_back();
    Insts.back().setOpcode(X86::CALL64r);
    Insts.back().addOperand(MCOperand::createReg(TempReg));
    Insts.emplace_back();
    createDirectCall(Insts.back(), IndCallHandler, Ctx, /*IsTailCall*/ true);
    return Insts;
  }

  InstructionListType createNumCountersGetter(MCContext *Ctx) const override {
    InstructionListType Insts(2);
    MCSymbol *NumLocs = Ctx->getOrCreateSymbol("__bolt_num_counters");
    createMove(Insts[0], NumLocs, X86::EAX, Ctx);
    createReturn(Insts[1]);
    return Insts;
  }

  InstructionListType
  createInstrLocationsGetter(MCContext *Ctx) const override {
    InstructionListType Insts(2);
    MCSymbol *Locs = Ctx->getOrCreateSymbol("__bolt_instr_locations");
    createLea(Insts[0], Locs, X86::EAX, Ctx);
    createReturn(Insts[1]);
    return Insts;
  }

  InstructionListType createInstrTablesGetter(MCContext *Ctx) const override {
    InstructionListType Insts(2);
    MCSymbol *Locs = Ctx->getOrCreateSymbol("__bolt_instr_tables");
    createLea(Insts[0], Locs, X86::EAX, Ctx);
    createReturn(Insts[1]);
    return Insts;
  }

  InstructionListType createInstrNumFuncsGetter(MCContext *Ctx) const override {
    InstructionListType Insts(2);
    MCSymbol *NumFuncs = Ctx->getOrCreateSymbol("__bolt_instr_num_funcs");
    createMove(Insts[0], NumFuncs, X86::EAX, Ctx);
    createReturn(Insts[1]);
    return Insts;
  }

  InstructionListType createSymbolTrampoline(const MCSymbol *TgtSym,
                                             MCContext *Ctx) override {
    InstructionListType Insts(1);
    createUncondBranch(Insts[0], TgtSym, Ctx);
    return Insts;
  }

  BlocksVectorTy indirectCallPromotion(
      const MCInst &CallInst,
      const std::vector<std::pair<MCSymbol *, uint64_t>> &Targets,
      const std::vector<std::pair<MCSymbol *, uint64_t>> &VtableSyms,
      const std::vector<MCInst *> &MethodFetchInsns,
      const bool MinimizeCodeSize, MCContext *Ctx) override {
    const bool IsTailCall = isTailCall(CallInst);
    const bool IsJumpTable = getJumpTable(CallInst) != 0;
    BlocksVectorTy Results;

    // Label for the current code block.
    MCSymbol *NextTarget = nullptr;

    // The join block which contains all the instructions following CallInst.
    // MergeBlock remains null if CallInst is a tail call.
    MCSymbol *MergeBlock = nullptr;

    unsigned FuncAddrReg = X86::R10;

    const bool LoadElim = !VtableSyms.empty();
    assert((!LoadElim || VtableSyms.size() == Targets.size()) &&
           "There must be a vtable entry for every method "
           "in the targets vector.");

    if (MinimizeCodeSize && !LoadElim) {
      std::set<unsigned> UsedRegs;

      for (unsigned int I = 0; I < MCPlus::getNumPrimeOperands(CallInst); ++I) {
        const MCOperand &Op = CallInst.getOperand(I);
        if (Op.isReg())
          UsedRegs.insert(Op.getReg());
      }

      if (UsedRegs.count(X86::R10) == 0)
        FuncAddrReg = X86::R10;
      else if (UsedRegs.count(X86::R11) == 0)
        FuncAddrReg = X86::R11;
      else
        return Results;
    }

    const auto jumpToMergeBlock = [&](InstructionListType &NewCall) {
      assert(MergeBlock);
      NewCall.push_back(CallInst);
      MCInst &Merge = NewCall.back();
      Merge.clear();
      createUncondBranch(Merge, MergeBlock, Ctx);
    };

    for (unsigned int i = 0; i < Targets.size(); ++i) {
      Results.emplace_back(NextTarget, InstructionListType());
      InstructionListType *NewCall = &Results.back().second;

      if (MinimizeCodeSize && !LoadElim) {
        // Load the call target into FuncAddrReg.
        NewCall->push_back(CallInst); // Copy CallInst in order to get SMLoc
        MCInst &Target = NewCall->back();
        Target.clear();
        Target.setOpcode(X86::MOV64ri32);
        Target.addOperand(MCOperand::createReg(FuncAddrReg));
        if (Targets[i].first) {
          // Is this OK?
          Target.addOperand(MCOperand::createExpr(MCSymbolRefExpr::create(
              Targets[i].first, MCSymbolRefExpr::VK_None, *Ctx)));
        } else {
          const uint64_t Addr = Targets[i].second;
          // Immediate address is out of sign extended 32 bit range.
          if (int64_t(Addr) != int64_t(int32_t(Addr)))
            return BlocksVectorTy();

          Target.addOperand(MCOperand::createImm(Addr));
        }

        // Compare current call target to a specific address.
        NewCall->push_back(CallInst);
        MCInst &Compare = NewCall->back();
        Compare.clear();
        if (isBranchOnReg(CallInst))
          Compare.setOpcode(X86::CMP64rr);
        else if (CallInst.getOpcode() == X86::CALL64pcrel32)
          Compare.setOpcode(X86::CMP64ri32);
        else
          Compare.setOpcode(X86::CMP64rm);

        Compare.addOperand(MCOperand::createReg(FuncAddrReg));

        // TODO: Would be preferable to only load this value once.
        for (unsigned i = 0;
             i < Info->get(CallInst.getOpcode()).getNumOperands(); ++i)
          if (!CallInst.getOperand(i).isInst())
            Compare.addOperand(CallInst.getOperand(i));
      } else {
        // Compare current call target to a specific address.
        NewCall->push_back(CallInst);
        MCInst &Compare = NewCall->back();
        Compare.clear();
        if (isBranchOnReg(CallInst))
          Compare.setOpcode(X86::CMP64ri32);
        else
          Compare.setOpcode(X86::CMP64mi32);

        // Original call address.
        for (unsigned i = 0;
             i < Info->get(CallInst.getOpcode()).getNumOperands(); ++i)
          if (!CallInst.getOperand(i).isInst())
            Compare.addOperand(CallInst.getOperand(i));

        // Target address.
        if (Targets[i].first || LoadElim) {
          const MCSymbol *Sym =
              LoadElim ? VtableSyms[i].first : Targets[i].first;
          const uint64_t Addend = LoadElim ? VtableSyms[i].second : 0;
          const MCExpr *Expr = MCSymbolRefExpr::create(Sym, *Ctx);
          if (Addend)
            Expr = MCBinaryExpr::createAdd(
                Expr, MCConstantExpr::create(Addend, *Ctx), *Ctx);
          Compare.addOperand(MCOperand::createExpr(Expr));
        } else {
          const uint64_t Addr = Targets[i].second;
          // Immediate address is out of sign extended 32 bit range.
          if (int64_t(Addr) != int64_t(int32_t(Addr)))
            return BlocksVectorTy();

          Compare.addOperand(MCOperand::createImm(Addr));
        }
      }

      // jump to next target compare.
      NextTarget =
          Ctx->createNamedTempSymbol(); // generate label for the next block
      NewCall->push_back(CallInst);

      if (IsJumpTable) {
        MCInst &Je = NewCall->back();

        // Jump to next compare if target addresses don't match.
        Je.clear();
        Je.setOpcode(X86::JCC_1);
        if (Targets[i].first)
          Je.addOperand(MCOperand::createExpr(MCSymbolRefExpr::create(
              Targets[i].first, MCSymbolRefExpr::VK_None, *Ctx)));
        else
          Je.addOperand(MCOperand::createImm(Targets[i].second));

        Je.addOperand(MCOperand::createImm(X86::COND_E));
        assert(!isInvoke(CallInst));
      } else {
        MCInst &Jne = NewCall->back();

        // Jump to next compare if target addresses don't match.
        Jne.clear();
        Jne.setOpcode(X86::JCC_1);
        Jne.addOperand(MCOperand::createExpr(MCSymbolRefExpr::create(
            NextTarget, MCSymbolRefExpr::VK_None, *Ctx)));
        Jne.addOperand(MCOperand::createImm(X86::COND_NE));

        // Call specific target directly.
        Results.emplace_back(Ctx->createNamedTempSymbol(),
                             InstructionListType());
        NewCall = &Results.back().second;
        NewCall->push_back(CallInst);
        MCInst &CallOrJmp = NewCall->back();

        CallOrJmp.clear();

        if (MinimizeCodeSize && !LoadElim) {
          CallOrJmp.setOpcode(IsTailCall ? X86::JMP32r : X86::CALL64r);
          CallOrJmp.addOperand(MCOperand::createReg(FuncAddrReg));
        } else {
          CallOrJmp.setOpcode(IsTailCall ? X86::JMP_4 : X86::CALL64pcrel32);

          if (Targets[i].first)
            CallOrJmp.addOperand(MCOperand::createExpr(MCSymbolRefExpr::create(
                Targets[i].first, MCSymbolRefExpr::VK_None, *Ctx)));
          else
            CallOrJmp.addOperand(MCOperand::createImm(Targets[i].second));
        }
        if (IsTailCall)
          setTailCall(CallOrJmp);

        if (CallOrJmp.getOpcode() == X86::CALL64r ||
            CallOrJmp.getOpcode() == X86::CALL64pcrel32) {
          if (std::optional<uint32_t> Offset = getOffset(CallInst))
            // Annotated as duplicated call
            setOffset(CallOrJmp, *Offset);
        }

        if (isInvoke(CallInst) && !isInvoke(CallOrJmp)) {
          // Copy over any EH or GNU args size information from the original
          // call.
          std::optional<MCPlus::MCLandingPad> EHInfo = getEHInfo(CallInst);
          if (EHInfo)
            addEHInfo(CallOrJmp, *EHInfo);
          int64_t GnuArgsSize = getGnuArgsSize(CallInst);
          if (GnuArgsSize >= 0)
            addGnuArgsSize(CallOrJmp, GnuArgsSize);
        }

        if (!IsTailCall) {
          // The fallthrough block for the most common target should be
          // the merge block.
          if (i == 0) {
            // Fallthrough to merge block.
            MergeBlock = Ctx->createNamedTempSymbol();
          } else {
            // Insert jump to the merge block if we are not doing a fallthrough.
            jumpToMergeBlock(*NewCall);
          }
        }
      }
    }

    // Cold call block.
    Results.emplace_back(NextTarget, InstructionListType());
    InstructionListType &NewCall = Results.back().second;
    for (const MCInst *Inst : MethodFetchInsns)
      if (Inst != &CallInst)
        NewCall.push_back(*Inst);
    NewCall.push_back(CallInst);

    // Jump to merge block from cold call block
    if (!IsTailCall && !IsJumpTable) {
      jumpToMergeBlock(NewCall);

      // Record merge block
      Results.emplace_back(MergeBlock, InstructionListType());
    }

    return Results;
  }

  BlocksVectorTy jumpTablePromotion(
      const MCInst &IJmpInst,
      const std::vector<std::pair<MCSymbol *, uint64_t>> &Targets,
      const std::vector<MCInst *> &TargetFetchInsns,
      MCContext *Ctx) const override {
    assert(getJumpTable(IJmpInst) != 0);
    uint16_t IndexReg = getAnnotationAs<uint16_t>(IJmpInst, "JTIndexReg");
    if (IndexReg == 0)
      return BlocksVectorTy();

    BlocksVectorTy Results;

    // Label for the current code block.
    MCSymbol *NextTarget = nullptr;

    for (unsigned int i = 0; i < Targets.size(); ++i) {
      Results.emplace_back(NextTarget, InstructionListType());
      InstructionListType *CurBB = &Results.back().second;

      // Compare current index to a specific index.
      CurBB->emplace_back(MCInst());
      MCInst &CompareInst = CurBB->back();
      CompareInst.setLoc(IJmpInst.getLoc());
      CompareInst.setOpcode(X86::CMP64ri32);
      CompareInst.addOperand(MCOperand::createReg(IndexReg));

      const uint64_t CaseIdx = Targets[i].second;
      // Immediate address is out of sign extended 32 bit range.
      if (int64_t(CaseIdx) != int64_t(int32_t(CaseIdx)))
        return BlocksVectorTy();

      CompareInst.addOperand(MCOperand::createImm(CaseIdx));
      shortenInstruction(CompareInst, *Ctx->getSubtargetInfo());

      // jump to next target compare.
      NextTarget =
          Ctx->createNamedTempSymbol(); // generate label for the next block
      CurBB->push_back(MCInst());

      MCInst &JEInst = CurBB->back();
      JEInst.setLoc(IJmpInst.getLoc());

      // Jump to target if indices match
      JEInst.setOpcode(X86::JCC_1);
      JEInst.addOperand(MCOperand::createExpr(MCSymbolRefExpr::create(
          Targets[i].first, MCSymbolRefExpr::VK_None, *Ctx)));
      JEInst.addOperand(MCOperand::createImm(X86::COND_E));
    }

    // Cold call block.
    Results.emplace_back(NextTarget, InstructionListType());
    InstructionListType &CurBB = Results.back().second;
    for (const MCInst *Inst : TargetFetchInsns)
      if (Inst != &IJmpInst)
        CurBB.push_back(*Inst);

    CurBB.push_back(IJmpInst);

    return Results;
  }

private:
  void createMove(MCInst &Inst, const MCSymbol *Src, unsigned Reg,
                  MCContext *Ctx) const {
    Inst.setOpcode(X86::MOV64rm);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(Reg));
    Inst.addOperand(MCOperand::createReg(X86::RIP));        // BaseReg
    Inst.addOperand(MCOperand::createImm(1));               // ScaleAmt
    Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // IndexReg
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(Src, MCSymbolRefExpr::VK_None,
                                *Ctx)));                    // Displacement
    Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // AddrSegmentReg
  }

  void createLea(MCInst &Inst, const MCSymbol *Src, unsigned Reg,
                 MCContext *Ctx) const {
    Inst.setOpcode(X86::LEA64r);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(Reg));
    Inst.addOperand(MCOperand::createReg(X86::RIP));        // BaseReg
    Inst.addOperand(MCOperand::createImm(1));               // ScaleAmt
    Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // IndexReg
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(Src, MCSymbolRefExpr::VK_None,
                                *Ctx)));                    // Displacement
    Inst.addOperand(MCOperand::createReg(X86::NoRegister)); // AddrSegmentReg
  }
};

} // namespace

namespace llvm {
namespace bolt {

MCPlusBuilder *createX86MCPlusBuilder(const MCInstrAnalysis *Analysis,
                                      const MCInstrInfo *Info,
                                      const MCRegisterInfo *RegInfo,
                                      const MCSubtargetInfo *STI) {
  return new X86MCPlusBuilder(Analysis, Info, RegInfo, STI);
}

} // namespace bolt
} // namespace llvm
