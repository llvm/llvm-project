//===- bolt/Core/MCPlusBuilder.h - Interface for MCPlus ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of MCPlusBuilder class, which provides
// means to create/analyze/modify instructions at MCPlus level.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_MCPLUSBUILDER_H
#define BOLT_CORE_MCPLUSBUILDER_H

#include "bolt/Core/MCPlus.h"
#include "bolt/Core/Relocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include <cassert>
#include <cstdint>
#include <map>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

namespace llvm {
class MCContext;
class MCFixup;
class MCRegisterInfo;
class MCSymbol;
class raw_ostream;

namespace bolt {

/// Different types of indirect branches encountered during disassembly.
enum class IndirectBranchType : char {
  UNKNOWN = 0,             /// Unable to determine type.
  POSSIBLE_TAIL_CALL,      /// Possibly a tail call.
  POSSIBLE_JUMP_TABLE,     /// Possibly a switch/jump table.
  POSSIBLE_PIC_JUMP_TABLE, /// Possibly a jump table for PIC.
  POSSIBLE_GOTO,           /// Possibly a gcc's computed goto.
  POSSIBLE_FIXED_BRANCH,   /// Possibly an indirect branch to a fixed location.
};

class MCPlusBuilder {
public:
  using AllocatorIdTy = uint16_t;

private:
  /// A struct that represents a single annotation allocator
  struct AnnotationAllocator {
    SpecificBumpPtrAllocator<MCInst> MCInstAllocator;
    BumpPtrAllocator ValueAllocator;
    std::unordered_set<MCPlus::MCAnnotation *> AnnotationPool;
  };

  /// A set of annotation allocators
  std::unordered_map<AllocatorIdTy, AnnotationAllocator> AnnotationAllocators;

  /// A variable that is used to generate unique ids for annotation allocators
  AllocatorIdTy MaxAllocatorId = 0;

  /// We encode Index and Value into a 64-bit immediate operand value.
  static int64_t encodeAnnotationImm(unsigned Index, int64_t Value) {
    assert(Index < 256 && "annotation index max value exceeded");
    assert((Value == (Value << 8) >> 8) && "annotation value out of range");

    Value &= 0xff'ffff'ffff'ffff;
    Value |= (int64_t)Index << 56;

    return Value;
  }

  /// Extract annotation index from immediate operand value.
  static unsigned extractAnnotationIndex(int64_t ImmValue) {
    return ImmValue >> 56;
  }

  /// Extract annotation value from immediate operand value.
  static int64_t extractAnnotationValue(int64_t ImmValue) {
    ImmValue &= 0xff'ffff'ffff'ffff;
    return (ImmValue << 8) >> 8;
  }

  MCInst *getAnnotationInst(const MCInst &Inst) const {
    if (Inst.getNumOperands() == 0)
      return nullptr;

    const MCOperand &LastOp = Inst.getOperand(Inst.getNumOperands() - 1);
    if (!LastOp.isInst())
      return nullptr;

    MCInst *AnnotationInst = const_cast<MCInst *>(LastOp.getInst());
    assert(AnnotationInst->getOpcode() == TargetOpcode::ANNOTATION_LABEL);

    return AnnotationInst;
  }

  void setAnnotationOpValue(MCInst &Inst, unsigned Index, int64_t Value,
                            AllocatorIdTy AllocatorId = 0) {
    MCInst *AnnotationInst = getAnnotationInst(Inst);
    if (!AnnotationInst) {
      AnnotationAllocator &Allocator = getAnnotationAllocator(AllocatorId);
      AnnotationInst = new (Allocator.MCInstAllocator.Allocate()) MCInst();
      AnnotationInst->setOpcode(TargetOpcode::ANNOTATION_LABEL);
      Inst.addOperand(MCOperand::createInst(AnnotationInst));
    }

    const int64_t AnnotationValue = encodeAnnotationImm(Index, Value);
    for (int I = AnnotationInst->getNumOperands() - 1; I >= 0; --I) {
      int64_t ImmValue = AnnotationInst->getOperand(I).getImm();
      if (extractAnnotationIndex(ImmValue) == Index) {
        AnnotationInst->getOperand(I).setImm(AnnotationValue);
        return;
      }
    }

    AnnotationInst->addOperand(MCOperand::createImm(AnnotationValue));
  }

  Optional<int64_t> getAnnotationOpValue(const MCInst &Inst,
                                         unsigned Index) const {
    const MCInst *AnnotationInst = getAnnotationInst(Inst);
    if (!AnnotationInst)
      return NoneType();

    for (int I = AnnotationInst->getNumOperands() - 1; I >= 0; --I) {
      int64_t ImmValue = AnnotationInst->getOperand(I).getImm();
      if (extractAnnotationIndex(ImmValue) == Index) {
        return extractAnnotationValue(ImmValue);
      }
    }

    return NoneType();
  }

protected:
  const MCInstrAnalysis *Analysis;
  const MCInstrInfo *Info;
  const MCRegisterInfo *RegInfo;

  /// Map annotation name into an annotation index.
  StringMap<uint64_t> AnnotationNameIndexMap;

  /// Names of non-standard annotations.
  SmallVector<std::string, 8> AnnotationNames;

  /// Allocate the TailCall annotation value. Clients of the target-specific
  /// MCPlusBuilder classes must use convert/lower/create* interfaces instead.
  void setTailCall(MCInst &Inst);

public:
  class InstructionIterator
      : public std::iterator<std::bidirectional_iterator_tag, MCInst> {
  public:
    class Impl {
    public:
      virtual Impl *Copy() const = 0;
      virtual void Next() = 0;
      virtual void Prev() = 0;
      virtual MCInst &Deref() = 0;
      virtual bool Compare(const Impl &Other) const = 0;
      virtual ~Impl() {}
    };

    template <typename T> class SeqImpl : public Impl {
    public:
      virtual Impl *Copy() const override { return new SeqImpl(Itr); }
      virtual void Next() override { ++Itr; }
      virtual void Prev() override { --Itr; }
      virtual MCInst &Deref() override { return const_cast<MCInst &>(*Itr); }
      virtual bool Compare(const Impl &Other) const override {
        // assumes that Other is same underlying type
        return Itr == static_cast<const SeqImpl<T> &>(Other).Itr;
      }
      explicit SeqImpl(T &&Itr) : Itr(std::move(Itr)) {}
      explicit SeqImpl(const T &Itr) : Itr(Itr) {}

    private:
      T Itr;
    };

    template <typename T> class MapImpl : public Impl {
    public:
      virtual Impl *Copy() const override { return new MapImpl(Itr); }
      virtual void Next() override { ++Itr; }
      virtual void Prev() override { --Itr; }
      virtual MCInst &Deref() override {
        return const_cast<MCInst &>(Itr->second);
      }
      virtual bool Compare(const Impl &Other) const override {
        // assumes that Other is same underlying type
        return Itr == static_cast<const MapImpl<T> &>(Other).Itr;
      }
      explicit MapImpl(T &&Itr) : Itr(std::move(Itr)) {}
      explicit MapImpl(const T &Itr) : Itr(Itr) {}

    private:
      T Itr;
    };

    InstructionIterator &operator++() {
      Itr->Next();
      return *this;
    }
    InstructionIterator &operator--() {
      Itr->Prev();
      return *this;
    }
    InstructionIterator operator++(int) {
      std::unique_ptr<Impl> Tmp(Itr->Copy());
      Itr->Next();
      return InstructionIterator(std::move(Tmp));
    }
    InstructionIterator operator--(int) {
      std::unique_ptr<Impl> Tmp(Itr->Copy());
      Itr->Prev();
      return InstructionIterator(std::move(Tmp));
    }
    bool operator==(const InstructionIterator &Other) const {
      return Itr->Compare(*Other.Itr);
    }
    bool operator!=(const InstructionIterator &Other) const {
      return !Itr->Compare(*Other.Itr);
    }
    MCInst &operator*() { return Itr->Deref(); }
    MCInst *operator->() { return &Itr->Deref(); }

    InstructionIterator &operator=(InstructionIterator &&Other) {
      Itr = std::move(Other.Itr);
      return *this;
    }
    InstructionIterator &operator=(const InstructionIterator &Other) {
      if (this != &Other)
        Itr.reset(Other.Itr->Copy());
      return *this;
    }
    InstructionIterator() {}
    InstructionIterator(const InstructionIterator &Other)
        : Itr(Other.Itr->Copy()) {}
    InstructionIterator(InstructionIterator &&Other)
        : Itr(std::move(Other.Itr)) {}
    explicit InstructionIterator(std::unique_ptr<Impl> Itr)
        : Itr(std::move(Itr)) {}

    InstructionIterator(InstructionListType::iterator Itr)
        : Itr(new SeqImpl<InstructionListType::iterator>(Itr)) {}

    template <typename T>
    InstructionIterator(T *Itr) : Itr(new SeqImpl<T *>(Itr)) {}

    InstructionIterator(ArrayRef<MCInst>::iterator Itr)
        : Itr(new SeqImpl<ArrayRef<MCInst>::iterator>(Itr)) {}

    InstructionIterator(MutableArrayRef<MCInst>::iterator Itr)
        : Itr(new SeqImpl<MutableArrayRef<MCInst>::iterator>(Itr)) {}

    // TODO: it would be nice to templatize this on the key type.
    InstructionIterator(std::map<uint32_t, MCInst>::iterator Itr)
        : Itr(new MapImpl<std::map<uint32_t, MCInst>::iterator>(Itr)) {}

  private:
    std::unique_ptr<Impl> Itr;
  };

public:
  MCPlusBuilder(const MCInstrAnalysis *Analysis, const MCInstrInfo *Info,
                const MCRegisterInfo *RegInfo)
      : Analysis(Analysis), Info(Info), RegInfo(RegInfo) {
    // Initialize the default annotation allocator with id 0
    AnnotationAllocators.emplace(0, AnnotationAllocator());
    MaxAllocatorId++;
  }

  /// Initialize a new annotation allocator and return its id
  AllocatorIdTy initializeNewAnnotationAllocator() {
    AnnotationAllocators.emplace(MaxAllocatorId, AnnotationAllocator());
    return MaxAllocatorId++;
  }

  /// Return the annotation allocator of a given id
  AnnotationAllocator &getAnnotationAllocator(AllocatorIdTy AllocatorId) {
    assert(AnnotationAllocators.count(AllocatorId) &&
           "allocator not initialized");
    return AnnotationAllocators.find(AllocatorId)->second;
  }

  // Check if an annotation allocator with the given id exists
  bool checkAllocatorExists(AllocatorIdTy AllocatorId) {
    return AnnotationAllocators.count(AllocatorId);
  }

  /// Free the values allocator within the annotation allocator
  void freeValuesAllocator(AllocatorIdTy AllocatorId) {
    AnnotationAllocator &Allocator = getAnnotationAllocator(AllocatorId);
    for (MCPlus::MCAnnotation *Annotation : Allocator.AnnotationPool)
      Annotation->~MCAnnotation();

    Allocator.AnnotationPool.clear();
    Allocator.ValueAllocator.Reset();
  }

  virtual ~MCPlusBuilder() { freeAnnotations(); }

  /// Free all memory allocated for annotations
  void freeAnnotations() {
    for (auto &Element : AnnotationAllocators) {
      AnnotationAllocator &Allocator = Element.second;
      for (MCPlus::MCAnnotation *Annotation : Allocator.AnnotationPool)
        Annotation->~MCAnnotation();

      Allocator.AnnotationPool.clear();
      Allocator.ValueAllocator.Reset();
      Allocator.MCInstAllocator.DestroyAll();
    }
  }

  using CompFuncTy = std::function<bool(const MCSymbol *, const MCSymbol *)>;

  bool equals(const MCInst &A, const MCInst &B, CompFuncTy Comp) const;

  bool equals(const MCOperand &A, const MCOperand &B, CompFuncTy Comp) const;

  bool equals(const MCExpr &A, const MCExpr &B, CompFuncTy Comp) const;

  virtual bool equals(const MCTargetExpr &A, const MCTargetExpr &B,
                      CompFuncTy Comp) const;

  virtual bool isBranch(const MCInst &Inst) const {
    return Analysis->isBranch(Inst);
  }

  virtual bool isConditionalBranch(const MCInst &Inst) const {
    return Analysis->isConditionalBranch(Inst);
  }

  /// Returns true if Inst is a condtional move instruction
  virtual bool isConditionalMove(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isUnconditionalBranch(const MCInst &Inst) const {
    return Analysis->isUnconditionalBranch(Inst);
  }

  virtual bool isIndirectBranch(const MCInst &Inst) const {
    return Analysis->isIndirectBranch(Inst);
  }

  /// Returns true if the instruction is memory indirect call or jump
  virtual bool isBranchOnMem(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Returns true if the instruction is register indirect call or jump
  virtual bool isBranchOnReg(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Check whether we support inverting this branch
  virtual bool isUnsupportedBranch(unsigned Opcode) const { return false; }

  /// Return true of the instruction is of pseudo kind.
  bool isPseudo(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isPseudo();
  }

  /// Creates x86 pause instruction.
  virtual void createPause(MCInst &Inst) const {
    llvm_unreachable("not implemented");
  }

  virtual void createLfence(MCInst &Inst) const {
    llvm_unreachable("not implemented");
  }

  virtual void createPushRegister(MCInst &Inst, MCPhysReg Reg,
                                  unsigned Size) const {
    llvm_unreachable("not implemented");
  }

  virtual void createPopRegister(MCInst &Inst, MCPhysReg Reg,
                                 unsigned Size) const {
    llvm_unreachable("not implemented");
  }

  virtual void createPushFlags(MCInst &Inst, unsigned Size) const {
    llvm_unreachable("not implemented");
  }

  virtual void createPopFlags(MCInst &Inst, unsigned Size) const {
    llvm_unreachable("not implemented");
  }

  virtual bool createDirectCall(MCInst &Inst, const MCSymbol *Target,
                                MCContext *Ctx, bool IsTailCall) {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual MCPhysReg getX86R11() const { llvm_unreachable("not implemented"); }

  /// Create increment contents of target by 1 for Instrumentation
  virtual void createInstrIncMemory(InstructionListType &Instrs,
                                    const MCSymbol *Target, MCContext *Ctx,
                                    bool IsLeaf) const {
    llvm_unreachable("not implemented");
  }

  /// Return a register number that is guaranteed to not match with
  /// any real register on the underlying architecture.
  virtual MCPhysReg getNoRegister() const {
    llvm_unreachable("not implemented");
  }

  /// Return a register corresponding to a function integer argument \p ArgNo
  /// if the argument is passed in a register. Or return the result of
  /// getNoRegister() otherwise. The enumeration starts at 0.
  ///
  /// Note: this should depend on a used calling convention.
  virtual MCPhysReg getIntArgRegister(unsigned ArgNo) const {
    llvm_unreachable("not implemented");
  }

  virtual bool isIndirectCall(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isCall(const MCInst &Inst) const {
    return Analysis->isCall(Inst) || isTailCall(Inst);
  }

  virtual bool isReturn(const MCInst &Inst) const {
    return Analysis->isReturn(Inst);
  }

  virtual bool isTerminator(const MCInst &Inst) const {
    return Analysis->isTerminator(Inst);
  }

  virtual bool isNoop(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isBreakpoint(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isPrefix(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isRep(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool deleteREPPrefix(MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isEHLabel(const MCInst &Inst) const {
    return Inst.getOpcode() == TargetOpcode::EH_LABEL;
  }

  virtual bool isPop(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return true if the instruction is used to terminate an indirect branch.
  virtual bool isTerminateBranch(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return the width, in bytes, of the memory access performed by \p Inst, if
  /// this is a pop instruction. Return zero otherwise.
  virtual int getPopSize(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return 0;
  }

  virtual bool isPush(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return the width, in bytes, of the memory access performed by \p Inst, if
  /// this is a push instruction. Return zero otherwise.
  virtual int getPushSize(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return 0;
  }

  virtual bool isADD64rr(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isSUB(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isLEA64r(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isMOVSX64rm32(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isLeave(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isADRP(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isADR(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual void getADRReg(const MCInst &Inst, MCPhysReg &RegName) const {
    llvm_unreachable("not implemented");
  }

  virtual bool isMoveMem2Reg(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isLoad(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isStore(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isCleanRegXOR(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isPacked(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// If non-zero, this is used to fill the executable space with instructions
  /// that will trap. Defaults to 0.
  virtual unsigned getTrapFillValue() const { return 0; }

  /// Interface and basic functionality of a MCInstMatcher. The idea is to make
  /// it easy to match one or more MCInsts against a tree-like pattern and
  /// extract the fragment operands. Example:
  ///
  ///   auto IndJmpMatcher =
  ///       matchIndJmp(matchAdd(matchAnyOperand(), matchAnyOperand()));
  ///   if (!IndJmpMatcher->match(...))
  ///     return false;
  ///
  /// This matches an indirect jump whose target register is defined by an
  /// add to form the target address. Matchers should also allow extraction
  /// of operands, for example:
  ///
  ///   uint64_t Scale;
  ///   auto IndJmpMatcher = BC.MIA->matchIndJmp(
  ///       BC.MIA->matchAnyOperand(), BC.MIA->matchImm(Scale),
  ///       BC.MIA->matchReg(), BC.MIA->matchAnyOperand());
  ///   if (!IndJmpMatcher->match(...))
  ///     return false;
  ///
  /// Here we are interesting in extracting the scale immediate in an indirect
  /// jump fragment.
  ///
  struct MCInstMatcher {
    MutableArrayRef<MCInst> InstrWindow;
    MutableArrayRef<MCInst>::iterator CurInst;
    virtual ~MCInstMatcher() {}

    /// Returns true if the pattern is matched. Needs MCRegisterInfo and
    /// MCInstrAnalysis for analysis. InstrWindow contains an array
    /// where the last instruction is always the instruction to start matching
    /// against a fragment, potentially matching more instructions before it.
    /// If OpNum is greater than 0, we will not match against the last
    /// instruction itself but against an operand of the last instruction given
    /// by the index OpNum. If this operand is a register, we will immediately
    /// look for a previous instruction defining this register and match against
    /// it instead. This parent member function contains common bookkeeping
    /// required to implement this behavior.
    virtual bool match(const MCRegisterInfo &MRI, MCPlusBuilder &MIA,
                       MutableArrayRef<MCInst> InInstrWindow, int OpNum) {
      InstrWindow = InInstrWindow;
      CurInst = InstrWindow.end();

      if (!next())
        return false;

      if (OpNum < 0)
        return true;

      if (static_cast<unsigned int>(OpNum) >=
          MCPlus::getNumPrimeOperands(*CurInst))
        return false;

      const MCOperand &Op = CurInst->getOperand(OpNum);
      if (!Op.isReg())
        return true;

      MCPhysReg Reg = Op.getReg();
      while (next()) {
        const MCInstrDesc &InstrDesc = MIA.Info->get(CurInst->getOpcode());
        if (InstrDesc.hasDefOfPhysReg(*CurInst, Reg, MRI)) {
          InstrWindow = InstrWindow.slice(0, CurInst - InstrWindow.begin() + 1);
          return true;
        }
      }
      return false;
    }

    /// If successfully matched, calling this function will add an annotation
    /// to all instructions that were matched. This is used to easily tag
    /// instructions for deletion and implement match-and-replace operations.
    virtual void annotate(MCPlusBuilder &MIA, StringRef Annotation) {}

    /// Moves internal instruction iterator to the next instruction, walking
    /// backwards for pattern matching (effectively the previous instruction in
    /// regular order).
    bool next() {
      if (CurInst == InstrWindow.begin())
        return false;
      --CurInst;
      return true;
    }
  };

  /// Matches any operand
  struct AnyOperandMatcher : MCInstMatcher {
    MCOperand &Op;
    AnyOperandMatcher(MCOperand &Op) : Op(Op) {}

    bool match(const MCRegisterInfo &MRI, MCPlusBuilder &MIA,
               MutableArrayRef<MCInst> InInstrWindow, int OpNum) override {
      auto I = InInstrWindow.end();
      if (I == InInstrWindow.begin())
        return false;
      --I;
      if (OpNum < 0 ||
          static_cast<unsigned int>(OpNum) >= MCPlus::getNumPrimeOperands(*I))
        return false;
      Op = I->getOperand(OpNum);
      return true;
    }
  };

  /// Matches operands that are immediates
  struct ImmMatcher : MCInstMatcher {
    uint64_t &Imm;
    ImmMatcher(uint64_t &Imm) : Imm(Imm) {}

    bool match(const MCRegisterInfo &MRI, MCPlusBuilder &MIA,
               MutableArrayRef<MCInst> InInstrWindow, int OpNum) override {
      if (!MCInstMatcher::match(MRI, MIA, InInstrWindow, OpNum))
        return false;

      if (OpNum < 0)
        return false;
      const MCOperand &Op = CurInst->getOperand(OpNum);
      if (!Op.isImm())
        return false;
      Imm = Op.getImm();
      return true;
    }
  };

  /// Matches operands that are MCSymbols
  struct SymbolMatcher : MCInstMatcher {
    const MCSymbol *&Sym;
    SymbolMatcher(const MCSymbol *&Sym) : Sym(Sym) {}

    bool match(const MCRegisterInfo &MRI, MCPlusBuilder &MIA,
               MutableArrayRef<MCInst> InInstrWindow, int OpNum) override {
      if (!MCInstMatcher::match(MRI, MIA, InInstrWindow, OpNum))
        return false;

      if (OpNum < 0)
        return false;
      Sym = MIA.getTargetSymbol(*CurInst, OpNum);
      return Sym != nullptr;
    }
  };

  /// Matches operands that are registers
  struct RegMatcher : MCInstMatcher {
    MCPhysReg &Reg;
    RegMatcher(MCPhysReg &Reg) : Reg(Reg) {}

    bool match(const MCRegisterInfo &MRI, MCPlusBuilder &MIA,
               MutableArrayRef<MCInst> InInstrWindow, int OpNum) override {
      auto I = InInstrWindow.end();
      if (I == InInstrWindow.begin())
        return false;
      --I;
      if (OpNum < 0 ||
          static_cast<unsigned int>(OpNum) >= MCPlus::getNumPrimeOperands(*I))
        return false;
      const MCOperand &Op = I->getOperand(OpNum);
      if (!Op.isReg())
        return false;
      Reg = Op.getReg();
      return true;
    }
  };

  std::unique_ptr<MCInstMatcher> matchAnyOperand(MCOperand &Op) const {
    return std::unique_ptr<MCInstMatcher>(new AnyOperandMatcher(Op));
  }

  std::unique_ptr<MCInstMatcher> matchAnyOperand() const {
    static MCOperand Unused;
    return std::unique_ptr<MCInstMatcher>(new AnyOperandMatcher(Unused));
  }

  std::unique_ptr<MCInstMatcher> matchReg(MCPhysReg &Reg) const {
    return std::unique_ptr<MCInstMatcher>(new RegMatcher(Reg));
  }

  std::unique_ptr<MCInstMatcher> matchReg() const {
    static MCPhysReg Unused;
    return std::unique_ptr<MCInstMatcher>(new RegMatcher(Unused));
  }

  std::unique_ptr<MCInstMatcher> matchImm(uint64_t &Imm) const {
    return std::unique_ptr<MCInstMatcher>(new ImmMatcher(Imm));
  }

  std::unique_ptr<MCInstMatcher> matchImm() const {
    static uint64_t Unused;
    return std::unique_ptr<MCInstMatcher>(new ImmMatcher(Unused));
  }

  std::unique_ptr<MCInstMatcher> matchSymbol(const MCSymbol *&Sym) const {
    return std::unique_ptr<MCInstMatcher>(new SymbolMatcher(Sym));
  }

  std::unique_ptr<MCInstMatcher> matchSymbol() const {
    static const MCSymbol *Unused;
    return std::unique_ptr<MCInstMatcher>(new SymbolMatcher(Unused));
  }

  virtual std::unique_ptr<MCInstMatcher>
  matchIndJmp(std::unique_ptr<MCInstMatcher> Target) const {
    llvm_unreachable("not implemented");
    return nullptr;
  }

  virtual std::unique_ptr<MCInstMatcher>
  matchIndJmp(std::unique_ptr<MCInstMatcher> Base,
              std::unique_ptr<MCInstMatcher> Scale,
              std::unique_ptr<MCInstMatcher> Index,
              std::unique_ptr<MCInstMatcher> Offset) const {
    llvm_unreachable("not implemented");
    return nullptr;
  }

  virtual std::unique_ptr<MCInstMatcher>
  matchAdd(std::unique_ptr<MCInstMatcher> A,
           std::unique_ptr<MCInstMatcher> B) const {
    llvm_unreachable("not implemented");
    return nullptr;
  }

  virtual std::unique_ptr<MCInstMatcher>
  matchLoadAddr(std::unique_ptr<MCInstMatcher> Target) const {
    llvm_unreachable("not implemented");
    return nullptr;
  }

  virtual std::unique_ptr<MCInstMatcher>
  matchLoad(std::unique_ptr<MCInstMatcher> Base,
            std::unique_ptr<MCInstMatcher> Scale,
            std::unique_ptr<MCInstMatcher> Index,
            std::unique_ptr<MCInstMatcher> Offset) const {
    llvm_unreachable("not implemented");
    return nullptr;
  }

  /// \brief Given a branch instruction try to get the address the branch
  /// targets. Return true on success, and the address in Target.
  virtual bool evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                              uint64_t &Target) const;

  /// Return true if one of the operands of the \p Inst instruction uses
  /// PC-relative addressing.
  /// Note that PC-relative branches do not fall into this category.
  virtual bool hasPCRelOperand(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return a number of the operand representing a memory.
  /// Return -1 if the instruction doesn't have an explicit memory field.
  virtual int getMemoryOperandNo(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return -1;
  }

  /// Return true if the instruction is encoded using EVEX (AVX-512).
  virtual bool hasEVEXEncoding(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return true if a pair of instructions represented by \p Insts
  /// could be fused into a single uop.
  virtual bool isMacroOpFusionPair(ArrayRef<MCInst> Insts) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Given an instruction with (compound) memory operand, evaluate and return
  /// the corresponding values. Note that the operand could be in any position,
  /// but there is an assumption there's only one compound memory operand.
  /// Return true upon success, return false if the instruction does not have
  /// a memory operand.
  ///
  /// Since a Displacement field could be either an immediate or an expression,
  /// the function sets either \p DispImm or \p DispExpr value.
  virtual bool
  evaluateX86MemoryOperand(const MCInst &Inst, unsigned *BaseRegNum,
                           int64_t *ScaleImm, unsigned *IndexRegNum,
                           int64_t *DispImm, unsigned *SegmentRegNum,
                           const MCExpr **DispExpr = nullptr) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Given an instruction with memory addressing attempt to statically compute
  /// the address being accessed. Return true on success, and the address in
  /// \p Target.
  ///
  /// For RIP-relative addressing the caller is required to pass instruction
  /// \p Address and \p Size.
  virtual bool evaluateMemOperandTarget(const MCInst &Inst, uint64_t &Target,
                                        uint64_t Address = 0,
                                        uint64_t Size = 0) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return operand iterator pointing to displacement in the compound memory
  /// operand if such exists. Return Inst.end() otherwise.
  virtual MCInst::iterator getMemOperandDisp(MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return Inst.end();
  }

  /// Analyze \p Inst and return true if this instruction accesses \p Size
  /// bytes of the stack frame at position \p StackOffset. \p IsLoad and
  /// \p IsStore are set accordingly. If both are set, it means it is a
  /// instruction that reads and updates the same memory location. \p Reg is set
  /// to the source register in case of a store or destination register in case
  /// of a load. If the store does not use a source register, \p SrcImm will
  /// contain the source immediate and \p IsStoreFromReg will be set to false.
  /// \p Simple is false if the instruction is not fully understood by
  /// companion functions "replaceMemOperandWithImm" or
  /// "replaceMemOperandWithReg".
  virtual bool isStackAccess(const MCInst &Inst, bool &IsLoad, bool &IsStore,
                             bool &IsStoreFromReg, MCPhysReg &Reg,
                             int32_t &SrcImm, uint16_t &StackPtrReg,
                             int64_t &StackOffset, uint8_t &Size,
                             bool &IsSimple, bool &IsIndexed) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Convert a stack accessing load/store instruction in \p Inst to a PUSH
  /// or POP saving/restoring the source/dest reg in \p Inst. The original
  /// stack offset in \p Inst is ignored.
  virtual void changeToPushOrPop(MCInst &Inst) const {
    llvm_unreachable("not implemented");
  }

  /// Identify stack adjustment instructions -- those that change the stack
  /// pointer by adding or subtracting an immediate.
  virtual bool isStackAdjustment(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Use \p Input1 or Input2 as the current value for the input register and
  /// put in \p Output the changes incurred by executing \p Inst. Return false
  /// if it was not possible to perform the evaluation.
  virtual bool evaluateSimple(const MCInst &Inst, int64_t &Output,
                              std::pair<MCPhysReg, int64_t> Input1,
                              std::pair<MCPhysReg, int64_t> Input2) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isRegToRegMove(const MCInst &Inst, MCPhysReg &From,
                              MCPhysReg &To) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual MCPhysReg getStackPointer() const {
    llvm_unreachable("not implemented");
    return 0;
  }

  virtual MCPhysReg getFramePointer() const {
    llvm_unreachable("not implemented");
    return 0;
  }

  virtual MCPhysReg getFlagsReg() const {
    llvm_unreachable("not implemented");
    return 0;
  }

  /// Return true if \p Inst is a instruction that copies either the frame
  /// pointer or the stack pointer to another general purpose register or
  /// writes it to a memory location.
  virtual bool escapesVariable(const MCInst &Inst, bool HasFramePointer) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Discard operand \p OpNum replacing it by a new MCOperand that is a
  /// MCExpr referencing \p Symbol + \p Addend.
  virtual bool setOperandToSymbolRef(MCInst &Inst, int OpNum,
                                     const MCSymbol *Symbol, int64_t Addend,
                                     MCContext *Ctx, uint64_t RelType) const;

  /// Replace an immediate operand in the instruction \p Inst with a reference
  /// of the passed \p Symbol plus \p Addend. If the instruction does not have
  /// an immediate operand or has more than one - then return false. Otherwise
  /// return true.
  virtual bool replaceImmWithSymbolRef(MCInst &Inst, const MCSymbol *Symbol,
                                       int64_t Addend, MCContext *Ctx,
                                       int64_t &Value, uint64_t RelType) const {
    llvm_unreachable("not implemented");
    return false;
  }

  // Replace Register in Inst with Imm. Returns true if successful
  virtual bool replaceRegWithImm(MCInst &Inst, unsigned Register,
                                 int64_t Imm) const {
    llvm_unreachable("not implemented");
    return false;
  }

  // Replace ToReplace in Inst with ReplaceWith. Returns true if successful
  virtual bool replaceRegWithReg(MCInst &Inst, unsigned ToReplace,
                                 unsigned ReplaceWith) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Add \p NewImm to the current immediate operand of \p Inst. If it is a
  /// memory accessing instruction, this immediate is the memory address
  /// displacement. Otherwise, the target operand is the first immediate
  /// operand found in \p Inst. Return false if no imm operand found.
  virtual bool addToImm(MCInst &Inst, int64_t &Amt, MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Replace the compound memory operand of Inst with an immediate operand.
  /// The value of the immediate operand is computed by reading the \p
  /// ConstantData array starting from \p offset and assuming little-endianess.
  /// Return true on success. The given instruction is modified in place.
  virtual bool replaceMemOperandWithImm(MCInst &Inst, StringRef ConstantData,
                                        uint64_t Offset) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Same as replaceMemOperandWithImm, but for registers.
  virtual bool replaceMemOperandWithReg(MCInst &Inst, MCPhysReg RegNum) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return true if a move instruction moves a register to itself.
  virtual bool isRedundantMove(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return true if the instruction is a tail call.
  bool isTailCall(const MCInst &Inst) const;

  /// Return true if the instruction is a call with an exception handling info.
  virtual bool isInvoke(const MCInst &Inst) const {
    return isCall(Inst) && getEHInfo(Inst);
  }

  /// Return true if \p Inst is an instruction that potentially traps when
  /// working with addresses not aligned to the size of the operand.
  virtual bool requiresAlignedAddress(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return handler and action info for invoke instruction if present.
  Optional<MCPlus::MCLandingPad> getEHInfo(const MCInst &Inst) const;

  // Add handler and action info for call instruction.
  void addEHInfo(MCInst &Inst, const MCPlus::MCLandingPad &LP);

  /// Return non-negative GNU_args_size associated with the instruction
  /// or -1 if there's no associated info.
  int64_t getGnuArgsSize(const MCInst &Inst) const;

  /// Add the value of GNU_args_size to Inst if it already has EH info.
  void addGnuArgsSize(MCInst &Inst, int64_t GnuArgsSize,
                      AllocatorIdTy AllocId = 0);

  /// Return jump table addressed by this instruction.
  uint64_t getJumpTable(const MCInst &Inst) const;

  /// Return index register for instruction that uses a jump table.
  uint16_t getJumpTableIndexReg(const MCInst &Inst) const;

  /// Set jump table addressed by this instruction.
  bool setJumpTable(MCInst &Inst, uint64_t Value, uint16_t IndexReg,
                    AllocatorIdTy AllocId = 0);

  /// Disassociate instruction with a jump table.
  bool unsetJumpTable(MCInst &Inst);

  /// Return destination of conditional tail call instruction if \p Inst is one.
  Optional<uint64_t> getConditionalTailCall(const MCInst &Inst) const;

  /// Mark the \p Instruction as a conditional tail call, and set its
  /// destination address if it is known. If \p Instruction was already marked,
  /// update its destination with \p Dest.
  bool setConditionalTailCall(MCInst &Inst, uint64_t Dest = 0);

  /// If \p Inst was marked as a conditional tail call convert it to a regular
  /// branch. Return true if the instruction was converted.
  bool unsetConditionalTailCall(MCInst &Inst);

  /// Return offset of \p Inst in the original function, if available.
  Optional<uint32_t> getOffset(const MCInst &Inst) const;

  /// Return the offset if the annotation is present, or \p Default otherwise.
  uint32_t getOffsetWithDefault(const MCInst &Inst, uint32_t Default) const;

  /// Set offset of \p Inst in the original function.
  bool setOffset(MCInst &Inst, uint32_t Offset, AllocatorIdTy AllocatorId = 0);

  /// Remove offset annotation.
  bool clearOffset(MCInst &Inst);

  /// Return MCSymbol that represents a target of this instruction at a given
  /// operand number \p OpNum. If there's no symbol associated with
  /// the operand - return nullptr.
  virtual const MCSymbol *getTargetSymbol(const MCInst &Inst,
                                          unsigned OpNum = 0) const {
    llvm_unreachable("not implemented");
    return nullptr;
  }

  /// Return MCSymbol extracted from a target expression
  virtual const MCSymbol *getTargetSymbol(const MCExpr *Expr) const {
    return &cast<const MCSymbolRefExpr>(Expr)->getSymbol();
  }

  /// Return addend that represents an offset from MCSymbol target
  /// of this instruction at a given operand number \p OpNum.
  /// If there's no symbol associated with  the operand - return 0
  virtual int64_t getTargetAddend(const MCInst &Inst,
                                  unsigned OpNum = 0) const {
    llvm_unreachable("not implemented");
    return 0;
  }

  /// Return MCSymbol addend extracted from a target expression
  virtual int64_t getTargetAddend(const MCExpr *Expr) const {
    llvm_unreachable("not implemented");
    return 0;
  }

  /// Return MCSymbol/offset extracted from a target expression
  virtual std::pair<const MCSymbol *, uint64_t>
  getTargetSymbolInfo(const MCExpr *Expr) const {
    if (auto *SymExpr = dyn_cast<MCSymbolRefExpr>(Expr)) {
      return std::make_pair(&SymExpr->getSymbol(), 0);
    } else if (auto *BinExpr = dyn_cast<MCBinaryExpr>(Expr)) {
      const auto *SymExpr = dyn_cast<MCSymbolRefExpr>(BinExpr->getLHS());
      const auto *ConstExpr = dyn_cast<MCConstantExpr>(BinExpr->getRHS());
      if (BinExpr->getOpcode() == MCBinaryExpr::Add && SymExpr && ConstExpr)
        return std::make_pair(&SymExpr->getSymbol(), ConstExpr->getValue());
    }
    return std::make_pair(nullptr, 0);
  }

  /// Replace displacement in compound memory operand with given \p Operand.
  virtual bool replaceMemOperandDisp(MCInst &Inst, MCOperand Operand) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return the MCExpr used for absolute references in this target
  virtual const MCExpr *getTargetExprFor(MCInst &Inst, const MCExpr *Expr,
                                         MCContext &Ctx,
                                         uint64_t RelType) const {
    return Expr;
  }

  /// Return a BitVector marking all sub or super registers of \p Reg, including
  /// itself.
  virtual const BitVector &getAliases(MCPhysReg Reg,
                                      bool OnlySmaller = false) const;

  /// Change \p Regs setting all registers used to pass parameters according
  /// to the host abi. Do nothing if not implemented.
  virtual BitVector getRegsUsedAsParams() const {
    llvm_unreachable("not implemented");
    return BitVector();
  }

  /// Change \p Regs setting all registers used as callee-saved according
  /// to the host abi. Do nothing if not implemented.
  virtual void getCalleeSavedRegs(BitVector &Regs) const {
    llvm_unreachable("not implemented");
  }

  /// Get the default def_in and live_out registers for the function
  /// Currently only used for the Stoke optimzation
  virtual void getDefaultDefIn(BitVector &Regs) const {
    llvm_unreachable("not implemented");
  }

  /// Similar to getDefaultDefIn
  virtual void getDefaultLiveOut(BitVector &Regs) const {
    llvm_unreachable("not implemented");
  }

  /// Change \p Regs with a bitmask with all general purpose regs
  virtual void getGPRegs(BitVector &Regs, bool IncludeAlias = true) const {
    llvm_unreachable("not implemented");
  }

  /// Change \p Regs with a bitmask with all general purpose regs that can be
  /// encoded without extra prefix bytes. For x86 only.
  virtual void getClassicGPRegs(BitVector &Regs) const {
    llvm_unreachable("not implemented");
  }

  /// Set of Registers used by the Rep instruction
  virtual void getRepRegs(BitVector &Regs) const {
    llvm_unreachable("not implemented");
  }

  /// Return the register width in bytes (1, 2, 4 or 8)
  virtual uint8_t getRegSize(MCPhysReg Reg) const;

  /// For aliased registers, return an alias of \p Reg that has the width of
  /// \p Size bytes
  virtual MCPhysReg getAliasSized(MCPhysReg Reg, uint8_t Size) const {
    llvm_unreachable("not implemented");
    return 0;
  }

  /// For X86, return whether this register is an upper 8-bit register, such as
  /// AH, BH, etc.
  virtual bool isUpper8BitReg(MCPhysReg Reg) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// For X86, return whether this instruction has special constraints that
  /// prevents it from encoding registers that require a REX prefix.
  virtual bool cannotUseREX(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Modifies the set \p Regs by adding registers \p Inst may rewrite. Caller
  /// is responsible for passing a valid BitVector with the size equivalent to
  /// the number of registers in the target.
  /// Since this function is called many times during clobber analysis, it
  /// expects the caller to manage BitVector creation to avoid extra overhead.
  virtual void getClobberedRegs(const MCInst &Inst, BitVector &Regs) const;

  /// Set of all registers touched by this instruction, including implicit uses
  /// and defs.
  virtual void getTouchedRegs(const MCInst &Inst, BitVector &Regs) const;

  /// Set of all registers being written to by this instruction -- includes
  /// aliases but only if they are strictly smaller than the actual reg
  virtual void getWrittenRegs(const MCInst &Inst, BitVector &Regs) const;

  /// Set of all registers being read by this instruction -- includes aliases
  /// but only if they are strictly smaller than the actual reg
  virtual void getUsedRegs(const MCInst &Inst, BitVector &Regs) const;

  /// Set of all src registers -- includes aliases but
  /// only if they are strictly smaller than the actual reg
  virtual void getSrcRegs(const MCInst &Inst, BitVector &Regs) const;

  /// Return true if this instruction defines the specified physical
  /// register either explicitly or implicitly.
  virtual bool hasDefOfPhysReg(const MCInst &MI, unsigned Reg) const;

  /// Return true if this instruction uses the specified physical
  /// register either explicitly or implicitly.
  virtual bool hasUseOfPhysReg(const MCInst &MI, unsigned Reg) const;

  /// Replace displacement in compound memory operand with given \p Label.
  bool replaceMemOperandDisp(MCInst &Inst, const MCSymbol *Label,
                             MCContext *Ctx) const {
    return replaceMemOperandDisp(
        Inst, MCOperand::createExpr(MCSymbolRefExpr::create(Label, *Ctx)));
  }

  /// Returns how many bits we have in this instruction to encode a PC-rel
  /// imm.
  virtual int getPCRelEncodingSize(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return 0;
  }

  /// Replace instruction opcode to be a tail call instead of jump.
  virtual bool convertJmpToTailCall(MCInst &Inst) {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Perform any additional actions to transform a (conditional) tail call
  /// into a (conditional) jump. Assume the target was already replaced with
  /// a local one, so the default is to do nothing more.
  virtual bool convertTailCallToJmp(MCInst &Inst) { return true; }

  /// Replace instruction opcode to be a regural call instead of tail call.
  virtual bool convertTailCallToCall(MCInst &Inst) {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Modify a direct call instruction \p Inst with an indirect call taking
  /// a destination from a memory location pointed by \p TargetLocation symbol.
  virtual bool convertCallToIndirectCall(MCInst &Inst,
                                         const MCSymbol *TargetLocation,
                                         MCContext *Ctx) {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Morph an indirect call into a load where \p Reg holds the call target.
  virtual void convertIndirectCallToLoad(MCInst &Inst, MCPhysReg Reg) {
    llvm_unreachable("not implemented");
  }

  /// Replace instruction with a shorter version that could be relaxed later
  /// if needed.
  virtual bool shortenInstruction(MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Lower a tail call instruction \p Inst if required by target.
  virtual bool lowerTailCall(MCInst &Inst) {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Receives a list of MCInst of the basic block to analyze and interpret the
  /// terminators of this basic block. TBB must be initialized with the original
  /// fall-through for this BB.
  virtual bool analyzeBranch(InstructionIterator Begin, InstructionIterator End,
                             const MCSymbol *&TBB, const MCSymbol *&FBB,
                             MCInst *&CondBranch, MCInst *&UncondBranch) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Analyze \p Instruction to try and determine what type of indirect branch
  /// it is.  It is assumed that \p Instruction passes isIndirectBranch().
  /// \p BB is an array of instructions immediately preceding \p Instruction.
  /// If \p Instruction can be successfully analyzed, the output parameters
  /// will be set to the different components of the branch.  \p MemLocInstr
  /// is the instruction that loads up the indirect function pointer.  It may
  /// or may not be same as \p Instruction.
  virtual IndirectBranchType
  analyzeIndirectBranch(MCInst &Instruction, InstructionIterator Begin,
                        InstructionIterator End, const unsigned PtrSize,
                        MCInst *&MemLocInstr, unsigned &BaseRegNum,
                        unsigned &IndexRegNum, int64_t &DispValue,
                        const MCExpr *&DispExpr, MCInst *&PCRelBaseOut) const {
    llvm_unreachable("not implemented");
    return IndirectBranchType::UNKNOWN;
  }

  virtual bool analyzeVirtualMethodCall(InstructionIterator Begin,
                                        InstructionIterator End,
                                        std::vector<MCInst *> &MethodFetchInsns,
                                        unsigned &VtableRegNum,
                                        unsigned &BaseRegNum,
                                        uint64_t &MethodOffset) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual void createLongJmp(InstructionListType &Seq, const MCSymbol *Target,
                             MCContext *Ctx, bool IsTailCall = false) {
    llvm_unreachable("not implemented");
  }

  virtual void createShortJmp(InstructionListType &Seq, const MCSymbol *Target,
                              MCContext *Ctx, bool IsTailCall = false) {
    llvm_unreachable("not implemented");
  }

  /// Return true if the instruction CurInst, in combination with the recent
  /// history of disassembled instructions supplied by [Begin, End), is a linker
  /// generated veneer/stub that needs patching. This happens in AArch64 when
  /// the code is large and the linker needs to generate stubs, but it does
  /// not put any extra relocation information that could help us to easily
  /// extract the real target. This function identifies and extract the real
  /// target in Tgt. The instruction that loads the lower bits of the target
  /// is put in TgtLowBits, and its pair in TgtHiBits. If the instruction in
  /// TgtHiBits does not have an immediate operand, but an expression, then
  /// this expression is put in TgtHiSym and Tgt only contains the lower bits.
  virtual bool matchLinkerVeneer(InstructionIterator Begin,
                                 InstructionIterator End, uint64_t Address,
                                 const MCInst &CurInst, MCInst *&TargetHiBits,
                                 MCInst *&TargetLowBits,
                                 uint64_t &Target) const {
    llvm_unreachable("not implemented");
  }

  virtual int getShortJmpEncodingSize() const {
    llvm_unreachable("not implemented");
  }

  virtual int getUncondBranchEncodingSize() const {
    llvm_unreachable("not implemented");
    return 0;
  }

  /// Create a no-op instruction.
  virtual bool createNoop(MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Create a return instruction.
  virtual bool createReturn(MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Store \p Target absolute adddress to \p RegName
  virtual InstructionListType materializeAddress(const MCSymbol *Target,
                                                 MCContext *Ctx,
                                                 MCPhysReg RegName,
                                                 int64_t Addend = 0) const {
    llvm_unreachable("not implemented");
    return {};
  }

  /// Creates a new unconditional branch instruction in Inst and set its operand
  /// to TBB.
  ///
  /// Returns true on success.
  virtual bool createUncondBranch(MCInst &Inst, const MCSymbol *TBB,
                                  MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Creates a new call instruction in Inst and sets its operand to
  /// Target.
  ///
  /// Returns true on success.
  virtual bool createCall(MCInst &Inst, const MCSymbol *Target,
                          MCContext *Ctx) {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Creates a new tail call instruction in Inst and sets its operand to
  /// Target.
  ///
  /// Returns true on success.
  virtual bool createTailCall(MCInst &Inst, const MCSymbol *Target,
                              MCContext *Ctx) {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual void createLongTailCall(InstructionListType &Seq,
                                  const MCSymbol *Target, MCContext *Ctx) {
    llvm_unreachable("not implemented");
  }

  /// Creates a trap instruction in Inst.
  ///
  /// Returns true on success.
  virtual bool createTrap(MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Creates an instruction to bump the stack pointer just like a call.
  virtual bool createStackPointerIncrement(MCInst &Inst, int Size = 8,
                                           bool NoFlagsClobber = false) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Creates an instruction to move the stack pointer just like a ret.
  virtual bool createStackPointerDecrement(MCInst &Inst, int Size = 8,
                                           bool NoFlagsClobber = false) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Create a store instruction using \p StackReg as the base register
  /// and \p Offset as the displacement.
  virtual bool createSaveToStack(MCInst &Inst, const MCPhysReg &StackReg,
                                 int Offset, const MCPhysReg &SrcReg,
                                 int Size) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool createLoad(MCInst &Inst, const MCPhysReg &BaseReg, int64_t Scale,
                          const MCPhysReg &IndexReg, int64_t Offset,
                          const MCExpr *OffsetExpr,
                          const MCPhysReg &AddrSegmentReg,
                          const MCPhysReg &DstReg, int Size) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual void createLoadImmediate(MCInst &Inst, const MCPhysReg Dest,
                                   uint32_t Imm) const {
    llvm_unreachable("not implemented");
  }

  /// Create instruction to increment contents of target by 1
  virtual bool createIncMemory(MCInst &Inst, const MCSymbol *Target,
                               MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Create a fragment of code (sequence of instructions) that load a 32-bit
  /// address from memory, zero-extends it to 64 and jump to it (indirect jump).
  virtual bool
  createIJmp32Frag(SmallVectorImpl<MCInst> &Insts, const MCOperand &BaseReg,
                   const MCOperand &Scale, const MCOperand &IndexReg,
                   const MCOperand &Offset, const MCOperand &TmpReg) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Create a load instruction using \p StackReg as the base register
  /// and \p Offset as the displacement.
  virtual bool createRestoreFromStack(MCInst &Inst, const MCPhysReg &StackReg,
                                      int Offset, const MCPhysReg &DstReg,
                                      int Size) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Creates a call frame pseudo instruction. A single operand identifies which
  /// MCCFIInstruction this MCInst is referring to.
  ///
  /// Returns true on success.
  virtual bool createCFI(MCInst &Inst, int64_t Offset) const {
    Inst.clear();
    Inst.setOpcode(TargetOpcode::CFI_INSTRUCTION);
    Inst.addOperand(MCOperand::createImm(Offset));
    return true;
  }

  /// Create an inline version of memcpy(dest, src, 1).
  virtual InstructionListType createOneByteMemcpy() const {
    llvm_unreachable("not implemented");
    return {};
  }

  /// Create a sequence of instructions to compare contents of a register
  /// \p RegNo to immediate \Imm and jump to \p Target if they are equal.
  virtual InstructionListType createCmpJE(MCPhysReg RegNo, int64_t Imm,
                                          const MCSymbol *Target,
                                          MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return {};
  }

  /// Creates inline memcpy instruction. If \p ReturnEnd is true, then return
  /// (dest + n) instead of dest.
  virtual InstructionListType createInlineMemcpy(bool ReturnEnd) const {
    llvm_unreachable("not implemented");
    return {};
  }

  /// Create a target-specific relocation out of the \p Fixup.
  /// Note that not every fixup could be converted into a relocation.
  virtual Optional<Relocation> createRelocation(const MCFixup &Fixup,
                                                const MCAsmBackend &MAB) const {
    llvm_unreachable("not implemented");
    return Relocation();
  }

  /// Returns true if instruction is a call frame pseudo instruction.
  virtual bool isCFI(const MCInst &Inst) const {
    return Inst.getOpcode() == TargetOpcode::CFI_INSTRUCTION;
  }

  /// Reverses the branch condition in Inst and update its taken target to TBB.
  ///
  /// Returns true on success.
  virtual bool reverseBranchCondition(MCInst &Inst, const MCSymbol *TBB,
                                      MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool replaceBranchCondition(MCInst &Inst, const MCSymbol *TBB,
                                      MCContext *Ctx, unsigned CC) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual unsigned getInvertedCondCode(unsigned CC) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual unsigned getCondCodesLogicalOr(unsigned CC1, unsigned CC2) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool isValidCondCode(unsigned CC) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return the conditional code used in a conditional jump instruction.
  /// Returns invalid code if not conditional jump.
  virtual unsigned getCondCode(const MCInst &Inst) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Return canonical branch opcode for a reversible branch opcode. For every
  /// opposite branch opcode pair Op <-> OpR this function returns one of the
  /// opcodes which is considered a canonical.
  virtual unsigned getCanonicalBranchCondCode(unsigned CC) const {
    llvm_unreachable("not implemented");
    return false;
  }

  /// Sets the taken target of the branch instruction to Target.
  ///
  /// Returns true on success.
  virtual bool replaceBranchTarget(MCInst &Inst, const MCSymbol *TBB,
                                   MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return false;
  }

  virtual bool createEHLabel(MCInst &Inst, const MCSymbol *Label,
                             MCContext *Ctx) const {
    Inst.setOpcode(TargetOpcode::EH_LABEL);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(Label, MCSymbolRefExpr::VK_None, *Ctx)));
    return true;
  }

  /// Return annotation index matching the \p Name.
  Optional<unsigned> getAnnotationIndex(StringRef Name) const {
    auto AI = AnnotationNameIndexMap.find(Name);
    if (AI != AnnotationNameIndexMap.end())
      return AI->second;
    return NoneType();
  }

  /// Return annotation index matching the \p Name. Create a new index if the
  /// \p Name wasn't registered previously.
  unsigned getOrCreateAnnotationIndex(StringRef Name) {
    auto AI = AnnotationNameIndexMap.find(Name);
    if (AI != AnnotationNameIndexMap.end())
      return AI->second;

    const unsigned Index =
        AnnotationNameIndexMap.size() + MCPlus::MCAnnotation::kGeneric;
    AnnotationNameIndexMap.insert(std::make_pair(Name, Index));
    AnnotationNames.emplace_back(std::string(Name));
    return Index;
  }

  /// Store an annotation value on an MCInst.  This assumes the annotation
  /// is not already present.
  template <typename ValueType>
  const ValueType &addAnnotation(MCInst &Inst, unsigned Index,
                                 const ValueType &Val,
                                 AllocatorIdTy AllocatorId = 0) {
    assert(!hasAnnotation(Inst, Index));
    AnnotationAllocator &Allocator = getAnnotationAllocator(AllocatorId);
    auto *A = new (Allocator.ValueAllocator)
        MCPlus::MCSimpleAnnotation<ValueType>(Val);

    if (!std::is_trivial<ValueType>::value)
      Allocator.AnnotationPool.insert(A);
    setAnnotationOpValue(Inst, Index, reinterpret_cast<int64_t>(A),
                         AllocatorId);
    return A->getValue();
  }

  /// Store an annotation value on an MCInst.  This assumes the annotation
  /// is not already present.
  template <typename ValueType>
  const ValueType &addAnnotation(MCInst &Inst, StringRef Name,
                                 const ValueType &Val,
                                 AllocatorIdTy AllocatorId = 0) {
    return addAnnotation(Inst, getOrCreateAnnotationIndex(Name), Val,
                         AllocatorId);
  }

  /// Get an annotation as a specific value, but if the annotation does not
  /// exist, create a new annotation with the default constructor for that type.
  /// Return a non-const ref so caller can freely modify its contents
  /// afterwards.
  template <typename ValueType>
  ValueType &getOrCreateAnnotationAs(MCInst &Inst, unsigned Index,
                                     AllocatorIdTy AllocatorId = 0) {
    auto Val =
        tryGetAnnotationAs<ValueType>(const_cast<const MCInst &>(Inst), Index);
    if (!Val)
      Val = addAnnotation(Inst, Index, ValueType(), AllocatorId);
    return const_cast<ValueType &>(*Val);
  }

  /// Get an annotation as a specific value, but if the annotation does not
  /// exist, create a new annotation with the default constructor for that type.
  /// Return a non-const ref so caller can freely modify its contents
  /// afterwards.
  template <typename ValueType>
  ValueType &getOrCreateAnnotationAs(MCInst &Inst, StringRef Name,
                                     AllocatorIdTy AllocatorId = 0) {
    const unsigned Index = getOrCreateAnnotationIndex(Name);
    return getOrCreateAnnotationAs<ValueType>(Inst, Index, AllocatorId);
  }

  /// Get an annotation as a specific value. Assumes that the annotation exists.
  /// Use hasAnnotation() if the annotation may not exist.
  template <typename ValueType>
  ValueType &getAnnotationAs(const MCInst &Inst, unsigned Index) const {
    Optional<int64_t> Value = getAnnotationOpValue(Inst, Index);
    assert(Value && "annotation should exist");
    return reinterpret_cast<MCPlus::MCSimpleAnnotation<ValueType> *>(*Value)
        ->getValue();
  }

  /// Get an annotation as a specific value. Assumes that the annotation exists.
  /// Use hasAnnotation() if the annotation may not exist.
  template <typename ValueType>
  ValueType &getAnnotationAs(const MCInst &Inst, StringRef Name) const {
    const auto Index = getAnnotationIndex(Name);
    assert(Index && "annotation should exist");
    return getAnnotationAs<ValueType>(Inst, *Index);
  }

  /// Get an annotation as a specific value. If the annotation does not exist,
  /// return the \p DefaultValue.
  template <typename ValueType>
  const ValueType &
  getAnnotationWithDefault(const MCInst &Inst, unsigned Index,
                           const ValueType &DefaultValue = ValueType()) {
    if (!hasAnnotation(Inst, Index))
      return DefaultValue;
    return getAnnotationAs<ValueType>(Inst, Index);
  }

  /// Get an annotation as a specific value. If the annotation does not exist,
  /// return the \p DefaultValue.
  template <typename ValueType>
  const ValueType &
  getAnnotationWithDefault(const MCInst &Inst, StringRef Name,
                           const ValueType &DefaultValue = ValueType()) {
    const unsigned Index = getOrCreateAnnotationIndex(Name);
    return getAnnotationWithDefault<ValueType>(Inst, Index, DefaultValue);
  }

  /// Check if the specified annotation exists on this instruction.
  bool hasAnnotation(const MCInst &Inst, unsigned Index) const;

  /// Check if an annotation with a specified \p Name exists on \p Inst.
  bool hasAnnotation(const MCInst &Inst, StringRef Name) const {
    const auto Index = getAnnotationIndex(Name);
    if (!Index)
      return false;
    return hasAnnotation(Inst, *Index);
  }

  /// Get an annotation as a specific value, but if the annotation does not
  /// exist, return errc::result_out_of_range.
  template <typename ValueType>
  ErrorOr<const ValueType &> tryGetAnnotationAs(const MCInst &Inst,
                                                unsigned Index) const {
    if (!hasAnnotation(Inst, Index))
      return make_error_code(std::errc::result_out_of_range);
    return getAnnotationAs<ValueType>(Inst, Index);
  }

  /// Get an annotation as a specific value, but if the annotation does not
  /// exist, return errc::result_out_of_range.
  template <typename ValueType>
  ErrorOr<const ValueType &> tryGetAnnotationAs(const MCInst &Inst,
                                                StringRef Name) const {
    const auto Index = getAnnotationIndex(Name);
    if (!Index)
      return make_error_code(std::errc::result_out_of_range);
    return tryGetAnnotationAs<ValueType>(Inst, *Index);
  }

  template <typename ValueType>
  ErrorOr<ValueType &> tryGetAnnotationAs(MCInst &Inst, unsigned Index) const {
    if (!hasAnnotation(Inst, Index))
      return make_error_code(std::errc::result_out_of_range);
    return const_cast<ValueType &>(getAnnotationAs<ValueType>(Inst, Index));
  }

  template <typename ValueType>
  ErrorOr<ValueType &> tryGetAnnotationAs(MCInst &Inst, StringRef Name) const {
    const auto Index = getAnnotationIndex(Name);
    if (!Index)
      return make_error_code(std::errc::result_out_of_range);
    return tryGetAnnotationAs<ValueType>(Inst, *Index);
  }

  /// Print each annotation attached to \p Inst.
  void printAnnotations(const MCInst &Inst, raw_ostream &OS) const;

  /// Remove annotation with a given \p Index.
  ///
  /// Return true if the annotation was removed, false if the annotation
  /// was not present.
  bool removeAnnotation(MCInst &Inst, unsigned Index);

  /// Remove annotation associated with \p Name.
  ///
  /// Return true if the annotation was removed, false if the annotation
  /// was not present.
  bool removeAnnotation(MCInst &Inst, StringRef Name) {
    const auto Index = getAnnotationIndex(Name);
    if (!Index)
      return false;
    return removeAnnotation(Inst, *Index);
  }

  /// Remove meta-data, but don't destroy it.
  void stripAnnotations(MCInst &Inst, bool KeepTC = false);

  virtual InstructionListType
  createInstrumentedIndirectCall(const MCInst &CallInst, bool TailCall,
                                 MCSymbol *HandlerFuncAddr, int CallSiteID,
                                 MCContext *Ctx) {
    llvm_unreachable("not implemented");
    return InstructionListType();
  }

  virtual InstructionListType createInstrumentedIndCallHandlerExitBB() const {
    llvm_unreachable("not implemented");
    return InstructionListType();
  }

  virtual InstructionListType
  createInstrumentedIndTailCallHandlerExitBB() const {
    llvm_unreachable("not implemented");
    return InstructionListType();
  }

  virtual InstructionListType
  createInstrumentedIndCallHandlerEntryBB(const MCSymbol *InstrTrampoline,
                                          const MCSymbol *IndCallHandler,
                                          MCContext *Ctx) {
    llvm_unreachable("not implemented");
    return InstructionListType();
  }

  virtual InstructionListType createNumCountersGetter(MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return {};
  }

  virtual InstructionListType createInstrLocationsGetter(MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return {};
  }

  virtual InstructionListType createInstrTablesGetter(MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return {};
  }

  virtual InstructionListType createInstrNumFuncsGetter(MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return {};
  }

  virtual InstructionListType createSymbolTrampoline(const MCSymbol *TgtSym,
                                                     MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return InstructionListType();
  }

  virtual InstructionListType createDummyReturnFunction(MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return InstructionListType();
  }

  /// This method takes an indirect call instruction and splits it up into an
  /// equivalent set of instructions that use direct calls for target
  /// symbols/addresses that are contained in the Targets vector.  This is done
  /// by guarding each direct call with a compare instruction to verify that
  /// the target is correct.
  /// If the VtableAddrs vector is not empty, the call will have the extra
  /// load of the method pointer from the vtable eliminated.  When non-empty
  /// the VtableAddrs vector must be the same size as Targets and include the
  /// address of a vtable for each corresponding method call in Targets.  The
  /// MethodFetchInsns vector holds instructions that are used to load the
  /// correct method for the cold call case.
  ///
  /// The return value is a vector of code snippets (essentially basic blocks).
  /// There is a symbol associated with each snippet except for the first.
  /// If the original call is not a tail call, the last snippet will have an
  /// empty vector of instructions.  The label is meant to indicate the basic
  /// block where all previous snippets are joined, i.e. the instructions that
  /// would immediate follow the original call.
  using BlocksVectorTy =
      std::vector<std::pair<MCSymbol *, InstructionListType>>;
  struct MultiBlocksCode {
    BlocksVectorTy Blocks;
    std::vector<MCSymbol *> Successors;
  };

  virtual BlocksVectorTy indirectCallPromotion(
      const MCInst &CallInst,
      const std::vector<std::pair<MCSymbol *, uint64_t>> &Targets,
      const std::vector<std::pair<MCSymbol *, uint64_t>> &VtableSyms,
      const std::vector<MCInst *> &MethodFetchInsns,
      const bool MinimizeCodeSize, MCContext *Ctx) {
    llvm_unreachable("not implemented");
    return BlocksVectorTy();
  }

  virtual BlocksVectorTy jumpTablePromotion(
      const MCInst &IJmpInst,
      const std::vector<std::pair<MCSymbol *, uint64_t>> &Targets,
      const std::vector<MCInst *> &TargetFetchInsns, MCContext *Ctx) const {
    llvm_unreachable("not implemented");
    return BlocksVectorTy();
  }
};

MCPlusBuilder *createX86MCPlusBuilder(const MCInstrAnalysis *,
                                      const MCInstrInfo *,
                                      const MCRegisterInfo *);

MCPlusBuilder *createAArch64MCPlusBuilder(const MCInstrAnalysis *,
                                          const MCInstrInfo *,
                                          const MCRegisterInfo *);

} // namespace bolt
} // namespace llvm

#endif
