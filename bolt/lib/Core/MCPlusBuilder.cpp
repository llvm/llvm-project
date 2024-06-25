//===- bolt/Core/MCPlusBuilder.cpp - Interface for MCPlus -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MCPlusBuilder class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Core/MCPlus.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#define DEBUG_TYPE "mcplus"

using namespace llvm;
using namespace bolt;
using namespace MCPlus;

namespace opts {
cl::opt<bool>
    TerminalTrap("terminal-trap",
                 cl::desc("Assume that execution stops at trap instruction"),
                 cl::init(true), cl::Hidden, cl::cat(BoltCategory));
}

bool MCPlusBuilder::equals(const MCInst &A, const MCInst &B,
                           CompFuncTy Comp) const {
  if (A.getOpcode() != B.getOpcode())
    return false;

  unsigned NumOperands = MCPlus::getNumPrimeOperands(A);
  if (NumOperands != MCPlus::getNumPrimeOperands(B))
    return false;

  for (unsigned Index = 0; Index < NumOperands; ++Index)
    if (!equals(A.getOperand(Index), B.getOperand(Index), Comp))
      return false;

  return true;
}

bool MCPlusBuilder::equals(const MCOperand &A, const MCOperand &B,
                           CompFuncTy Comp) const {
  if (A.isReg()) {
    if (!B.isReg())
      return false;
    return A.getReg() == B.getReg();
  } else if (A.isImm()) {
    if (!B.isImm())
      return false;
    return A.getImm() == B.getImm();
  } else if (A.isSFPImm()) {
    if (!B.isSFPImm())
      return false;
    return A.getSFPImm() == B.getSFPImm();
  } else if (A.isDFPImm()) {
    if (!B.isDFPImm())
      return false;
    return A.getDFPImm() == B.getDFPImm();
  } else if (A.isExpr()) {
    if (!B.isExpr())
      return false;
    return equals(*A.getExpr(), *B.getExpr(), Comp);
  } else {
    llvm_unreachable("unexpected operand kind");
    return false;
  }
}

bool MCPlusBuilder::equals(const MCExpr &A, const MCExpr &B,
                           CompFuncTy Comp) const {
  if (A.getKind() != B.getKind())
    return false;

  switch (A.getKind()) {
  case MCExpr::Constant: {
    const auto &ConstA = cast<MCConstantExpr>(A);
    const auto &ConstB = cast<MCConstantExpr>(B);
    return ConstA.getValue() == ConstB.getValue();
  }

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr &SymbolA = cast<MCSymbolRefExpr>(A);
    const MCSymbolRefExpr &SymbolB = cast<MCSymbolRefExpr>(B);
    return SymbolA.getKind() == SymbolB.getKind() &&
           Comp(&SymbolA.getSymbol(), &SymbolB.getSymbol());
  }

  case MCExpr::Unary: {
    const auto &UnaryA = cast<MCUnaryExpr>(A);
    const auto &UnaryB = cast<MCUnaryExpr>(B);
    return UnaryA.getOpcode() == UnaryB.getOpcode() &&
           equals(*UnaryA.getSubExpr(), *UnaryB.getSubExpr(), Comp);
  }

  case MCExpr::Binary: {
    const auto &BinaryA = cast<MCBinaryExpr>(A);
    const auto &BinaryB = cast<MCBinaryExpr>(B);
    return BinaryA.getOpcode() == BinaryB.getOpcode() &&
           equals(*BinaryA.getLHS(), *BinaryB.getLHS(), Comp) &&
           equals(*BinaryA.getRHS(), *BinaryB.getRHS(), Comp);
  }

  case MCExpr::Target: {
    const auto &TargetExprA = cast<MCTargetExpr>(A);
    const auto &TargetExprB = cast<MCTargetExpr>(B);
    return equals(TargetExprA, TargetExprB, Comp);
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

bool MCPlusBuilder::equals(const MCTargetExpr &A, const MCTargetExpr &B,
                           CompFuncTy Comp) const {
  llvm_unreachable("target-specific expressions are unsupported");
}

bool MCPlusBuilder::isTerminator(const MCInst &Inst) const {
  return Analysis->isTerminator(Inst) ||
         (opts::TerminalTrap && Info->get(Inst.getOpcode()).isTrap());
}

void MCPlusBuilder::setTailCall(MCInst &Inst) const {
  assert(!hasAnnotation(Inst, MCAnnotation::kTailCall));
  setAnnotationOpValue(Inst, MCAnnotation::kTailCall, true);
}

bool MCPlusBuilder::isTailCall(const MCInst &Inst) const {
  if (hasAnnotation(Inst, MCAnnotation::kTailCall))
    return true;
  if (getConditionalTailCall(Inst))
    return true;
  return false;
}

std::optional<MCLandingPad> MCPlusBuilder::getEHInfo(const MCInst &Inst) const {
  if (!isCall(Inst))
    return std::nullopt;
  std::optional<int64_t> LPSym =
      getAnnotationOpValue(Inst, MCAnnotation::kEHLandingPad);
  if (!LPSym)
    return std::nullopt;
  std::optional<int64_t> Action =
      getAnnotationOpValue(Inst, MCAnnotation::kEHAction);
  if (!Action)
    return std::nullopt;

  return std::make_pair(reinterpret_cast<const MCSymbol *>(*LPSym),
                        static_cast<uint64_t>(*Action));
}

void MCPlusBuilder::addEHInfo(MCInst &Inst, const MCLandingPad &LP) const {
  if (isCall(Inst)) {
    assert(!getEHInfo(Inst));
    setAnnotationOpValue(Inst, MCAnnotation::kEHLandingPad,
                         reinterpret_cast<int64_t>(LP.first));
    setAnnotationOpValue(Inst, MCAnnotation::kEHAction,
                         static_cast<int64_t>(LP.second));
  }
}

bool MCPlusBuilder::updateEHInfo(MCInst &Inst, const MCLandingPad &LP) const {
  if (!isInvoke(Inst))
    return false;

  setAnnotationOpValue(Inst, MCAnnotation::kEHLandingPad,
                       reinterpret_cast<int64_t>(LP.first));
  setAnnotationOpValue(Inst, MCAnnotation::kEHAction,
                       static_cast<int64_t>(LP.second));
  return true;
}

int64_t MCPlusBuilder::getGnuArgsSize(const MCInst &Inst) const {
  std::optional<int64_t> Value =
      getAnnotationOpValue(Inst, MCAnnotation::kGnuArgsSize);
  if (!Value)
    return -1LL;
  return *Value;
}

void MCPlusBuilder::addGnuArgsSize(MCInst &Inst, int64_t GnuArgsSize) const {
  assert(GnuArgsSize >= 0 && "cannot set GNU_args_size to negative value");
  assert(getGnuArgsSize(Inst) == -1LL && "GNU_args_size already set");
  assert(isInvoke(Inst) && "GNU_args_size can only be set for invoke");

  setAnnotationOpValue(Inst, MCAnnotation::kGnuArgsSize, GnuArgsSize);
}

uint64_t MCPlusBuilder::getJumpTable(const MCInst &Inst) const {
  std::optional<int64_t> Value =
      getAnnotationOpValue(Inst, MCAnnotation::kJumpTable);
  if (!Value)
    return 0;
  return *Value;
}

uint16_t MCPlusBuilder::getJumpTableIndexReg(const MCInst &Inst) const {
  return getAnnotationAs<uint16_t>(Inst, "JTIndexReg");
}

bool MCPlusBuilder::setJumpTable(MCInst &Inst, uint64_t Value,
                                 uint16_t IndexReg, AllocatorIdTy AllocId) {
  if (!isIndirectBranch(Inst))
    return false;
  setAnnotationOpValue(Inst, MCAnnotation::kJumpTable, Value);
  getOrCreateAnnotationAs<uint16_t>(Inst, "JTIndexReg", AllocId) = IndexReg;
  return true;
}

bool MCPlusBuilder::unsetJumpTable(MCInst &Inst) const {
  if (!getJumpTable(Inst))
    return false;
  removeAnnotation(Inst, MCAnnotation::kJumpTable);
  removeAnnotation(Inst, "JTIndexReg");
  return true;
}

std::optional<uint64_t>
MCPlusBuilder::getConditionalTailCall(const MCInst &Inst) const {
  std::optional<int64_t> Value =
      getAnnotationOpValue(Inst, MCAnnotation::kConditionalTailCall);
  if (!Value)
    return std::nullopt;
  return static_cast<uint64_t>(*Value);
}

bool MCPlusBuilder::setConditionalTailCall(MCInst &Inst, uint64_t Dest) const {
  if (!isConditionalBranch(Inst))
    return false;

  setAnnotationOpValue(Inst, MCAnnotation::kConditionalTailCall, Dest);
  return true;
}

bool MCPlusBuilder::unsetConditionalTailCall(MCInst &Inst) const {
  if (!getConditionalTailCall(Inst))
    return false;
  removeAnnotation(Inst, MCAnnotation::kConditionalTailCall);
  return true;
}

std::optional<uint32_t> MCPlusBuilder::getOffset(const MCInst &Inst) const {
  std::optional<int64_t> Value =
      getAnnotationOpValue(Inst, MCAnnotation::kOffset);
  if (!Value)
    return std::nullopt;
  return static_cast<uint32_t>(*Value);
}

uint32_t MCPlusBuilder::getOffsetWithDefault(const MCInst &Inst,
                                             uint32_t Default) const {
  if (std::optional<uint32_t> Offset = getOffset(Inst))
    return *Offset;
  return Default;
}

bool MCPlusBuilder::setOffset(MCInst &Inst, uint32_t Offset) const {
  setAnnotationOpValue(Inst, MCAnnotation::kOffset, Offset);
  return true;
}

bool MCPlusBuilder::clearOffset(MCInst &Inst) const {
  if (!hasAnnotation(Inst, MCAnnotation::kOffset))
    return false;
  removeAnnotation(Inst, MCAnnotation::kOffset);
  return true;
}

MCSymbol *MCPlusBuilder::getInstLabel(const MCInst &Inst) const {
  if (std::optional<int64_t> Label =
          getAnnotationOpValue(Inst, MCAnnotation::kLabel))
    return reinterpret_cast<MCSymbol *>(*Label);
  return nullptr;
}

MCSymbol *MCPlusBuilder::getOrCreateInstLabel(MCInst &Inst, const Twine &Name,
                                              MCContext *Ctx) const {
  MCSymbol *Label = getInstLabel(Inst);
  if (Label)
    return Label;

  Label = Ctx->createNamedTempSymbol(Name);
  setAnnotationOpValue(Inst, MCAnnotation::kLabel,
                       reinterpret_cast<int64_t>(Label));
  return Label;
}

void MCPlusBuilder::setInstLabel(MCInst &Inst, MCSymbol *Label) const {
  assert(!getInstLabel(Inst) && "Instruction already has assigned label.");
  setAnnotationOpValue(Inst, MCAnnotation::kLabel,
                       reinterpret_cast<int64_t>(Label));
}

std::optional<uint32_t> MCPlusBuilder::getSize(const MCInst &Inst) const {
  if (std::optional<int64_t> Value =
          getAnnotationOpValue(Inst, MCAnnotation::kSize))
    return static_cast<uint32_t>(*Value);
  return std::nullopt;
}

void MCPlusBuilder::setSize(MCInst &Inst, uint32_t Size) const {
  setAnnotationOpValue(Inst, MCAnnotation::kSize, Size);
}

bool MCPlusBuilder::isDynamicBranch(const MCInst &Inst) const {
  if (!hasAnnotation(Inst, MCAnnotation::kDynamicBranch))
    return false;
  assert(isBranch(Inst) && "Branch expected.");
  return true;
}

std::optional<uint32_t>
MCPlusBuilder::getDynamicBranchID(const MCInst &Inst) const {
  if (std::optional<int64_t> Value =
          getAnnotationOpValue(Inst, MCAnnotation::kDynamicBranch)) {
    assert(isBranch(Inst) && "Branch expected.");
    return static_cast<uint32_t>(*Value);
  }
  return std::nullopt;
}

void MCPlusBuilder::setDynamicBranch(MCInst &Inst, uint32_t ID) const {
  assert(isBranch(Inst) && "Branch expected.");
  setAnnotationOpValue(Inst, MCAnnotation::kDynamicBranch, ID);
}

bool MCPlusBuilder::hasAnnotation(const MCInst &Inst, unsigned Index) const {
  return (bool)getAnnotationOpValue(Inst, Index);
}

bool MCPlusBuilder::removeAnnotation(MCInst &Inst, unsigned Index) const {
  std::optional<unsigned> FirstAnnotationOp = getFirstAnnotationOpIndex(Inst);
  if (!FirstAnnotationOp)
    return false;

  for (unsigned I = Inst.getNumOperands() - 1; I >= *FirstAnnotationOp; --I) {
    const int64_t ImmValue = Inst.getOperand(I).getImm();
    if (extractAnnotationIndex(ImmValue) == Index) {
      Inst.erase(Inst.begin() + I);
      return true;
    }
  }
  return false;
}

void MCPlusBuilder::stripAnnotations(MCInst &Inst, bool KeepTC) const {
  KeepTC &= hasAnnotation(Inst, MCAnnotation::kTailCall);

  removeAnnotations(Inst);

  if (KeepTC)
    setTailCall(Inst);
}

void MCPlusBuilder::printAnnotations(const MCInst &Inst,
                                     raw_ostream &OS) const {
  std::optional<unsigned> FirstAnnotationOp = getFirstAnnotationOpIndex(Inst);
  if (!FirstAnnotationOp)
    return;

  for (unsigned I = *FirstAnnotationOp; I < Inst.getNumOperands(); ++I) {
    const int64_t Imm = Inst.getOperand(I).getImm();
    const unsigned Index = extractAnnotationIndex(Imm);
    const int64_t Value = extractAnnotationValue(Imm);
    const auto *Annotation = reinterpret_cast<const MCAnnotation *>(Value);
    if (Index >= MCAnnotation::kGeneric) {
      OS << " # " << AnnotationNames[Index - MCAnnotation::kGeneric] << ": ";
      Annotation->print(OS);
    }
  }
}

void MCPlusBuilder::getClobberedRegs(const MCInst &Inst,
                                     BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  const MCInstrDesc &InstInfo = Info->get(Inst.getOpcode());

  for (MCPhysReg ImplicitDef : InstInfo.implicit_defs())
    Regs |= getAliases(ImplicitDef, /*OnlySmaller=*/false);

  for (const MCOperand &Operand : defOperands(Inst)) {
    assert(Operand.isReg());
    Regs |= getAliases(Operand.getReg(), /*OnlySmaller=*/false);
  }
}

void MCPlusBuilder::getTouchedRegs(const MCInst &Inst, BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  const MCInstrDesc &InstInfo = Info->get(Inst.getOpcode());

  for (MCPhysReg ImplicitDef : InstInfo.implicit_defs())
    Regs |= getAliases(ImplicitDef, /*OnlySmaller=*/false);
  for (MCPhysReg ImplicitUse : InstInfo.implicit_uses())
    Regs |= getAliases(ImplicitUse, /*OnlySmaller=*/false);

  for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
    if (!Inst.getOperand(I).isReg())
      continue;
    Regs |= getAliases(Inst.getOperand(I).getReg(), /*OnlySmaller=*/false);
  }
}

void MCPlusBuilder::getWrittenRegs(const MCInst &Inst, BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  const MCInstrDesc &InstInfo = Info->get(Inst.getOpcode());

  for (MCPhysReg ImplicitDef : InstInfo.implicit_defs())
    Regs |= getAliases(ImplicitDef, /*OnlySmaller=*/true);

  for (const MCOperand &Operand : defOperands(Inst)) {
    assert(Operand.isReg());
    Regs |= getAliases(Operand.getReg(), /*OnlySmaller=*/true);
  }
}

void MCPlusBuilder::getUsedRegs(const MCInst &Inst, BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  const MCInstrDesc &InstInfo = Info->get(Inst.getOpcode());

  for (MCPhysReg ImplicitUse : InstInfo.implicit_uses())
    Regs |= getAliases(ImplicitUse, /*OnlySmaller=*/true);

  for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
    if (!Inst.getOperand(I).isReg())
      continue;
    Regs |= getAliases(Inst.getOperand(I).getReg(), /*OnlySmaller=*/true);
  }
}

void MCPlusBuilder::getSrcRegs(const MCInst &Inst, BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  if (isCall(Inst)) {
    BitVector CallRegs = BitVector(Regs.size(), false);
    getCalleeSavedRegs(CallRegs);
    CallRegs.flip();
    Regs |= CallRegs;
    return;
  }

  if (isReturn(Inst)) {
    getDefaultLiveOut(Regs);
    return;
  }

  if (isRep(Inst))
    getRepRegs(Regs);

  const MCInstrDesc &InstInfo = Info->get(Inst.getOpcode());

  for (MCPhysReg ImplicitUse : InstInfo.implicit_uses())
    Regs |= getAliases(ImplicitUse, /*OnlySmaller=*/true);

  for (const MCOperand &Operand : useOperands(Inst))
    if (Operand.isReg())
      Regs |= getAliases(Operand.getReg(), /*OnlySmaller=*/true);
}

bool MCPlusBuilder::hasDefOfPhysReg(const MCInst &MI, unsigned Reg) const {
  const MCInstrDesc &InstInfo = Info->get(MI.getOpcode());
  return InstInfo.hasDefOfPhysReg(MI, Reg, *RegInfo);
}

bool MCPlusBuilder::hasUseOfPhysReg(const MCInst &MI, unsigned Reg) const {
  const MCInstrDesc &InstInfo = Info->get(MI.getOpcode());
  for (int I = InstInfo.NumDefs; I < InstInfo.NumOperands; ++I)
    if (MI.getOperand(I).isReg() && MI.getOperand(I).getReg() &&
        RegInfo->isSubRegisterEq(Reg, MI.getOperand(I).getReg()))
      return true;
  for (MCPhysReg ImplicitUse : InstInfo.implicit_uses()) {
    if (ImplicitUse == Reg || RegInfo->isSubRegister(Reg, ImplicitUse))
      return true;
  }
  return false;
}

const BitVector &MCPlusBuilder::getAliases(MCPhysReg Reg,
                                           bool OnlySmaller) const {
  if (OnlySmaller)
    return SmallerAliasMap[Reg];
  return AliasMap[Reg];
}

void MCPlusBuilder::initAliases() {
  assert(AliasMap.size() == 0 && SmallerAliasMap.size() == 0);
  // Build alias map
  for (MCPhysReg I = 0, E = RegInfo->getNumRegs(); I != E; ++I) {
    BitVector BV(RegInfo->getNumRegs(), false);
    BV.set(I);
    AliasMap.emplace_back(BV);
    SmallerAliasMap.emplace_back(BV);
  }

  // Cache all aliases for each register
  for (MCPhysReg I = 1, E = RegInfo->getNumRegs(); I != E; ++I) {
    for (MCRegAliasIterator AI(I, RegInfo, true); AI.isValid(); ++AI)
      AliasMap[I].set(*AI);
  }

  // Propagate smaller alias info upwards. Skip reg 0 (mapped to NoRegister)
  for (MCPhysReg I = 1, E = RegInfo->getNumRegs(); I < E; ++I)
    for (MCSubRegIterator SI(I, RegInfo); SI.isValid(); ++SI)
      SmallerAliasMap[I] |= SmallerAliasMap[*SI];

  LLVM_DEBUG({
    dbgs() << "Dumping reg alias table:\n";
    for (MCPhysReg I = 0, E = RegInfo->getNumRegs(); I != E; ++I) {
      dbgs() << "Reg " << I << ": ";
      const BitVector &BV = AliasMap[I];
      int Idx = BV.find_first();
      while (Idx != -1) {
        dbgs() << Idx << " ";
        Idx = BV.find_next(Idx);
      }
      dbgs() << "\n";
    }
  });
}

void MCPlusBuilder::initSizeMap() {
  SizeMap.resize(RegInfo->getNumRegs());
  // Build size map
  for (auto RC : RegInfo->regclasses())
    for (MCPhysReg Reg : RC)
      SizeMap[Reg] = RC.getSizeInBits() / 8;
}

bool MCPlusBuilder::setOperandToSymbolRef(MCInst &Inst, int OpNum,
                                          const MCSymbol *Symbol,
                                          int64_t Addend, MCContext *Ctx,
                                          uint64_t RelType) const {
  MCOperand Operand;
  if (!Addend) {
    Operand = MCOperand::createExpr(getTargetExprFor(
        Inst, MCSymbolRefExpr::create(Symbol, *Ctx), *Ctx, RelType));
  } else {
    Operand = MCOperand::createExpr(getTargetExprFor(
        Inst,
        MCBinaryExpr::createAdd(MCSymbolRefExpr::create(Symbol, *Ctx),
                                MCConstantExpr::create(Addend, *Ctx), *Ctx),
        *Ctx, RelType));
  }
  Inst.getOperand(OpNum) = Operand;
  return true;
}
