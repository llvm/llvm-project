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
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <queue>

#define DEBUG_TYPE "mcplus"

using namespace llvm;
using namespace bolt;
using namespace MCPlus;

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

void MCPlusBuilder::setTailCall(MCInst &Inst) {
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

Optional<MCLandingPad> MCPlusBuilder::getEHInfo(const MCInst &Inst) const {
  if (!isCall(Inst))
    return NoneType();
  Optional<int64_t> LPSym =
      getAnnotationOpValue(Inst, MCAnnotation::kEHLandingPad);
  if (!LPSym)
    return NoneType();
  Optional<int64_t> Action =
      getAnnotationOpValue(Inst, MCAnnotation::kEHAction);
  if (!Action)
    return NoneType();

  return std::make_pair(reinterpret_cast<const MCSymbol *>(*LPSym),
                        static_cast<uint64_t>(*Action));
}

void MCPlusBuilder::addEHInfo(MCInst &Inst, const MCLandingPad &LP) {
  if (isCall(Inst)) {
    assert(!getEHInfo(Inst));
    setAnnotationOpValue(Inst, MCAnnotation::kEHLandingPad,
                         reinterpret_cast<int64_t>(LP.first));
    setAnnotationOpValue(Inst, MCAnnotation::kEHAction,
                         static_cast<int64_t>(LP.second));
  }
}

int64_t MCPlusBuilder::getGnuArgsSize(const MCInst &Inst) const {
  Optional<int64_t> Value =
      getAnnotationOpValue(Inst, MCAnnotation::kGnuArgsSize);
  if (!Value)
    return -1LL;
  return *Value;
}

void MCPlusBuilder::addGnuArgsSize(MCInst &Inst, int64_t GnuArgsSize,
                                   AllocatorIdTy AllocId) {
  assert(GnuArgsSize >= 0 && "cannot set GNU_args_size to negative value");
  assert(getGnuArgsSize(Inst) == -1LL && "GNU_args_size already set");
  assert(isInvoke(Inst) && "GNU_args_size can only be set for invoke");

  setAnnotationOpValue(Inst, MCAnnotation::kGnuArgsSize, GnuArgsSize, AllocId);
}

uint64_t MCPlusBuilder::getJumpTable(const MCInst &Inst) const {
  Optional<int64_t> Value =
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
  setAnnotationOpValue(Inst, MCAnnotation::kJumpTable, Value, AllocId);
  getOrCreateAnnotationAs<uint16_t>(Inst, "JTIndexReg", AllocId) = IndexReg;
  return true;
}

bool MCPlusBuilder::unsetJumpTable(MCInst &Inst) {
  if (!getJumpTable(Inst))
    return false;
  removeAnnotation(Inst, MCAnnotation::kJumpTable);
  removeAnnotation(Inst, "JTIndexReg");
  return true;
}

Optional<uint64_t>
MCPlusBuilder::getConditionalTailCall(const MCInst &Inst) const {
  Optional<int64_t> Value =
      getAnnotationOpValue(Inst, MCAnnotation::kConditionalTailCall);
  if (!Value)
    return NoneType();
  return static_cast<uint64_t>(*Value);
}

bool MCPlusBuilder::setConditionalTailCall(MCInst &Inst, uint64_t Dest) {
  if (!isConditionalBranch(Inst))
    return false;

  setAnnotationOpValue(Inst, MCAnnotation::kConditionalTailCall, Dest);
  return true;
}

bool MCPlusBuilder::unsetConditionalTailCall(MCInst &Inst) {
  if (!getConditionalTailCall(Inst))
    return false;
  removeAnnotation(Inst, MCAnnotation::kConditionalTailCall);
  return true;
}

Optional<uint32_t> MCPlusBuilder::getOffset(const MCInst &Inst) const {
  Optional<int64_t> Value = getAnnotationOpValue(Inst, MCAnnotation::kOffset);
  if (!Value)
    return NoneType();
  return static_cast<uint32_t>(*Value);
}

uint32_t MCPlusBuilder::getOffsetWithDefault(const MCInst &Inst,
                                             uint32_t Default) const {
  if (Optional<uint32_t> Offset = getOffset(Inst))
    return *Offset;
  return Default;
}

bool MCPlusBuilder::setOffset(MCInst &Inst, uint32_t Offset,
                              AllocatorIdTy AllocatorId) {
  setAnnotationOpValue(Inst, MCAnnotation::kOffset, Offset, AllocatorId);
  return true;
}

bool MCPlusBuilder::clearOffset(MCInst &Inst) {
  if (!hasAnnotation(Inst, MCAnnotation::kOffset))
    return false;
  removeAnnotation(Inst, MCAnnotation::kOffset);
  return true;
}

bool MCPlusBuilder::hasAnnotation(const MCInst &Inst, unsigned Index) const {
  const MCInst *AnnotationInst = getAnnotationInst(Inst);
  if (!AnnotationInst)
    return false;

  return (bool)getAnnotationOpValue(Inst, Index);
}

bool MCPlusBuilder::removeAnnotation(MCInst &Inst, unsigned Index) {
  MCInst *AnnotationInst = getAnnotationInst(Inst);
  if (!AnnotationInst)
    return false;

  for (int I = AnnotationInst->getNumOperands() - 1; I >= 0; --I) {
    int64_t ImmValue = AnnotationInst->getOperand(I).getImm();
    if (extractAnnotationIndex(ImmValue) == Index) {
      AnnotationInst->erase(AnnotationInst->begin() + I);
      return true;
    }
  }
  return false;
}

void MCPlusBuilder::stripAnnotations(MCInst &Inst, bool KeepTC) {
  MCInst *AnnotationInst = getAnnotationInst(Inst);
  if (!AnnotationInst)
    return;
  // Preserve TailCall annotation.
  auto IsTC = hasAnnotation(Inst, MCAnnotation::kTailCall);

  Inst.erase(std::prev(Inst.end()));
  if (KeepTC && IsTC)
    setTailCall(Inst);
}

void MCPlusBuilder::printAnnotations(const MCInst &Inst,
                                     raw_ostream &OS) const {
  const MCInst *AnnotationInst = getAnnotationInst(Inst);
  if (!AnnotationInst)
    return;

  for (unsigned I = 0; I < AnnotationInst->getNumOperands(); ++I) {
    const int64_t Imm = AnnotationInst->getOperand(I).getImm();
    const unsigned Index = extractAnnotationIndex(Imm);
    const int64_t Value = extractAnnotationValue(Imm);
    const auto *Annotation = reinterpret_cast<const MCAnnotation *>(Value);
    if (Index >= MCAnnotation::kGeneric) {
      OS << " # " << AnnotationNames[Index - MCAnnotation::kGeneric] << ": ";
      Annotation->print(OS);
    }
  }
}

bool MCPlusBuilder::evaluateBranch(const MCInst &Inst, uint64_t Addr,
                                   uint64_t Size, uint64_t &Target) const {
  return Analysis->evaluateBranch(Inst, Addr, Size, Target);
}

void MCPlusBuilder::getClobberedRegs(const MCInst &Inst,
                                     BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  const MCInstrDesc &InstInfo = Info->get(Inst.getOpcode());

  const MCPhysReg *ImplicitDefs = InstInfo.getImplicitDefs();
  for (unsigned I = 0, E = InstInfo.getNumImplicitDefs(); I != E; ++I)
    Regs |= getAliases(ImplicitDefs[I], /*OnlySmaller=*/false);

  for (unsigned I = 0, E = InstInfo.getNumDefs(); I != E; ++I) {
    const MCOperand &Operand = Inst.getOperand(I);
    assert(Operand.isReg());
    Regs |= getAliases(Operand.getReg(), /*OnlySmaller=*/false);
  }
}

void MCPlusBuilder::getTouchedRegs(const MCInst &Inst, BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  const MCInstrDesc &InstInfo = Info->get(Inst.getOpcode());

  const MCPhysReg *ImplicitDefs = InstInfo.getImplicitDefs();
  for (unsigned I = 0, E = InstInfo.getNumImplicitDefs(); I != E; ++I)
    Regs |= getAliases(ImplicitDefs[I], /*OnlySmaller=*/false);
  const MCPhysReg *ImplicitUses = InstInfo.getImplicitUses();
  for (unsigned I = 0, E = InstInfo.getNumImplicitUses(); I != E; ++I)
    Regs |= getAliases(ImplicitUses[I], /*OnlySmaller=*/false);

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

  const MCPhysReg *ImplicitDefs = InstInfo.getImplicitDefs();
  for (unsigned I = 0, E = InstInfo.getNumImplicitDefs(); I != E; ++I)
    Regs |= getAliases(ImplicitDefs[I], /*OnlySmaller=*/true);

  for (unsigned I = 0, E = InstInfo.getNumDefs(); I != E; ++I) {
    const MCOperand &Operand = Inst.getOperand(I);
    assert(Operand.isReg());
    Regs |= getAliases(Operand.getReg(), /*OnlySmaller=*/true);
  }
}

void MCPlusBuilder::getUsedRegs(const MCInst &Inst, BitVector &Regs) const {
  if (isPrefix(Inst) || isCFI(Inst))
    return;

  const MCInstrDesc &InstInfo = Info->get(Inst.getOpcode());

  const MCPhysReg *ImplicitUses = InstInfo.getImplicitUses();
  for (unsigned I = 0, E = InstInfo.getNumImplicitUses(); I != E; ++I)
    Regs |= getAliases(ImplicitUses[I], /*OnlySmaller=*/true);

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

  const MCPhysReg *ImplicitUses = InstInfo.getImplicitUses();
  for (unsigned I = 0, E = InstInfo.getNumImplicitUses(); I != E; ++I)
    Regs |= getAliases(ImplicitUses[I], /*OnlySmaller=*/true);

  for (unsigned I = InstInfo.getNumDefs(), E = InstInfo.getNumOperands();
       I != E; ++I) {
    if (!Inst.getOperand(I).isReg())
      continue;
    Regs |= getAliases(Inst.getOperand(I).getReg(), /*OnlySmaller=*/true);
  }
}

bool MCPlusBuilder::hasDefOfPhysReg(const MCInst &MI, unsigned Reg) const {
  const MCInstrDesc &InstInfo = Info->get(MI.getOpcode());
  return InstInfo.hasDefOfPhysReg(MI, Reg, *RegInfo);
}

bool MCPlusBuilder::hasUseOfPhysReg(const MCInst &MI, unsigned Reg) const {
  const MCInstrDesc &InstInfo = Info->get(MI.getOpcode());
  for (int I = InstInfo.NumDefs; I < InstInfo.NumOperands; ++I)
    if (MI.getOperand(I).isReg() &&
        RegInfo->isSubRegisterEq(Reg, MI.getOperand(I).getReg()))
      return true;
  if (const uint16_t *ImpUses = InstInfo.ImplicitUses) {
    for (; *ImpUses; ++ImpUses)
      if (*ImpUses == Reg || RegInfo->isSubRegister(Reg, *ImpUses))
        return true;
  }
  return false;
}

const BitVector &MCPlusBuilder::getAliases(MCPhysReg Reg,
                                           bool OnlySmaller) const {
  // AliasMap caches a mapping of registers to the set of registers that
  // alias (are sub or superregs of itself, including itself).
  static std::vector<BitVector> AliasMap;
  static std::vector<BitVector> SmallerAliasMap;

  if (AliasMap.size() > 0) {
    if (OnlySmaller)
      return SmallerAliasMap[Reg];
    return AliasMap[Reg];
  }

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
  std::queue<MCPhysReg> Worklist;
  for (MCPhysReg I = 1, E = RegInfo->getNumRegs(); I < E; ++I)
    Worklist.push(I);
  while (!Worklist.empty()) {
    MCPhysReg I = Worklist.front();
    Worklist.pop();
    for (MCSubRegIterator SI(I, RegInfo); SI.isValid(); ++SI)
      SmallerAliasMap[I] |= SmallerAliasMap[*SI];
    for (MCSuperRegIterator SI(I, RegInfo); SI.isValid(); ++SI)
      Worklist.push(*SI);
  }

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

  if (OnlySmaller)
    return SmallerAliasMap[Reg];
  return AliasMap[Reg];
}

uint8_t MCPlusBuilder::getRegSize(MCPhysReg Reg) const {
  // SizeMap caches a mapping of registers to their sizes
  static std::vector<uint8_t> SizeMap;

  if (SizeMap.size() > 0) {
    return SizeMap[Reg];
  }
  SizeMap = std::vector<uint8_t>(RegInfo->getNumRegs());
  // Build size map
  for (auto I = RegInfo->regclass_begin(), E = RegInfo->regclass_end(); I != E;
       ++I) {
    for (MCPhysReg Reg : *I)
      SizeMap[Reg] = I->getSizeInBits() / 8;
  }

  return SizeMap[Reg];
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
