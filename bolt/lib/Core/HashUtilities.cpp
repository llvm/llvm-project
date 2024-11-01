//===- bolt/Core/HashUtilities.cpp - Misc hash utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computation of hash values over BinaryFunction and BinaryBasicBlock.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/HashUtilities.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/MC/MCInstPrinter.h"

namespace llvm {
namespace bolt {

/// Hashing a 64-bit integer to a 16-bit one.
uint16_t hash_64_to_16(const uint64_t Hash) {
  uint16_t Res = (uint16_t)(Hash & 0xFFFF);
  Res ^= (uint16_t)((Hash >> 16) & 0xFFFF);
  Res ^= (uint16_t)((Hash >> 32) & 0xFFFF);
  Res ^= (uint16_t)((Hash >> 48) & 0xFFFF);
  return Res;
}

std::string hashInteger(uint64_t Value) {
  std::string HashString;
  if (Value == 0)
    HashString.push_back(0);

  while (Value) {
    uint8_t LSB = Value & 0xff;
    HashString.push_back(LSB);
    Value >>= 8;
  }

  return HashString;
}

std::string hashSymbol(BinaryContext &BC, const MCSymbol &Symbol) {
  std::string HashString;

  // Ignore function references.
  if (BC.getFunctionForSymbol(&Symbol))
    return HashString;

  llvm::ErrorOr<uint64_t> ErrorOrValue = BC.getSymbolValue(Symbol);
  if (!ErrorOrValue)
    return HashString;

  // Ignore jump table references.
  if (BC.getJumpTableContainingAddress(*ErrorOrValue))
    return HashString;

  return HashString.append(hashInteger(*ErrorOrValue));
}

std::string hashExpr(BinaryContext &BC, const MCExpr &Expr) {
  switch (Expr.getKind()) {
  case MCExpr::Constant:
    return hashInteger(cast<MCConstantExpr>(Expr).getValue());
  case MCExpr::SymbolRef:
    return hashSymbol(BC, cast<MCSymbolRefExpr>(Expr).getSymbol());
  case MCExpr::Unary: {
    const auto &UnaryExpr = cast<MCUnaryExpr>(Expr);
    return hashInteger(UnaryExpr.getOpcode())
        .append(hashExpr(BC, *UnaryExpr.getSubExpr()));
  }
  case MCExpr::Binary: {
    const auto &BinaryExpr = cast<MCBinaryExpr>(Expr);
    return hashExpr(BC, *BinaryExpr.getLHS())
        .append(hashInteger(BinaryExpr.getOpcode()))
        .append(hashExpr(BC, *BinaryExpr.getRHS()));
  }
  case MCExpr::Target:
    return std::string();
  }

  llvm_unreachable("invalid expression kind");
}

std::string hashInstOperand(BinaryContext &BC, const MCOperand &Operand) {
  if (Operand.isImm())
    return hashInteger(Operand.getImm());
  if (Operand.isReg())
    return hashInteger(Operand.getReg());
  if (Operand.isExpr())
    return hashExpr(BC, *Operand.getExpr());

  return std::string();
}

std::string hashBlock(BinaryContext &BC, const BinaryBasicBlock &BB,
                      OperandHashFuncTy OperandHashFunc) {
  const bool IsX86 = BC.isX86();

  // The hash is computed by creating a string of all instruction opcodes and
  // possibly their operands and then hashing that string with std::hash.
  std::string HashString;

  for (const MCInst &Inst : BB) {
    if (BC.MIB->isPseudo(Inst))
      continue;

    unsigned Opcode = Inst.getOpcode();

    // Ignore unconditional jumps since we check CFG consistency by processing
    // basic blocks in order and do not rely on branches to be in-sync with
    // CFG. Note that we still use condition code of conditional jumps.
    if (BC.MIB->isUnconditionalBranch(Inst))
      continue;

    if (IsX86 && BC.MIB->isConditionalBranch(Inst))
      Opcode = BC.MIB->getShortBranchOpcode(Opcode);

    if (Opcode == 0) {
      HashString.push_back(0);
    } else {
      StringRef OpcodeName = BC.InstPrinter->getOpcodeName(Opcode);
      HashString.append(OpcodeName.str());
    }

    for (const MCOperand &Op : MCPlus::primeOperands(Inst))
      HashString.append(OperandHashFunc(Op));
  }
  return HashString;
}

/// A "loose" hash of a basic block to use with the stale profile matching. The
/// computed value will be the same for blocks with minor changes (such as
/// reordering of instructions or using different operands) but may result in
/// collisions that need to be resolved by a stronger hashing.
std::string hashBlockLoose(BinaryContext &BC, const BinaryBasicBlock &BB) {
  // The hash is computed by creating a string of all lexicographically ordered
  // instruction opcodes, which is then hashed with std::hash.
  std::set<std::string> Opcodes;
  for (const MCInst &Inst : BB) {
    // Skip pseudo instructions and nops.
    if (BC.MIB->isPseudo(Inst) || BC.MIB->isNoop(Inst))
      continue;

    // Ignore unconditional jumps, as they can be added / removed as a result
    // of basic block reordering.
    if (BC.MIB->isUnconditionalBranch(Inst))
      continue;

    // Do not distinguish different types of conditional jumps.
    if (BC.MIB->isConditionalBranch(Inst)) {
      Opcodes.insert("JMP");
      continue;
    }

    std::string Mnemonic = BC.InstPrinter->getMnemonic(&Inst).first;
    Mnemonic.erase(
        std::remove_if(Mnemonic.begin(), Mnemonic.end(),
                       [](unsigned char ch) { return std::isspace(ch); }),
        Mnemonic.end());
    Opcodes.insert(Mnemonic);
  }

  std::string HashString;
  for (const std::string &Opcode : Opcodes)
    HashString.append(Opcode);
  return HashString;
}

} // namespace bolt
} // namespace llvm
