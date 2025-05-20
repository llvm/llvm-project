//===- bolt/Target/PPC/PPCMCSymbolizer.cpp ----------------------*- C++ -*-===//
//
// Minimal PowerPC Symbolizer for BOLT "Hello World" Programs
//
//===----------------------------------------------------------------------===//

#include "PPCMCSymbolizer.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/Relocation.h"
#include "llvm/MC/MCInst.h"

using namespace llvm;
using namespace bolt;

PPCMCSymbolizer::~PPCMCSymbolizer() = default;

bool PPCMCSymbolizer::tryAddingSymbolicOperand(
    MCInst &Inst, raw_ostream &CStream, int64_t Value, uint64_t Address,
    bool IsBranch, uint64_t Offset, uint64_t OpSize, uint64_t InstSize) {
  // 1) Normalize to function-relative offset
  BinaryContext &BC = Function.getBinaryContext();
  MCContext *Ctx = BC.Ctx.get();
  const uint64_t InstOffset = Address - Function.getAddress();

  // 2) Find relocation at "instruction start + immediate offset"
  const Relocation *Rel = Function.getRelocationAt(InstOffset + Offset);
  if (!Rel)
    return false;

  // 3) Build MCExpr = Symbol [+ Addend] and attach as a real operand
  const MCSymbol *Sym = Rel->Symbol; // prefer the pointer, not a name string
  const MCExpr *Expr = MCSymbolRefExpr::create(Sym, *Ctx);
  if (Rel->Addend)
    Expr = MCBinaryExpr::createAdd(
        Expr, MCConstantExpr::create(Rel->Addend, *Ctx), *Ctx);

  Inst.addOperand(MCOperand::createExpr(Expr));
  return true;
}

void PPCMCSymbolizer::tryAddingPcLoadReferenceComment(raw_ostream &CStream,
                                                      int64_t Value,
                                                      uint64_t Address) {
  // For "Hello World": no special PC-relative loads, leave empty for now
}