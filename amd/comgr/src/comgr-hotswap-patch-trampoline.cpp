//===- comgr-hotswap-patch-trampoline.cpp - B0-to-A0 trampoline patches ---===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Strong-symbol override for applyTrampolinePatches. Handles B0 errata
/// whose fix is larger than the original instruction:
///   - ds_*_2addr_stride64_*  : one 8B DS instruction -> two single-address
///     DS instructions
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

// MSVC does not support weak symbols; LLVM_ATTRIBUTE_WEAK expands to nothing,
// so the stub in comgr-hotswap-b0a0.cpp becomes a regular definition and
// this file would produce a duplicate-symbol link error (LNK2005). Guard
// the strong override until a proper registration mechanism replaces the
// weak-symbol pattern on Windows (tracked in #2294 / #2285).
#if !defined(_MSC_VER)

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace llvm;

namespace COMGR {
namespace hotswap {

// -- DS stride64 swap table (StringSwitch) ----------------------------------
//
// Maps each 2-address DS mnemonic to its single-address replacement.

static StringRef getDs2AddrReplacement(StringRef Mnemonic) {
  return StringSwitch<StringRef>(Mnemonic)
      .Case("ds_load_2addr_stride64_b32", "ds_load_b32")
      .Case("ds_load_2addr_stride64_b64", "ds_load_b64")
      .Case("ds_store_2addr_stride64_b32", "ds_store_b32")
      .Case("ds_store_2addr_stride64_b64", "ds_store_b64")
      .Case("ds_storexchg_2addr_stride64_rtn_b32", "ds_storexchg_rtn_b32")
      .Case("ds_storexchg_2addr_stride64_rtn_b64", "ds_storexchg_rtn_b64")
      .Default("");
}

// -- MC-layer register helpers ----------------------------------------------
//
// MCRegisterInfo::getName() returns internal LLVM names (e.g. "VGPR0",
// "SGPR4"). We convert these to assembly syntax ("v0", "s4") for instruction
// building. Sub-register iteration returns ALL fragments (including lo16/hi16);
// getDirectSubRegs filters to only scalar 32-bit components.

static std::string toAsmRegName(const MCRegisterInfo &MRI, MCRegister Reg) {
  const char *N = MRI.getName(Reg);
  if (!N)
    return {};
  StringRef Name(N);
  if (Name.starts_with("VGPR") && !Name.contains('_'))
    return ("v" + Name.drop_front(4)).str();
  if (Name.starts_with("SGPR") && !Name.contains('_'))
    return ("s" + Name.drop_front(4)).str();
  return Name.str();
}

static SmallVector<MCRegister, 4> getDirectSubRegs(MCRegister Reg,
                                                   const MCRegisterInfo &MRI) {
  SmallVector<MCRegister, 4> Result;
  for (MCPhysReg Sub : MRI.subregs(Reg)) {
    StringRef Name = MRI.getName(Sub);
    if ((Name.starts_with("VGPR") || Name.starts_with("SGPR")) &&
        !Name.contains("LO") && !Name.contains("HI") && !Name.contains('_'))
      Result.push_back(MCRegister(Sub));
  }
  return Result;
}

// Format a VGPR pair as a range expression: (VGPR0, VGPR1) -> "v[0:1]".
static std::string fmtRegPair(const MCRegisterInfo &MRI, MCRegister Lo,
                              MCRegister Hi) {
  std::string LoName = toAsmRegName(MRI, Lo);
  std::string HiName = toAsmRegName(MRI, Hi);
  char Prefix = LoName[0];
  StringRef LoIdx = StringRef(LoName).drop_front(1);
  StringRef HiIdx = StringRef(HiName).drop_front(1);
  return std::string(1, Prefix) + "[" + LoIdx.str() + ":" + HiIdx.str() + "]";
}

// Format a register operand for assembly. Single registers (VGPR0) produce
// "v0"; register tuples (VGPR0_VGPR1) produce "v[0:1]" by decomposing into
// their scalar sub-registers.
static std::string fmtRegOperand(const MCRegisterInfo &MRI, MCRegister Reg) {
  const char *N = MRI.getName(Reg);
  if (!N)
    return {};
  StringRef Name(N);
  if (!Name.contains('_'))
    return toAsmRegName(MRI, Reg);
  SmallVector<MCRegister, 4> Subs = getDirectSubRegs(Reg, MRI);
  if (Subs.size() < 2)
    return toAsmRegName(MRI, Reg);
  return fmtRegPair(MRI, Subs.front(), Subs.back());
}

// Format an optional byte offset as " offset:N" (empty string when zero).
static std::string fmtOffset(uint32_t Offset) {
  return Offset ? " offset:" + std::to_string(Offset) : "";
}

// -- DS expansion -----------------------------------------------------------
//
// Expands one DS 2-address instruction into two single-address assembly
// strings. The three operation types have different operand layouts:
//   Load:  ds_load_2addr_stride64  vdst_pair, addr, off0, off1
//   Store: ds_store_2addr_stride64 addr, data0, data1, off0, off1
//   Xchg:  ds_storexchg_2addr_stride64_rtn vdst_pair, addr, data0, data1, ...
//
// For b32 operations, destinations are split into individual VGPRs.
// For b64 operations, destinations are split into VGPR pairs (v[X:Y]).

struct DsOperands {
  SmallVector<MCRegister, 4> Regs;
  uint32_t Off0;
  uint32_t Off1;
  bool IsB64;
  const MCRegisterInfo *MRI;
};

// Extract register operands and scaled offsets from a DS 2-address MCInst.
// Offsets are scaled by 64 * element_size (stride64 encoding).
static DsOperands extractDsOperands(const MCInst &Inst, StringRef FromMnem,
                                    const LLVMState &LS) {
  DsOperands Ops;
  Ops.MRI = LS.MRI.get();

  int64_t RawOff0 = 0, RawOff1 = 0;
  unsigned ImmsSeen = 0;
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() != 0)
      Ops.Regs.push_back(MCRegister(Op.getReg()));
    else if (Op.isImm()) {
      if (ImmsSeen == 0)
        RawOff0 = Op.getImm();
      else if (ImmsSeen == 1)
        RawOff1 = Op.getImm();
      ++ImmsSeen;
    }
  }

  uint32_t ElemBytes = FromMnem.contains("_b64") ? 8 : 4;
  uint32_t Scale = 64 * ElemBytes;
  Ops.Off0 = static_cast<uint32_t>(RawOff0) * Scale;
  Ops.Off1 = static_cast<uint32_t>(RawOff1) * Scale;
  Ops.IsB64 = (ElemBytes == 8);
  return Ops;
}

// Split a compound destination register into two formatted destination strings.
// b32: VReg_64 -> ("v0", "v1"); b64: VReg_128 -> ("v[0:1]", "v[2:3]")
static std::pair<std::string, std::string>
splitDstPair(MCRegister CompoundReg, bool IsB64, const MCRegisterInfo &MRI) {
  SmallVector<MCRegister, 4> Subs = getDirectSubRegs(CompoundReg, MRI);
  if (IsB64) {
    if (Subs.size() < 4)
      return {};
    return {fmtRegPair(MRI, Subs[0], Subs[1]),
            fmtRegPair(MRI, Subs[2], Subs[3])};
  }
  if (Subs.size() < 2)
    return {};
  return {toAsmRegName(MRI, Subs[0]), toAsmRegName(MRI, Subs[1])};
}

// Expand a DS 2-address load into two single-address loads (dst, addr).
static std::vector<std::string> expandDs2AddrLoad(const DsOperands &Ops,
                                                  StringRef ToMnem) {
  if (Ops.Regs.size() < 2)
    return {};
  std::pair<std::string, std::string> Dst =
      splitDstPair(Ops.Regs[0], Ops.IsB64, *Ops.MRI);
  if (Dst.first.empty())
    return {};
  std::string Addr = toAsmRegName(*Ops.MRI, Ops.Regs[1]);
  return {
      ToMnem.str() + " " + Dst.first + ", " + Addr + fmtOffset(Ops.Off0),
      ToMnem.str() + " " + Dst.second + ", " + Addr + fmtOffset(Ops.Off1),
  };
}

// Expand a DS 2-address store into two single-address stores (addr, data).
static std::vector<std::string> expandDs2AddrStore(const DsOperands &Ops,
                                                   StringRef ToMnem) {
  if (Ops.Regs.size() < 3)
    return {};
  const MCRegisterInfo &MRI = *Ops.MRI;
  std::string Addr = toAsmRegName(MRI, Ops.Regs[0]);
  std::string Data0 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[1])
                                : toAsmRegName(MRI, Ops.Regs[1]);
  std::string Data1 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[2])
                                : toAsmRegName(MRI, Ops.Regs[2]);
  return {
      ToMnem.str() + " " + Addr + ", " + Data0 + fmtOffset(Ops.Off0),
      ToMnem.str() + " " + Addr + ", " + Data1 + fmtOffset(Ops.Off1),
  };
}

// Expand a DS 2-address exchange into two single-address exchanges
// (dst, addr, data).
static std::vector<std::string> expandDs2AddrXchg(const DsOperands &Ops,
                                                  StringRef ToMnem) {
  if (Ops.Regs.size() < 4)
    return {};
  const MCRegisterInfo &MRI = *Ops.MRI;
  std::pair<std::string, std::string> Dst =
      splitDstPair(Ops.Regs[0], Ops.IsB64, MRI);
  if (Dst.first.empty())
    return {};
  std::string Addr = toAsmRegName(MRI, Ops.Regs[1]);
  std::string Data0 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[2])
                                : toAsmRegName(MRI, Ops.Regs[2]);
  std::string Data1 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[3])
                                : toAsmRegName(MRI, Ops.Regs[3]);
  return {
      ToMnem.str() + " " + Dst.first + ", " + Addr + ", " + Data0 +
          fmtOffset(Ops.Off0),
      ToMnem.str() + " " + Dst.second + ", " + Addr + ", " + Data1 +
          fmtOffset(Ops.Off1),
  };
}

// -- expandDs2Addr ----------------------------------------------------------
//
// Top-level expansion: extracts operands from the decoded MCInst, computes
// scaled offsets, then dispatches to the appropriate layout-specific helper.

static std::vector<std::string> expandDs2Addr(const MCInst &Inst,
                                              StringRef FromMnem,
                                              StringRef ToMnem,
                                              const LLVMState &LS) {
  DsOperands Ops = extractDsOperands(Inst, FromMnem, LS);

  if (FromMnem.starts_with("ds_load"))
    return expandDs2AddrLoad(Ops, ToMnem);
  if (FromMnem.starts_with("ds_storexchg"))
    return expandDs2AddrXchg(Ops, ToMnem);
  if (FromMnem.starts_with("ds_store"))
    return expandDs2AddrStore(Ops, ToMnem);

  log() << "hotswap: error: unrecognized DS mnemonic: " << FromMnem << "\n";
  return {};
}

// -- bumpNextWaitDscnt ------------------------------------------------------
//
// After splitting one DS 2-addr instruction into two, the next s_wait_dscnt
// in the same straight-line block must be incremented by 1 to account for the
// extra outstanding DS operation.
//
// Returns true if a wait was found and bumped, false otherwise.
//
// If the wait is past a branch or join point, we conservatively do nothing:
// the compiler guarantees a straight-line s_wait_dscnt follows each DS op in
// well-formed kernels. If absent (e.g. s_endpgm terminates first), skipping
// the bump is safe — the hardware wait counter saturates harmlessly.

static bool bumpNextWaitDscnt(PatchContext &Ctx, size_t Idx) {
  const MCInstrInfo &MCII = *Ctx.LS.MCII;
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;

  for (size_t I = Idx + 1; I < Ctx.Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Ctx.Decoded[I];
    if (DI.Mnemonic == "<unknown>" || DI.Mnemonic == "<replaced>")
      continue;
    if (DI.Mnemonic == "s_endpgm")
      return false;

    // Stop at any control-flow instruction (branches, jumps, calls) to
    // avoid bumping a wait that belongs to a different execution path.
    const MCInstrDesc &Desc = MCII.get(DI.Inst.getOpcode());
    if (Desc.mayAffectControlFlow(DI.Inst, MRI))
      return false;

    if (DI.Mnemonic != "s_wait_dscnt")
      continue;

    // s_wait_dscnt has a single immediate operand (the wait count) at
    // index 0. Increment it directly.
    MCInst NewInst = DI.Inst;
    MCOperand &Op = NewInst.getOperand(0);
    if (!Op.isImm())
      return false;
    Op.setImm(Op.getImm() + 1);

    SmallVector<char, 8> Bytes;
    SmallVector<MCFixup, 2> Fixups;
    Ctx.LS.MCE->encodeInstruction(NewInst, Bytes, Fixups, *Ctx.LS.STI);

    uint64_t Off = Ctx.Decoded[I].Offset;
    std::memcpy(Ctx.Text + Off, Bytes.data(), Bytes.size());

    Ctx.Decoded[I].Inst = NewInst;
    return true;
  }

  return false;
}

// -- patchDs2AddrStride64 ---------------------------------------------------
//
// Expand one ds_*_2addr_stride64_* instruction into two single-address DS
// instructions. The split doubles the outstanding DS operation count, so
// bumpNextWaitDscnt adjusts the next s_wait_dscnt accordingly.

static bool patchDs2AddrStride64(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  StringRef ToMnem = getDs2AddrReplacement(DI.Mnemonic);
  if (ToMnem.empty())
    return false;
  std::vector<std::string> Expanded =
      expandDs2Addr(DI.Inst, DI.Mnemonic, ToMnem, Ctx.LS);
  if (Expanded.empty()) {
    log() << "hotswap: error: ds_2addr_stride64 expansion failed for: "
          << DI.Mnemonic << "\n";
    return false;
  }

  std::string Combined;
  for (const std::string &Line : Expanded)
    Combined += Line + "\n";
  SmallVector<uint8_t> Bytes = assembleSingleInst(Combined, Ctx.LS);
  if (Bytes.empty()) {
    log() << "hotswap: error: ds_2addr_stride64: assembly failed: " << Combined
          << "\n";
    return false;
  }

  std::vector<uint8_t> Replacement(Bytes.begin(), Bytes.end());
  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
    return false;

  bumpNextWaitDscnt(Ctx, Idx);
  DI.Mnemonic = "<replaced>";
  return true;
}

// -- applyTrampolinePatches -------------------------------------------------
//
// Strong-symbol override. Handles B0 errata that produce replacement code
// larger than the original instruction slot:
//
//   ds_*_2addr_stride64_*  -> split into two single-address DS ops

uint32_t applyTrampolinePatches(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (!getDs2AddrReplacement(Mnem).empty())
    return patchDs2AddrStride64(Ctx, Idx) ? 1 : 0;

  return 0;
}

} // namespace hotswap
} // namespace COMGR

#endif // !defined(_MSC_VER)
