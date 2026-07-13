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
///   - ds_*_2addr_*           : one 8B DS instruction -> two single-address
///     DS instructions. Covers both the stride64 and non-stride64 encodings:
///     A0 requires DS2 addresses to be aligned to the payload size, while
///     B0 dropped that restriction, so a B0-compiled binary may emit a
///     2-address DS instruction with unaligned offsets that silently
///     corrupts LDS on A0. The expansion uses two single-address ops with
///     byte offsets scaled appropriately for each encoding.
///   - tensor_load_to_lds     : clear multicast routing bits in the group
///     descriptor's base SGPR. A0 clears unconditionally; B0 clears only when
///     runtime cluster state reports a non-cluster wave.
///   - cluster_load*          : for cluster-load forms that remain cluster
///     loads after in-place demotion on A0, save M0, clear wg_mask bits
///     [15:0], issue the original load, then restore M0
///   - ds_*_addtid_b32        : compute the LDS address through the ALU and
///     issue a regular ds_*_b32, bypassing the gfx1250 A0 16-bit M0
///     truncation. On B0 the DS unit reads 20 bits of M0; on A0 it reads only
///     16, silently dropping bits [19:16].
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;

namespace COMGR {
namespace hotswap {
namespace {

bool failRequiredPatch(PatchContext &Ctx) {
  Ctx.RequiredPatchFailed = true;
  return false;
}

// -- DS 2-address swap table (StringSwitch) ---------------------------------
//
// Maps each 2-address DS mnemonic to its single-address replacement. Covers
// both encodings -- the stride64 variants pack the index*64*ElemBytes
// stride into each per-operand offset field, while the non-stride64
// variants encode raw index*ElemBytes byte offsets. The single-address
// replacement is the same regardless of encoding; only the offset scale
// differs (see extractDsOperands).

StringRef getDs2AddrReplacement(StringRef Mnemonic) {
  return StringSwitch<StringRef>(Mnemonic)
      .Case("ds_load_2addr_b32", "ds_load_b32")
      .Case("ds_load_2addr_b64", "ds_load_b64")
      .Case("ds_load_2addr_stride64_b32", "ds_load_b32")
      .Case("ds_load_2addr_stride64_b64", "ds_load_b64")
      .Case("ds_store_2addr_b32", "ds_store_b32")
      .Case("ds_store_2addr_b64", "ds_store_b64")
      .Case("ds_store_2addr_stride64_b32", "ds_store_b32")
      .Case("ds_store_2addr_stride64_b64", "ds_store_b64")
      .Case("ds_storexchg_2addr_rtn_b32", "ds_storexchg_rtn_b32")
      .Case("ds_storexchg_2addr_rtn_b64", "ds_storexchg_rtn_b64")
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

std::string toAsmRegName(const MCRegisterInfo &MRI, MCRegister Reg) {
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

bool isM0Reg(MCRegister Reg, const MCRegisterInfo &MRI) {
  const char *N = MRI.getName(Reg);
  return N && StringRef(N).starts_with("M0");
}

SmallVector<MCRegister, 4> getDirectSubRegs(MCRegister Reg,
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
std::string fmtRegPair(const MCRegisterInfo &MRI, MCRegister Lo,
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
std::string fmtRegOperand(const MCRegisterInfo &MRI, MCRegister Reg) {
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
std::string fmtOffset(uint32_t Offset) {
  return Offset ? " offset:" + std::to_string(Offset) : "";
}

// -- DS expansion -----------------------------------------------------------
//
// Expands one DS 2-address instruction into two single-address assembly
// strings. The three operation types have different operand layouts (the
// stride64 and non-stride64 encodings share identical operand layouts;
// only the offset scale differs):
//   Load:  ds_load_2addr[_stride64]  vdst_pair, addr, off0, off1
//   Store: ds_store_2addr[_stride64] addr, data0, data1, off0, off1
//   Xchg:  ds_storexchg_2addr[_stride64]_rtn vdst_pair, addr, data0, data1, ...
//
// For b32 operations, destinations are split into individual VGPRs.
// For b64 operations, destinations are split into VGPR pairs (v[X:Y]).

// Maximum byte offset encodable in a single-address DS instruction's
// 16-bit immediate offset field on gfx1250. The replacement we emit uses
// this field directly, so any scaled byte offset that exceeds it cannot
// be represented and the patch must be skipped.
constexpr uint32_t Ds1AddrOffsetMax = 0xFFFF;

struct DsOperands {
  SmallVector<MCRegister, 4> Regs;
  uint32_t Off0 = 0;
  uint32_t Off1 = 0;
  bool IsB64 = false;
  const MCRegisterInfo *MRI = nullptr;
};

// Extract register operands and scaled offsets from a DS 2-address MCInst.
// The per-operand immediate fields hold dword indices that the hardware
// scales differently for the two encodings: the non-stride64 forms encode
// (index * ElemBytes) byte offsets, while the stride64 forms encode
// (index * 64 * ElemBytes) byte offsets. The replacement single-address
// instructions take byte offsets directly, so we materialise the scaled
// value here once and let the layout-specific helpers consume it.
//
// Range check: the stride64 b64 encoding can scale a raw 8-bit index up to
// 255 * 64 * 8 = 130560 bytes, which overflows the single-address 16-bit
// offset field (max 0xFFFF = 65535). When that happens the patch is not
// representable in this expansion shape; std::nullopt signals the failure
// to the caller, which leaves the original (broken-on-A0) instruction in
// place rather than emitting a silently-truncated replacement.
std::optional<DsOperands>
extractDsOperands(const MCInst &Inst, StringRef FromMnem, const LLVMState &LS) {
  DsOperands Ops;
  Ops.MRI = LS.MRI.get();

  int64_t RawOff0 = 0, RawOff1 = 0;
  unsigned ImmsSeen = 0;
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg())
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
  uint32_t Scale = FromMnem.contains("_stride64_") ? 64 * ElemBytes : ElemBytes;
  // Compute scaled offsets in 64-bit so an oversize stride64_b64 index
  // does not silently wrap when assigned to Off*.
  uint64_t Scaled0 = static_cast<uint64_t>(RawOff0) * Scale;
  uint64_t Scaled1 = static_cast<uint64_t>(RawOff1) * Scale;
  if (Scaled0 > Ds1AddrOffsetMax || Scaled1 > Ds1AddrOffsetMax) {
    log() << "hotswap: error: " << FromMnem
          << " scaled offsets exceed the single-address DS 16-bit field "
             "(off0=raw "
          << RawOff0 << " * scale " << Scale << " = " << Scaled0
          << ", off1=raw " << RawOff1 << " * scale " << Scale << " = "
          << Scaled1 << ", max " << Ds1AddrOffsetMax
          << "); required A0 rewrite cannot continue\n";
    return std::nullopt;
  }
  Ops.Off0 = static_cast<uint32_t>(Scaled0);
  Ops.Off1 = static_cast<uint32_t>(Scaled1);
  Ops.IsB64 = (ElemBytes == 8);
  return Ops;
}

// Split a compound destination register into two formatted destination strings.
// b32: VReg_64 -> ("v0", "v1"); b64: VReg_128 -> ("v[0:1]", "v[2:3]")
std::optional<std::pair<std::string, std::string>>
splitDstPair(MCRegister CompoundReg, bool IsB64, const MCRegisterInfo &MRI) {
  SmallVector<MCRegister, 4> Subs = getDirectSubRegs(CompoundReg, MRI);
  if (IsB64) {
    if (Subs.size() < 4) {
      log() << "hotswap: error: DS b64 destination " << MRI.getName(CompoundReg)
            << " has " << Subs.size()
            << " direct subregisters; expected at least 4\n";
      return std::nullopt;
    }
    return std::pair<std::string, std::string>{
        fmtRegPair(MRI, Subs[0], Subs[1]), fmtRegPair(MRI, Subs[2], Subs[3])};
  }
  if (Subs.size() < 2) {
    log() << "hotswap: error: DS b32 destination " << MRI.getName(CompoundReg)
          << " has " << Subs.size()
          << " direct subregisters; expected at least 2\n";
    return std::nullopt;
  }
  return std::pair<std::string, std::string>{toAsmRegName(MRI, Subs[0]),
                                             toAsmRegName(MRI, Subs[1])};
}

// Expand a DS 2-address load into two single-address loads (dst, addr).
std::optional<std::vector<std::string>> expandDs2AddrLoad(const DsOperands &Ops,
                                                          StringRef ToMnem) {
  if (Ops.Regs.size() < 2) {
    log() << "hotswap: error: " << ToMnem << " expansion found "
          << Ops.Regs.size() << " register operands; expected at least 2\n";
    return std::nullopt;
  }
  std::optional<std::pair<std::string, std::string>> Dst =
      splitDstPair(Ops.Regs[0], Ops.IsB64, *Ops.MRI);
  if (!Dst)
    return std::nullopt;
  std::string Addr = toAsmRegName(*Ops.MRI, Ops.Regs[1]);
  std::string First =
      ToMnem.str() + " " + Dst->first + ", " + Addr + fmtOffset(Ops.Off0);
  std::string Second =
      ToMnem.str() + " " + Dst->second + ", " + Addr + fmtOffset(Ops.Off1);

  // A compound DS load reads its address once before writing either half of
  // the destination. After splitting, the first single-address load must not
  // overwrite the address needed by the second. If the address overlaps the
  // first destination half, issue the independent second half first and put
  // the self-overlapping load last. (If it overlaps the second half, the
  // natural order is already safe.)
  SmallVector<MCRegister, 4> DstSubs = getDirectSubRegs(Ops.Regs[0], *Ops.MRI);
  const unsigned FirstHalfWidth = Ops.IsB64 ? 2 : 1;
  bool AddrOverlapsFirst = llvm::any_of(
      ArrayRef(DstSubs).take_front(FirstHalfWidth),
      [&](MCRegister Reg) { return Ops.MRI->regsOverlap(Reg, Ops.Regs[1]); });
  if (AddrOverlapsFirst)
    return std::vector<std::string>{std::move(Second), std::move(First)};
  return std::vector<std::string>{std::move(First), std::move(Second)};
}

// Expand a DS 2-address store into two single-address stores (addr, data).
std::optional<std::vector<std::string>>
expandDs2AddrStore(const DsOperands &Ops, StringRef ToMnem) {
  if (Ops.Regs.size() < 3) {
    log() << "hotswap: error: " << ToMnem << " expansion found "
          << Ops.Regs.size() << " register operands; expected at least 3\n";
    return std::nullopt;
  }
  const MCRegisterInfo &MRI = *Ops.MRI;
  std::string Addr = toAsmRegName(MRI, Ops.Regs[0]);
  std::string Data0 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[1])
                                : toAsmRegName(MRI, Ops.Regs[1]);
  std::string Data1 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[2])
                                : toAsmRegName(MRI, Ops.Regs[2]);
  return std::vector<std::string>{
      ToMnem.str() + " " + Addr + ", " + Data0 + fmtOffset(Ops.Off0),
      ToMnem.str() + " " + Addr + ", " + Data1 + fmtOffset(Ops.Off1),
  };
}

bool registerSliceOverlaps(ArrayRef<MCRegister> Registers, unsigned Begin,
                           unsigned Count, MCRegister Reg,
                           const MCRegisterInfo &MRI) {
  return llvm::any_of(Registers.slice(Begin, Count), [&](MCRegister DstReg) {
    return MRI.regsOverlap(DstReg, Reg);
  });
}

// Expand a DS 2-address exchange into two single-address exchanges
// (dst, addr, data).
std::optional<std::vector<std::string>> expandDs2AddrXchg(const DsOperands &Ops,
                                                          StringRef ToMnem) {
  if (Ops.Regs.size() < 4) {
    log() << "hotswap: error: " << ToMnem << " expansion found "
          << Ops.Regs.size() << " register operands; expected at least 4\n";
    return std::nullopt;
  }
  const MCRegisterInfo &MRI = *Ops.MRI;
  std::optional<std::pair<std::string, std::string>> Dst =
      splitDstPair(Ops.Regs[0], Ops.IsB64, MRI);
  if (!Dst)
    return std::nullopt;
  std::string Addr = toAsmRegName(MRI, Ops.Regs[1]);
  std::string Data0 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[2])
                                : toAsmRegName(MRI, Ops.Regs[2]);
  std::string Data1 = Ops.IsB64 ? fmtRegOperand(MRI, Ops.Regs[3])
                                : toAsmRegName(MRI, Ops.Regs[3]);
  std::string First = ToMnem.str() + " " + Dst->first + ", " + Addr + ", " +
                      Data0 + fmtOffset(Ops.Off0);
  std::string Second = ToMnem.str() + " " + Dst->second + ", " + Addr + ", " +
                       Data1 + fmtOffset(Ops.Off1);

  SmallVector<MCRegister, 4> DstSubs = getDirectSubRegs(Ops.Regs[0], MRI);
  const unsigned HalfWidth = Ops.IsB64 ? 2 : 1;
  if (DstSubs.size() < 2 * HalfWidth) {
    log() << "hotswap: error: " << ToMnem << " destination "
          << MRI.getName(Ops.Regs[0]) << " has " << DstSubs.size()
          << " direct subregisters; expected at least " << 2 * HalfWidth
          << "\n";
    return std::nullopt;
  }

  // Op0 writes the first destination half and op1 still needs addr + data1;
  // op1 writes the second half and op0 still needs addr + data0. Pick the safe
  // order when only one direction has a dependency. If both directions do,
  // neither ordering preserves the compound instruction's read-before-write
  // semantics without a scratch VGPR, so decline the rewrite.
  const bool FirstClobbersSecond =
      registerSliceOverlaps(DstSubs, 0, HalfWidth, Ops.Regs[1], MRI) ||
      registerSliceOverlaps(DstSubs, 0, HalfWidth, Ops.Regs[3], MRI);
  const bool SecondClobbersFirst =
      registerSliceOverlaps(DstSubs, HalfWidth, HalfWidth, Ops.Regs[1], MRI) ||
      registerSliceOverlaps(DstSubs, HalfWidth, HalfWidth, Ops.Regs[2], MRI);
  if (FirstClobbersSecond && SecondClobbersFirst) {
    log() << "hotswap: error: ds_storexchg_2addr has cyclic "
             "destination/source overlap and cannot be split without scratch "
             "VGPRs\n";
    return std::nullopt;
  }
  if (FirstClobbersSecond)
    return std::vector<std::string>{std::move(Second), std::move(First)};
  return std::vector<std::string>{std::move(First), std::move(Second)};
}

// -- expandDs2Addr ----------------------------------------------------------
//
// Top-level expansion: extracts operands from the decoded MCInst, computes
// scaled offsets, then dispatches to the appropriate layout-specific helper.

std::optional<std::vector<std::string>> expandDs2AddrImpl(const MCInst &Inst,
                                                          StringRef FromMnem,
                                                          StringRef ToMnem,
                                                          const LLVMState &LS) {
  std::optional<DsOperands> Ops = extractDsOperands(Inst, FromMnem, LS);
  if (!Ops)
    return std::nullopt;

  // Use the trailing underscore so the three prefixes are disjoint
  // ("ds_load_", "ds_store_", "ds_storexchg_"); without it "ds_store" is a
  // prefix of "ds_storexchg" and the dispatch order would matter.
  if (FromMnem.starts_with("ds_load_"))
    return expandDs2AddrLoad(*Ops, ToMnem);
  if (FromMnem.starts_with("ds_storexchg_"))
    return expandDs2AddrXchg(*Ops, ToMnem);
  if (FromMnem.starts_with("ds_store_"))
    return expandDs2AddrStore(*Ops, ToMnem);

  log() << "hotswap: error: unrecognized DS mnemonic: " << FromMnem << "\n";
  return std::nullopt;
}

// -- patchDs2Addr -----------------------------------------------------------
//
// Expand one ds_*_2addr_* instruction (stride64 or non-stride64) into two
// single-address DS instructions, followed by an s_wait_dscnt 0 drain so both
// halves are guaranteed complete before any downstream DS consumer. Splitting
// one DS instruction into two perturbs the outstanding-DS instruction count
// that later s_wait_dscnt immediates encode; the local drain sidesteps that
// entirely (see the rationale in the body below).

bool patchDs2Addr(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  StringRef ToMnem = getDs2AddrReplacement(DI.Mnemonic);
  if (ToMnem.empty())
    return false;
  std::optional<std::vector<std::string>> Expanded =
      expandDs2AddrImpl(DI.Inst, DI.Mnemonic, ToMnem, Ctx.LS);
  if (!Expanded)
    return failRequiredPatch(Ctx);

  std::string Combined;
  for (const std::string &Line : *Expanded)
    Combined += Line + "\n";
  // Drain the DS counter right after the split pair so both halves are
  // guaranteed complete before any downstream consumer. The original code
  // tracked completion of the single 2-addr instruction via a later
  // s_wait_dscnt whose immediate counts outstanding DS *instructions*;
  // splitting one instruction into two perturbs that count. Adjusting the
  // downstream wait by +1 (the previous bumpNextWaitDscnt approach) relaxes
  // the wait (s_wait_dscnt K stalls until outstanding <= K, so a larger K
  // waits for FEWER ops), which lets a consumer read the second half's LDS
  // slot before it lands -- observed as NaN in MIOpen layernormbfp16. A
  // local drain is unconditionally correct; a precise per-wait dataflow
  // recomputation is the eventual optimization (tracked separately).
  Combined += "s_wait_dscnt 0\n";
  SmallVector<uint8_t> Bytes = assembleSingleInst(Combined, Ctx.LS);
  if (Bytes.empty()) {
    log() << "hotswap: error: ds_2addr: assembly failed: " << Combined << "\n";
    return failRequiredPatch(Ctx);
  }

  SmallVector<uint8_t> Replacement(Bytes.begin(), Bytes.end());
  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
    return failRequiredPatch(Ctx);

  DI.Mnemonic = "<replaced>";
  return true;
}

// -- getDescriptorBaseSgpr --------------------------------------------------
//
// Extract the base SGPR MCRegister from the second operand of a
// tensor_load_to_lds instruction. The second operand is an 8-SGPR group
// descriptor (SReg_256); we need its first sub-register for the
// s_pack_hh_b32_b16 fix.

MCRegister getDescriptorBaseSgpr(const MCInst &Inst,
                                 const MCRegisterInfo &MRI) {
  if (Inst.getNumOperands() < 2 || !Inst.getOperand(1).isReg())
    return MCRegister();
  MCRegister Tuple = MCRegister(Inst.getOperand(1).getReg());
  SmallVector<MCRegister, 4> Subs = getDirectSubRegs(Tuple, MRI);
  return Subs.empty() ? MCRegister() : Subs[0];
}

std::optional<unsigned> getSgprIndex(MCRegister Reg,
                                     const MCRegisterInfo &MRI) {
  const char *N = MRI.getName(Reg);
  if (!N)
    return std::nullopt;
  StringRef Name(N);
  if (!Name.starts_with("SGPR") || Name.contains('_'))
    return std::nullopt;
  unsigned Index = 0;
  if (Name.drop_front(4).getAsInteger(10, Index))
    return std::nullopt;
  return Index;
}

SmallVector<unsigned, 8> getDescriptorSgprIndices(const MCInst &Inst,
                                                  const MCRegisterInfo &MRI) {
  SmallVector<unsigned, 8> Result;
  if (Inst.getNumOperands() < 2 || !Inst.getOperand(1).isReg())
    return Result;

  MCRegister Tuple = MCRegister(Inst.getOperand(1).getReg());
  for (MCRegister Sub : getDirectSubRegs(Tuple, MRI)) {
    if (std::optional<unsigned> Index = getSgprIndex(Sub, MRI))
      Result.push_back(*Index);
  }
  return Result;
}

SmallVector<unsigned, 8> getSgprOperandIndices(const MCInst &Inst,
                                               const MCRegisterInfo &MRI) {
  SmallVector<unsigned, 8> Result;
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (!Op.isReg() || !Op.getReg())
      continue;

    MCRegister Reg = MCRegister(Op.getReg());
    if (std::optional<unsigned> Index = getSgprIndex(Reg, MRI)) {
      Result.push_back(*Index);
      continue;
    }

    for (MCRegister Sub : getDirectSubRegs(Reg, MRI)) {
      if (std::optional<unsigned> Index = getSgprIndex(Sub, MRI))
        Result.push_back(*Index);
    }
  }
  return Result;
}

bool isAlreadyTensorMaskPatched(const PatchContext &Ctx, size_t Idx,
                                MCRegister BaseMCReg) {
  if (Idx == 0)
    return false;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  const InternalDecodedInst &Prev = Ctx.Decoded[Idx - 1];
  const MCInst &PI = Prev.Inst;
  if (Prev.Mnemonic != "s_pack_hh_b32_b16" || PI.getNumOperands() < 3)
    return false;
  if (!PI.getOperand(0).isReg() ||
      !MRI.regsOverlap(PI.getOperand(0).getReg(), BaseMCReg.id()))
    return false;
  return PI.getOperand(1).isImm() && PI.getOperand(1).getImm() == 0;
}

bool isSccReg(MCRegister Reg, const MCRegisterInfo &MRI) {
  const char *N = MRI.getName(Reg);
  return N && StringRef(N) == "SCC";
}

bool hasSccReg(ArrayRef<MCPhysReg> Regs, const MCRegisterInfo &MRI) {
  for (MCPhysReg Reg : Regs) {
    if (isSccReg(MCRegister(Reg), MRI))
      return true;
  }
  return false;
}

bool explicitDefsScc(const MCInst &Inst, const MCInstrDesc &Desc,
                     const MCRegisterInfo &MRI) {
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), Inst.getNumOperands());
  for (unsigned I = 0; I < NumDefs; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() && isSccReg(MCRegister(Op.getReg()), MRI))
      return true;
  }
  return false;
}

bool explicitUsesScc(const MCInst &Inst, const MCInstrDesc &Desc,
                     const MCRegisterInfo &MRI) {
  unsigned NumDefs =
      std::min<unsigned>(Desc.getNumDefs(), Inst.getNumOperands());
  for (unsigned I = NumDefs, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() && isSccReg(MCRegister(Op.getReg()), MRI))
      return true;
  }
  return false;
}

bool instReadsScc(const MCInst &Inst, const MCInstrDesc &Desc,
                  const MCRegisterInfo &MRI) {
  return explicitUsesScc(Inst, Desc, MRI) ||
         hasSccReg(Desc.implicit_uses(), MRI);
}

bool instWritesScc(const MCInst &Inst, const MCInstrDesc &Desc,
                   const MCRegisterInfo &MRI) {
  return explicitDefsScc(Inst, Desc, MRI) ||
         hasSccReg(Desc.implicit_defs(), MRI);
}

// -- isSgprLiveAfter --------------------------------------------------------
//
// Conservative forward-scan heuristic. Returns true if the given SGPR
// (identified by its MCRegister) is used before being redefined in the
// instruction stream following Idx. Conservatively returns true on
// control-flow-affecting instructions or end of stream.

bool isSgprLiveAfter(const PatchContext &Ctx, size_t Idx,
                     MCRegister SgprMCReg) {
  if (!SgprMCReg.isValid())
    return true;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  const MCInstrInfo &MCII = *Ctx.LS.MCII;

  for (size_t I = Idx + 1; I < Ctx.Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Ctx.Decoded[I];
    if (DI.Mnemonic == "<unknown>" || DI.Mnemonic == "<replaced>")
      continue;

    const MCInst &Inst = DI.Inst;
    const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());

    if (DI.Mnemonic == "s_endpgm")
      return false;

    if (Desc.mayAffectControlFlow(Inst, MRI))
      return true;

    unsigned NumDefs = Desc.getNumDefs();
    auto RegInRange = [&](ArrayRef<MCOperand> Ops) {
      for (const MCOperand &Op : Ops) {
        if (!Op.isReg() || !Op.getReg())
          continue;
        if (MRI.regsOverlap(Op.getReg(), SgprMCReg.id()))
          return true;
      }
      return false;
    };
    ArrayRef<MCOperand> Operands = Inst.getOperands();
    ArrayRef<MCOperand> Defs = Operands.slice(0, NumDefs);
    ArrayRef<MCOperand> Uses = Operands.slice(NumDefs);
    if (RegInRange(Uses))
      return true;
    if (RegInRange(Defs))
      return false;
  }

  return true;
}

// -- isSccLiveAfter ---------------------------------------------------------
//
// Conservative forward-scan heuristic for the scalar condition code. Returns
// true if SCC is read before the next instruction that defines SCC. Returns
// true at control-flow boundaries because the linear stream alone cannot prove
// the branch target does not consume the incoming SCC value.

bool isSccLiveAfter(const PatchContext &Ctx, size_t Idx) {
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  const MCInstrInfo &MCII = *Ctx.LS.MCII;

  for (size_t I = Idx + 1; I < Ctx.Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Ctx.Decoded[I];
    if (DI.Mnemonic == "<unknown>" || DI.Mnemonic == "<replaced>")
      continue;

    const MCInst &Inst = DI.Inst;
    const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());

    if (DI.Mnemonic == "s_endpgm")
      return false;

    if (instReadsScc(Inst, Desc, MRI))
      return true;
    if (instWritesScc(Inst, Desc, MRI))
      return false;
    if (Desc.mayAffectControlFlow(Inst, MRI))
      return true;
  }

  return true;
}

// -- scratch-VGPR allocation ------------------------------------------------
//
// Allocation is split into a pure try-step and a commit-step so callers can
// decide a scratch VGPR before assembling/emitting the patch and then only
// charge the kernel descriptor for the extra VGPRs once the patch is known
// to have landed. Bumping KernelPatchStats inside the try-step would leave
// orphan VGPR reservations in the kernel descriptor whenever assembly or
// emission failed downstream.

struct ScratchAlloc {
  unsigned Vgpr = 0;
  std::string KernelName;
  unsigned ExtraVgprsNeeded = 0;
};

std::optional<ScratchAlloc> tryAllocScratchVgpr(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  // findKernelAtAddress matches against symbol virtual addresses, so bias the
  // .text-relative DI.Offset by textAddr() (matching the other patches). A
  // bare offset misses when .text has a non-zero sh_addr, leaving KdVgprs ==
  // 0 and handing the allocator a live register.
  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  unsigned KdVgprs = 0;
  if (std::optional<unsigned> Opt =
          Ctx.Elf.getKernelVgprCount(KernelName, Ctx.Config.VgprGranuleSize))
    KdVgprs = *Opt;

  VgprAllocator Alloc(Ctx.Liveness.LiveBefore[Idx], KdVgprs,
                      Ctx.Config.MaxVgprs);
  std::optional<unsigned> ScratchOpt = Alloc.alloc();
  if (!ScratchOpt)
    return std::nullopt;

  ScratchAlloc Out;
  Out.Vgpr = *ScratchOpt;
  Out.KernelName = std::move(KernelName);
  Out.ExtraVgprsNeeded = Alloc.extraVgprsNeeded();
  return Out;
}

// Apply the kernel-descriptor accounting for a scratch VGPR. Must be called
// only after the corresponding patch has been emitted successfully.
void commitScratchVgpr(PatchContext &Ctx, const ScratchAlloc &Alloc) {
  if (Alloc.ExtraVgprsNeeded == 0 || Alloc.KernelName.empty())
    return;
  KernelPatchStats &Stats = Ctx.KernelStats[Alloc.KernelName];
  Stats.ExtraVgprs = std::max(Stats.ExtraVgprs, Alloc.ExtraVgprsNeeded);
  Stats.ScratchAboveKd += Alloc.ExtraVgprsNeeded;
}

// -- scratch-SGPR allocation ------------------------------------------------
//
// Allocate a scratch SGPR above the kernel's .sgpr_count. Those SGPRs are
// never used by the kernel, and GFX10+ waves always have the full SGPR file
// (no KD bump needed), so unlike VGPRs this needs no liveness. Same strategy
// the E5M3 patch uses.
//
// TODO: the E5M3 patch open-codes this same scratch-SGPR reservation. Hoist
// SgprScratchAlloc / tryAllocScratchSgpr / commitScratchSgpr into shared
// infrastructure both patches call, rather than duplicating it.

struct SgprScratchAlloc {
  unsigned Sgpr = 0;
  std::string KernelName;
  unsigned ExtraSgprsNeeded = 0;
};

std::optional<SgprScratchAlloc>
tryAllocScratchSgpr(PatchContext &Ctx, size_t Idx,
                    ArrayRef<unsigned> ExcludedSgprs = {}) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdSgprs = Ctx.Elf.getKernelSgprCount(KernelName);
  unsigned SgprKdCount = KdSgprs.value_or(Ctx.Config.MaxSgprs);

  SgprAllocator Alloc(SgprKdCount, Ctx.Config.MaxSgprs);
  while (std::optional<unsigned> S = Alloc.alloc()) {
    if (llvm::is_contained(ExcludedSgprs, *S))
      continue;

    SgprScratchAlloc Out;
    Out.Sgpr = *S;
    Out.KernelName = std::move(KernelName);
    Out.ExtraSgprsNeeded = Alloc.extraSgprsNeeded();
    return Out;
  }

  return std::nullopt;
}

void commitScratchSgpr(PatchContext &Ctx, const SgprScratchAlloc &Alloc) {
  if (Alloc.ExtraSgprsNeeded == 0 || Alloc.KernelName.empty())
    return;
  KernelPatchStats &Stats = Ctx.KernelStats[Alloc.KernelName];
  Stats.ExtraSgprs = std::max(Stats.ExtraSgprs, Alloc.ExtraSgprsNeeded);
}

// -- patchTensorLoadToLdsA0 -------------------------------------------------
//
// Prepend s_pack_hh_b32_b16 to clear multicast routing bits in the group
// descriptor's base SGPR. Preserve the architectural SGPR value when it is live
// after the tensor instruction: the descriptor is a source operand, so the
// rewrite must not make its temporary normalization visible to later code.

bool patchTensorLoadToLdsA0(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;

  MCRegister BaseMCReg = getDescriptorBaseSgpr(DI.Inst, MRI);
  if (!BaseMCReg.isValid()) {
    log() << "hotswap: error: tensor_load_to_lds: could not extract descriptor "
             "base register\n";
    return failRequiredPatch(Ctx);
  }

  if (isAlreadyTensorMaskPatched(Ctx, Idx, BaseMCReg))
    return false;

  std::string BaseSreg = toAsmRegName(MRI, BaseMCReg);

  std::string PackAsm = "s_pack_hh_b32_b16 " + BaseSreg + ", 0, " + BaseSreg;
  SmallVector<uint8_t> PackBytes = assembleSingleInst(PackAsm, Ctx.LS);
  if (PackBytes.empty()) {
    log() << "hotswap: tensor_load_to_lds pack: assembly failed: " << PackAsm
          << "\n";
    return failRequiredPatch(Ctx);
  }

  bool SgprLive = isSgprLiveAfter(Ctx, Idx, BaseMCReg);
  const uint8_t *OrigInst = Ctx.Text + DI.Offset;

  if (SgprLive) {
    std::optional<SafeSgprScratchBlock> Scratch =
        findSafeSgprScratchBlock(Ctx, DI.Offset, /*Count=*/1, /*Alignment=*/1,
                                 "tensor_load_to_lds descriptor save");
    if (!Scratch) {
      log() << "hotswap: error: tensor_load_to_lds: no scratch SGPR "
               "available\n";
      return failRequiredPatch(Ctx);
    }

    std::string ScratchName = "s" + std::to_string(Scratch->Base);
    SmallVector<uint8_t> Save = assembleSingleInst(
        "s_mov_b32 " + ScratchName + ", " + BaseSreg, Ctx.LS);
    SmallVector<uint8_t> Restore = assembleSingleInst(
        "s_mov_b32 " + BaseSreg + ", " + ScratchName, Ctx.LS);
    if (Save.empty() || Restore.empty()) {
      log() << "hotswap: error: tensor_load_to_lds: failed to assemble "
               "descriptor save/restore through "
            << ScratchName << "\n";
      return failRequiredPatch(Ctx);
    }

    SmallVector<uint8_t> Replacement;
    Replacement.append(Save.begin(), Save.end());
    Replacement.append(PackBytes.begin(), PackBytes.end());
    Replacement.append(OrigInst, OrigInst + DI.Size);
    Replacement.append(Restore.begin(), Restore.end());
    if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
      return failRequiredPatch(Ctx);

    if (!commitSafeSgprScratchBlock(Ctx, DI.Offset, *Scratch,
                                    "tensor_load_to_lds descriptor save"))
      return failRequiredPatch(Ctx);
    log() << "hotswap: tensor_load_to_lds: " << BaseSreg
          << " live, save/restore via " << ScratchName << "\n";
  } else {
    SmallVector<uint8_t> Replacement;
    Replacement.append(PackBytes.begin(), PackBytes.end());
    Replacement.append(OrigInst, OrigInst + DI.Size);
    if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
      return failRequiredPatch(Ctx);

    log() << "hotswap: tensor_load_to_lds: " << BaseSreg
          << " dead, no save/restore needed\n";
  }

  Ctx.RequiredPatchApplied = true;
  DI.Mnemonic = "<replaced>";
  return true;
}

// -- Cluster/TDM mask helpers ------------------------------------------------
//
// In-place patching demotes off-form cluster_load* instructions to
// global_load* first. Any cluster_load* that reaches this trampoline pass is
// still a real cluster load on A0 and must see M0.wg_mask[15:0] cleared. B0
// does not need the cluster-load M0 workaround; its hotswap mask rule applies
// only to tensor_load_to_lds when the wave is effectively non-cluster.

// MI400 SPG section 3.4: SQ_WAVE_IB_STS2.CLUSTER_ID is bits [9:6].
constexpr unsigned IbSts2ClusterIdOffset = 6;
constexpr unsigned IbSts2ClusterIdWidth = 4;

bool isClusterLoad(StringRef Mnemonic) {
  return StringSwitch<bool>(Mnemonic)
      .Case("cluster_load_b32", true)
      .Case("cluster_load_b64", true)
      .Case("cluster_load_b128", true)
      .Case("cluster_load_async_to_lds_b8", true)
      .Case("cluster_load_async_to_lds_b32", true)
      .Case("cluster_load_async_to_lds_b64", true)
      .Case("cluster_load_async_to_lds_b128", true)
      .Default(false);
}

bool operandIsM0(const MCInst &Inst, const MCRegisterInfo &MRI,
                 unsigned OperandIdx) {
  if (OperandIdx >= Inst.getNumOperands())
    return false;
  const MCOperand &Op = Inst.getOperand(OperandIdx);
  return Op.isReg() && isM0Reg(MCRegister(Op.getReg()), MRI);
}

bool isAlreadyClusterMaskPatched(const PatchContext &Ctx, size_t Idx) {
  if (Idx == 0)
    return false;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  const InternalDecodedInst &Prev = Ctx.Decoded[Idx - 1];
  const MCInst &PI = Prev.Inst;

  if (Prev.Mnemonic == "s_pack_hh_b32_b16") {
    if (PI.getNumOperands() < 3 || !operandIsM0(PI, MRI, 0))
      return false;
    if (!PI.getOperand(1).isImm() || PI.getOperand(1).getImm() != 0)
      return false;
    return operandIsM0(PI, MRI, 2);
  }

  if (Prev.Mnemonic != "s_and_b32" || PI.getNumOperands() < 3 ||
      !operandIsM0(PI, MRI, 0))
    return false;

  for (unsigned OpIdx = 1; OpIdx < PI.getNumOperands(); ++OpIdx) {
    if (operandIsM0(PI, MRI, OpIdx))
      return true;
  }
  return false;
}

std::optional<uint64_t> getFlatClusterSize(const KernelClusterDims &Dims,
                                           StringRef KernelName) {
  if (Dims.X == 0 && Dims.Y == 0 && Dims.Z == 0)
    return 0;

  if (Dims.X == 0 || Dims.Y == 0 || Dims.Z == 0) {
    log() << "hotswap: error: .cluster_dims for '" << KernelName
          << "' contains a zero dimension in a nonzero fixed cluster ("
          << Dims.X << ", " << Dims.Y << ", " << Dims.Z
          << "); falling back to dynamic cluster-id check\n";
    return std::nullopt;
  }

  uint64_t Flat = Dims.X;
  if (Dims.Y > std::numeric_limits<uint64_t>::max() / Flat) {
    log() << "hotswap: error: .cluster_dims for '" << KernelName
          << "' overflows uint64_t; falling back to dynamic cluster-id check\n";
    return std::nullopt;
  }
  Flat *= Dims.Y;
  if (Dims.Z > std::numeric_limits<uint64_t>::max() / Flat) {
    log() << "hotswap: error: .cluster_dims for '" << KernelName
          << "' overflows uint64_t; falling back to dynamic cluster-id check\n";
    return std::nullopt;
  }
  Flat *= Dims.Z;
  return Flat;
}

bool appendAsmBytes(SmallVectorImpl<uint8_t> &Out, StringRef Asm,
                    const LLVMState &LS, StringRef Context) {
  SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, LS);
  if (Bytes.empty()) {
    log() << "hotswap: error: " << Context << ": assembly failed: " << Asm
          << "\n";
    return false;
  }
  Out.append(Bytes.begin(), Bytes.end());
  return true;
}

bool appendRequiredAsm(PatchContext &Ctx, SmallVectorImpl<uint8_t> &Out,
                       StringRef Asm, StringRef Context) {
  if (appendAsmBytes(Out, Asm, Ctx.LS, Context))
    return true;
  return failRequiredPatch(Ctx);
}

bool hasKnownNonClusterDispatch(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];
  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  std::optional<KernelClusterDims> ClusterDims =
      Ctx.Elf.getKernelClusterDims(KernelName);
  if (!ClusterDims)
    return false;

  std::optional<uint64_t> Flat = getFlatClusterSize(*ClusterDims, KernelName);
  return Flat && *Flat <= 1;
}

bool patchTensorLoadToLdsB0(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;

  MCRegister BaseMCReg = getDescriptorBaseSgpr(DI.Inst, MRI);
  if (!BaseMCReg.isValid()) {
    log() << "hotswap: error: tensor_load_to_lds: could not extract descriptor "
             "base register\n";
    return failRequiredPatch(Ctx);
  }

  if (isAlreadyTensorMaskPatched(Ctx, Idx, BaseMCReg))
    return false;

  if (hasKnownNonClusterDispatch(Ctx, Idx))
    return patchTensorLoadToLdsA0(Ctx, Idx);

  SmallVector<unsigned, 8> DescriptorSgprs =
      getDescriptorSgprIndices(DI.Inst, MRI);
  std::optional<SgprScratchAlloc> ScratchSgpr =
      tryAllocScratchSgpr(Ctx, Idx, DescriptorSgprs);
  if (!ScratchSgpr) {
    log() << "hotswap: error: tensor_load_to_lds: no scratch SGPR available "
             "for B0 cluster-id check\n";
    return failRequiredPatch(Ctx);
  }

  bool SccLive = isSccLiveAfter(Ctx, Idx);
  std::optional<SgprScratchAlloc> SccScratchSgpr;
  SmallVector<unsigned, 9> SccExcludedSgprs;
  SccExcludedSgprs.append(DescriptorSgprs.begin(), DescriptorSgprs.end());
  SccExcludedSgprs.push_back(ScratchSgpr->Sgpr);
  if (SccLive) {
    SccScratchSgpr = tryAllocScratchSgpr(Ctx, Idx, SccExcludedSgprs);
    if (!SccScratchSgpr) {
      log() << "hotswap: error: tensor_load_to_lds: no scratch SGPR available "
               "to preserve SCC for B0 cluster-id check\n";
      return failRequiredPatch(Ctx);
    }
  }

  std::string BaseSreg = toAsmRegName(MRI, BaseMCReg);
  std::string S = "s" + std::to_string(ScratchSgpr->Sgpr);
  std::string SccS =
      SccScratchSgpr ? "s" + std::to_string(SccScratchSgpr->Sgpr) : "";
  std::string Context =
      "tensor_load_to_lds B0 mask at 0x" + utohexstr(DI.Offset);

  SmallVector<uint8_t> Prefix;
  std::string ReadClusterIdAsm = "s_getreg_b32 " + BaseSreg +
                                 ", hwreg(HW_REG_IB_STS2, " +
                                 std::to_string(IbSts2ClusterIdOffset) + ", " +
                                 std::to_string(IbSts2ClusterIdWidth) + ")";
  if (!appendRequiredAsm(Ctx, Prefix, "s_mov_b32 " + S + ", " + BaseSreg,
                         Context))
    return false;

  if (SccLive) {
    if (!appendRequiredAsm(Ctx, Prefix, "s_cselect_b32 " + SccS + ", 1, 0",
                           Context))
      return false;
  }

  if (!appendRequiredAsm(Ctx, Prefix, ReadClusterIdAsm, Context))
    return false;
  if (!appendRequiredAsm(Ctx, Prefix, "s_cmp_eq_u32 " + BaseSreg + ", 0",
                         Context))
    return false;
  if (!appendRequiredAsm(
          Ctx, Prefix, "s_pack_hh_b32_b16 " + BaseSreg + ", 0, " + S, Context))
    return false;
  if (!appendRequiredAsm(
          Ctx, Prefix, "s_cselect_b32 " + BaseSreg + ", " + BaseSreg + ", " + S,
          Context))
    return false;

  if (SccLive) {
    if (!appendRequiredAsm(Ctx, Prefix, "s_cmp_lg_u32 " + SccS + ", 0",
                           Context))
      return false;
  }

  const uint8_t *OrigInst = Ctx.Text + DI.Offset;
  SmallVector<uint8_t> Replacement;
  Replacement.append(Prefix.begin(), Prefix.end());
  Replacement.append(OrigInst, OrigInst + DI.Size);

  bool SgprLive = isSgprLiveAfter(Ctx, Idx, BaseMCReg);
  if (SgprLive) {
    SmallVector<uint8_t> Restore =
        assembleSingleInst("s_mov_b32 " + BaseSreg + ", " + S, Ctx.LS);
    if (Restore.empty()) {
      log() << "hotswap: error: tensor_load_to_lds: B0 restore assembly "
               "failed\n";
      return failRequiredPatch(Ctx);
    }
    Replacement.append(Restore.begin(), Restore.end());
  }

  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
    return failRequiredPatch(Ctx);

  commitScratchSgpr(Ctx, *ScratchSgpr);
  if (SccScratchSgpr)
    commitScratchSgpr(Ctx, *SccScratchSgpr);
  Ctx.RequiredPatchApplied = true;

  log() << "hotswap: tensor_load_to_lds: B0 cluster-id conditional mask for "
        << BaseSreg << ", save/restore via " << S << " at 0x"
        << utohexstr(DI.Offset) << "\n";
  DI.Mnemonic = "<replaced>";
  return true;
}

std::optional<SmallVector<uint8_t>>
buildClusterLoadA0MaskPrefix(PatchContext &Ctx, StringRef ScratchSgpr,
                             StringRef Context) {
  SmallVector<uint8_t> Prefix;
  std::string SaveAsm = "s_mov_b32 ";
  SaveAsm += ScratchSgpr;
  SaveAsm += ", m0";
  std::string MaskAsm = "s_pack_hh_b32_b16 m0, 0, m0";
  if (!appendAsmBytes(Prefix, SaveAsm, Ctx.LS, Context))
    return std::nullopt;
  if (!appendAsmBytes(Prefix, MaskAsm, Ctx.LS, Context))
    return std::nullopt;
  return Prefix;
}

bool patchClusterLoadMaskA0(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  if (isAlreadyClusterMaskPatched(Ctx, Idx))
    return false;

  SmallVector<unsigned, 8> ClusterLoadSgprs =
      getSgprOperandIndices(DI.Inst, MRI);
  std::optional<SgprScratchAlloc> ScratchSgpr =
      tryAllocScratchSgpr(Ctx, Idx, ClusterLoadSgprs);
  if (!ScratchSgpr) {
    log() << "hotswap: error: " << DI.Mnemonic
          << ": no scratch SGPR available for M0 mask save/restore at 0x"
          << utohexstr(DI.Offset) << "\n";
    return failRequiredPatch(Ctx);
  }

  std::string S = "s" + std::to_string(ScratchSgpr->Sgpr);
  std::string Context = DI.Mnemonic + " M0 mask at 0x" + utohexstr(DI.Offset);
  std::optional<SmallVector<uint8_t>> Prefix =
      buildClusterLoadA0MaskPrefix(Ctx, S, Context);
  std::string RestoreAsm = "s_mov_b32 m0, " + S;
  SmallVector<uint8_t> Restore = assembleSingleInst(RestoreAsm, Ctx.LS);
  if (!Prefix || Restore.empty()) {
    log() << "hotswap: error: " << DI.Mnemonic
          << ": M0 mask save/restore assembly failed at 0x"
          << utohexstr(DI.Offset) << "\n";
    return failRequiredPatch(Ctx);
  }

  const uint8_t *OrigInst = Ctx.Text + DI.Offset;
  SmallVector<uint8_t> Replacement;
  Replacement.append(Prefix->begin(), Prefix->end());
  Replacement.append(OrigInst, OrigInst + DI.Size);
  Replacement.append(Restore.begin(), Restore.end());

  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Replacement))
    return failRequiredPatch(Ctx);

  commitScratchSgpr(Ctx, *ScratchSgpr);
  Ctx.RequiredPatchApplied = true;

  log() << "hotswap: cluster_load M0 mask: " << DI.Mnemonic
        << " clears A0 wg_mask bits, save/restore via " << S << " at 0x"
        << utohexstr(DI.Offset) << "\n";
  DI.Mnemonic = "<replaced>";
  return true;
}

// -- ADDTID swap table (StringSwitch) ---------------------------------------
//
// Maps each ADDTID DS mnemonic to its plain DS replacement. The lane-id
// expression that ADDTID encodes implicitly is materialised in the ALU by
// the trampoline body, then a regular DS op consumes the computed address.

StringRef getAddtidReplacement(StringRef Mnemonic) {
  return StringSwitch<StringRef>(Mnemonic)
      .Case("ds_load_addtid_b32", "ds_load_b32")
      .Case("ds_store_addtid_b32", "ds_store_b32")
      .Default("");
}

// Predicate that pins the load/store dispatch alongside getAddtidReplacement
// so the two stay in sync if the table grows. Avoids a string compare in
// patchDsAddtid that would silently diverge from the StringSwitch above.
bool isAddtidLoad(StringRef Mnemonic) {
  return Mnemonic == "ds_load_addtid_b32";
}

// LDS allocations strictly above this threshold are unreachable through
// ADDTID once hotswapped to A0, because A0 truncates M0 to 16 bits. The
// patch itself is still applied (the lane-id math runs through the ALU);
// this constant only gates a diagnostic so users with oversized LDS
// allocations are warned that values may still be silently wrong.
// Derived from the M0 bit-width on A0 so the magic number stays out of
// the source: 1 << 16 = 65536 bytes addressable per ADDTID encoding.
constexpr uint32_t AddtidLdsLimitA0 = 1u << 16;

// ADDTID MCInst operand layout (AddtidOpReg / AddtidOpOffset / AddtidOpGds)
// lives in comgr-hotswap-internal.h so the layout pin is shared with the unit
// tests in HotswapMCTest.cpp.

// GDS=1 ADDTID is not reachable through the gfx12 assembler -- the asm
// parser rejects the `gds` modifier on this subtarget, so any MCInst
// produced by clang/llvm-mc has GDS=0. This predicate stays as
// defense-in-depth for hand-crafted byte input or future subtargets that
// re-enable the encoding through the same MCInst slot. Because the path
// is unreachable on gfx12 it is not exercised by lit; coverage exists via
// AddTid.{Load,Store}AddTidDecodesWithExpectedLayout pinning the operand
// shape that this predicate consumes.
bool isAddtidGds(const MCInst &Inst) {
  if (Inst.getNumOperands() <= AddtidOpGds)
    return false;
  const MCOperand &Op = Inst.getOperand(AddtidOpGds);
  return Op.isImm() && Op.getImm() != 0;
}

// The DS offset field is a 16-bit immediate per the gfx12 ISA encoding;
// returning uint16_t keeps the field width visible at the type level and
// lets callers widen explicitly when needed.
std::optional<uint16_t> getAddtidOffset(const MCInst &Inst) {
  if (Inst.getNumOperands() <= AddtidOpOffset)
    return std::nullopt;
  const MCOperand &Op = Inst.getOperand(AddtidOpOffset);
  if (!Op.isImm())
    return std::nullopt;
  return static_cast<uint16_t>(Op.getImm());
}

// Build the trampoline asm for a ds_load_addtid_b32 site. The destination
// VGPR is reused as the address-compute scratch because the load overwrites
// it, so no extra VGPR allocation is needed for the load path. Reusing the
// destination as both source operands of ds_load_b32 (`ds_load_b32 vN, vN`)
// is well-defined on gfx12: the DS unit reads vaddr from the operand file
// before vdst is written, so the same VGPR can serve both roles.
//
// The replacement reproduces the ADDTID address computation in the ALU:
//   lane_id = mbcnt_lo(-1, 0)    ; lanes 0-31 contribute via exec_lo
//             mbcnt_hi(-1, V)    ;   lanes 32-63 (wave64) extend through
//                                ;   exec_hi; in wave32 exec_hi is zero so
//                                ;   the hi step is a no-op (the sequence
//                                ;   is identical for both wave sizes)
//   addr    = m0 + lane_id * 4   ; + offset (folded into the DS encoding by
//                                ;   the assembler when ToMnem is emitted)
//
// Address mask: B0 hardware reads only 20 bits of M0 at the DS unit, so any
// junk in M0[31:20] (e.g. left over from s_sendmsg or other M0 producers) is
// ignored. v_add_nc_u32 reads M0 as a full 32-bit scalar source, so we mask
// the post-add result to the same 20 bits to stay bit-exact with B0 across
// the entire reachable LDS range (gfx1250 LDS <= 320 KiB and lane_id*4 <=
// 0xFC, so the sum fits comfortably below 1 MiB and the mask is a no-op for
// any conforming M0 -- the mask only fires defensively when M0[31:20] is
// non-zero on entry).
SmallVector<std::string> buildAddtidLoadAsm(StringRef VName, uint16_t Offset,
                                            StringRef ToMnem) {
  std::string V(VName);
  SmallVector<std::string> Lines;
  Lines.push_back("v_mbcnt_lo_u32_b32 " + V + ", -1, 0");
  Lines.push_back("v_mbcnt_hi_u32_b32 " + V + ", -1, " + V);
  Lines.push_back("v_lshlrev_b32 " + V + ", 2, " + V);
  Lines.push_back("v_add_nc_u32 " + V + ", m0, " + V);
  Lines.push_back("v_and_b32 " + V + ", 0xfffff, " + V);
  Lines.push_back(ToMnem.str() + " " + V + ", " + V + fmtOffset(Offset));
  return Lines;
}

// Build the trampoline asm for a ds_store_addtid_b32 site. \p VTmpName is a
// scratch VGPR holding the computed address; \p VDataName is the original
// data VGPR. Operand order for ds_store_b32 is (addr, data).
//
// Same mbcnt_lo/mbcnt_hi pair and 20-bit M0 mask as the load path; see
// buildAddtidLoadAsm above for the full rationale.
SmallVector<std::string> buildAddtidStoreAsm(StringRef VTmpName,
                                             StringRef VDataName,
                                             uint16_t Offset,
                                             StringRef ToMnem) {
  std::string VTmp(VTmpName);
  std::string VData(VDataName);
  SmallVector<std::string> Lines;
  Lines.push_back("v_mbcnt_lo_u32_b32 " + VTmp + ", -1, 0");
  Lines.push_back("v_mbcnt_hi_u32_b32 " + VTmp + ", -1, " + VTmp);
  Lines.push_back("v_lshlrev_b32 " + VTmp + ", 2, " + VTmp);
  Lines.push_back("v_add_nc_u32 " + VTmp + ", m0, " + VTmp);
  Lines.push_back("v_and_b32 " + VTmp + ", 0xfffff, " + VTmp);
  Lines.push_back(ToMnem.str() + " " + VTmp + ", " + VData + fmtOffset(Offset));
  return Lines;
}

// -- patchDsAddtid ----------------------------------------------------------
//
// Trampoline expansion for ds_load_addtid_b32 / ds_store_addtid_b32 on
// A0. The replacement materialises the ADDTID address through the ALU
// (so the full 32-bit M0 is used) and issues a regular ds_*_b32. GDS=1
// is rejected: the rewrite stays a no-op so the original (broken on A0)
// instruction is preserved and the failure is loud in the verbose log.

bool patchDsAddtid(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  // The dispatcher in applyTrampolinePatchesImpl already gates on
  // !getAddtidReplacement(Mnem).empty(), so by contract we only see
  // ds_load_addtid_b32 / ds_store_addtid_b32 here.
  StringRef ToMnem = getAddtidReplacement(DI.Mnemonic);
  assert(!ToMnem.empty() &&
         "patchDsAddtid called for non-ADDTID mnemonic; caller must filter");

  if (isAddtidGds(DI.Inst)) {
    log() << "hotswap: error: " << DI.Mnemonic << " with GDS=1 at 0x"
          << utohexstr(DI.Offset)
          << " is not supported; leaving original instruction in place\n";
    return false;
  }

  std::optional<uint16_t> OffsetOpt = getAddtidOffset(DI.Inst);
  if (!OffsetOpt) {
    log() << "hotswap: error: " << DI.Mnemonic << " at 0x"
          << utohexstr(DI.Offset) << ": missing/non-immediate offset\n";
    return false;
  }
  uint16_t Offset = *OffsetOpt;

  if (DI.Inst.getNumOperands() <= AddtidOpReg ||
      !DI.Inst.getOperand(AddtidOpReg).isReg() ||
      !DI.Inst.getOperand(AddtidOpReg).getReg()) {
    log() << "hotswap: error: " << DI.Mnemonic << " at 0x"
          << utohexstr(DI.Offset) << ": missing register operand\n";
    return false;
  }

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  MCRegister Reg = MCRegister(DI.Inst.getOperand(AddtidOpReg).getReg());
  std::string RegName = toAsmRegName(MRI, Reg);
  if (RegName.empty()) {
    log() << "hotswap: error: " << DI.Mnemonic << " at 0x"
          << utohexstr(DI.Offset) << ": cannot resolve register name\n";
    return false;
  }

  bool IsLoad = isAddtidLoad(DI.Mnemonic);
  SmallVector<std::string> AsmLines;
  std::optional<ScratchAlloc> StoreScratch;

  if (IsLoad) {
    AsmLines = buildAddtidLoadAsm(RegName, Offset, ToMnem);
  } else {
    // Store path needs a scratch VGPR for the address-compute temporary
    // because the original data VGPR must be preserved as the store source.
    StoreScratch = tryAllocScratchVgpr(Ctx, Idx);
    if (!StoreScratch) {
      std::string KernelName =
          Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
      StringRef KernelDisplay =
          KernelName.empty() ? StringRef("<unknown>") : StringRef(KernelName);
      std::optional<uint32_t> LdsSize =
          Ctx.Elf.getKernelStaticLdsSize(KernelName);
      // Trampoline could not be applied: the original ds_*_addtid_b32 stays
      // in the code object and will silently truncate M0 to 16 bits on gfx1250
      // A0 whenever the runtime LDS layout exceeds 64 KiB.
      // Static LDS is visible in the kernel descriptor; dynamic LDS added
      // by the host at dispatch (hidden_dynamic_lds_size kernarg or a
      // dynamic_shared_pointer user arg) is not. The warning therefore
      // fires unconditionally rather than gating on the visible lower
      // bound -- a follow-up will use ElfView::kernelUsesDynamicLds to
      // tighten the condition to (static>64KiB || dynamicUsed).
      log() << "hotswap: warning: kernel '" << KernelDisplay << "' uses "
            << DI.Mnemonic
            << "; trampoline could not be applied, so A0 16-bit M0"
               " truncation may produce silently wrong results when runtime"
               " LDS (static + dynamic) exceeds "
            << AddtidLdsLimitA0 << " bytes";
      if (LdsSize)
        log() << " (static LDS = " << *LdsSize << " bytes)";
      log() << " at 0x" << utohexstr(DI.Offset) << "\n";
      log() << "hotswap: error: " << DI.Mnemonic << " at 0x"
            << utohexstr(DI.Offset) << ": no scratch VGPR available\n";
      return false;
    }

    std::string TmpName = ("v" + Twine(StoreScratch->Vgpr)).str();
    AsmLines = buildAddtidStoreAsm(TmpName, RegName, Offset, ToMnem);
  }

  std::string Combined;
  for (const std::string &Line : AsmLines)
    Combined += Line + "\n";
  SmallVector<uint8_t> Bytes = assembleSingleInst(Combined, Ctx.LS);
  if (Bytes.empty()) {
    log() << "hotswap: error: " << DI.Mnemonic
          << " trampoline assembly failed at 0x" << utohexstr(DI.Offset)
          << "\n";
    return false;
  }

  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, Bytes))
    return false;

  // Commit the scratch-VGPR reservation only after the patch is in place:
  // any earlier failure (assembly, sled/trampoline emission) leaves no
  // bytes at DI.Offset to back the reservation, so neither the descriptor
  // accounting nor OutScratchPatches must advertise a slot for it.
  if (StoreScratch) {
    ScratchPatchInfo SPI;
    SPI.Offset = DI.Offset;
    SPI.ScratchRegs.resize(Ctx.Config.MaxVgprs);
    SPI.ScratchRegs.set(StoreScratch->Vgpr);
    Ctx.OutScratchPatches.push_back(std::move(SPI));
    commitScratchVgpr(Ctx, *StoreScratch);
  }

  log() << "hotswap: trampoline: " << DI.Mnemonic << " -> " << ToMnem
        << " at 0x" << utohexstr(DI.Offset) << " (offset=" << Offset << ", "
        << RegName << ")\n";
  DI.Mnemonic = "<replaced>";
  return true;
}

} // anonymous namespace

std::optional<std::vector<std::string>> expandDs2Addr(const MCInst &Inst,
                                                      StringRef FromMnem,
                                                      StringRef ToMnem,
                                                      const LLVMState &LS) {
  return expandDs2AddrImpl(Inst, FromMnem, ToMnem, LS);
}

// -- applyTrampolinePatches -------------------------------------------------
//
// Strong-symbol override. Handles B0 errata that produce replacement code
// larger than the original instruction slot:
//
//   ds_*_2addr_*           -> split into two single-address DS ops
//     (covers both the stride64 and non-stride64 encodings)
//   tensor_load_to_lds     -> apply the selected target stepping's multicast
//                             mask rule
//   cluster_load*          -> in A0 mask mode, save/clear/restore M0 for
//                             remaining cluster ops
//   ds_*_addtid_b32        -> materialise lane-id math in ALU, then ds_*_b32

static uint32_t applyTrampolinePatchesImpl(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (Ctx.Config.RunB0A0Patches && !getDs2AddrReplacement(Mnem).empty())
    return patchDs2Addr(Ctx, Idx) ? 1 : 0;

  if (Mnem == "tensor_load_to_lds") {
    if (Ctx.Config.MaskPolicy == MaskWorkaroundPolicy::A0)
      return patchTensorLoadToLdsA0(Ctx, Idx) ? 1 : 0;
    if (Ctx.Config.MaskPolicy == MaskWorkaroundPolicy::B0)
      return patchTensorLoadToLdsB0(Ctx, Idx) ? 1 : 0;
  }

  if (Ctx.Config.MaskPolicy == MaskWorkaroundPolicy::A0 && isClusterLoad(Mnem))
    return patchClusterLoadMaskA0(Ctx, Idx) ? 1 : 0;

  if (Ctx.Config.RunB0A0Patches && !getAddtidReplacement(Mnem).empty())
    return patchDsAddtid(Ctx, Idx) ? 1 : 0;

  return 0;
}

void registerTrampolinePatch(HotswapPatchVTable &VT) {
  VT.applyTrampolinePatches = &applyTrampolinePatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
