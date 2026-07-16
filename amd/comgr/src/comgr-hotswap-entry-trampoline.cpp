//===- comgr-hotswap-entry-trampoline.cpp - Kernel-entry stubs ------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Kernel-entry redirection pass for COMGR HotSwap. This pass is
/// independent of the gfx1250 B0-to-A0 instruction patcher: it appends one
/// PC-relative entry stub per kernel descriptor and rewrites the descriptor's
/// kernel_code_entry_byte_offset to point at that stub.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <limits>

using namespace llvm;

namespace COMGR {
namespace hotswap {

static bool appendAsm(SmallVectorImpl<uint8_t> &Out, StringRef Asm,
                      const LLVMState &LS) {
  SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, LS);
  if (Bytes.empty()) {
    log() << "hotswap: error: failed to assemble entry-stub instruction: "
          << Asm << "\n";
    return false;
  }
  Out.append(Bytes.begin(), Bytes.end());
  return true;
}

static SmallVector<uint8_t> getCodeEndBytes(const LLVMState &LS) {
  SmallVector<uint8_t> CodeEnd = assembleSingleInst("s_code_end", LS);
  if (CodeEnd.empty())
    log() << "hotswap: error: failed to assemble s_code_end for entry-stub "
          << "padding.\n";
  return CodeEnd;
}

SmallVector<uint8_t> buildKernelEntryTrampoline(uint64_t StubVAddr,
                                                uint64_t EntryVAddr,
                                                unsigned ScratchSgpr,
                                                const LLVMState &LS) {
  if (ScratchSgpr == std::numeric_limits<unsigned>::max()) {
    log() << "hotswap: error: kernel-entry stub scratch SGPR pair overflows "
          << "unsigned.\n";
    return {};
  }

  SmallVector<uint8_t> Bytes;
  std::string ScratchPair =
      (Twine("s[") + Twine(ScratchSgpr) + ":" + Twine(ScratchSgpr + 1) + "]")
          .str();
  std::string ScratchLo = (Twine("s") + Twine(ScratchSgpr)).str();
  std::string ScratchHi = (Twine("s") + Twine(ScratchSgpr + 1)).str();

  // Assemble through the MC layer instead of spelling encoded bytes; the LIT
  // test pins the generated stub's disassembly.
  if (!appendAsm(Bytes, "global_wb", LS))
    return {};
  if (!appendAsm(Bytes, "v_nop", LS))
    return {};
  if (!appendAsm(Bytes, "s_get_pc_i64 " + ScratchPair, LS))
    return {};

  // s_get_pc_i64 returns the address of the following s_add_u32 instruction.
  // Materialize the original entry with a 64-bit PC-relative add so the code
  // object can be rewritten before ROCR knows final device addresses.
  std::optional<uint64_t> PcBase =
      checkedAddUint64(StubVAddr, static_cast<uint64_t>(Bytes.size()),
                       "kernel-entry stub PC base");
  if (!PcBase)
    return {};
  // Unsigned subtraction is intentional: the immediate pair materializes the
  // 64-bit two's-complement delta, including backward jumps.
  const uint64_t Delta = EntryVAddr - *PcBase;
  const uint32_t Lo = static_cast<uint32_t>(Delta);
  const uint32_t Hi = static_cast<uint32_t>(Delta >> 32);

  if (!appendAsm(Bytes,
                 "s_add_u32 " + ScratchLo + ", " + ScratchLo + ", 0x" +
                     utohexstr(Lo),
                 LS))
    return {};
  if (!appendAsm(Bytes,
                 "s_addc_u32 " + ScratchHi + ", " + ScratchHi + ", 0x" +
                     utohexstr(Hi),
                 LS))
    return {};
  if (!appendAsm(Bytes, "s_set_pc_i64 " + ScratchPair, LS))
    return {};

  SmallVector<uint8_t> CodeEnd = getCodeEndBytes(LS);
  if (CodeEnd.empty())
    return {};
  if (Bytes.size() > KernelEntryStubStride) {
    log() << "hotswap: error: kernel-entry stub grew past "
          << KernelEntryStubStride << " bytes.\n";
    return {};
  }
  while (Bytes.size() < KernelEntryStubStride) {
    if (Bytes.size() + CodeEnd.size() > KernelEntryStubStride) {
      log() << "hotswap: error: s_code_end padding does not evenly fill "
            << "kernel-entry stub stride " << KernelEntryStubStride << ".\n";
      return {};
    }
    Bytes.append(CodeEnd.begin(), CodeEnd.end());
  }
  return Bytes;
}

uint64_t computeKernelEntryPrefetchGuardBytes(uint32_t InstPrefLines) {
  const uint64_t PrefetchBytes =
      static_cast<uint64_t>(InstPrefLines) * KernelEntryInstPrefUnitBytes;
  if (PrefetchBytes <= KernelEntryStubStride)
    return 0;
  return PrefetchBytes - KernelEntryStubStride;
}

static bool hasResolvedEntryStubState(const LLVMState &LS, StringRef Context) {
  if (!LS.MCII || LS.GlobalWbOpcode >= LS.MCII->getNumOpcodes() ||
      LS.SGetPcI64Opcode >= LS.MCII->getNumOpcodes() ||
      LS.SAddU32Opcode >= LS.MCII->getNumOpcodes() ||
      LS.SAddcU32Opcode >= LS.MCII->getNumOpcodes() ||
      LS.SSetPcI64Opcode >= LS.MCII->getNumOpcodes()) {
    log() << "hotswap: error: " << Context
          << ": LLVMState lacks resolved entry-stub opcodes.\n";
    return false;
  }

  if (!LS.MRI) {
    log() << "hotswap: error: " << Context
          << ": LLVMState lacks register info.\n";
    return false;
  }

  return true;
}

static bool decodeKernelEntryStub(ArrayRef<uint8_t> Bytes, const LLVMState &LS,
                                  std::vector<InternalDecodedInst> &Decoded,
                                  StringRef Context) {
  if (Bytes.size() < KernelEntryStubStride)
    return false;

  if (!hasResolvedEntryStubState(LS, Context))
    return false;

  if (!decodeTextSection(Bytes.data(), KernelEntryStubStride, LS, Decoded)) {
    log() << "hotswap: error: " << Context << ": failed to decode "
          << KernelEntryStubStride << "-byte candidate.\n";
    return false;
  }
  return Decoded.size() >= 6;
}

static bool startsWithBytes(ArrayRef<uint8_t> Bytes, ArrayRef<uint8_t> Prefix) {
  return Bytes.size() >= Prefix.size() &&
         Bytes.take_front(Prefix.size()).equals(Prefix);
}

static SmallVector<uint8_t> buildEntryStubBytePrefix(const LLVMState &LS) {
  SmallVector<uint8_t> GlobalWb = assembleSingleInst("global_wb", LS);
  SmallVector<uint8_t> VNop = assembleSingleInst("v_nop", LS);
  if (GlobalWb.empty() || VNop.empty())
    return {};

  SmallVector<uint8_t> Prefix;
  Prefix.append(GlobalWb.begin(), GlobalWb.end());
  Prefix.append(VNop.begin(), VNop.end());
  return Prefix;
}

static bool hasRegOperand(const MCInst &Inst, unsigned Index) {
  return Inst.getNumOperands() > Index && Inst.getOperand(Index).isReg();
}

static bool hasImmOperand(const MCInst &Inst, unsigned Index) {
  return Inst.getNumOperands() > Index && Inst.getOperand(Index).isImm();
}

static bool sameRegOperand(const MCInst &LHS, unsigned LHSIndex,
                           const MCInst &RHS, unsigned RHSIndex) {
  return hasRegOperand(LHS, LHSIndex) && hasRegOperand(RHS, RHSIndex) &&
         LHS.getOperand(LHSIndex).getReg() == RHS.getOperand(RHSIndex).getReg();
}

static bool hasEntryStubOperandShape(ArrayRef<InternalDecodedInst> Decoded,
                                     const LLVMState &LS) {
  if (Decoded.size() < 6)
    return false;

  if (Decoded[0].Inst.getOpcode() != LS.GlobalWbOpcode ||
      Decoded[1].Inst.getOpcode() != LS.VNopInst.getOpcode() ||
      Decoded[2].Inst.getOpcode() != LS.SGetPcI64Opcode ||
      Decoded[3].Inst.getOpcode() != LS.SAddU32Opcode ||
      Decoded[4].Inst.getOpcode() != LS.SAddcU32Opcode ||
      Decoded[5].Inst.getOpcode() != LS.SSetPcI64Opcode)
    return false;

  const MCInst &GlobalWb = Decoded[0].Inst;
  const MCInst &VNop = Decoded[1].Inst;
  const MCInst &GetPc = Decoded[2].Inst;
  const MCInst &AddLo = Decoded[3].Inst;
  const MCInst &AddHi = Decoded[4].Inst;
  const MCInst &SetPc = Decoded[5].Inst;

  if (GlobalWb.getNumOperands() != 1 || !GlobalWb.getOperand(0).isImm() ||
      GlobalWb.getOperand(0).getImm() != 0 || VNop.getNumOperands() != 0)
    return false;

  if (GetPc.getNumOperands() != 1 || SetPc.getNumOperands() != 1 ||
      !sameRegOperand(GetPc, 0, SetPc, 0))
    return false;

  if (AddLo.getNumOperands() != 3 || AddHi.getNumOperands() != 3 ||
      !sameRegOperand(AddLo, 0, AddLo, 1) ||
      !sameRegOperand(AddHi, 0, AddHi, 1) || !hasImmOperand(AddLo, 2) ||
      !hasImmOperand(AddHi, 2))
    return false;

  MCRegister PairReg = GetPc.getOperand(0).getReg();
  MCRegister LoReg = AddLo.getOperand(0).getReg();
  MCRegister HiReg = AddHi.getOperand(0).getReg();
  unsigned LoSubRegIndex = LS.MRI->getSubRegIndex(PairReg, LoReg);
  unsigned HiSubRegIndex = LS.MRI->getSubRegIndex(PairReg, HiReg);
  return LoSubRegIndex != 0 && HiSubRegIndex != 0 &&
         LoSubRegIndex != HiSubRegIndex && LoSubRegIndex < HiSubRegIndex;
}

static std::optional<uint64_t>
decodeEntryStubTargetVAddr(ArrayRef<InternalDecodedInst> Decoded,
                           uint64_t StubVAddr) {
  std::optional<uint64_t> PcBaseOffset =
      checkedAddUint64(Decoded[2].Offset, Decoded[2].Size,
                       "decoded kernel-entry stub PC-base offset");
  if (!PcBaseOffset)
    return std::nullopt;
  std::optional<uint64_t> PcBase = checkedAddUint64(
      StubVAddr, *PcBaseOffset, "decoded kernel-entry stub PC base");
  if (!PcBase)
    return std::nullopt;

  const uint64_t Lo =
      static_cast<uint32_t>(Decoded[3].Inst.getOperand(2).getImm());
  const uint64_t Hi =
      static_cast<uint32_t>(Decoded[4].Inst.getOperand(2).getImm());
  const uint64_t Delta = Lo | (Hi << 32);
  return *PcBase + Delta;
}

bool isKernelEntryTrampoline(ArrayRef<uint8_t> Bytes, const LLVMState &LS) {
  if (!hasKernelEntryTrampolinePrefix(Bytes, LS))
    return false;

  std::vector<InternalDecodedInst> Decoded;
  return decodeKernelEntryStub(Bytes, LS, Decoded, "isKernelEntryTrampoline") &&
         hasEntryStubOperandShape(Decoded, LS);
}

bool hasKernelEntryTrampolinePrefix(ArrayRef<uint8_t> Bytes,
                                    const LLVMState &LS) {
  SmallVector<uint8_t> Prefix;
  if (!appendAsm(Prefix, "global_wb", LS))
    return false;
  if (!appendAsm(Prefix, "v_nop", LS))
    return false;

  return Bytes.size() >= Prefix.size() &&
         std::equal(Prefix.begin(), Prefix.end(), Bytes.begin());
}

std::optional<uint64_t> checkedAlignTo(uint64_t Value, uint64_t Alignment,
                                       StringRef Context) {
  if (Alignment == 0)
    return Value;

  uint64_t Remainder = Value % Alignment;
  if (Remainder == 0)
    return Value;
  return checkedAddUint64(Value, Alignment - Remainder, Context);
}

std::optional<uint64_t> entryVAddr(const KernelDescriptorInfo &KD) {
  if (KD.EntryOffset >= 0)
    return checkedAddUint64(
        KD.VAddr, static_cast<uint64_t>(KD.EntryOffset),
        (Twine("kernel entry vaddr for '") + KD.KernelName + "'").str());

  const uint64_t Magnitude =
      KD.EntryOffset == std::numeric_limits<int64_t>::min()
          ? static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1
          : static_cast<uint64_t>(-KD.EntryOffset);
  if (KD.VAddr < Magnitude) {
    log() << "hotswap: error: kernel entry vaddr for '" << KD.KernelName
          << "' underflows uint64_t.\n";
    return std::nullopt;
  }
  return KD.VAddr - Magnitude;
}

static std::optional<bool>
descriptorAlreadyTargetsEntryStub(const ElfView &Elf,
                                  const KernelDescriptorInfo &KD,
                                  const LLVMState &LS,
                                  ArrayRef<uint8_t> EntryStubPrefix) {
  std::optional<uint64_t> Entry = entryVAddr(KD);
  if (!Entry)
    return std::nullopt;

  std::optional<uint64_t> TextEnd = checkedAddUint64(
      Elf.textAddr(), Elf.textSize(), "entry trampoline text end");
  if (!TextEnd)
    return std::nullopt;

  // Read whatever the descriptor's entry points at: the real kernel prologue in
  // .text on a never-rewritten object, or the entry stub in the appended
  // trampoline pool on an already-rewritten one. dataAtVAddr resolves either
  // through the covering allocatable section.
  const uint8_t *StubBytes = Elf.dataAtVAddr(*Entry, KernelEntryStubStride);
  if (!StubBytes)
    return false;
  ArrayRef<uint8_t> Candidate(StubBytes, KernelEntryStubStride);
  // Avoid feeding arbitrary kernel-entry instructions into the stub matcher.
  // The full decode below is only needed once the bytes look like a hotswap
  // entry stub.
  if (!startsWithBytes(Candidate, EntryStubPrefix))
    return false;

  std::vector<InternalDecodedInst> Decoded;
  if (!decodeKernelEntryStub(Candidate, LS,
                             Decoded, "entry trampoline idempotency matcher"))
    return false;
  if (!hasEntryStubOperandShape(Decoded, LS))
    return false;

  std::optional<uint64_t> Target = decodeEntryStubTargetVAddr(Decoded, *Entry);
  if (!Target)
    return std::nullopt;

  // A genuine entry stub jumps back to the original kernel body in .text.
  return *Target >= Elf.textAddr() && *Target < *TextEnd;
}

// True when the prologue already begins with the compile-time GFX1250
// unclaused-VMEM workaround (llvm/llvm-project#208467): `global_wb` (cpol 0)
// then `v_nop`. Unlike descriptorAlreadyTargetsEntryStub(), the descriptor
// still points at the real kernel body, not a hotswap stub.
static std::optional<bool> entryPrologueHasVmemWorkaround(
    const ElfView &Elf, const KernelDescriptorInfo &KD, const LLVMState &LS,
    ArrayRef<uint8_t> EntryStubPrefix) {
  if (EntryStubPrefix.empty())
    return false;

  std::optional<uint64_t> Entry = entryVAddr(KD);
  if (!Entry)
    return std::nullopt;

  const uint8_t *Bytes = Elf.dataAtVAddr(*Entry, EntryStubPrefix.size());
  if (!Bytes)
    return false;
  ArrayRef<uint8_t> Candidate(Bytes, EntryStubPrefix.size());
  if (!startsWithBytes(Candidate, EntryStubPrefix))
    return false;

  // Confirm the prefix decodes to global_wb (cpol 0) + v_nop, not a byte match.
  if (!hasResolvedEntryStubState(LS, "entry prologue workaround matcher"))
    return false;
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Bytes, EntryStubPrefix.size(), LS, Decoded) ||
      Decoded.size() < 2)
    return false;
  const MCInst &GlobalWb = Decoded[0].Inst;
  const MCInst &VNop = Decoded[1].Inst;
  return GlobalWb.getOpcode() == LS.GlobalWbOpcode &&
         GlobalWb.getNumOperands() == 1 && GlobalWb.getOperand(0).isImm() &&
         GlobalWb.getOperand(0).getImm() == 0 &&
         VNop.getOpcode() == LS.VNopInst.getOpcode() &&
         VNop.getNumOperands() == 0;
}

static std::optional<uint64_t>
totalTrampolineBytes(ArrayRef<Trampoline> Trampolines) {
  uint64_t Total = 0;
  for (const Trampoline &T : Trampolines) {
    std::optional<uint64_t> NewTotal =
        checkedAddUint64(Total, static_cast<uint64_t>(T.Bytes.size()),
                         "existing trampoline byte count");
    if (!NewTotal)
      return std::nullopt;
    Total = *NewTotal;
  }
  return Total;
}

static std::optional<int64_t>
checkedSignedDifference(uint64_t LHS, uint64_t RHS, StringRef Context) {
  if (LHS >= RHS) {
    uint64_t Diff = LHS - RHS;
    if (Diff > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      log() << "hotswap: error: " << Context
            << " positive offset is not representable as int64_t.\n";
      return std::nullopt;
    }
    return static_cast<int64_t>(Diff);
  }

  uint64_t Diff = RHS - LHS;
  constexpr uint64_t Int64MinMagnitude =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1;
  if (Diff > Int64MinMagnitude) {
    log() << "hotswap: error: " << Context
          << " negative offset is not representable as int64_t.\n";
    return std::nullopt;
  }
  if (Diff == Int64MinMagnitude)
    return std::numeric_limits<int64_t>::min();
  return -static_cast<int64_t>(Diff);
}

static std::optional<unsigned> allocateEntryStubScratchSgprs(
    const ElfView &Elf, const KernelDescriptorInfo &KD, unsigned MaxSgprs) {
  constexpr unsigned ScratchSgprs = 2;
  std::optional<unsigned> SgprCount = Elf.getKernelSgprCount(KD.KernelName);
  if (!SgprCount) {
    log() << "hotswap: error: entry trampoline: failed to read SGPR count for '"
          << KD.KernelName << "'.\n";
    return std::nullopt;
  }
  if (*SgprCount > MaxSgprs) {
    log() << "hotswap: error: entry trampoline: kernel '" << KD.KernelName
          << "' uses " << *SgprCount << " SGPRs, above max " << MaxSgprs
          << ".\n";
    return std::nullopt;
  }

  unsigned ScratchBase = (*SgprCount + 1) & ~1u;
  if (ScratchBase > MaxSgprs || MaxSgprs - ScratchBase < ScratchSgprs) {
    log() << "hotswap: error: entry trampoline: kernel '" << KD.KernelName
          << "' uses " << *SgprCount << " SGPRs; no aligned scratch pair fits "
          << "below max " << MaxSgprs << ".\n";
    return std::nullopt;
  }
  return ScratchBase;
}

static bool appendPaddingTrampoline(std::vector<Trampoline> &Out,
                                    uint64_t PadBytes, ArrayRef<uint8_t> Fill) {
  if (PadBytes == 0)
    return true;
  if (Fill.empty()) {
    log() << "hotswap: error: entry-stub alignment padding requested without "
          << "cached s_nop bytes.\n";
    return false;
  }
  if (PadBytes % Fill.size() != 0) {
    log() << "hotswap: error: entry-stub alignment padding size " << PadBytes
          << " is not a multiple of cached s_nop size " << Fill.size() << ".\n";
    return false;
  }
  if (PadBytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    log() << "hotswap: error: entry-stub alignment padding size " << PadBytes
          << " exceeds size_t.\n";
    return false;
  }

  Trampoline Pad;
  while (static_cast<uint64_t>(Pad.Bytes.size()) < PadBytes)
    Pad.Bytes.append(Fill.begin(), Fill.end());
  Out.push_back(std::move(Pad));
  return true;
}

std::optional<uint32_t> appendKernelEntryTrampolines(
    const ElfView &Elf, const LLVMState &LS, unsigned MaxSgprs,
    std::vector<Trampoline> &Growth,
    std::vector<KernelEntryTrampolineFixup> &OutFixups) {
  ArrayRef<KernelDescriptorInfo> Descriptors = Elf.kernelDescriptors();
  if (Descriptors.empty())
    return 0;

  SmallVector<uint8_t> EntryStubPrefix = buildEntryStubBytePrefix(LS);
  if (EntryStubPrefix.empty()) {
    log() << "hotswap: error: entry trampoline: failed to assemble byte "
          << "prefix for idempotency matching.\n";
    return std::nullopt;
  }

  struct WorkItem {
    KernelDescriptorInfo KD;
    uint32_t StubInstPrefLines = 0;
  };

  std::vector<WorkItem> Work;
  uint32_t MaxStubInstPrefLines = 0;
  for (const KernelDescriptorInfo &KD : Descriptors) {
    std::optional<bool> AlreadyHasEntryStub =
        descriptorAlreadyTargetsEntryStub(Elf, KD, LS, EntryStubPrefix);
    if (!AlreadyHasEntryStub)
      return std::nullopt;
    if (*AlreadyHasEntryStub)
      continue;
    // Skip if the compiler already applied the workaround (#208467)
    // in-prologue.
    std::optional<bool> PrologueHasWorkaround =
        entryPrologueHasVmemWorkaround(Elf, KD, LS, EntryStubPrefix);
    if (!PrologueHasWorkaround)
      return std::nullopt;
    if (*PrologueHasWorkaround) {
      log() << "hotswap: kernel '" << KD.KernelName
            << "' prologue already carries the unclaused-VMEM workaround "
            << "(global_wb; v_nop); skipping entry trampoline\n";
      continue;
    }
    std::optional<uint32_t> OriginalInstPrefLines =
        Elf.getKernelDescriptorInstPrefSize(KD.KernelName, LS.Cpu);
    if (!OriginalInstPrefLines)
      return std::nullopt;
    // Entry stubs are 256-byte aligned and fit inside one stride, so clamp the
    // descriptor prefetch to the stub stride. Deferred non-entry trampolines
    // keep using the original descriptor prefetch guard.
    uint32_t StubInstPrefLines =
        std::min(*OriginalInstPrefLines, KernelEntryStubInstPrefLines);
    MaxStubInstPrefLines = std::max(MaxStubInstPrefLines, StubInstPrefLines);
    Work.push_back({KD, StubInstPrefLines});
  }
  if (Work.empty())
    return 0;

  std::optional<uint64_t> ExistingGrowthBytes = totalTrampolineBytes(Growth);
  if (!ExistingGrowthBytes)
    return std::nullopt;
  uint64_t AppendOffset = *ExistingGrowthBytes;
  // Stubs live in the appended trampoline pool at its fresh virtual address
  // (trampolinePoolVAddr()), no longer immediately after .text.
  std::optional<uint64_t> PoolVAddrOr = Elf.trampolinePoolVAddr();
  if (!PoolVAddrOr)
    return std::nullopt;
  const uint64_t PoolVAddr = *PoolVAddrOr;
  std::optional<uint64_t> StubPoolBaseVAddr = checkedAddUint64(
      PoolVAddr, AppendOffset, "entry trampoline stub-pool base");
  if (!StubPoolBaseVAddr)
    return std::nullopt;
  std::optional<uint64_t> AlignedStubPoolBaseVAddr =
      checkedAlignTo(*StubPoolBaseVAddr, KernelEntryStubStride,
                     "entry trampoline aligned stub-pool base");
  if (!AlignedStubPoolBaseVAddr)
    return std::nullopt;
  const uint64_t StubStart = *AlignedStubPoolBaseVAddr - PoolVAddr;
  std::vector<Trampoline> LocalGrowth;
  std::vector<KernelEntryTrampolineFixup> LocalFixups;
  if (!appendPaddingTrampoline(LocalGrowth, StubStart - AppendOffset,
                               LS.SNopBytes))
    return std::nullopt;
  AppendOffset = StubStart;

  for (const WorkItem &Item : Work) {
    const KernelDescriptorInfo &KD = Item.KD;
    std::optional<uint64_t> StubVAddr = checkedAddUint64(
        PoolVAddr, AppendOffset,
        (Twine("entry trampoline vaddr for '") + KD.KernelName + "'").str());
    if (!StubVAddr)
      return std::nullopt;
    std::optional<unsigned> ScratchSgpr =
        allocateEntryStubScratchSgprs(Elf, KD, MaxSgprs);
    if (!ScratchSgpr)
      return std::nullopt;
    std::optional<uint64_t> Entry = entryVAddr(KD);
    if (!Entry)
      return std::nullopt;
    SmallVector<uint8_t> Stub =
        buildKernelEntryTrampoline(*StubVAddr, *Entry, *ScratchSgpr, LS);
    if (Stub.empty()) {
      log() << "hotswap: error: failed to build kernel-entry trampoline for '"
            << KD.KernelName << "' at original entry vaddr 0x"
            << utohexstr(*Entry) << ".\n";
      return std::nullopt;
    }

    Trampoline T;
    T.Bytes.assign(Stub.begin(), Stub.end());
    LocalGrowth.push_back(std::move(T));
    LocalFixups.push_back({KD.KernelName, AppendOffset, *ScratchSgpr + 2,
                           Item.StubInstPrefLines});
    std::optional<uint64_t> NewAppendOffset = checkedAddUint64(
        AppendOffset, KernelEntryStubStride,
        (Twine("entry trampoline append offset after '") + KD.KernelName + "'")
            .str());
    if (!NewAppendOffset)
      return std::nullopt;
    AppendOffset = *NewAppendOffset;
  }

  const uint64_t GuardBytes =
      computeKernelEntryPrefetchGuardBytes(MaxStubInstPrefLines);
  if (GuardBytes != 0) {
    SmallVector<uint8_t> CodeEnd = getCodeEndBytes(LS);
    if (CodeEnd.empty() ||
        !appendPaddingTrampoline(LocalGrowth, GuardBytes, CodeEnd))
      return std::nullopt;
  }

  if (LocalFixups.empty())
    return 0;

  if (LocalFixups.size() > std::numeric_limits<uint32_t>::max()) {
    log() << "hotswap: error: kernel-entry trampoline count "
          << LocalFixups.size() << " exceeds uint32_t.\n";
    return std::nullopt;
  }

  for (Trampoline &T : LocalGrowth)
    Growth.push_back(std::move(T));
  OutFixups.insert(OutFixups.end(), LocalFixups.begin(), LocalFixups.end());

  log() << "hotswap: installed " << LocalFixups.size()
        << " kernel-entry trampoline" << (LocalFixups.size() == 1 ? "" : "s")
        << " with " << GuardBytes << " prefetch guard bytes\n";
  return static_cast<uint32_t>(LocalFixups.size());
}

bool rewriteKernelEntryDescriptorOffsets(
    WritableMemoryBuffer &OutBuf, uint64_t PoolVAddr, StringRef TargetCpu,
    ArrayRef<KernelEntryTrampolineFixup> Fixups) {
  if (Fixups.empty())
    return true;

  uint8_t *Data = reinterpret_cast<uint8_t *>(OutBuf.getBufferStart());
  Expected<ElfView> ViewOrErr = ElfView::create(Data, OutBuf.getBufferSize());
  if (!ViewOrErr) {
    log() << "hotswap: error: failed to reparse grown ELF for entry "
          << "descriptor rewrites: " << toString(ViewOrErr.takeError()) << "\n";
    return false;
  }

  bool Ok = true;
  ElfView &OutElf = *ViewOrErr;
  // Collect SGPR bumps and apply them in one batched metadata rewrite after the
  // loop; a per-fixup update reparses/reserializes the whole note (O(n^2)).
  StringMap<unsigned> SgprBumps;
  for (const KernelEntryTrampolineFixup &Fixup : Fixups) {
    std::optional<uint64_t> KdVAddr =
        OutElf.getKernelDescriptorVAddr(Fixup.KernelName);
    if (!KdVAddr) {
      log() << "hotswap: error: missing kernel descriptor for entry "
            << "trampoline fixup '" << Fixup.KernelName << "'.\n";
      Ok = false;
      continue;
    }
    std::optional<uint64_t> StubVAddr = checkedAddUint64(
        PoolVAddr, Fixup.StubTextOffset,
        (Twine("entry trampoline vaddr for '") + Fixup.KernelName + "'").str());
    if (!StubVAddr) {
      Ok = false;
      continue;
    }
    std::optional<int64_t> NewOffset = checkedSignedDifference(
        *StubVAddr, *KdVAddr,
        (Twine("entry trampoline descriptor offset for '") + Fixup.KernelName +
         "'")
            .str());
    if (!NewOffset) {
      Ok = false;
      continue;
    }
    bool UpdatedEntry =
        OutElf.updateKernelDescriptorEntryOffset(Fixup.KernelName, *NewOffset);
    if (!Fixup.SkipSgprReservation && Fixup.RequiredSgprs != 0) {
      unsigned &Bump = SgprBumps[Fixup.KernelName];
      Bump = std::max(Bump, Fixup.RequiredSgprs);
    }
    bool UpdatedInstPref = OutElf.updateKernelDescriptorInstPrefSize(
        Fixup.KernelName, TargetCpu, Fixup.InstPrefLines);
    Ok = UpdatedEntry && UpdatedInstPref && Ok;
  }

  if (!SgprBumps.empty())
    Ok = OutElf.updateKernelMetadataSgprCounts(SgprBumps) && Ok;
  return Ok;
}

} // namespace hotswap
} // namespace COMGR
