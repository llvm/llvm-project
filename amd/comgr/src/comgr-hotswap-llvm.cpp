//===- comgr-hotswap-llvm.cpp - LLVM MC infrastructure, decode/encode -----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// LLVM MC-layer infrastructure for the HotSwap ISA rewriting subsystem:
/// per-ISA target / context initialization, disassembly, single-instruction
/// assembly, and trampoline assembly.
///
/// The pieces here form the assembly-side counterpart of comgr-disassembly.cpp
/// (which wraps DisassemblyInfo over the same MC object set). Extracting a
/// shared Comgr MC toolchain module that both sides embed is tracked in
/// ROCm/llvm-project#2253 and is a follow-up to this PR.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "comgr.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {

static constexpr StringLiteral UnknownMnemonic("<unknown>");

namespace {
// The amdgcn Target is the same for every AMDGPU subtarget (the per-CPU /
// per-feature differences live in MCSubtargetInfo), so a fixed triple is fine
// for the one-time TargetRegistry lookup below. The per-call ISA-specific
// triple is built from the caller's TargetIdentifier inside initLLVM().
static const Triple AmdgcnLookupTriple("amdgcn-amd-amdhsa");

/// Resolve the amdgcn Target once per process, after delegating AMDGPU MC
/// registration to the shared Comgr init path. Function-local static init is
/// thread-safe per [basic.stc.static]/[stmt.dcl], so no explicit mutex or
/// call_once is required.
const Target *getAmdgcnTarget() {
  static const Target *const Tgt = []() -> const Target * {
    COMGR::ensureLLVMInitialized();
    std::string Err;
    Triple T(AmdgcnLookupTriple);
    return TargetRegistry::lookupTarget("amdgcn", T, Err);
  }();
  return Tgt;
}

/// Build the LLVM Triple for \p TI by concatenating its Arch/Vendor/OS/Environ
/// fields. Mirrors DisassemblyInfo::create() so the hotswap and disassembly
/// paths see the same triple for the same TargetIdentifier.
Triple buildTriple(const TargetIdentifier &TI) {
  std::string TT =
      (Twine(TI.Arch) + "-" + TI.Vendor + "-" + TI.OS + "-" + TI.Environ).str();
  return Triple(TT);
}

/// Join \p Features into an LLVM MC feature string such as "+sramecc,-xnack".
/// Comgr stores each feature as "<name><polarity>" (polarity char last), so we
/// move the trailing polarity character to the front per LLVM's convention.
/// Mirrors DisassemblyInfo::create()'s feature-string build.
std::string buildFeatureString(ArrayRef<StringRef> Features) {
  SmallVector<std::string, 2> Parts;
  Parts.reserve(Features.size());
  for (StringRef F : Features) {
    if (F.empty())
      continue;
    Parts.emplace_back((Twine(F.take_back()) + F.drop_back()).str());
  }
  return join(Parts, ",");
}

/// MCStreamer that captures matched MCInsts instead of emitting to an object
/// file. Mirrors MCNullStreamer's minimal pure-virtual surface (see
/// llvm/lib/MC/MCNullStreamer.cpp) plus an emitInstruction override that
/// records each matched instruction so the caller can encode it directly via
/// MCCodeEmitter. Used by assembleSingleInst to avoid the object-file round
/// trip.
class InstCapturingStreamer final : public MCStreamer {
public:
  explicit InstCapturingStreamer(MCContext &Ctx) : MCStreamer(Ctx) {}

  ArrayRef<MCInst> captured() const { return Captured; }

  void emitInstruction(const MCInst &Inst,
                       const MCSubtargetInfo & /*STI*/) override {
    Captured.emplace_back(Inst);
  }

  bool hasRawTextSupport() const override { return true; }
  void emitRawTextImpl(StringRef /*String*/) override {}

  bool emitSymbolAttribute(MCSymbol * /*Symbol*/,
                           MCSymbolAttr /*Attribute*/) override {
    return true;
  }
  void emitCommonSymbol(MCSymbol * /*Symbol*/, uint64_t /*Size*/,
                        Align /*ByteAlignment*/) override {}
  void emitSubsectionsViaSymbols() override {}
  void beginCOFFSymbolDef(const MCSymbol * /*Symbol*/) override {}
  void emitCOFFSymbolStorageClass(int /*StorageClass*/) override {}
  void emitCOFFSymbolType(int /*Type*/) override {}
  void endCOFFSymbolDef() override {}
  void
  emitXCOFFSymbolLinkageWithVisibility(MCSymbol * /*Symbol*/,
                                       MCSymbolAttr /*Linkage*/,
                                       MCSymbolAttr /*Visibility*/) override {}

private:
  SmallVector<MCInst, 8> Captured;
};
} // namespace

// -- Instruction helpers ------------------------------------------------------

/// Encode \p Inst to raw bytes via the cached MCCodeEmitter. This is the
/// canonical "MCInst -> bytes" primitive; mirrors the encoding sequence used
/// by AMDGPUMCInstLower and the MC object streamer.
static SmallVector<uint8_t> encodeMCInst(const MCInst &Inst,
                                         const LLVMState &S) {
  SmallVector<char, 16> Code;
  SmallVector<MCFixup, 4> Fixups;
  S.MCE->encodeInstruction(Inst, Code, Fixups, *S.STI);
  return SmallVector<uint8_t>(Code.begin(), Code.end());
}

/// Run the AMDGPU asm parser over \p AsmStr and return the captured MCInsts.
/// Used by assembleSingleInst() for the full parse-and-encode path, and by
/// initLLVM() / resolveOpcodeViaParse() to pick subtarget-specific opcodes
/// (e.g. s_branch, s_nop) without hardcoding opcode numbers or doing fragile
/// case-insensitive name matching over `MCInstrInfo::getName` (which returns
/// enum-style names such as `S_BRANCH_gfx12`, not the assembly mnemonic).
static SmallVector<MCInst, 2> parseAsmToMCInsts(StringRef AsmStr,
                                                const LLVMState &S) {
  S.Ctx->reset();

  // Register the buffer with the context's inline SourceMgr so that
  // MCContext::diagnose() can resolve source locations on the error path.
  // A bare local SourceMgr would be invisible to MCContext, and the asm
  // parser hits MCContext::diagnose() (via SourceMgr::PrintMessage) when
  // it encounters bad input -- without a registered SourceMgr that path
  // aborts at `Either SourceMgr should be available` in MCContext.cpp.
  S.Ctx->initInlineSourceManager();
  SourceMgr *SrcMgr = S.Ctx->getInlineSourceManager();

  std::string FullAsm = (".text\n" + AsmStr).str();
  std::unique_ptr<MemoryBuffer> Buf =
      MemoryBuffer::getMemBuffer(FullAsm, "", false);
  SrcMgr->AddNewSourceBuffer(std::move(Buf), SMLoc());

  InstCapturingStreamer Streamer(*S.Ctx);

  MCTargetOptions McOpts;
  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(*SrcMgr, *S.Ctx, Streamer, *S.MAI));
  std::unique_ptr<MCTargetAsmParser> TAP(
      S.Target->createMCAsmParser(*S.STI, *Parser, *S.MCII));
  if (!TAP) {
    log() << "hotswap: error: parseAsmToMCInsts: createMCAsmParser returned "
          << "null for asm:\n    " << AsmStr << "\n";
    return {};
  }
  Parser->setTargetParser(*TAP);

  if (Parser->Run(true)) {
    log() << "hotswap: error: parseAsmToMCInsts: Parser->Run failed for "
          << "asm:\n    " << AsmStr << "\n";
    return {};
  }

  SmallVector<MCInst, 2> Result;
  Result.reserve(Streamer.captured().size());
  for (const MCInst &Inst : Streamer.captured())
    Result.emplace_back(Inst);
  return Result;
}

/// Resolve the subtarget-appropriate MC opcode for \p AsmSnippet by letting
/// the AMDGPU asm parser pick it. \p AsmSnippet should be a minimal well-
/// formed instruction (e.g. "s_nop 0"). Returns `MCII::getNumOpcodes()` as
/// a "not found" sentinel.
static unsigned resolveOpcodeViaParse(StringRef AsmSnippet,
                                      const LLVMState &S) {
  SmallVector<MCInst, 2> Parsed = parseAsmToMCInsts(AsmSnippet, S);
  if (Parsed.size() != 1)
    return S.MCII->getNumOpcodes();
  return Parsed[0].getOpcode();
}

// -- LLVM MC target init ------------------------------------------------------

LLVMState initLLVM(const TargetIdentifier &TI) {
  LLVMState S;
  if (TI.Processor.empty()) {
    log() << "hotswap: error: initLLVM: empty CPU name in TargetIdentifier.\n";
    return S;
  }
  S.Cpu = TI.Processor.str();

  S.Target = getAmdgcnTarget();
  if (!S.Target) {
    log() << "hotswap: error: initLLVM: TargetRegistry::lookupTarget "
          << "(\"amdgcn\") failed; no AMDGPU backend registered.\n";
    return S;
  }

  Triple TT = buildTriple(TI);
  std::string Features = buildFeatureString(TI.Features);

  S.MRI.reset(S.Target->createMCRegInfo(TT));
  if (!S.MRI) {
    log() << "hotswap: error: initLLVM: createMCRegInfo failed for CPU '"
          << S.Cpu << "'.\n";
    return S;
  }

  MCTargetOptions McOpts;
  S.MAI.reset(S.Target->createMCAsmInfo(*S.MRI, TT, McOpts));
  if (!S.MAI) {
    log() << "hotswap: error: initLLVM: createMCAsmInfo failed.\n";
    return S;
  }

  S.MCII.reset(S.Target->createMCInstrInfo());
  if (!S.MCII) {
    log() << "hotswap: error: initLLVM: createMCInstrInfo failed.\n";
    return S;
  }

  S.STI.reset(S.Target->createMCSubtargetInfo(TT, S.Cpu, Features));
  if (!S.STI || !S.STI->isCPUStringValid(S.Cpu)) {
    log() << "hotswap: error: initLLVM: MCSubtargetInfo invalid for CPU '"
          << S.Cpu << "' with features '" << Features << "'.\n";
    return S;
  }

  S.Ctx = std::make_unique<MCContext>(TT, *S.MAI, S.MRI.get(), S.STI.get());
  S.MOFI = std::make_unique<MCObjectFileInfo>();
  S.MOFI->initMCObjectFileInfo(*S.Ctx, false);
  S.Ctx->setObjectFileInfo(S.MOFI.get());

  S.MCD.reset(S.Target->createMCDisassembler(*S.STI, *S.Ctx));
  if (!S.MCD) {
    log() << "hotswap: error: initLLVM: createMCDisassembler failed for "
          << "CPU '" << S.Cpu << "'.\n";
    return S;
  }

  unsigned AsmVariant = S.MAI->getAssemblerDialect();
  S.MCIP.reset(
      S.Target->createMCInstPrinter(TT, AsmVariant, *S.MAI, *S.MCII, *S.MRI));
  if (!S.MCIP) {
    log() << "hotswap: error: initLLVM: createMCInstPrinter failed for CPU '"
          << S.Cpu << "'.\n";
    return S;
  }

  S.MCE.reset(S.Target->createMCCodeEmitter(*S.MCII, *S.Ctx));
  if (!S.MCE) {
    log() << "hotswap: error: initLLVM: createMCCodeEmitter failed for CPU '"
          << S.Cpu << "'.\n";
    return S;
  }

  // MCInstrAnalysis is optional -- AMDGPU may not implement one -- so we
  // don't fail initLLVM if it comes back null. Consumers must null-check.
  S.MIA.reset(S.Target->createMCInstrAnalysis(S.MCII.get()));

  // Resolve AMDGPU instruction primitives through the asm parser so we pick
  // up the subtarget-appropriate opcode variant (e.g. S_BRANCH_gfx12 vs
  // S_BRANCH_gfx10) without hardcoding names or bits. s_branch / s_nop are
  // cached as MC opcode indices; s_nop is additionally pre-encoded to 4
  // bytes since its representation is a constant and pad loops memcpy it
  // directly.
  S.SBranchOpcode = resolveOpcodeViaParse("s_branch 0", S);
  if (S.SBranchOpcode >= S.MCII->getNumOpcodes()) {
    log() << "hotswap: error: initLLVM: failed to resolve 's_branch' opcode "
          << "via asm parser for CPU '" << S.Cpu << "'.\n";
    return S;
  }

  SmallVector<MCInst, 2> NopInsts = parseAsmToMCInsts("s_nop 0", S);
  if (NopInsts.size() != 1) {
    log() << "hotswap: error: initLLVM: failed to parse 's_nop 0' for CPU '"
          << S.Cpu << "'.\n";
    return S;
  }
  S.SNopOpcode = NopInsts[0].getOpcode();
  SmallVector<uint8_t> NopBytes = encodeMCInst(NopInsts[0], S);
  if (NopBytes.size() != MinInstSize) {
    log() << "hotswap: error: initLLVM: 's_nop 0' encoded to "
          << NopBytes.size() << " bytes; expected " << MinInstSize
          << " for CPU '" << S.Cpu << "'.\n";
    return S;
  }
  S.SNopBytes.assign(NopBytes.begin(), NopBytes.end());

  SmallVector<MCInst, 2> VNopInsts = parseAsmToMCInsts("v_nop", S);
  if (VNopInsts.size() != 1) {
    log() << "hotswap: error: initLLVM: failed to parse 'v_nop' for CPU '"
          << S.Cpu << "'.\n";
    return S;
  }
  S.VNopInst = VNopInsts[0];

  S.Valid = true;
  return S;
}

// -- LLVMState::encodeSBranch -------------------------------------------------

SmallVector<uint8_t> LLVMState::encodeSBranch(uint64_t FromOffset,
                                              uint64_t ToOffset) const {
  if (!Valid || !MCE || !MCII || SBranchOpcode >= MCII->getNumOpcodes()) {
    log() << "hotswap: error: encodeSBranch: LLVMState is not ready "
          << "(Valid=" << Valid << ", has MCE=" << (MCE != nullptr)
          << ", has MCII=" << (MCII != nullptr)
          << ", SBranchOpcode=" << SBranchOpcode << ").\n";
    return {};
  }
  int64_t ByteDelta = static_cast<int64_t>(ToOffset) -
                      static_cast<int64_t>(FromOffset) - MinInstSize;
  if (ByteDelta % MinInstSize != 0) {
    log() << "hotswap: error: encodeSBranch: unaligned byte delta " << ByteDelta
          << " from 0x" << utohexstr(FromOffset) << " to 0x"
          << utohexstr(ToOffset) << "; must be a multiple of " << MinInstSize
          << ".\n";
    return {};
  }
  int64_t DwordOffset = ByteDelta / MinInstSize;
  if (DwordOffset < BranchOffsetMin || DwordOffset > BranchOffsetMax) {
    log() << "hotswap: error: encodeSBranch: dword offset " << DwordOffset
          << " out of s_branch simm16 range [" << BranchOffsetMin << ", "
          << BranchOffsetMax << "] (from 0x" << utohexstr(FromOffset)
          << " to 0x" << utohexstr(ToOffset) << ").\n";
    return {};
  }

  MCInst Inst;
  Inst.setOpcode(SBranchOpcode);
  Inst.addOperand(MCOperand::createImm(DwordOffset));
  SmallVector<uint8_t> Bytes = encodeMCInst(Inst, *this);
  if (Bytes.size() != MinInstSize) {
    log() << "hotswap: error: encodeSBranch: MCCodeEmitter produced "
          << Bytes.size() << " bytes for s_branch (opcode index "
          << SBranchOpcode << "); expected " << MinInstSize << ".\n";
    return {};
  }
  return Bytes;
}

// -- Instruction decode -------------------------------------------------------

bool decodeTextSection(const uint8_t *Text, uint64_t TextSize,
                       const LLVMState &S,
                       std::vector<InternalDecodedInst> &Decoded) {
  Decoded.reserve(Decoded.size() + TextSize / MinInstSize);
  uint64_t Pos = 0;
  while (Pos < TextSize) {
    InternalDecodedInst DI;
    DI.Offset = Pos;

    ArrayRef<uint8_t> Bytes(Text + Pos, TextSize - Pos);
    uint64_t InstSize = 0;
    MCDisassembler::DecodeStatus Status =
        S.MCD->getInstruction(DI.Inst, InstSize, Bytes, Pos, nulls());

    if (Status == MCDisassembler::Fail) {
      DI.Size = MinInstSize;
      DI.Mnemonic = UnknownMnemonic.str();
    } else {
      DI.Size = static_cast<uint32_t>(InstSize);
      // MCInstPrinter::getMnemonic returns a pointer into the tblgen-generated
      // AsmStrs table (see AMDGPUGenAsmWriter.inc). Storage is process-
      // lifetime static; the trailing whitespace baked into AsmStrs must be
      // trimmed. Falls back to MCII->getName for targets that leave it null.
      if (S.MCIP) {
        std::pair<const char *, uint64_t> Mnem = S.MCIP->getMnemonic(DI.Inst);
        DI.Mnemonic = Mnem.first ? StringRef(Mnem.first).rtrim().str()
                                 : S.MCII->getName(DI.Inst.getOpcode()).str();
      } else {
        DI.Mnemonic = S.MCII->getName(DI.Inst.getOpcode()).str();
      }
    }
    Pos += DI.Size;
    Decoded.emplace_back(std::move(DI));
  }
  return true;
}

// -- assembleSingleInst -------------------------------------------------------

SmallVector<uint8_t> assembleSingleInst(StringRef AsmStr, const LLVMState &S) {
  // Parse \p AsmStr through the shared parseAsmToMCInsts helper, then encode
  // each captured MCInst via the cached MCCodeEmitter. Avoids the old
  // createMCObjectStreamer -> ELF parse -> extract .text round trip.
  SmallVector<MCInst, 2> Insts = parseAsmToMCInsts(AsmStr, S);
  if (Insts.empty()) {
    log() << "hotswap: error: assembleSingleInst: parser produced no "
          << "instructions for asm:\n    " << AsmStr << "\n";
    return {};
  }

  SmallVector<uint8_t> Bytes;
  for (const MCInst &Inst : Insts) {
    SmallVector<uint8_t> InstBytes = encodeMCInst(Inst, S);
    Bytes.append(InstBytes.begin(), InstBytes.end());
  }
  return Bytes;
}

// -- buildTrampoline ----------------------------------------------------------

Trampoline buildTrampoline(ArrayRef<std::string> AsmLines,
                           uint64_t OriginalOffset, uint32_t OriginalSize,
                           uint64_t TrampolineTextOffset, const LLVMState &S) {
  Trampoline Result;
  Result.OriginalOffset = OriginalOffset;
  Result.OriginalSize = OriginalSize;

  std::string AsmSource;
  for (StringRef Line : AsmLines) {
    AsmSource += Line;
    AsmSource += '\n';
  }

  SmallVector<uint8_t> Bytes = assembleSingleInst(AsmSource, S);
  if (Bytes.empty()) {
    log() << "hotswap: error: buildTrampoline: assembleSingleInst returned "
          << "empty for trampoline originating at offset 0x"
          << utohexstr(OriginalOffset) << " (" << AsmLines.size()
          << " asm lines).\n";
    return Result;
  }

  Result.Bytes = std::move(Bytes);

  uint64_t BranchBackFrom = TrampolineTextOffset + Result.Bytes.size();
  uint64_t BranchBackTo = OriginalOffset + OriginalSize;

  SmallVector<uint8_t> BranchBytes =
      S.encodeSBranch(BranchBackFrom, BranchBackTo);
  if (BranchBytes.empty()) {
    log() << "hotswap: error: buildTrampoline: encodeSBranch failed for "
          << "branch-back from trampoline offset 0x"
          << utohexstr(BranchBackFrom) << " to original offset 0x"
          << utohexstr(BranchBackTo) << "; clearing trampoline.\n";
    Result.Bytes.clear();
    return Result;
  }

  Result.Bytes.append(BranchBytes.begin(), BranchBytes.end());
  return Result;
}

Trampoline buildTrampoline(ArrayRef<MCInst> Insts, uint64_t OriginalOffset,
                           uint32_t OriginalSize, uint64_t TrampolineTextOffset,
                           const LLVMState &S) {
  Trampoline Result;
  Result.OriginalOffset = OriginalOffset;
  Result.OriginalSize = OriginalSize;

  for (const MCInst &Inst : Insts) {
    SmallVector<uint8_t> InstBytes = encodeMCInst(Inst, S);
    if (InstBytes.empty()) {
      log() << "hotswap: error: buildTrampoline(MCInst): encodeMCInst failed "
            << "for opcode " << Inst.getOpcode() << " at trampoline for 0x"
            << utohexstr(OriginalOffset) << "\n";
      Result.Bytes.clear();
      return Result;
    }
    Result.Bytes.append(InstBytes.begin(), InstBytes.end());
  }

  uint64_t BranchBackFrom = TrampolineTextOffset + Result.Bytes.size();
  uint64_t BranchBackTo = OriginalOffset + OriginalSize;

  SmallVector<uint8_t> BranchBytes =
      S.encodeSBranch(BranchBackFrom, BranchBackTo);
  if (BranchBytes.empty()) {
    log() << "hotswap: error: buildTrampoline(MCInst): encodeSBranch failed "
          << "for branch-back from 0x" << utohexstr(BranchBackFrom) << " to 0x"
          << utohexstr(BranchBackTo) << "; clearing trampoline.\n";
    Result.Bytes.clear();
    return Result;
  }

  Result.Bytes.append(BranchBytes.begin(), BranchBytes.end());
  return Result;
}

// -- WMMA co-execution hazard overlap check -----------------------------------

bool checkVgprOverlap(const MCInst &WmmaInst, const MCInst &ValuInst,
                      const MCRegisterInfo &MRI) {
  // Delegates register-aliasing to MCRegisterInfo::regsOverlap, which walks
  // regunits and handles VGPR tuples, sub-registers, and alias classes. Mirrors
  // the upstream pattern used by GCNHazardRecognizer::hasWMMAToVALURegOverlap.
  static constexpr unsigned DestOperandIdx = 0;
  if (ValuInst.getNumOperands() <= DestOperandIdx)
    return false;
  const MCOperand &DestOp = ValuInst.getOperand(DestOperandIdx);
  if (!DestOp.isReg())
    return false;

  for (const MCOperand &Op : WmmaInst)
    if (Op.isReg() && MRI.regsOverlap(Op.getReg(), DestOp.getReg()))
      return true;
  return false;
}

} // namespace hotswap
} // namespace COMGR
