//===- HotswapMCTest.cpp - Unit tests for HotSwap LLVM MC layer -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for the hotswap MC/LLVM infrastructure in comgr-hotswap-llvm.cpp:
/// initLLVM construction, LLVMState::encodeSBranch, assembleSingleInst /
/// decodeTextSection round-trip, applyMnemonicSwap, applyByteReplace, and
/// checkVgprOverlap.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "comgr-test-elf-utils.h"
#include "comgr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

using namespace COMGR;
using namespace COMGR::hotswap;

// --------------------------------------------------------------------------
// Test-only stub definition of COMGR::ensureLLVMInitialized.
//
// hotswap::initLLVM() calls COMGR::ensureLLVMInitialized() (normally defined
// in comgr.cpp) to register the AMDGPU target. The production definition
// lives in libamd_comgr, which we don't want to link into the unit-test
// binary (it drags in the full Comgr compiler pipeline). Providing this
// stub here keeps the test binary minimal while matching the production
// registration behaviour for the target components we exercise.
//
// Stubbing is safe because this translation unit is linked into
// HotswapMCTests only, never into libamd_comgr.
// --------------------------------------------------------------------------
namespace COMGR {
void ensureLLVMInitialized() {
  static std::once_flag Once;
  std::call_once(Once, []() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTarget();
  });
}
} // namespace COMGR

// Build a TargetIdentifier for the gfx1250 test subtarget without features --
// production callers go through parseTargetIdentifier; here we populate
// directly so the tests stay self-contained.
static TargetIdentifier makeGfx1250Ident() {
  TargetIdentifier TI;
  TI.Arch = "amdgcn";
  TI.Vendor = "amd";
  TI.OS = "amdhsa";
  TI.Environ = "";
  TI.Processor = "gfx1250";
  return TI;
}

// Helper: decode the little-endian 32-bit dword at \p Bytes.
static uint32_t readDword(const uint8_t *Bytes) {
  uint32_t V;
  std::memcpy(&V, Bytes, sizeof(V));
  return V;
}

// -- initLLVM ----------------------------------------------------------------

TEST(InitLLVM, ValidGfx1250) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_EQ(S.Cpu, "gfx1250");
  EXPECT_NE(S.Target, nullptr);
  ASSERT_NE(S.MCII, nullptr);
  EXPECT_LT(S.SBranchOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SClauseOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SDelayAluOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SEndPgmOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SEndPgmSavedOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SAddPcI64Opcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SCallI64Opcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SSwapPcI64Opcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SPrefetchInstPcRelOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SPrefetchDataPcRelOpcode, S.MCII->getNumOpcodes());
  EXPECT_TRUE(S.SCCRegister.isValid());
  EXPECT_EQ(S.SNopBytes.size(), MinInstSize);
}

TEST(InitLLVM, EmptyProcessorFails) {
  TargetIdentifier TI = makeGfx1250Ident();
  TI.Processor = "";
  LLVMState S = initLLVM(TI);
  EXPECT_FALSE(S.Valid);
}

TEST(InitLLVM, UnknownProcessorFails) {
  TargetIdentifier TI = makeGfx1250Ident();
  TI.Processor = "gfxbogus";
  LLVMState S = initLLVM(TI);
  EXPECT_FALSE(S.Valid);
}

// -- LLVMState::encodeSBranch -------------------------------------------------
//
// Exact byte checks are avoided here -- tblgen encodings can be reshuffled
// across LLVM versions. Instead we assert the structural invariants that
// downstream callers rely on: the encoded delta round-trips to the expected
// simm16 field, the size is MinInstSize, and out-of-range / unaligned deltas
// are rejected.

TEST(EncodeSBranch, ForwardBranchRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // s_branch SIMM16 -> PC += (SIMM16 + 1) * 4; From=0, To=8 => SIMM16=1.
  llvm::SmallVector<uint8_t> Out = S.encodeSBranch(0, 8);
  ASSERT_EQ(Out.size(), MinInstSize);
  uint32_t Encoded = readDword(Out.data());
  EXPECT_EQ(static_cast<uint16_t>(Encoded & 0xFFFFu), 1u);
}

TEST(EncodeSBranch, BackwardBranchRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // From=16, To=0 => delta=-5 dwords.
  llvm::SmallVector<uint8_t> Out = S.encodeSBranch(16, 0);
  ASSERT_EQ(Out.size(), MinInstSize);
  uint32_t Encoded = readDword(Out.data());
  EXPECT_EQ(static_cast<int16_t>(Encoded & 0xFFFFu), -5);
}

TEST(EncodeSBranch, ZeroOffsetBranch) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // PC advance of MinInstSize: SIMM16 should be 0.
  llvm::SmallVector<uint8_t> Out = S.encodeSBranch(0, MinInstSize);
  ASSERT_EQ(Out.size(), MinInstSize);
  EXPECT_EQ(readDword(Out.data()) & 0xFFFFu, 0u);
}

TEST(EncodeSBranch, UnalignedDeltaFails) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_TRUE(S.encodeSBranch(0, 7).empty());
}

TEST(EncodeSBranch, OutOfRangeFails) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_TRUE(S.encodeSBranch(0, 500000).empty());
}

TEST(EncodeSBranch, PositiveBoundaryRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  constexpr uint64_t To =
      static_cast<uint64_t>(BranchOffsetMax + 1) * MinInstSize;
  llvm::SmallVector<uint8_t> Out = S.encodeSBranch(0, To);
  ASSERT_EQ(Out.size(), MinInstSize);
  uint32_t Encoded = readDword(Out.data());
  EXPECT_EQ(static_cast<int16_t>(Encoded & 0xFFFFu), BranchOffsetMax);
  EXPECT_TRUE(S.encodeSBranch(0, To + MinInstSize).empty());
}

TEST(EncodeSBranch, NegativeBoundaryRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  constexpr uint64_t From =
      static_cast<uint64_t>(-(BranchOffsetMin + 1)) * MinInstSize;
  llvm::SmallVector<uint8_t> Out = S.encodeSBranch(From, 0);
  ASSERT_EQ(Out.size(), MinInstSize);
  uint32_t Encoded = readDword(Out.data());
  EXPECT_EQ(static_cast<int16_t>(Encoded & 0xFFFFu), BranchOffsetMin);
  EXPECT_TRUE(S.encodeSBranch(From + MinInstSize, 0).empty());
}

TEST(EncodeSBranch, FailsOnInvalidState) {
  LLVMState S; // default-constructed, Valid = false
  EXPECT_TRUE(S.encodeSBranch(0, 8).empty());
}

// -- encodeSetPCLongBranch ---------------------------------------------------

TEST(EncodeSetPCLongBranch, UsesSccPreservingSequenceWithoutAddPc) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  const uint64_t From = 0x81000;
  const uint64_t To = 0x1004;
  std::optional<llvm::SmallVector<uint8_t>> Out =
      encodeSetPCLongBranch(S, From, To, /*SgprBase=*/12);
  ASSERT_TRUE(Out);

  std::vector<InternalDecodedInst> Dec;
  ASSERT_TRUE(decodeTextSection(Out->data(), Out->size(), S, Dec));
  ASSERT_EQ(Dec.size(), 6u);
  EXPECT_EQ(Dec[0].Mnemonic, "s_cselect_b32");
  EXPECT_EQ(Dec[1].Mnemonic, "s_get_pc_i64");
  EXPECT_EQ(Dec[2].Mnemonic, "s_add_co_u32");
  EXPECT_EQ(Dec[3].Mnemonic, "s_add_co_ci_u32");
  EXPECT_EQ(Dec[4].Mnemonic, "s_cmp_lg_u32");
  EXPECT_EQ(Dec[5].Mnemonic, "s_set_pc_i64");
  for (const InternalDecodedInst &DI : Dec)
    EXPECT_NE(DI.Mnemonic, "s_add_pc_i64");

  // s_get_pc_i64 is the second dword and captures From + 8. The two add
  // immediates materialize this exact two's-complement displacement.
  uint64_t Delta = To - (From + 2 * MinInstSize);
  EXPECT_EQ(static_cast<uint32_t>(Delta), 0xFFF7FFFCu);
  EXPECT_EQ(static_cast<uint32_t>(Delta >> 32), 0xFFFFFFFFu);
}

TEST(EncodeSetPCLongBranch, ForwardLandsOnTarget) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t From = 0x1000;
  constexpr uint64_t To = 0x81000;
  std::optional<llvm::SmallVector<uint8_t>> Out =
      encodeSetPCLongBranch(S, From, To, /*SgprBase=*/12);
  ASSERT_TRUE(Out);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Out->data(), Out->size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 6u);
  ASSERT_TRUE(Decoded[2].Inst.getOperand(2).isImm());
  ASSERT_TRUE(Decoded[3].Inst.getOperand(2).isImm());
  uint64_t Lo = static_cast<uint32_t>(Decoded[2].Inst.getOperand(2).getImm());
  uint64_t Hi = static_cast<uint32_t>(Decoded[3].Inst.getOperand(2).getImm());
  uint64_t Delta = Lo | (Hi << 32);
  EXPECT_EQ(From + 2 * MinInstSize + Delta, To);
}

TEST(EncodeSetPCLongBranch, RejectsPcBaseOverflow) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_FALSE(encodeSetPCLongBranch(
      S, std::numeric_limits<uint64_t>::max() - MinInstSize, 0,
      /*SgprBase=*/12));
}

TEST(EncodeSetPCLongBranch, RejectsMisalignedScratchPair) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_FALSE(encodeSetPCLongBranch(S, 0, 0x1000, /*SgprBase=*/3));
}

TEST(IsSBranchReachable, CoversBoundariesAlignmentAndPcOverflow) {
  constexpr uint64_t PositiveLimit =
      static_cast<uint64_t>(BranchOffsetMax + 1) * MinInstSize;
  EXPECT_TRUE(isSBranchReachable(/*From=*/0, PositiveLimit));
  EXPECT_FALSE(isSBranchReachable(/*From=*/0, PositiveLimit + MinInstSize));
  EXPECT_FALSE(isSBranchReachable(/*From=*/0, /*To=*/7));

  constexpr uint64_t NegativeFrom =
      static_cast<uint64_t>(-(BranchOffsetMin + 1)) * MinInstSize;
  EXPECT_TRUE(isSBranchReachable(NegativeFrom, /*To=*/0));
  EXPECT_FALSE(isSBranchReachable(NegativeFrom + MinInstSize, /*To=*/0));
  EXPECT_FALSE(isSBranchReachable(std::numeric_limits<uint64_t>::max() - 1,
                                  /*To=*/0));
}

TEST(EvaluateDirectControlFlowTarget, EvaluatesImmediateBranch) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst("s_branch 1", S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  Decoded[0].Offset = 0x100;
  EXPECT_EQ(evaluateDirectControlFlowTarget(Decoded[0], S), 0x108u);
}

TEST(EvaluateDirectControlFlowTarget, EvaluatesGfx1250CallOperandFallback) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("s_call_i64 s[0:1], 2", S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  Decoded[0].Offset = 0x200;
  EXPECT_EQ(evaluateDirectControlFlowTarget(Decoded[0], S),
            0x200u + Decoded[0].Size + 2 * MinInstSize);
}

TEST(SafeSgprScratchBlock, RejectsRegisterBeyondAddressableLimit) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_mov_b32 s4, s0", S);
  ASSERT_FALSE(Text.empty());

  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  ElfView &View = *ViewOrErr;

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(View.textData(), View.textSize(), S, Decoded));
  RewriteConfig Config;
  Config.MaxSgprs = 4;
  std::vector<Trampoline> Trampolines;
  std::vector<NopSled> Sleds;
  LivenessInfo Liveness;
  llvm::StringMap<KernelPatchStats> KernelStats;
  std::vector<ScratchPatchInfo> ScratchPatches;
  PatchContext Ctx{Config,
                   Decoded,
                   View.textData(),
                   View.textSize(),
                   /*PoolBaseOffset=*/0,
                   S,
                   Trampolines,
                   Sleds,
                   View,
                   Liveness,
                   KernelStats,
                   ScratchPatches};

  EXPECT_FALSE(findSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, /*Count=*/1,
                                        /*Alignment=*/1, "unit test"));
}

TEST(SafeSgprScratchBlock, CommitRejectsObjectWithoutKernelDescriptor) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_FALSE(Text.empty());

  comgr_test::KernelDescriptorElfOptions Options;
  Options.EmitKernelDescriptorSymbol = false;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Options);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  ElfView &View = *ViewOrErr;

  std::vector<InternalDecodedInst> Decoded;
  RewriteConfig Config;
  Config.MaxSgprs = 106;
  std::vector<Trampoline> Trampolines;
  std::vector<NopSled> Sleds;
  LivenessInfo Liveness;
  llvm::StringMap<KernelPatchStats> KernelStats;
  std::vector<ScratchPatchInfo> ScratchPatches;
  PatchContext Ctx{Config,
                   Decoded,
                   View.textData(),
                   View.textSize(),
                   /*PoolBaseOffset=*/0,
                   S,
                   Trampolines,
                   Sleds,
                   View,
                   Liveness,
                   KernelStats,
                   ScratchPatches};

  const SafeSgprScratchBlock Block{/*Base=*/4, /*Count=*/1};
  EXPECT_FALSE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Block, "unit test"));
}

TEST(FindNearestSled, RejectsOverflowingHeadroom) {
  std::vector<NopSled> Sleds = {{0, 64, 60, 0, 64}, {100, 128, 100, 100, 128}};
  EXPECT_EQ(findNearestSled(Sleds, 0, std::numeric_limits<uint64_t>::max()),
            nullptr);
}

TEST(FindNearestSled, HandlesLargeUnsignedOffsets) {
  std::vector<NopSled> Sleds = {{100, 128, 100, 100, 128},
                                {std::numeric_limits<uint64_t>::max() - 32,
                                 std::numeric_limits<uint64_t>::max(),
                                 std::numeric_limits<uint64_t>::max() - 32,
                                 std::numeric_limits<uint64_t>::max() - 64,
                                 std::numeric_limits<uint64_t>::max()}};
  NopSled *Sled =
      findNearestSled(Sleds, std::numeric_limits<uint64_t>::max() - 40,
                      /*Needed=*/8);
  ASSERT_NE(Sled, nullptr);
  EXPECT_EQ(Sled, &Sleds[1]);
}

// -- assembleSingleInst / decodeTextSection round-trip ------------------------

TEST(AssembleDecode, SNopRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst("s_nop 0", S);
  ASSERT_EQ(Bytes.size(), MinInstSize);
  // Must match the pre-encoded bytes cached in LLVMState at init time.
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(Bytes),
            llvm::ArrayRef<uint8_t>(S.SNopBytes));

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  EXPECT_TRUE(Decoded[0].DecodeSucceeded);
  EXPECT_EQ(Decoded[0].Size, MinInstSize);
  EXPECT_EQ(Decoded[0].Mnemonic, "s_nop");
}

TEST(AssembleDecode, CvtPkFp8LiteralSourcesDecodeAsTwelveBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(
      "v_cvt_pk_fp8_f32 v4, 0x477f0000, 0x477f0000 clamp", S);
  ASSERT_EQ(Bytes.size(), 3u * MinInstSize);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  const InternalDecodedInst &DI = Decoded[0];
  EXPECT_EQ(DI.Size, 3u * MinInstSize);
  EXPECT_EQ(DI.Mnemonic, "v_cvt_pk_fp8_f32");

  const llvm::MCInst &Inst = DI.Inst;
  ASSERT_GE(Inst.getNumOperands(), 7u);
  EXPECT_TRUE(Inst.getOperand(0).isReg());
  ASSERT_TRUE(Inst.getOperand(2).isImm());
  EXPECT_EQ(Inst.getOperand(2).getImm(), 0x477f0000);
  ASSERT_TRUE(Inst.getOperand(4).isImm());
  EXPECT_EQ(Inst.getOperand(4).getImm(), 0x477f0000);
  ASSERT_TRUE(Inst.getOperand(5).isImm());
  EXPECT_EQ(Inst.getOperand(5).getImm(), 1);
}

TEST(AssembleDecode, CvtPkFp8MixedLiteralSourcesDecodeAsTwelveBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Src0LiteralBytes =
      assembleSingleInst("v_cvt_pk_fp8_f32 v4, 0x477f0000, v5 clamp", S);
  ASSERT_EQ(Src0LiteralBytes.size(), 3u * MinInstSize);

  std::vector<InternalDecodedInst> Src0LiteralDecoded;
  ASSERT_TRUE(decodeTextSection(
      Src0LiteralBytes.data(), Src0LiteralBytes.size(), S, Src0LiteralDecoded));
  ASSERT_EQ(Src0LiteralDecoded.size(), 1u);
  const llvm::MCInst &Src0LiteralInst = Src0LiteralDecoded[0].Inst;
  ASSERT_GE(Src0LiteralInst.getNumOperands(), 7u);
  ASSERT_TRUE(Src0LiteralInst.getOperand(2).isImm());
  EXPECT_EQ(Src0LiteralInst.getOperand(2).getImm(), 0x477f0000);
  EXPECT_TRUE(Src0LiteralInst.getOperand(4).isReg());

  llvm::SmallVector<uint8_t> Src1LiteralBytes = assembleSingleInst(
      "v_cvt_pk_fp8_f32 v4, v5, 0.3333333432674408 clamp", S);
  ASSERT_EQ(Src1LiteralBytes.size(), 3u * MinInstSize);

  std::vector<InternalDecodedInst> Src1LiteralDecoded;
  ASSERT_TRUE(decodeTextSection(
      Src1LiteralBytes.data(), Src1LiteralBytes.size(), S, Src1LiteralDecoded));
  ASSERT_EQ(Src1LiteralDecoded.size(), 1u);
  const llvm::MCInst &Src1LiteralInst = Src1LiteralDecoded[0].Inst;
  ASSERT_GE(Src1LiteralInst.getNumOperands(), 7u);
  EXPECT_TRUE(Src1LiteralInst.getOperand(2).isReg());
  ASSERT_TRUE(Src1LiteralInst.getOperand(4).isImm());
  EXPECT_EQ(Src1LiteralInst.getOperand(4).getImm(), 0x3eaaaaab);
}

TEST(AssembleDecode, CvtPkFp8InlineConstantsDecodeAsEightBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("v_cvt_pk_fp8_f32 v4, 1.0, 0.5 clamp", S);
  ASSERT_EQ(Bytes.size(), 2u * MinInstSize);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  const InternalDecodedInst &DI = Decoded[0];
  EXPECT_EQ(DI.Size, 2u * MinInstSize);
  EXPECT_EQ(DI.Mnemonic, "v_cvt_pk_fp8_f32");

  const llvm::MCInst &Inst = DI.Inst;
  ASSERT_GE(Inst.getNumOperands(), 7u);
  ASSERT_TRUE(Inst.getOperand(2).isImm());
  EXPECT_EQ(Inst.getOperand(2).getImm(), 0x3f800000);
  ASSERT_TRUE(Inst.getOperand(4).isImm());
  EXPECT_EQ(Inst.getOperand(4).getImm(), 0x3f000000);
  ASSERT_TRUE(Inst.getOperand(5).isImm());
  EXPECT_EQ(Inst.getOperand(5).getImm(), 1);
}

TEST(AssembleDecode, RejectsGarbageAsm) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst("not_a_real_op", S);
  EXPECT_TRUE(Bytes.empty());
}

// -- applyByteReplace ---------------------------------------------------------

TEST(ApplyByteReplace, PadsWithSNop) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // 8 bytes of zeroed "text", simulate replacing the first 8 bytes with a
  // 4-byte rule and expecting the remainder to be padded with s_nop.
  uint8_t Text[8] = {};
  RewriteRule Rule;
  Rule.ReplaceBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());
  ASSERT_TRUE(applyByteReplace(Rule, /*InstOffset=*/0, /*InstSize=*/8, Text,
                               sizeof(Text), S));
  // Both halves should be s_nop bytes now.
  EXPECT_EQ(std::memcmp(Text, S.SNopBytes.data(), MinInstSize), 0);
  EXPECT_EQ(std::memcmp(Text + MinInstSize, S.SNopBytes.data(), MinInstSize),
            0);
}

TEST(ApplyByteReplace, RejectsOutOfBounds) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  uint8_t Text[4] = {};
  RewriteRule Rule;
  Rule.ReplaceBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());
  // InstOffset+InstSize (8) exceeds TextSize (4).
  EXPECT_FALSE(applyByteReplace(Rule, /*InstOffset=*/0, /*InstSize=*/8, Text,
                                sizeof(Text), S));
}

// -- checkVgprOverlap ---------------------------------------------------------
//
// checkVgprOverlap checks whether any register operand of a "WMMA-like"
// MCInst overlaps the destination (operand 0) of a "VALU-like" MCInst.
// We drive it with real MCInsts produced by assembling + decoding simple
// AMDGPU instructions so the register operands are populated the way the
// production code sees them.

// Assemble \p Asm and decode the first resulting MCInst. Aborts the test if
// either step fails, so callers can rely on the return value being populated.
static llvm::MCInst assembleOne(llvm::StringRef Asm, const LLVMState &S) {
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, S);
  EXPECT_FALSE(Bytes.empty()) << "failed to assemble: " << Asm.str();
  std::vector<InternalDecodedInst> Decoded;
  EXPECT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded))
      << "failed to decode: " << Asm.str();
  EXPECT_EQ(Decoded.size(), 1u) << "expected one inst for: " << Asm.str();
  return Decoded.empty() ? llvm::MCInst() : Decoded[0].Inst;
}

static void expectSameOperands(const llvm::MCInst &Actual,
                               const llvm::MCInst &Expected,
                               llvm::StringRef Context) {
  EXPECT_EQ(Actual.getOpcode(), Expected.getOpcode()) << Context.str();
  ASSERT_EQ(Actual.getNumOperands(), Expected.getNumOperands())
      << Context.str();
  for (unsigned I = 0, E = Actual.getNumOperands(); I != E; ++I) {
    const llvm::MCOperand &ActualOp = Actual.getOperand(I);
    const llvm::MCOperand &ExpectedOp = Expected.getOperand(I);
    EXPECT_EQ(ActualOp.isReg(), ExpectedOp.isReg())
        << Context.str() << " operand " << I;
    EXPECT_EQ(ActualOp.isImm(), ExpectedOp.isImm())
        << Context.str() << " operand " << I;
    EXPECT_EQ(ActualOp.isSFPImm(), ExpectedOp.isSFPImm())
        << Context.str() << " operand " << I;
    EXPECT_EQ(ActualOp.isDFPImm(), ExpectedOp.isDFPImm())
        << Context.str() << " operand " << I;
    EXPECT_EQ(ActualOp.isExpr(), ExpectedOp.isExpr())
        << Context.str() << " operand " << I;
    if (ExpectedOp.isReg()) {
      EXPECT_EQ(ActualOp.getReg(), ExpectedOp.getReg())
          << Context.str() << " operand " << I;
    } else if (ExpectedOp.isImm()) {
      EXPECT_EQ(ActualOp.getImm(), ExpectedOp.getImm())
          << Context.str() << " operand " << I;
    } else if (ExpectedOp.isSFPImm()) {
      EXPECT_EQ(ActualOp.getSFPImm(), ExpectedOp.getSFPImm())
          << Context.str() << " operand " << I;
    } else if (ExpectedOp.isDFPImm()) {
      EXPECT_EQ(ActualOp.getDFPImm(), ExpectedOp.getDFPImm())
          << Context.str() << " operand " << I;
    }
  }
}

static void expectInstMatchesAsm(const llvm::MCInst &Actual,
                                 llvm::StringRef Asm, const LLVMState &S) {
  llvm::MCInst Expected = assembleOne(Asm, S);
  expectSameOperands(Actual, Expected, Asm);
}

static bool appendSingleInstBytes(llvm::SmallVectorImpl<uint8_t> &Bytes,
                                  llvm::StringRef Asm, const LLVMState &S) {
  llvm::SmallVector<uint8_t> Inst = assembleSingleInst(Asm, S);
  if (Inst.empty()) {
    ADD_FAILURE() << "failed to assemble: " << Asm.str();
    return false;
  }
  Bytes.append(Inst.begin(), Inst.end());
  return true;
}

TEST(CheckVgprOverlap, DetectsDirectOverlap) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // Wmma-like inst references v5 and v10; Valu-like inst writes v10.
  llvm::MCInst Wmma = assembleOne("v_mov_b32 v5, v10", S);
  llvm::MCInst Valu = assembleOne("v_mov_b32 v10, v20", S);
  EXPECT_TRUE(checkVgprOverlap(Wmma, Valu, *S.MRI));
}

TEST(CheckVgprOverlap, NoOverlapForDisjointVgprs) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // Wmma-like inst references v0, v1; Valu-like inst writes v10.
  llvm::MCInst Wmma = assembleOne("v_mov_b32 v0, v1", S);
  llvm::MCInst Valu = assembleOne("v_mov_b32 v10, v20", S);
  EXPECT_FALSE(checkVgprOverlap(Wmma, Valu, *S.MRI));
}

TEST(CheckVgprOverlap, HandlesEmptyValuInst) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::MCInst Wmma = assembleOne("v_mov_b32 v0, v1", S);
  llvm::MCInst Empty; // no operands
  EXPECT_FALSE(checkVgprOverlap(Wmma, Empty, *S.MRI));
}

// -- buildTrampoline ----------------------------------------------------------
//
// buildTrampoline assembles one or more asm lines and appends a branch-back
// s_branch to the instruction immediately following the original site. We
// verify the size / structure of the result rather than the exact bytes
// (which are target-specific and captured separately in the encodeSBranch /
// SNopBytes tests).

TEST(BuildTrampoline, AppendsBranchBackAfterAssembledAsm) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::string AsmLine = "s_nop 0";
  std::vector<std::string> AsmLines = {AsmLine};
  constexpr uint64_t OriginalOffset = 0;
  constexpr uint32_t OriginalSize = MinInstSize;
  constexpr uint64_t TrampolineTextOffset = 0x1000;

  Trampoline T = buildTrampoline(AsmLines, OriginalOffset, OriginalSize,
                                 TrampolineTextOffset, S);

  EXPECT_EQ(T.OriginalOffset, OriginalOffset);
  EXPECT_EQ(T.OriginalSize, OriginalSize);
  // One assembled inst (s_nop 0, 4 bytes) + one branch-back (4 bytes).
  ASSERT_EQ(T.Bytes.size(), 2u * MinInstSize);
  // The first MinInstSize bytes should match the cached s_nop encoding.
  EXPECT_EQ(std::memcmp(T.Bytes.data(), S.SNopBytes.data(), MinInstSize), 0);
}

TEST(BuildTrampoline, EmptyOnBadAsm) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<std::string> AsmLines = {"this_is_not_a_valid_instruction"};
  Trampoline T = buildTrampoline(AsmLines, /*OriginalOffset=*/0,
                                 /*OriginalSize=*/MinInstSize,
                                 /*TrampolineTextOffset=*/0x1000, S);
  EXPECT_TRUE(T.Bytes.empty());
}

// -- DS two-address expansion ------------------------------------------------

TEST(ExpandDs2Addr, PreservesAddressNeededBySecondLoad) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(
      "ds_load_2addr_b64 v[12:15], v12 offset0:0 offset1:1", S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);

  std::optional<std::vector<std::string>> Expanded =
      expandDs2Addr(Decoded[0].Inst, Decoded[0].Mnemonic, "ds_load_b64", S);
  ASSERT_TRUE(Expanded);
  ASSERT_EQ(Expanded->size(), 2u);
  EXPECT_EQ((*Expanded)[0], "ds_load_b64 v[14:15], v12 offset:8");
  EXPECT_EQ((*Expanded)[1], "ds_load_b64 v[12:13], v12");
}

TEST(ExpandDs2Addr, RejectsCyclicExchangeDependency) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(
      "ds_storexchg_2addr_rtn_b64 v[20:23], v24, v[22:23], v[20:21] "
      "offset0:0 offset1:1",
      S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);

  EXPECT_FALSE(expandDs2Addr(Decoded[0].Inst, Decoded[0].Mnemonic,
                             "ds_storexchg_rtn_b64", S));
}

// -- buildKernelEntryTrampoline -----------------------------------------------

TEST(BuildKernelEntryTrampoline, BuildsRecognizedPcRelativeStub) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t StubVAddr = 0x200000;
  constexpr uint64_t EntryVAddr = 0x10100;
  llvm::SmallVector<uint8_t> GlobalWb = assembleSingleInst("global_wb", S);
  ASSERT_EQ(GlobalWb.size(), 3 * MinInstSize);

  llvm::SmallVector<uint8_t> Bytes =
      buildKernelEntryTrampoline(StubVAddr, EntryVAddr, /*ScratchSgpr=*/8, S);

  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);
  EXPECT_TRUE(isKernelEntryTrampoline(Bytes, S));

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_GE(Decoded.size(), 6u);
  EXPECT_EQ(Decoded[0].Inst.getOpcode(), S.GlobalWbOpcode);
  EXPECT_EQ(Decoded[1].Inst.getOpcode(), S.VNopInst.getOpcode());
  EXPECT_EQ(Decoded[2].Inst.getOpcode(), S.SGetPcI64Opcode);
  EXPECT_EQ(Decoded[3].Inst.getOpcode(), S.SAddU32Opcode);
  EXPECT_EQ(Decoded[4].Inst.getOpcode(), S.SAddcU32Opcode);
  EXPECT_EQ(Decoded[5].Inst.getOpcode(), S.SSetPcI64Opcode);

  const uint64_t PcBase = StubVAddr + Decoded[2].Offset + Decoded[2].Size;
  const uint64_t Delta = EntryVAddr - PcBase;
  const uint32_t Lo = static_cast<uint32_t>(Delta);
  const uint32_t Hi = static_cast<uint32_t>(Delta >> 32);
  expectInstMatchesAsm(Decoded[0].Inst, "global_wb", S);
  expectInstMatchesAsm(Decoded[1].Inst, "v_nop", S);
  expectInstMatchesAsm(Decoded[2].Inst, "s_get_pc_i64 s[8:9]", S);
  expectInstMatchesAsm(
      Decoded[3].Inst,
      (llvm::Twine("s_add_u32 s8, s8, 0x") + llvm::utohexstr(Lo)).str(), S);
  expectInstMatchesAsm(
      Decoded[4].Inst,
      (llvm::Twine("s_addc_u32 s9, s9, 0x") + llvm::utohexstr(Hi)).str(), S);
  expectInstMatchesAsm(Decoded[5].Inst, "s_set_pc_i64 s[8:9]", S);
}

TEST(BuildKernelEntryTrampoline, PrefixPrefiltersNonStubBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Stub =
      buildKernelEntryTrampoline(/*StubVAddr=*/0x200000,
                                 /*EntryVAddr=*/0x10100,
                                 /*ScratchSgpr=*/8, S);
  ASSERT_EQ(Stub.size(), KernelEntryStubStride);
  EXPECT_TRUE(hasKernelEntryTrampolinePrefix(Stub, S));

  llvm::SmallVector<uint8_t> NonStub;
  ASSERT_TRUE(appendSingleInstBytes(NonStub, "s_endpgm", S));
  while (NonStub.size() < KernelEntryStubStride)
    NonStub.append(S.SNopBytes.begin(), S.SNopBytes.end());
  ASSERT_EQ(NonStub.size(), KernelEntryStubStride);

  EXPECT_FALSE(hasKernelEntryTrampolinePrefix(NonStub, S));
  EXPECT_FALSE(isKernelEntryTrampoline(NonStub, S));

  llvm::ArrayRef<uint8_t> ShortCandidate(Stub.data(), MinInstSize);
  EXPECT_FALSE(hasKernelEntryTrampolinePrefix(ShortCandidate, S));
}

TEST(BuildKernelEntryTrampoline, PrefixPrefiltersHipblasltSmokeEntryBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Reduced from the gfx1250 hipBLASLt MXF8/BF16 smoke kernel entry. The
  // idempotency path should reject this by raw prefix before classifying it as
  // a possible appended entry stub.
  const uint8_t EntryBytes[] = {
      0x1a, 0x08, 0x80, 0xb9, 0x02, 0x00, 0x00, 0x00, 0x1a, 0x08, 0x80,
      0xb9, 0x02, 0x00, 0x00, 0x00, 0xff, 0x02, 0x3f, 0x8b, 0xff, 0xff,
      0xff, 0x3f, 0x02, 0x9e, 0x40, 0x85, 0x03, 0x00, 0xc1, 0xbe,
  };

  llvm::SmallVector<uint8_t> Candidate;
  Candidate.append(EntryBytes, EntryBytes + sizeof(EntryBytes));
  while (Candidate.size() < KernelEntryStubStride)
    Candidate.append(S.SNopBytes.begin(), S.SNopBytes.end());
  ASSERT_EQ(Candidate.size(), KernelEntryStubStride);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(
      decodeTextSection(Candidate.data(), sizeof(EntryBytes), S, Decoded));
  ASSERT_GE(Decoded.size(), 5u);
  EXPECT_EQ(Decoded[0].Mnemonic, "s_setreg_imm32_b32");
  EXPECT_EQ(Decoded[1].Mnemonic, "s_setreg_imm32_b32");
  EXPECT_EQ(Decoded[2].Mnemonic, "s_and_b32");
  EXPECT_FALSE(hasKernelEntryTrampolinePrefix(Candidate, S));
  EXPECT_FALSE(isKernelEntryTrampoline(Candidate, S));
}

TEST(BuildKernelEntryTrampoline, PrefixPrefiltersUnknownDecodeBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  const uint8_t UnknownInst[] = {0xff, 0xff, 0xff, 0xff};

  llvm::SmallVector<uint8_t> Candidate;
  Candidate.append(UnknownInst, UnknownInst + sizeof(UnknownInst));
  while (Candidate.size() < KernelEntryStubStride)
    Candidate.append(S.SNopBytes.begin(), S.SNopBytes.end());
  ASSERT_EQ(Candidate.size(), KernelEntryStubStride);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Candidate.data(), MinInstSize, S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  EXPECT_EQ(Decoded[0].Mnemonic, "<unknown>");
  EXPECT_FALSE(hasKernelEntryTrampolinePrefix(Candidate, S));
  EXPECT_FALSE(isKernelEntryTrampoline(Candidate, S));
}

TEST(BuildKernelEntryTrampoline, MatcherRejectsNonStubBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<uint8_t> Bytes(KernelEntryStubStride, 0);
  for (size_t I = 0; I < Bytes.size(); I += MinInstSize)
    std::memcpy(Bytes.data() + I, S.SNopBytes.data(), MinInstSize);

  EXPECT_FALSE(isKernelEntryTrampoline(Bytes, S));
}

TEST(BuildKernelEntryTrampoline, MatcherRejectsWrongOperandShape) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes;
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "global_wb", S));
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "v_nop", S));
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "s_get_pc_i64 s[8:9]", S));
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "s_add_u32 s8, s8, 0", S));
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "s_addc_u32 s10, s10, 0", S));
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "s_set_pc_i64 s[8:9]", S));

  llvm::SmallVector<uint8_t> CodeEnd = assembleSingleInst("s_code_end", S);
  ASSERT_EQ(CodeEnd.size(), MinInstSize);
  while (Bytes.size() < KernelEntryStubStride)
    Bytes.append(CodeEnd.begin(), CodeEnd.end());
  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);

  EXPECT_TRUE(hasKernelEntryTrampolinePrefix(Bytes, S));
  EXPECT_FALSE(isKernelEntryTrampoline(Bytes, S));
}

TEST(KernelEntryTrampoline, ClampsInstPrefSizeAndAvoidsPrefetchGuard) {
  namespace hsa = llvm::amdhsa;

  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  uint32_t Rsrc3 = 0;
  AMDHSA_BITS_SET(Rsrc3, hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE, 7);
  Rsrc3 |= hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_GLG_EN;
  AMDHSA_BITS_SET(Rsrc3, hsa::COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT, 3);
  AMDHSA_BITS_SET(Rsrc3, hsa::COMPUTE_PGM_RSRC3_GFX125_TCP_SPLIT, 5);
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.ComputePgmRsrc3 = Rsrc3;
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  uint8_t *Kd = ViewOrErr->findKernelDescriptor("kernel");
  ASSERT_NE(Kd, nullptr);
  uint32_t Rsrc1Before = 0;
  std::memcpy(&Rsrc1Before,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1Before));

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, Growth, Fixups);
  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 1u);
  ASSERT_EQ(Fixups.size(), 1u);
  EXPECT_EQ(Fixups[0].InstPrefLines, KernelEntryStubInstPrefLines);

  const uint64_t ExpectedGuard =
      computeKernelEntryPrefetchGuardBytes(KernelEntryStubInstPrefLines);
  EXPECT_EQ(ExpectedGuard, 0u);
  ASSERT_FALSE(Growth.empty());

  // Stubs live in the appended pool at trampolinePoolVAddr(); the first stub's
  // offset is the padding needed to reach a KernelEntryStubStride boundary from
  // the pool base.
  std::optional<uint64_t> PoolVAddrOr = ViewOrErr->trampolinePoolVAddr();
  ASSERT_TRUE(PoolVAddrOr.has_value());
  const uint64_t PoolVAddr = *PoolVAddrOr;
  const uint64_t ExpectedStubOffset =
      ((PoolVAddr + KernelEntryStubStride - 1) & ~(KernelEntryStubStride - 1)) -
      PoolVAddr;
  EXPECT_EQ(Fixups[0].StubTextOffset, ExpectedStubOffset);

  uint64_t GrowthTotal = 0;
  for (const Trampoline &T : Growth)
    GrowthTotal += T.Bytes.size();
  EXPECT_EQ(GrowthTotal,
            ExpectedStubOffset + KernelEntryStubStride + ExpectedGuard);

  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      ViewOrErr->growWithTrampolines(Growth, S.SNopBytes);
  ASSERT_NE(Out, nullptr);

  ASSERT_TRUE(
      rewriteKernelEntryDescriptorOffsets(*Out, PoolVAddr, S.Cpu, Fixups));

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());

  uint8_t *OutKd = OutView->findKernelDescriptor("kernel");
  ASSERT_NE(OutKd, nullptr);
  uint32_t OutRsrc3 = 0;
  std::memcpy(&OutRsrc3,
              OutKd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc3),
              sizeof(OutRsrc3));
  uint32_t ExpectedRsrc3 = Rsrc3;
  AMDHSA_BITS_SET(ExpectedRsrc3,
                  hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE,
                  KernelEntryStubInstPrefLines);
  EXPECT_EQ(OutRsrc3, ExpectedRsrc3);
  EXPECT_EQ(AMDHSA_BITS_GET(OutRsrc3,
                            hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE),
            KernelEntryStubInstPrefLines);
  EXPECT_NE(OutRsrc3 & hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_GLG_EN, 0u);
  EXPECT_EQ(Fixups[0].RequiredSgprs, 10u);
  uint32_t OutRsrc1 = 0;
  std::memcpy(&OutRsrc1,
              OutKd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(OutRsrc1));
  EXPECT_EQ(OutRsrc1, Rsrc1Before);
  EXPECT_EQ(OutView->getKernelSgprCount("kernel"), Fixups[0].RequiredSgprs);

  llvm::ArrayRef<KernelDescriptorInfo> KDs = OutView->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  std::optional<uint64_t> KdVAddr = OutView->getKernelDescriptorVAddr("kernel");
  ASSERT_TRUE(KdVAddr.has_value());
  const uint64_t StubVAddr = PoolVAddr + Fixups[0].StubTextOffset;
  EXPECT_EQ(KDs[0].EntryOffset, static_cast<int64_t>(StubVAddr - *KdVAddr));
}

// Count symbols named \p Name in the .symtab of the ELF held in \p Buf.
// Returns ~0u if the ELF or its symbol table cannot be parsed, so a mis-parse
// surfaces as a failed expectation rather than a silent zero.
static unsigned countSymtabSymbolsNamed(llvm::WritableMemoryBuffer &Buf,
                                        llvm::StringRef Name) {
  using ELFT = llvm::object::ELF64LE;
  llvm::Expected<llvm::object::ELFFile<ELFT>> FileOrErr =
      llvm::object::ELFFile<ELFT>::create(
          llvm::StringRef(reinterpret_cast<const char *>(Buf.getBufferStart()),
                          Buf.getBufferSize()));
  if (!FileOrErr) {
    llvm::consumeError(FileOrErr.takeError());
    return ~0u;
  }
  llvm::object::ELFFile<ELFT> &File = *FileOrErr;
  llvm::Expected<ELFT::ShdrRange> Secs = File.sections();
  if (!Secs) {
    llvm::consumeError(Secs.takeError());
    return ~0u;
  }
  const ELFT::Shdr *Symtab = nullptr;
  for (const ELFT::Shdr &Sh : *Secs)
    if (Sh.sh_type == llvm::ELF::SHT_SYMTAB) {
      Symtab = &Sh;
      break;
    }
  if (!Symtab)
    return 0;
  llvm::Expected<ELFT::SymRange> Syms = File.symbols(Symtab);
  llvm::Expected<llvm::StringRef> Str = File.getStringTableForSymtab(*Symtab);
  if (!Syms || !Str) {
    if (!Syms)
      llvm::consumeError(Syms.takeError());
    if (!Str)
      llvm::consumeError(Str.takeError());
    return ~0u;
  }
  unsigned Count = 0;
  for (const ELFT::Sym &Sym : *Syms) {
    llvm::Expected<llvm::StringRef> N = Sym.getName(*Str);
    if (!N) {
      llvm::consumeError(N.takeError());
      continue;
    }
    if (*N == Name)
      ++Count;
  }
  return Count;
}

// Covers: the entry-trampoline rewrite is idempotent -- a second pass over an
// already-rewritten code object installs no new stub, and therefore defines no
// duplicate `<kernel>.stub` symbol. This backs the idempotency claim made by
// the change that adds stub symbols.
//
// How: run the full first pass on a synthetic gfx1250 object
// (appendKernelEntryTrampolines -> growWithTrampolines ->
// rewriteKernelEntryDescriptorOffsets -> addKernelEntryTrampolineSymbols) and
// confirm exactly one "kernel.stub" symbol. Then re-parse that output and run
// appendKernelEntryTrampolines again: because the descriptor already targets
// the appended stub, the second pass must report zero new stubs and produce no
// fixups, so the symbol pass never runs. Feeding those empty fixups to
// addKernelEntryTrampolineSymbols returns nullptr (no new buffer), and
// "kernel.stub" remains defined exactly once -- i.e. no duplicate name.
TEST(KernelEntryTrampoline, SecondPassAddsNoDuplicateStubSymbol) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);

  // -- First pass: append one stub, grow .text, rewrite the descriptor, and
  //    attach the stub symbol. --
  llvm::Expected<ElfView> View1 =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View1) << llvm::toString(View1.takeError());
  const unsigned TextIdx = View1->textSectionIndex();
  const uint64_t TextAddr = View1->textAddr();
  const uint64_t OldTextSize = View1->textSize();

  std::vector<Trampoline> Growth1;
  std::vector<KernelEntryTrampolineFixup> Fixups1;
  std::optional<uint32_t> Count1 = appendKernelEntryTrampolines(
      *View1, S, /*MaxSgprs=*/106, Growth1, Fixups1);
  ASSERT_TRUE(Count1.has_value());
  ASSERT_EQ(*Count1, 1u);
  std::optional<uint64_t> PoolVAddr = View1->trampolinePoolVAddr();
  ASSERT_TRUE(PoolVAddr.has_value());

  std::unique_ptr<llvm::WritableMemoryBuffer> Grown =
      View1->growWithTrampolines(Growth1, S.SNopBytes);
  ASSERT_NE(Grown, nullptr);
  ASSERT_TRUE(
      rewriteKernelEntryDescriptorOffsets(*Grown, *PoolVAddr, S.Cpu, Fixups1));
  std::unique_ptr<llvm::WritableMemoryBuffer> Pass1 =
      addKernelEntryTrampolineSymbols(*Grown, TextIdx, TextAddr, OldTextSize,
                                      Fixups1);
  ASSERT_NE(Pass1, nullptr);
  ASSERT_EQ(countSymtabSymbolsNamed(*Pass1, "kernel.stub"), 1u);

  // -- Second pass over the already-rewritten object. --
  uint8_t *Pass1Data = reinterpret_cast<uint8_t *>(Pass1->getBufferStart());
  llvm::Expected<ElfView> View2 =
      ElfView::create(Pass1Data, Pass1->getBufferSize());
  ASSERT_TRUE((bool)View2) << llvm::toString(View2.takeError());

  std::vector<Trampoline> Growth2;
  std::vector<KernelEntryTrampolineFixup> Fixups2;
  std::optional<uint32_t> Count2 = appendKernelEntryTrampolines(
      *View2, S, /*MaxSgprs=*/106, Growth2, Fixups2);
  ASSERT_TRUE(Count2.has_value());
  // The descriptor already targets a stub, so nothing new is installed.
  EXPECT_EQ(*Count2, 0u);
  EXPECT_TRUE(Fixups2.empty());

  // With no fixups the symbol pass is a no-op (returns nullptr, keeping the
  // existing buffer), so no second "kernel.stub" can be defined.
  std::unique_ptr<llvm::WritableMemoryBuffer> Pass2 =
      addKernelEntryTrampolineSymbols(*Pass1, TextIdx, TextAddr,
                                      View2->textSize(), Fixups2);
  EXPECT_EQ(Pass2, nullptr);
  EXPECT_EQ(countSymtabSymbolsNamed(*Pass1, "kernel.stub"), 1u);
}

// A `global_wb; v_nop` prologue (llvm/llvm-project#208467) already satisfies
// the workaround, so no trampoline is installed.
TEST(KernelEntryTrampoline, SkipsWhenPrologueAlreadyHasVmemWorkaround) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> GlobalWb = assembleSingleInst("global_wb", S);
  llvm::SmallVector<uint8_t> VNop = assembleSingleInst("v_nop", S);
  llvm::SmallVector<uint8_t> EndPgm = assembleSingleInst("s_endpgm", S);
  ASSERT_FALSE(GlobalWb.empty());
  ASSERT_FALSE(VNop.empty());
  ASSERT_EQ(EndPgm.size(), MinInstSize);

  llvm::SmallVector<uint8_t> Text;
  Text.append(GlobalWb.begin(), GlobalWb.end());
  Text.append(VNop.begin(), VNop.end());
  Text.append(EndPgm.begin(), EndPgm.end());

  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text);
  llvm::Expected<ElfView> View =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count =
      appendKernelEntryTrampolines(*View, S, /*MaxSgprs=*/106, Growth, Fixups);
  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 0u);
  EXPECT_TRUE(Fixups.empty());
  EXPECT_TRUE(Growth.empty());
}

// The same two instructions in the wrong order are not the workaround, so a
// trampoline is still installed.
TEST(KernelEntryTrampoline, InstallsWhenPrologueLacksVmemWorkaround) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> VNop = assembleSingleInst("v_nop", S);
  llvm::SmallVector<uint8_t> GlobalWb = assembleSingleInst("global_wb", S);
  llvm::SmallVector<uint8_t> EndPgm = assembleSingleInst("s_endpgm", S);
  ASSERT_FALSE(VNop.empty());
  ASSERT_FALSE(GlobalWb.empty());
  ASSERT_EQ(EndPgm.size(), MinInstSize);

  llvm::SmallVector<uint8_t> Text;
  Text.append(VNop.begin(), VNop.end());
  Text.append(GlobalWb.begin(), GlobalWb.end());
  Text.append(EndPgm.begin(), EndPgm.end());

  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text);
  llvm::Expected<ElfView> View =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count =
      appendKernelEntryTrampolines(*View, S, /*MaxSgprs=*/106, Growth, Fixups);
  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 1u);
  EXPECT_EQ(Fixups.size(), 1u);
}

TEST(KernelEntryTrampoline, AlignsStubByVirtualAddress) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.TextAddr = 0x1080;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, Growth, Fixups);

  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 1u);
  ASSERT_EQ(Fixups.size(), 1u);
  // The stub is aligned by its virtual address: the pool base plus the stub's
  // offset lands on a KernelEntryStubStride boundary.
  std::optional<uint64_t> PoolVAddrOr = ViewOrErr->trampolinePoolVAddr();
  ASSERT_TRUE(PoolVAddrOr.has_value());
  const uint64_t StubVAddr = *PoolVAddrOr + Fixups[0].StubTextOffset;
  EXPECT_EQ(StubVAddr % KernelEntryStubStride, 0u);
}

TEST(KernelEntryTrampoline, AppendReturnsZeroWhenNoDescriptorsExist) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.EmitKernelDescriptorSymbol = false;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, Growth, Fixups);

  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 0u);
  EXPECT_TRUE(Growth.empty());
  EXPECT_TRUE(Fixups.empty());
}

TEST(KernelEntryTrampoline, AppendFailsWithoutSgprScratchPair) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = 105;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  Trampoline Existing;
  Existing.Bytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());
  std::vector<Trampoline> Growth;
  Growth.push_back(Existing);
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, Growth, Fixups);

  EXPECT_FALSE(Count.has_value());
  ASSERT_EQ(Growth.size(), 1u);
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(Growth[0].Bytes),
            llvm::ArrayRef<uint8_t>(Existing.Bytes));
  EXPECT_TRUE(Fixups.empty());
}

// -- classifyWmmaNops ---------------------------------------------------------

TEST(ClassifyWmmaNops, CoversKnownMnemonics) {
  struct Case {
    llvm::StringLiteral Mnemonic;
    int A0Nops;
    int B0Nops;
  };
  const Case Cases[] = {
      {"v_add_f32", 4, 4},
      {"v_wmma_i32_16x16x32_iu8", 8, 4},
      {"v_wmma_i32_16x16x64_iu4", 8, 4},
      {"v_wmma_f32_16x16x128_f8f6f4", 1, 4},
      {"v_wmma_f32_16x16x128_fp8_fp8", 3, 4},
      {"v_wmma_f32_16x16x32_fp8_fp8", 1, 4},
      {"v_wmma_f32_16x16x16_f16", 4, 4},
      {"v_wmma_f32_16x16x16_bf16", 4, 4},
      {"v_swmmac_i32_16x16x64_iu8", 8, 4},
      {"v_wmma_f32_16x16x4_f32", 4, 4},
      {"v_wmma_f16_something_iu8", 8, 4},
  };

  for (const Case &C : Cases) {
    WmmaNopReq Req = classifyWmmaNops(C.Mnemonic);
    EXPECT_EQ(Req.A0Nops, C.A0Nops) << C.Mnemonic.str();
    EXPECT_EQ(Req.B0Nops, C.B0Nops) << C.Mnemonic.str();
  }
}

// -- patchScaleSrc2 -----------------------------------------------------------
//
// Pure byte-level tests for the VOP3PX2 scale_src2 bit-field fix.
// The function patches bits [58:50] of a 16-byte VOP3PX2 encoding to
// VGPR0 (0x100): byte 6 bits [7:2] cleared, byte 7 bit [2] set,
// byte 7 bits [1:0] cleared.

TEST(PatchScaleSrc2, ZeroedFieldGetsPatched) {
  uint8_t Inst[16] = {};
  EXPECT_TRUE(patchScaleSrc2(Inst));
  EXPECT_EQ(Inst[6] & 0xFC, 0x00);
  EXPECT_EQ(Inst[7] & 0x07, 0x04);
}

TEST(PatchScaleSrc2, PreservesOtherBytes) {
  uint8_t Inst[16];
  std::memset(Inst, 0xAA, sizeof(Inst));
  EXPECT_TRUE(patchScaleSrc2(Inst));
  for (size_t I = 0; I < 16; ++I) {
    if (I == 6 || I == 7)
      continue;
    EXPECT_EQ(Inst[I], 0xAA) << "byte " << I << " unexpectedly modified";
  }
}

TEST(PatchScaleSrc2, AllOnesFieldGetsPatched) {
  uint8_t Inst[16] = {};
  Inst[6] = 0xFF;
  Inst[7] = 0xFF;
  EXPECT_TRUE(patchScaleSrc2(Inst));
  EXPECT_EQ(Inst[6] & 0xFC, 0x00);
  EXPECT_EQ(Inst[7] & 0x07, 0x04);
  EXPECT_EQ(Inst[7] & 0xF8, 0xF8);
}

TEST(PatchScaleSrc2, AlreadyVgpr0ReturnsFalse) {
  uint8_t Inst[16] = {};
  Inst[7] = 0x04;
  EXPECT_FALSE(patchScaleSrc2(Inst));
  EXPECT_EQ(Inst[6], 0x00);
  EXPECT_EQ(Inst[7], 0x04);
}

TEST(PatchScaleSrc2, IsIdempotent) {
  uint8_t Inst[16] = {};
  Inst[6] = 0xAB;
  Inst[7] = 0xCD;
  EXPECT_TRUE(patchScaleSrc2(Inst));
  uint8_t AfterFirst6 = Inst[6];
  uint8_t AfterFirst7 = Inst[7];
  EXPECT_FALSE(patchScaleSrc2(Inst));
  EXPECT_EQ(Inst[6], AfterFirst6);
  EXPECT_EQ(Inst[7], AfterFirst7);
}

TEST(PatchScaleSrc2, PreservesNonScaleSrc2Bits) {
  uint8_t Inst[16] = {};
  Inst[6] = 0x03 | 0xA0;
  Inst[7] = 0xF8 | 0x02;
  EXPECT_TRUE(patchScaleSrc2(Inst));
  EXPECT_EQ(Inst[6] & 0x03, 0x03);
  EXPECT_EQ(Inst[7] & 0xF8, 0xF8);
  EXPECT_EQ(Inst[6] & 0xFC, 0x00);
  EXPECT_EQ(Inst[7] & 0x07, 0x04);
}

// -- HotswapPatchVTable -------------------------------------------------------
//
// Tests for the .def-driven patch registry that replaced the
// LLVM_ATTRIBUTE_WEAK override pattern (issue ROCm/llvm-project#2479).
//
// Coverage strategy: link errors already catch missing register*Patch
// definitions and missing comgr-hotswap-patches.def entries, so we only
// test what the linker cannot:
//   1. One canonical per-installer "binds only its own slot" check,
//      kept as a worked example for future patch authors. Wrong-slot
//      bugs in the other register*Patch functions are caught via the
//      install end-to-end test below.
//   2. End-to-end install: a default-constructed vtable has null slots,
//      installHotswapPatches() binds every .def entry, and slots without
//      a .def entry stay null (the dispatcher's no-op contract).
//   3. The production singleton accessor returns the same fully-bound
//      vtable on every call -- the initializer eagerly runs the install
//      under the C++11 magic-static rule, so production code never sees
//      an empty vtable.

TEST(HotswapPatchVTable, RegisterInPlaceBindsOnlyInPlaceSlot) {
  HotswapPatchVTable VT;
  registerInPlacePatch(VT);
  EXPECT_NE(VT.applyInPlacePatches, nullptr);
  EXPECT_EQ(VT.applyTrampolinePatches, nullptr);
  EXPECT_EQ(VT.applyWmmaHazardPatch, nullptr);
  EXPECT_EQ(VT.applyVop3px2Src2Fix, nullptr);
}

TEST(HotswapPatchVTable, InstallBindsRegisteredAndLeavesUnregisteredNull) {
  HotswapPatchVTable VT;

  // Defaults: every slot null (no patch implementation linked yet).
  EXPECT_EQ(VT.applyInPlacePatches, nullptr);
  EXPECT_EQ(VT.applyTrampolinePatches, nullptr);
  EXPECT_EQ(VT.applyWmmaHazardPatch, nullptr);
  EXPECT_EQ(VT.applyVop3px2Src2Fix, nullptr);
  EXPECT_EQ(VT.applyWmmaSplitPatches, nullptr);
  EXPECT_EQ(VT.applyScratchPatches, nullptr);

  installHotswapPatches(VT);

  // Slots backed by a comgr-hotswap-patches.def entry get bound. If a
  // register*Patch fails to set its slot (or sets the wrong one), one
  // of these EXPECT_NEs catches it.
  EXPECT_NE(VT.applyInPlacePatches, nullptr);
  EXPECT_NE(VT.applyTrampolinePatches, nullptr);
  EXPECT_NE(VT.applyWmmaHazardPatch, nullptr);
  EXPECT_NE(VT.applyVop3px2Src2Fix, nullptr);
  EXPECT_NE(VT.applyWmmaSplitPatches, nullptr);
  EXPECT_NE(VT.applyScratchPatches, nullptr);
}

TEST(HotswapPatchVTable, ProcessSingletonIdentityAndEagerInstall) {
  HotswapPatchVTable &VT1 = getHotswapPatchVTable();
  HotswapPatchVTable &VT2 = getHotswapPatchVTable();
  EXPECT_EQ(&VT1, &VT2);

  // The singleton's initializer runs installHotswapPatches() on first
  // access, so every .def-backed slot is already bound by the time the
  // first reference is handed out. Pinning this contract here keeps the
  // dispatcher safe to call getHotswapPatchVTable() without any explicit
  // install step at the entry point.
  EXPECT_NE(VT1.applyInPlacePatches, nullptr);
  EXPECT_NE(VT1.applyTrampolinePatches, nullptr);
  EXPECT_NE(VT1.applyWmmaHazardPatch, nullptr);
  EXPECT_NE(VT1.applyVop3px2Src2Fix, nullptr);
  EXPECT_NE(VT1.applyWmmaSplitPatches, nullptr);
  EXPECT_NE(VT1.applyScratchPatches, nullptr);
}

// -- DS ADDTID trampoline support ---------------------------------------------
//
// Tests for the ds_load_addtid_b32 / ds_store_addtid_b32 gfx1250 trampoline
// patch. Coverage is bottom-up: first that the encode/decode of ADDTID
// instructions exposes the expected MCInst operand layout, then that
// buildTrampoline assembles and decodes a full ADDTID replacement body plus
// its branch-back tail.

namespace {

// AddtidOpReg / AddtidOpOffset / AddtidOpGds operand-layout constants live
// in comgr-hotswap-internal.h and are imported by the COMGR::hotswap using-
// declaration at the top of this file.

// Decode a single instruction string and return the resulting MCInst, or
// llvm::None on failure. Aborts the test if assemble/decode fail so the
// caller can dereference unconditionally.
llvm::MCInst decodeOne(llvm::StringRef Asm, const LLVMState &S) {
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, S);
  EXPECT_FALSE(Bytes.empty()) << "failed to assemble: " << Asm.str();
  std::vector<InternalDecodedInst> Decoded;
  EXPECT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded))
      << "failed to decode: " << Asm.str();
  EXPECT_EQ(Decoded.size(), 1u) << "expected one inst for: " << Asm.str();
  return Decoded.empty() ? llvm::MCInst() : Decoded[0].Inst;
}

void expectAddTidLayout(llvm::StringRef Asm, int64_t Offset,
                        llvm::StringRef RegName, const LLVMState &S) {
  llvm::MCInst Inst = decodeOne(Asm, S);
  ASSERT_GE(Inst.getNumOperands(), 3u);

  EXPECT_TRUE(Inst.getOperand(AddtidOpReg).isReg());
  EXPECT_NE(Inst.getOperand(AddtidOpReg).getReg(), 0u);
  EXPECT_TRUE(Inst.getOperand(AddtidOpOffset).isImm());
  EXPECT_EQ(Inst.getOperand(AddtidOpOffset).getImm(), Offset);
  EXPECT_TRUE(Inst.getOperand(AddtidOpGds).isImm());
  EXPECT_EQ(Inst.getOperand(AddtidOpGds).getImm(), 0);

  const char *N = S.MRI->getName(Inst.getOperand(AddtidOpReg).getReg());
  ASSERT_NE(N, nullptr);
  EXPECT_EQ(llvm::StringRef(N).str(), RegName.str());
}

void expectDecodedMnemonics(llvm::ArrayRef<InternalDecodedInst> Decoded,
                            llvm::ArrayRef<llvm::StringRef> Expected) {
  ASSERT_EQ(Decoded.size(), Expected.size());
  for (size_t I = 0; I < Expected.size(); ++I)
    EXPECT_EQ(Decoded[I].Mnemonic, Expected[I].str()) << "index " << I;
}

void expectDecodedBodyMatchesAsm(llvm::ArrayRef<InternalDecodedInst> Decoded,
                                 llvm::ArrayRef<std::string> AsmLines,
                                 const LLVMState &S) {
  ASSERT_GE(Decoded.size(), AsmLines.size());
  for (size_t I = 0; I < AsmLines.size(); ++I) {
    llvm::MCInst Expected = decodeOne(AsmLines[I], S);
    expectSameOperands(Decoded[I].Inst, Expected, AsmLines[I]);
  }
}

} // namespace

TEST(AddTid, AddTidDecodesWithExpectedLayout) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Direct operand access: register, then offset, then gds bit. No
  // print-and-parse round-trip -- production code uses the same operand
  // indices to reach the destination VGPR.
  // Production code uses MRI.getName() to resolve the VGPR identifier
  // ("VGPR5" for v5, etc.); pin that so a tablegen rename catches here.
  expectAddTidLayout("ds_load_addtid_b32 v5 offset:128", 128, "VGPR5", S);
  expectAddTidLayout("ds_store_addtid_b32 v10 offset:256", 256, "VGPR10", S);
}

TEST(AddTid, LoadTrampolineThroughBuildTrampoline) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<std::string> AsmLines = {
      "v_mbcnt_lo_u32_b32 v3, -1, 0", "v_mbcnt_hi_u32_b32 v3, -1, v3",
      "v_lshlrev_b32 v3, 2, v3",      "v_add_nc_u32 v3, m0, v3",
      "v_and_b32 v3, 0xfffff, v3",    "ds_load_b32 v3, v3 offset:0",
  };

  Trampoline T = buildTrampoline(AsmLines, /*OriginalOffset=*/0x100,
                                 /*OriginalSize=*/4,
                                 /*TrampolineTextOffset=*/0x2000, S);

  ASSERT_FALSE(T.Bytes.empty());
  EXPECT_EQ(T.OriginalOffset, 0x100u);
  EXPECT_EQ(T.OriginalSize, 4u);

  // 6 body instructions + 1 branch-back tail.
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(T.Bytes.data(), T.Bytes.size(), S, Decoded));
  const llvm::StringRef Expected[] = {"v_mbcnt_lo_u32_b32",
                                      "v_mbcnt_hi_u32_b32",
                                      "v_lshlrev_b32",
                                      "v_add_nc_u32",
                                      "v_and_b32",
                                      "ds_load_b32",
                                      "s_branch"};
  expectDecodedMnemonics(Decoded, Expected);
  expectDecodedBodyMatchesAsm(Decoded, AsmLines, S);
}

TEST(AddTid, StoreTrampolineThroughBuildTrampoline) {
  // Mirror of LoadTrampolineThroughBuildTrampoline for the store path, where
  // the data VGPR (v10) must be preserved and an allocator-supplied scratch
  // VGPR (v42) holds the computed address. The two register operands of
  // ds_store_b32 carry independent VGPR indices, which is what distinguishes
  // this from the load case (which can fold dst back into address).
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<std::string> AsmLines = {
      "v_mbcnt_lo_u32_b32 v42, -1, 0", "v_mbcnt_hi_u32_b32 v42, -1, v42",
      "v_lshlrev_b32 v42, 2, v42",     "v_add_nc_u32 v42, m0, v42",
      "v_and_b32 v42, 0xfffff, v42",   "ds_store_b32 v42, v10",
  };

  Trampoline T = buildTrampoline(AsmLines, /*OriginalOffset=*/0x180,
                                 /*OriginalSize=*/4,
                                 /*TrampolineTextOffset=*/0x2040, S);

  ASSERT_FALSE(T.Bytes.empty());
  EXPECT_EQ(T.OriginalOffset, 0x180u);
  EXPECT_EQ(T.OriginalSize, 4u);

  // 6 body instructions + 1 branch-back tail, matching the load variant.
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(T.Bytes.data(), T.Bytes.size(), S, Decoded));
  const llvm::StringRef Expected[] = {"v_mbcnt_lo_u32_b32",
                                      "v_mbcnt_hi_u32_b32",
                                      "v_lshlrev_b32",
                                      "v_add_nc_u32",
                                      "v_and_b32",
                                      "ds_store_b32",
                                      "s_branch"};
  expectDecodedMnemonics(Decoded, Expected);
  expectDecodedBodyMatchesAsm(Decoded, AsmLines, S);
}
