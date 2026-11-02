//===- bolt/unittests/Core/Relocation.cpp
//----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/Relocation.h"
#include "llvm/BinaryFormat/ELF.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::bolt;

namespace {

#ifdef AARCH64_AVAILABLE

TEST(AArch64RelocationHandlerTest, EncodeAndExtractCall26) {
  std::unique_ptr<RelocationHandler> Handler = createAArch64RelocationHandler();
  constexpr uint64_t PC = 0x1000;
  constexpr uint64_t Target = 0x1100;

  const uint64_t Encoded =
      Handler->encodeValue(ELF::R_AARCH64_CALL26, Target, PC);
  EXPECT_EQ(Encoded, 0x94000040u);
  EXPECT_EQ(Handler->extractValue(ELF::R_AARCH64_CALL26, Encoded, PC), Target);
}

TEST(AArch64RelocationHandlerTest, EncodeAndExtractPrel32) {
  std::unique_ptr<RelocationHandler> Handler = createAArch64RelocationHandler();
  constexpr uint64_t PC = 0x1000;
  constexpr uint64_t Target = 0xf00;

  const uint64_t Encoded =
      Handler->encodeValue(ELF::R_AARCH64_PREL32, Target, PC);
  EXPECT_EQ(Encoded, static_cast<uint64_t>(-0x100));
  EXPECT_EQ(Handler->extractValue(ELF::R_AARCH64_PREL32, Encoded, PC), Target);
}

TEST(AArch64RelocationHandlerTest, Call26EncodingRange) {
  std::unique_ptr<RelocationHandler> Handler = createAArch64RelocationHandler();
  constexpr uint64_t PC = 0x1000;
  constexpr uint64_t Range = uint64_t{1} << 27;

  EXPECT_TRUE(
      Handler->canEncodeValue(ELF::R_AARCH64_CALL26, PC + Range - 4, PC));
  EXPECT_FALSE(Handler->canEncodeValue(ELF::R_AARCH64_CALL26, PC + Range, PC));
}

TEST(AArch64RelocationTest, CONDBR19) {
  std::unique_ptr<RelocationHandler> Handler = createAArch64RelocationHandler();

  constexpr uint64_t PC = 0x200000;
  constexpr uint32_t Type = ELF::R_AARCH64_CONDBR19;
  constexpr uint64_t BackwardTarget = PC - 0x100000;
  constexpr uint64_t ForwardTarget = PC + 0xffffc;
  EXPECT_TRUE(Handler->canEncodeValue(Type, BackwardTarget, PC));
  EXPECT_TRUE(Handler->canEncodeValue(Type, ForwardTarget, PC));
  EXPECT_FALSE(Handler->canEncodeValue(Type, BackwardTarget - 4, PC));
  EXPECT_FALSE(Handler->canEncodeValue(Type, ForwardTarget + 4, PC));

  // Each original instruction branches back four bytes. The expected words,
  // assembled independently, replace this nonzero displacement with the signed
  // limits while preserving the condition, register, and 32/64-bit form.
  const struct {
    uint32_t OriginalInst;
    uint32_t BackwardInst;
    uint32_t ForwardInst;
  } Cases[] = {
      {0x54ffffea, 0x5480000a, 0x547fffea}, // b.ge
      {0x54ffffe5, 0x54800005, 0x547fffe5}, // b.pl
      {0x34ffffe0, 0x34800000, 0x347fffe0}, // cbz w0
      {0xb4fffff5, 0xb4800015, 0xb47ffff5}, // cbz x21
      {0xb5ffffea, 0xb580000a, 0xb57fffea}, // cbnz x10
      {0x35ffffff, 0x3580001f, 0x357fffff}, // cbnz wzr
  };
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.OriginalInst);
    EXPECT_EQ(Case.BackwardInst, Handler->encodeValue(Type, BackwardTarget, PC,
                                                      Case.OriginalInst));
    EXPECT_EQ(Case.ForwardInst,
              Handler->encodeValue(Type, ForwardTarget, PC, Case.OriginalInst));
  }
}

TEST(AArch64RelocationTest, TSTBR14) {
  std::unique_ptr<RelocationHandler> Handler = createAArch64RelocationHandler();

  constexpr uint64_t PC = 0x200000;
  constexpr uint32_t Type = ELF::R_AARCH64_TSTBR14;
  constexpr uint64_t BackwardTarget = PC - 0x8000;
  constexpr uint64_t ForwardTarget = PC + 0x7ffc;
  EXPECT_TRUE(Handler->canEncodeValue(Type, BackwardTarget, PC));
  EXPECT_TRUE(Handler->canEncodeValue(Type, ForwardTarget, PC));
  EXPECT_FALSE(Handler->canEncodeValue(Type, BackwardTarget - 4, PC));
  EXPECT_FALSE(Handler->canEncodeValue(Type, ForwardTarget + 4, PC));

  // Replace a nonzero displacement at both signed limits without changing the
  // opcode, register, or tested-bit fields. Expected words were assembled
  // independently, including tested bits 0, 31, 32, and 63.
  const struct {
    uint32_t OriginalInst;
    uint32_t BackwardInst;
    uint32_t ForwardInst;
  } Cases[] = {
      {0x3607ffe0, 0x36040000, 0x3603ffe0}, // tbz w0, #0
      {0xb607ffea, 0xb604000a, 0xb603ffea}, // tbz x10, #32
      {0x37fffff5, 0x37fc0015, 0x37fbfff5}, // tbnz w21, #31
      {0xb7ffffff, 0xb7fc001f, 0xb7fbffff}, // tbnz xzr, #63
  };
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.OriginalInst);
    EXPECT_EQ(Case.BackwardInst, Handler->encodeValue(Type, BackwardTarget, PC,
                                                      Case.OriginalInst));
    EXPECT_EQ(Case.ForwardInst,
              Handler->encodeValue(Type, ForwardTarget, PC, Case.OriginalInst));
  }
}

#endif

#ifdef X86_AVAILABLE

TEST(X86RelocationHandlerTest, ExtractSigned32) {
  std::unique_ptr<RelocationHandler> Handler = createX86RelocationHandler();
  EXPECT_EQ(Handler->extractValue(ELF::R_X86_64_32S, 0x80000000, 0),
            0xffffffff80000000ULL);
}

#endif

#ifdef RISCV_AVAILABLE

TEST(RISCVRelocationHandlerTest, ExtractCallAddend) {
  std::unique_ptr<RelocationHandler> Handler =
      createRISCVRelocationHandler(true);
  // AUIPC ra, 0x12345 followed by JALR ra, 0x678(ra).
  constexpr uint64_t Instructions = 0x678080e712345097ULL;
  EXPECT_EQ(Handler->extractValue(ELF::R_RISCV_CALL, Instructions, 0),
            0x12345678u);
}

#endif

} // namespace
