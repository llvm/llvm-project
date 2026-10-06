//===- bolt/unittest/Core/Relocation.cpp ----------------------------------===//
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
