//===- bolt/unittest/Core/Relocation.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/Relocation.h"
#ifdef AARCH64_AVAILABLE
#include "bolt/Target/AArch64/AArch64RelocationHandler.h"
#endif
#ifdef RISCV_AVAILABLE
#include "bolt/Target/RISCV/RISCVRelocationHandler.h"
#endif
#ifdef X86_AVAILABLE
#include "bolt/Target/X86/X86RelocationHandler.h"
#endif
#include "llvm/BinaryFormat/ELF.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::bolt;

#ifdef AARCH64_AVAILABLE

namespace {

TEST(AArch64RelocationHandlerTest, ClassificationAndSize) {
  std::unique_ptr<RelocationHandler> Handler = createAArch64RelocationHandler();

  EXPECT_TRUE(Handler->isSupported(ELF::R_AARCH64_CALL26));
  EXPECT_TRUE(Handler->isSupported(ELF::R_AARCH64_PREL32));
  EXPECT_FALSE(Handler->isSupported(ELF::R_AARCH64_NONE));

  EXPECT_EQ(Handler->getSizeForType(ELF::R_AARCH64_CALL26), 4u);
  EXPECT_EQ(Handler->getSizeForType(ELF::R_AARCH64_PREL32), 4u);
  EXPECT_EQ(Handler->getSizeForType(ELF::R_AARCH64_ABS64), 8u);

  EXPECT_TRUE(Handler->isPCRelative(ELF::R_AARCH64_CALL26));
  EXPECT_TRUE(Handler->isPCRelative(ELF::R_AARCH64_PREL32));
  EXPECT_FALSE(Handler->isPCRelative(ELF::R_AARCH64_ABS64));
}

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

} // namespace

#endif

#ifdef X86_AVAILABLE

namespace {

TEST(X86RelocationHandlerTest, ClassificationAndEncoding) {
  std::unique_ptr<RelocationHandler> Handler = createX86RelocationHandler();
  constexpr uint64_t PC = 0x1000;
  constexpr uint64_t Target = 0xf00;

  EXPECT_TRUE(Handler->isSupported(ELF::R_X86_64_PC32));
  EXPECT_EQ(Handler->getSizeForType(ELF::R_X86_64_PC32), 4u);
  EXPECT_TRUE(Handler->isPCRelative(ELF::R_X86_64_PC32));

  const uint64_t Encoded = Handler->encodeValue(ELF::R_X86_64_PC32, Target, PC);
  EXPECT_EQ(Encoded, static_cast<uint64_t>(-0x100));
  EXPECT_EQ(Handler->extractValue(ELF::R_X86_64_PC32, Encoded, PC),
            static_cast<uint64_t>(-0x100));
}

} // namespace

#endif

#ifdef RISCV_AVAILABLE

namespace {

TEST(RISCVRelocationHandlerTest, ClassificationAndEncoding) {
  std::unique_ptr<RelocationHandler> Handler =
      createRISCVRelocationHandler(true);
  constexpr uint64_t Value = 0x12345678;

  EXPECT_TRUE(Handler->isSupported(ELF::R_RISCV_CALL));
  EXPECT_EQ(Handler->getSizeForType(ELF::R_RISCV_CALL), 8u);
  EXPECT_TRUE(Handler->isPCRelative(ELF::R_RISCV_CALL));
  EXPECT_TRUE(Handler->isInstructionReference(ELF::R_RISCV_PCREL_LO12_I));

  const uint64_t Encoded = Handler->encodeValue(ELF::R_RISCV_32, Value, 0);
  EXPECT_EQ(Encoded, Value);
  EXPECT_EQ(Handler->extractValue(ELF::R_RISCV_32, Encoded, 0), Value);
}

TEST(RISCVRelocationHandlerTest, Preserves32And64BitBehavior) {
  std::unique_ptr<RelocationHandler> RISCV32Handler =
      createRISCVRelocationHandler(false);
  std::unique_ptr<RelocationHandler> RISCV64Handler =
      createRISCVRelocationHandler(true);

  EXPECT_TRUE(
      RISCV32Handler->isInstructionReference(ELF::R_RISCV_PCREL_LO12_S));
  EXPECT_TRUE(
      RISCV64Handler->isInstructionReference(ELF::R_RISCV_PCREL_LO12_S));
  EXPECT_EQ(RISCV64Handler->getRelative(), ELF::R_RISCV_RELATIVE);
  EXPECT_TRUE(RISCV64Handler->isIRelative(ELF::R_RISCV_IRELATIVE));
}

} // namespace

#endif
