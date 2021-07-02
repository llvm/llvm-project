// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mutagen/Mutagen.h"
#include "mutagen/MutagenDispatcher.h"
#include "mutagen/MutagenSequence.h"
#include "mutagen/MutagenUtil.h"
#include "gtest/gtest.h"
#include <chrono>

// This test doesn't set Config.MsanUnpoison*, so ensure MSan isn't present.
// Avoid using fuzzer::ExternalFunctions, since it may not be linked against
// the test binary.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#error MemorySanitizer is not supported for the mutagen unit tests.
#endif // __has_feature(memory_sanitizer)
#endif // defined(__has_feature)

namespace mutagen {
namespace {

using fuzzer::Set;

std::unique_ptr<MutationDispatcher> CreateMutationDispatcher() {
  LLVMMutagenConfiguration Config;
  memset(&Config, 0, sizeof(Config));
  return std::unique_ptr<MutationDispatcher>(new MutationDispatcher(&Config));
}

typedef size_t (MutationDispatcher::*Mutator)(uint8_t *Data, size_t Size,
                                              size_t MaxSize);

TEST(MutationDispatcher, CrossOver) {
  auto MD = CreateMutationDispatcher();
  Unit A({0, 1, 2}), B({5, 6, 7});
  Unit C;
  Unit Expected[] = {{0},
                     {0, 1},
                     {0, 5},
                     {0, 1, 2},
                     {0, 1, 5},
                     {0, 5, 1},
                     {0, 5, 6},
                     {0, 1, 2, 5},
                     {0, 1, 5, 2},
                     {0, 1, 5, 6},
                     {0, 5, 1, 2},
                     {0, 5, 1, 6},
                     {0, 5, 6, 1},
                     {0, 5, 6, 7},
                     {0, 1, 2, 5, 6},
                     {0, 1, 5, 2, 6},
                     {0, 1, 5, 6, 2},
                     {0, 1, 5, 6, 7},
                     {0, 5, 1, 2, 6},
                     {0, 5, 1, 6, 2},
                     {0, 5, 1, 6, 7},
                     {0, 5, 6, 1, 2},
                     {0, 5, 6, 1, 7},
                     {0, 5, 6, 7, 1},
                     {0, 1, 2, 5, 6, 7},
                     {0, 1, 5, 2, 6, 7},
                     {0, 1, 5, 6, 2, 7},
                     {0, 1, 5, 6, 7, 2},
                     {0, 5, 1, 2, 6, 7},
                     {0, 5, 1, 6, 2, 7},
                     {0, 5, 1, 6, 7, 2},
                     {0, 5, 6, 1, 2, 7},
                     {0, 5, 6, 1, 7, 2},
                     {0, 5, 6, 7, 1, 2}};
  for (size_t Len = 1; Len < 8; Len++) {
    Set<Unit> FoundUnits, ExpectedUnitsWitThisLength;
    for (int Iter = 0; Iter < 3000; Iter++) {
      C.resize(Len);
      size_t NewSize = MD->CrossOver(A.data(), A.size(), B.data(), B.size(),
                                     C.data(), C.size());
      C.resize(NewSize);
      FoundUnits.insert(C);
    }
    for (const Unit &U : Expected)
      if (U.size() <= Len)
        ExpectedUnitsWitThisLength.insert(U);
    EXPECT_EQ(ExpectedUnitsWitThisLength, FoundUnits);
  }
}

void TestEraseBytes(Mutator M, int NumIter) {
  uint8_t REM0[8] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM1[8] = {0x00, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM2[8] = {0x00, 0x11, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM3[8] = {0x00, 0x11, 0x22, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM4[8] = {0x00, 0x11, 0x22, 0x33, 0x55, 0x66, 0x77};
  uint8_t REM5[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x66, 0x77};
  uint8_t REM6[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x77};
  uint8_t REM7[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

  uint8_t REM8[6] = {0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM9[6] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55};
  uint8_t REM10[6] = {0x00, 0x11, 0x22, 0x55, 0x66, 0x77};

  uint8_t REM11[5] = {0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM12[5] = {0x00, 0x11, 0x22, 0x33, 0x44};
  uint8_t REM13[5] = {0x00, 0x44, 0x55, 0x66, 0x77};

  auto MD = CreateMutationDispatcher();
  int FoundMask = 0;
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
    size_t NewSize = (*MD.*M)(T, sizeof(T), sizeof(T));
    if (NewSize == 7 && !memcmp(REM0, T, 7))
      FoundMask |= 1 << 0;
    if (NewSize == 7 && !memcmp(REM1, T, 7))
      FoundMask |= 1 << 1;
    if (NewSize == 7 && !memcmp(REM2, T, 7))
      FoundMask |= 1 << 2;
    if (NewSize == 7 && !memcmp(REM3, T, 7))
      FoundMask |= 1 << 3;
    if (NewSize == 7 && !memcmp(REM4, T, 7))
      FoundMask |= 1 << 4;
    if (NewSize == 7 && !memcmp(REM5, T, 7))
      FoundMask |= 1 << 5;
    if (NewSize == 7 && !memcmp(REM6, T, 7))
      FoundMask |= 1 << 6;
    if (NewSize == 7 && !memcmp(REM7, T, 7))
      FoundMask |= 1 << 7;

    if (NewSize == 6 && !memcmp(REM8, T, 6))
      FoundMask |= 1 << 8;
    if (NewSize == 6 && !memcmp(REM9, T, 6))
      FoundMask |= 1 << 9;
    if (NewSize == 6 && !memcmp(REM10, T, 6))
      FoundMask |= 1 << 10;

    if (NewSize == 5 && !memcmp(REM11, T, 5))
      FoundMask |= 1 << 11;
    if (NewSize == 5 && !memcmp(REM12, T, 5))
      FoundMask |= 1 << 12;
    if (NewSize == 5 && !memcmp(REM13, T, 5))
      FoundMask |= 1 << 13;
  }
  EXPECT_EQ(FoundMask, (1 << 14) - 1);
}

TEST(MutationDispatcher, EraseBytes1) {
  TestEraseBytes(&MutationDispatcher::Mutate_EraseBytes, 200);
}
TEST(MutationDispatcher, EraseBytes2) {
  TestEraseBytes(&MutationDispatcher::Mutate, 2000);
}

void TestInsertByte(Mutator M, int NumIter) {
  auto MD = CreateMutationDispatcher();
  int FoundMask = 0;
  uint8_t INS0[8] = {0xF1, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  uint8_t INS1[8] = {0x00, 0xF2, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  uint8_t INS2[8] = {0x00, 0x11, 0xF3, 0x22, 0x33, 0x44, 0x55, 0x66};
  uint8_t INS3[8] = {0x00, 0x11, 0x22, 0xF4, 0x33, 0x44, 0x55, 0x66};
  uint8_t INS4[8] = {0x00, 0x11, 0x22, 0x33, 0xF5, 0x44, 0x55, 0x66};
  uint8_t INS5[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0xF6, 0x55, 0x66};
  uint8_t INS6[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0xF7, 0x66};
  uint8_t INS7[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0xF8};
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    size_t NewSize = (*MD.*M)(T, 7, 8);
    if (NewSize == 8 && !memcmp(INS0, T, 8))
      FoundMask |= 1 << 0;
    if (NewSize == 8 && !memcmp(INS1, T, 8))
      FoundMask |= 1 << 1;
    if (NewSize == 8 && !memcmp(INS2, T, 8))
      FoundMask |= 1 << 2;
    if (NewSize == 8 && !memcmp(INS3, T, 8))
      FoundMask |= 1 << 3;
    if (NewSize == 8 && !memcmp(INS4, T, 8))
      FoundMask |= 1 << 4;
    if (NewSize == 8 && !memcmp(INS5, T, 8))
      FoundMask |= 1 << 5;
    if (NewSize == 8 && !memcmp(INS6, T, 8))
      FoundMask |= 1 << 6;
    if (NewSize == 8 && !memcmp(INS7, T, 8))
      FoundMask |= 1 << 7;
  }
  EXPECT_EQ(FoundMask, 255);
}

TEST(MutationDispatcher, InsertByte1) {
  TestInsertByte(&MutationDispatcher::Mutate_InsertByte, 1 << 15);
}
TEST(MutationDispatcher, InsertByte2) {
  TestInsertByte(&MutationDispatcher::Mutate, 1 << 17);
}

void TestInsertRepeatedBytes(Mutator M, int NumIter) {
  auto MD = CreateMutationDispatcher();
  int FoundMask = 0;
  uint8_t INS0[7] = {0x00, 0x11, 0x22, 0x33, 'a', 'a', 'a'};
  uint8_t INS1[7] = {0x00, 0x11, 0x22, 'a', 'a', 'a', 0x33};
  uint8_t INS2[7] = {0x00, 0x11, 'a', 'a', 'a', 0x22, 0x33};
  uint8_t INS3[7] = {0x00, 'a', 'a', 'a', 0x11, 0x22, 0x33};
  uint8_t INS4[7] = {'a', 'a', 'a', 0x00, 0x11, 0x22, 0x33};

  uint8_t INS5[8] = {0x00, 0x11, 0x22, 0x33, 'b', 'b', 'b', 'b'};
  uint8_t INS6[8] = {0x00, 0x11, 0x22, 'b', 'b', 'b', 'b', 0x33};
  uint8_t INS7[8] = {0x00, 0x11, 'b', 'b', 'b', 'b', 0x22, 0x33};
  uint8_t INS8[8] = {0x00, 'b', 'b', 'b', 'b', 0x11, 0x22, 0x33};
  uint8_t INS9[8] = {'b', 'b', 'b', 'b', 0x00, 0x11, 0x22, 0x33};

  for (int i = 0; i < NumIter; i++) {
    uint8_t T[8] = {0x00, 0x11, 0x22, 0x33};
    size_t NewSize = (*MD.*M)(T, 4, 8);
    if (NewSize == 7 && !memcmp(INS0, T, 7))
      FoundMask |= 1 << 0;
    if (NewSize == 7 && !memcmp(INS1, T, 7))
      FoundMask |= 1 << 1;
    if (NewSize == 7 && !memcmp(INS2, T, 7))
      FoundMask |= 1 << 2;
    if (NewSize == 7 && !memcmp(INS3, T, 7))
      FoundMask |= 1 << 3;
    if (NewSize == 7 && !memcmp(INS4, T, 7))
      FoundMask |= 1 << 4;

    if (NewSize == 8 && !memcmp(INS5, T, 8))
      FoundMask |= 1 << 5;
    if (NewSize == 8 && !memcmp(INS6, T, 8))
      FoundMask |= 1 << 6;
    if (NewSize == 8 && !memcmp(INS7, T, 8))
      FoundMask |= 1 << 7;
    if (NewSize == 8 && !memcmp(INS8, T, 8))
      FoundMask |= 1 << 8;
    if (NewSize == 8 && !memcmp(INS9, T, 8))
      FoundMask |= 1 << 9;
  }
  EXPECT_EQ(FoundMask, (1 << 10) - 1);
}

TEST(MutationDispatcher, InsertRepeatedBytes1) {
  TestInsertRepeatedBytes(&MutationDispatcher::Mutate_InsertRepeatedBytes,
                          10000);
}
TEST(MutationDispatcher, InsertRepeatedBytes2) {
  TestInsertRepeatedBytes(&MutationDispatcher::Mutate, 300000);
}

void TestChangeByte(Mutator M, int NumIter) {
  auto MD = CreateMutationDispatcher();
  int FoundMask = 0;
  uint8_t CH0[8] = {0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH1[8] = {0x00, 0xF1, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH2[8] = {0x00, 0x11, 0xF2, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH3[8] = {0x00, 0x11, 0x22, 0xF3, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH4[8] = {0x00, 0x11, 0x22, 0x33, 0xF4, 0x55, 0x66, 0x77};
  uint8_t CH5[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0xF5, 0x66, 0x77};
  uint8_t CH6[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0xF5, 0x77};
  uint8_t CH7[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0xF7};
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[9] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
    size_t NewSize = (*MD.*M)(T, 8, 9);
    if (NewSize == 8 && !memcmp(CH0, T, 8))
      FoundMask |= 1 << 0;
    if (NewSize == 8 && !memcmp(CH1, T, 8))
      FoundMask |= 1 << 1;
    if (NewSize == 8 && !memcmp(CH2, T, 8))
      FoundMask |= 1 << 2;
    if (NewSize == 8 && !memcmp(CH3, T, 8))
      FoundMask |= 1 << 3;
    if (NewSize == 8 && !memcmp(CH4, T, 8))
      FoundMask |= 1 << 4;
    if (NewSize == 8 && !memcmp(CH5, T, 8))
      FoundMask |= 1 << 5;
    if (NewSize == 8 && !memcmp(CH6, T, 8))
      FoundMask |= 1 << 6;
    if (NewSize == 8 && !memcmp(CH7, T, 8))
      FoundMask |= 1 << 7;
  }
  EXPECT_EQ(FoundMask, 255);
}

TEST(MutationDispatcher, ChangeByte1) {
  TestChangeByte(&MutationDispatcher::Mutate_ChangeByte, 1 << 15);
}
TEST(MutationDispatcher, ChangeByte2) {
  TestChangeByte(&MutationDispatcher::Mutate, 1 << 17);
}

void TestChangeBit(Mutator M, int NumIter) {
  auto MD = CreateMutationDispatcher();
  int FoundMask = 0;
  uint8_t CH0[8] = {0x01, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH1[8] = {0x00, 0x13, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH2[8] = {0x00, 0x11, 0x02, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH3[8] = {0x00, 0x11, 0x22, 0x37, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH4[8] = {0x00, 0x11, 0x22, 0x33, 0x54, 0x55, 0x66, 0x77};
  uint8_t CH5[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x54, 0x66, 0x77};
  uint8_t CH6[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x76, 0x77};
  uint8_t CH7[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0xF7};
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[9] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
    size_t NewSize = (*MD.*M)(T, 8, 9);
    if (NewSize == 8 && !memcmp(CH0, T, 8))
      FoundMask |= 1 << 0;
    if (NewSize == 8 && !memcmp(CH1, T, 8))
      FoundMask |= 1 << 1;
    if (NewSize == 8 && !memcmp(CH2, T, 8))
      FoundMask |= 1 << 2;
    if (NewSize == 8 && !memcmp(CH3, T, 8))
      FoundMask |= 1 << 3;
    if (NewSize == 8 && !memcmp(CH4, T, 8))
      FoundMask |= 1 << 4;
    if (NewSize == 8 && !memcmp(CH5, T, 8))
      FoundMask |= 1 << 5;
    if (NewSize == 8 && !memcmp(CH6, T, 8))
      FoundMask |= 1 << 6;
    if (NewSize == 8 && !memcmp(CH7, T, 8))
      FoundMask |= 1 << 7;
  }
  EXPECT_EQ(FoundMask, 255);
}

TEST(MutationDispatcher, ChangeBit1) {
  TestChangeBit(&MutationDispatcher::Mutate_ChangeBit, 1 << 16);
}
TEST(MutationDispatcher, ChangeBit2) {
  TestChangeBit(&MutationDispatcher::Mutate, 1 << 18);
}

void TestShuffleBytes(Mutator M, int NumIter) {
  auto MD = CreateMutationDispatcher();
  int FoundMask = 0;
  uint8_t CH0[7] = {0x00, 0x22, 0x11, 0x33, 0x44, 0x55, 0x66};
  uint8_t CH1[7] = {0x11, 0x00, 0x33, 0x22, 0x44, 0x55, 0x66};
  uint8_t CH2[7] = {0x00, 0x33, 0x11, 0x22, 0x44, 0x55, 0x66};
  uint8_t CH3[7] = {0x00, 0x11, 0x22, 0x44, 0x55, 0x66, 0x33};
  uint8_t CH4[7] = {0x00, 0x11, 0x22, 0x33, 0x55, 0x44, 0x66};
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[7] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    size_t NewSize = (*MD.*M)(T, 7, 7);
    if (NewSize == 7 && !memcmp(CH0, T, 7))
      FoundMask |= 1 << 0;
    if (NewSize == 7 && !memcmp(CH1, T, 7))
      FoundMask |= 1 << 1;
    if (NewSize == 7 && !memcmp(CH2, T, 7))
      FoundMask |= 1 << 2;
    if (NewSize == 7 && !memcmp(CH3, T, 7))
      FoundMask |= 1 << 3;
    if (NewSize == 7 && !memcmp(CH4, T, 7))
      FoundMask |= 1 << 4;
  }
  EXPECT_EQ(FoundMask, 31);
}

TEST(MutationDispatcher, ShuffleBytes1) {
  TestShuffleBytes(&MutationDispatcher::Mutate_ShuffleBytes, 1 << 17);
}
TEST(MutationDispatcher, ShuffleBytes2) {
  TestShuffleBytes(&MutationDispatcher::Mutate, 1 << 20);
}

void TestCopyPart(Mutator M, int NumIter) {
  auto MD = CreateMutationDispatcher();
  int FoundMask = 0;
  uint8_t CH0[7] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x00, 0x11};
  uint8_t CH1[7] = {0x55, 0x66, 0x22, 0x33, 0x44, 0x55, 0x66};
  uint8_t CH2[7] = {0x00, 0x55, 0x66, 0x33, 0x44, 0x55, 0x66};
  uint8_t CH3[7] = {0x00, 0x11, 0x22, 0x00, 0x11, 0x22, 0x66};
  uint8_t CH4[7] = {0x00, 0x11, 0x11, 0x22, 0x33, 0x55, 0x66};

  for (int i = 0; i < NumIter; i++) {
    uint8_t T[7] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    size_t NewSize = (*MD.*M)(T, 7, 7);
    if (NewSize == 7 && !memcmp(CH0, T, 7))
      FoundMask |= 1 << 0;
    if (NewSize == 7 && !memcmp(CH1, T, 7))
      FoundMask |= 1 << 1;
    if (NewSize == 7 && !memcmp(CH2, T, 7))
      FoundMask |= 1 << 2;
    if (NewSize == 7 && !memcmp(CH3, T, 7))
      FoundMask |= 1 << 3;
    if (NewSize == 7 && !memcmp(CH4, T, 7))
      FoundMask |= 1 << 4;
  }

  uint8_t CH5[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x00, 0x11, 0x22};
  uint8_t CH6[8] = {0x22, 0x33, 0x44, 0x00, 0x11, 0x22, 0x33, 0x44};
  uint8_t CH7[8] = {0x00, 0x11, 0x22, 0x00, 0x11, 0x22, 0x33, 0x44};
  uint8_t CH8[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x22, 0x33, 0x44};
  uint8_t CH9[8] = {0x00, 0x11, 0x22, 0x22, 0x33, 0x44, 0x33, 0x44};

  for (int i = 0; i < NumIter; i++) {
    uint8_t T[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
    size_t NewSize = (*MD.*M)(T, 5, 8);
    if (NewSize == 8 && !memcmp(CH5, T, 8))
      FoundMask |= 1 << 5;
    if (NewSize == 8 && !memcmp(CH6, T, 8))
      FoundMask |= 1 << 6;
    if (NewSize == 8 && !memcmp(CH7, T, 8))
      FoundMask |= 1 << 7;
    if (NewSize == 8 && !memcmp(CH8, T, 8))
      FoundMask |= 1 << 8;
    if (NewSize == 8 && !memcmp(CH9, T, 8))
      FoundMask |= 1 << 9;
  }

  EXPECT_EQ(FoundMask, 1023);
}

TEST(MutationDispatcher, CopyPart1) {
  TestCopyPart(&MutationDispatcher::Mutate_CopyPart, 1 << 10);
}
TEST(MutationDispatcher, CopyPart2) {
  TestCopyPart(&MutationDispatcher::Mutate, 1 << 13);
}
TEST(MutationDispatcher, CopyPartNoInsertAtMaxSize) {
  // This (non exhaustively) tests if `Mutate_CopyPart` tries to perform an
  // insert on an input of size `MaxSize`.  Performing an insert in this case
  // will lead to the mutation failing.
  auto MD = CreateMutationDispatcher();
  uint8_t Data[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x00, 0x11, 0x22};
  size_t MaxSize = sizeof(Data);
  for (int count = 0; count < (1 << 18); ++count) {
    size_t NewSize = MD->Mutate_CopyPart(Data, MaxSize, MaxSize);
    ASSERT_EQ(NewSize, MaxSize);
  }
}

void TestAddWordFromDictionary(Mutator M, int NumIter) {
  auto MD = CreateMutationDispatcher();
  uint8_t Word1[4] = {0xAA, 0xBB, 0xCC, 0xDD};
  uint8_t Word2[3] = {0xFF, 0xEE, 0xEF};
  MD->AddWordToManualDictionary(Word(Word1, sizeof(Word1)));
  MD->AddWordToManualDictionary(Word(Word2, sizeof(Word2)));
  int FoundMask = 0;
  uint8_t CH0[7] = {0x00, 0x11, 0x22, 0xAA, 0xBB, 0xCC, 0xDD};
  uint8_t CH1[7] = {0x00, 0x11, 0xAA, 0xBB, 0xCC, 0xDD, 0x22};
  uint8_t CH2[7] = {0x00, 0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22};
  uint8_t CH3[7] = {0xAA, 0xBB, 0xCC, 0xDD, 0x00, 0x11, 0x22};
  uint8_t CH4[6] = {0x00, 0x11, 0x22, 0xFF, 0xEE, 0xEF};
  uint8_t CH5[6] = {0x00, 0x11, 0xFF, 0xEE, 0xEF, 0x22};
  uint8_t CH6[6] = {0x00, 0xFF, 0xEE, 0xEF, 0x11, 0x22};
  uint8_t CH7[6] = {0xFF, 0xEE, 0xEF, 0x00, 0x11, 0x22};
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[7] = {0x00, 0x11, 0x22};
    size_t NewSize = (*MD.*M)(T, 3, 7);
    if (NewSize == 7 && !memcmp(CH0, T, 7))
      FoundMask |= 1 << 0;
    if (NewSize == 7 && !memcmp(CH1, T, 7))
      FoundMask |= 1 << 1;
    if (NewSize == 7 && !memcmp(CH2, T, 7))
      FoundMask |= 1 << 2;
    if (NewSize == 7 && !memcmp(CH3, T, 7))
      FoundMask |= 1 << 3;
    if (NewSize == 6 && !memcmp(CH4, T, 6))
      FoundMask |= 1 << 4;
    if (NewSize == 6 && !memcmp(CH5, T, 6))
      FoundMask |= 1 << 5;
    if (NewSize == 6 && !memcmp(CH6, T, 6))
      FoundMask |= 1 << 6;
    if (NewSize == 6 && !memcmp(CH7, T, 6))
      FoundMask |= 1 << 7;
  }
  EXPECT_EQ(FoundMask, 255);
}

TEST(MutationDispatcher, AddWordFromDictionary1) {
  TestAddWordFromDictionary(
      &MutationDispatcher::Mutate_AddWordFromManualDictionary, 1 << 15);
}

TEST(MutationDispatcher, AddWordFromDictionary2) {
  TestAddWordFromDictionary(&MutationDispatcher::Mutate, 1 << 15);
}

void TestChangeASCIIInteger(Mutator M, int NumIter) {
  auto MD = CreateMutationDispatcher();

  uint8_t CH0[8] = {'1', '2', '3', '4', '5', '6', '7', '7'};
  uint8_t CH1[8] = {'1', '2', '3', '4', '5', '6', '7', '9'};
  uint8_t CH2[8] = {'2', '4', '6', '9', '1', '3', '5', '6'};
  uint8_t CH3[8] = {'0', '6', '1', '7', '2', '8', '3', '9'};
  int FoundMask = 0;
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[8] = {'1', '2', '3', '4', '5', '6', '7', '8'};
    size_t NewSize = (*MD.*M)(T, 8, 8);
    /**/ if (NewSize == 8 && !memcmp(CH0, T, 8))
      FoundMask |= 1 << 0;
    else if (NewSize == 8 && !memcmp(CH1, T, 8))
      FoundMask |= 1 << 1;
    else if (NewSize == 8 && !memcmp(CH2, T, 8))
      FoundMask |= 1 << 2;
    else if (NewSize == 8 && !memcmp(CH3, T, 8))
      FoundMask |= 1 << 3;
    else if (NewSize == 8)
      FoundMask |= 1 << 4;
  }
  EXPECT_EQ(FoundMask, 31);
}

TEST(MutationDispatcher, ChangeASCIIInteger1) {
  TestChangeASCIIInteger(&MutationDispatcher::Mutate_ChangeASCIIInteger,
                         1 << 15);
}

TEST(MutationDispatcher, ChangeASCIIInteger2) {
  TestChangeASCIIInteger(&MutationDispatcher::Mutate, 1 << 15);
}

void TestChangeBinaryInteger(Mutator M, int NumIter) {
  auto MD = CreateMutationDispatcher();

  uint8_t CH0[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x79};
  uint8_t CH1[8] = {0x00, 0x11, 0x22, 0x31, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH2[8] = {0xff, 0x10, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH3[8] = {0x00, 0x11, 0x2a, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t CH4[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x4f, 0x66, 0x77};
  uint8_t CH5[8] = {0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88};
  uint8_t CH6[8] = {0x00, 0x11, 0x22, 0x00, 0x00, 0x00, 0x08, 0x77}; // Size
  uint8_t CH7[8] = {0x00, 0x08, 0x00, 0x33, 0x44, 0x55, 0x66, 0x77}; // Sw(Size)

  int FoundMask = 0;
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
    size_t NewSize = (*MD.*M)(T, 8, 8);
    /**/ if (NewSize == 8 && !memcmp(CH0, T, 8))
      FoundMask |= 1 << 0;
    else if (NewSize == 8 && !memcmp(CH1, T, 8))
      FoundMask |= 1 << 1;
    else if (NewSize == 8 && !memcmp(CH2, T, 8))
      FoundMask |= 1 << 2;
    else if (NewSize == 8 && !memcmp(CH3, T, 8))
      FoundMask |= 1 << 3;
    else if (NewSize == 8 && !memcmp(CH4, T, 8))
      FoundMask |= 1 << 4;
    else if (NewSize == 8 && !memcmp(CH5, T, 8))
      FoundMask |= 1 << 5;
    else if (NewSize == 8 && !memcmp(CH6, T, 8))
      FoundMask |= 1 << 6;
    else if (NewSize == 8 && !memcmp(CH7, T, 8))
      FoundMask |= 1 << 7;
  }
  EXPECT_EQ(FoundMask, 255);
}

TEST(MutationDispatcher, ChangeBinaryInteger1) {
  TestChangeBinaryInteger(&MutationDispatcher::Mutate_ChangeBinaryInteger,
                          1 << 12);
}

TEST(MutationDispatcher, ChangeBinaryInteger2) {
  TestChangeBinaryInteger(&MutationDispatcher::Mutate, 1 << 15);
}

// Test fixture for MutagenInterface unit tests.
static const char *kWord1 = "word1";
static const char *kWord2 = "word2";

class MutagenInterface : public ::testing::Test {
protected:
  void SetUp() override {
    Current = this;
    memset(&Config, 0, sizeof(Config));

    Config.Seed = 1;

    Config.UseCmp = 1;
    Config.FromTORC4 = [](size_t Idx, uint32_t *Arg1, uint32_t *Arg2) {
      ++(Current->FromTORC4Calls);
      *Arg1 = 0x0401;
      *Arg2 = 0x0402;
    };
    Config.FromTORC8 = [](size_t Idx, uint64_t *Arg1, uint64_t *Arg2) {
      ++(Current->FromTORC8Calls);
      *Arg1 = 0x0801;
      *Arg2 = 0x0802;
    };
    Config.FromTORCW = [](size_t Idx, const uint8_t **Data1, size_t *Size1,
                          const uint8_t **Data2, size_t *Size2) {
      ++(Current->FromTORCWCalls);
      *Data1 = reinterpret_cast<const uint8_t *>(kWord1);
      *Size1 = strlen(kWord1);
      *Data2 = reinterpret_cast<const uint8_t *>(kWord2);
      *Size2 = strlen(kWord2);
    };

    Config.UseMemmem = 0;
    Config.FromMMT = [](size_t Idx, const uint8_t **Data, size_t *Size) {
      ++(Current->FromMMTCalls);
      *Data = reinterpret_cast<const uint8_t *>(kWord1);
      *Size = strlen(kWord1);
    };

    Config.OnlyASCII = 0;

    Config.CustomMutator = [](uint8_t *Data, size_t Size, size_t MaxSize,
                              unsigned int Seed) {
      ++(Current->CustomMutatorCalls);
      return LLVMMutagenDefaultMutate(Data, Size, MaxSize);
    };

    Config.CustomCrossOver =
        [](const uint8_t *Data1, size_t Size1, const uint8_t *Data2,
           size_t Size2, uint8_t *Out, size_t MaxOutSize, unsigned int Seed) {
          ++(Current->CustomCrossOverCalls);
          auto *MD = GetMutationDispatcherForTest();
          return MD->CrossOver(Data1, Size1, Data2, Size2, Out, MaxOutSize);
        };

    U = Unit({1, 2, 3, 4});
    U.reserve(8);
  }

  void TearDown() override {
    Current = nullptr;
    memset(&Config, 0, sizeof(Config));
    LLVMMutagenConfigure(&Config);
  }

  LLVMMutagenConfiguration Config;
  Unit U;

  size_t FromTORC4Calls = 0;
  size_t FromTORC8Calls = 0;
  size_t FromTORCWCalls = 0;
  size_t FromMMTCalls = 0;
  size_t CustomMutatorCalls = 0;
  size_t CustomCrossOverCalls = 0;

private:
  static MutagenInterface *Current;
};

MutagenInterface *MutagenInterface::Current = nullptr;

// Unit tests for MutagenInterface.

TEST_F(MutagenInterface, Configure) {
  Config.OnlyASCII = 1;
  LLVMMutagenConfigure(&Config);
  auto *MD = GetMutationDispatcherForTest();
  ASSERT_NE(MD, nullptr);

  Random Rand1(Config.Seed);
  Random &Rand2 = MD->GetRand();
  for (size_t i = 0; i < 10; ++i)
    EXPECT_EQ(Rand1(), Rand2());

  Config.Seed = static_cast<unsigned>(
      std::chrono::system_clock::now().time_since_epoch().count());
  Config.OnlyASCII = 0;
  LLVMMutagenConfigure(&Config);
  MD = GetMutationDispatcherForTest();
  ASSERT_NE(MD, nullptr);

  Random Rand3(Config.Seed);
  Random &Rand4 = MD->GetRand();
  for (size_t i = 0; i < 10; ++i)
    EXPECT_EQ(Rand3(), Rand4());
}

TEST_F(MutagenInterface, UseTORCs) {
  // If !UseCmp, none of the TORC/MMT callbacks are called, regardless of
  // UseMemmem.
  Config.UseCmp = 0;
  Config.UseMemmem = 1;
  LLVMMutagenConfigure(&Config);
  for (size_t i = 0; i < 200; ++i)
    LLVMMutagenMutate(U.data(), U.size(), U.capacity());
  EXPECT_EQ(FromTORC4Calls, 0U);
  EXPECT_EQ(FromTORC8Calls, 0U);
  EXPECT_EQ(FromTORCWCalls, 0U);
  EXPECT_EQ(FromMMTCalls, 0U);

  // If UseCmp, but !UseMemmem, only the TORC callbacks are invoked.
  Config.UseCmp = 1;
  Config.UseMemmem = 0;
  LLVMMutagenConfigure(&Config);
  for (size_t i = 0; i < 200; ++i)
    LLVMMutagenMutate(U.data(), U.size(), U.capacity());
  EXPECT_NE(FromTORC4Calls, 0U);
  EXPECT_NE(FromTORC8Calls, 0U);
  EXPECT_NE(FromTORCWCalls, 0U);
  EXPECT_EQ(FromMMTCalls, 0U);

  // If UseCmp and UseMemmem, all the TORC/MMT callbacks are invoked.
  Config.UseCmp = 1;
  Config.UseMemmem = 1;
  LLVMMutagenConfigure(&Config);
  for (size_t i = 0; i < 200; ++i)
    LLVMMutagenMutate(U.data(), U.size(), U.capacity());
  EXPECT_NE(FromTORC4Calls, 0U);
  EXPECT_NE(FromTORC8Calls, 0U);
  EXPECT_NE(FromTORCWCalls, 0U);
  EXPECT_NE(FromMMTCalls, 0U);
}

TEST_F(MutagenInterface, CustomCallbacks) {
  // DefaultMutate never selects custom callbacks.
  LLVMMutagenConfigure(&Config);
  for (size_t i = 0; i < 200; ++i)
    LLVMMutagenDefaultMutate(U.data(), U.size(), U.capacity());

  // Valid.
  auto *MD = GetMutationDispatcherForTest();
  EXPECT_EQ(CustomMutatorCalls, 0U);
  MD->Mutate_Custom(U.data(), U.size(), U.capacity());
  EXPECT_EQ(CustomMutatorCalls, 1U);

  // Null cross-over input disables CustomCrossOver.
  LLVMMutagenSetCrossOverWith(nullptr, 0);
  MD->Mutate_CustomCrossOver(U.data(), U.size(), U.capacity());
  EXPECT_EQ(CustomCrossOverCalls, 0U);

  // Zero-length cross-over input disables CustomCrossOver.
  Unit CrossOverWith = {4, 3, 2, 1};
  LLVMMutagenSetCrossOverWith(CrossOverWith.data(), 0);
  MD->Mutate_CustomCrossOver(U.data(), U.size(), U.capacity());
  EXPECT_EQ(CustomCrossOverCalls, 0U);

  // Valid.
  LLVMMutagenSetCrossOverWith(CrossOverWith.data(), CrossOverWith.size());
  MD->Mutate_CustomCrossOver(U.data(), U.size(), U.capacity());
  EXPECT_EQ(CustomCrossOverCalls, 1U);

  // Can mutate without custom callbacks.
  Config.CustomMutator = nullptr;
  Config.CustomCrossOver = nullptr;
  LLVMMutagenConfigure(&Config);
  for (size_t i = 0; i < 200; ++i)
    LLVMMutagenMutate(U.data(), U.size(), U.capacity());
}

TEST_F(MutagenInterface, MutationSequence) {
  LLVMMutagenConfigure(&Config);
  char Buf[1024];
  size_t NumItems;

  Set<std::string> Names = {
      "ShuffleBytes", "EraseBytes", "InsertBytes", "InsertRepeatedBytes",
      "ChangeByte",   "ChangeBit",  "CopyPart",    "ChangeASCIIInt",
      "ChangeBinInt",
  };
  std::string Name;
  std::istringstream ISS;

  // Empty sequences
  auto Size = LLVMMutagenGetMutationSequence(true, Buf, sizeof(Buf), &NumItems);
  EXPECT_STREQ(Buf, "");
  EXPECT_EQ(Size, 0U);
  EXPECT_EQ(NumItems, 0U);

  while (true) {
    // Can get size without output parameters.
    Size = LLVMMutagenGetMutationSequence(true, nullptr, 0, &NumItems);
    if (NumItems > Sequence<Mutator>::kMaxBriefItems)
      break;
    // !Verbose has no effect for <= 10 items.
    EXPECT_EQ(LLVMMutagenGetMutationSequence(false, nullptr, 0, nullptr), Size);
    EXPECT_GT(LLVMMutagenDefaultMutate(U.data(), U.size(), U.capacity()), 0U);
  }

  // All items are valid.
  LLVMMutagenGetMutationSequence(true, Buf, sizeof(Buf), nullptr);
  ISS.str(Buf);
  size_t N = 0;
  while (std::getline(ISS, Name, '-')) {
    EXPECT_GT(Names.count(Name), 0U);
    ++N;
  }
  EXPECT_EQ(N, NumItems);

  // !Verbose truncates, but items are still valid.
  EXPECT_LT(LLVMMutagenGetMutationSequence(false, Buf, sizeof(Buf), nullptr),
            Size);
  ISS.str(Buf);
  N = 0;
  while (std::getline(ISS, Name, '-')) {
    EXPECT_GT(Names.count(Name), 0U);
    ++N;
  }
  EXPECT_LT(N, NumItems);

  // Truncated sequence is a prefix of its untruncated equivalent.
  std::string Truncated(Buf);
  LLVMMutagenGetMutationSequence(true, Buf, sizeof(Buf), &NumItems);
  Buf[Truncated.size()] = '\0';
  EXPECT_STREQ(Truncated.c_str(), Buf);

  // Stops at the end of |Buf|, and null terminates.
  EXPECT_EQ(LLVMMutagenGetMutationSequence(true, Buf, Size - 1, nullptr), Size);
  EXPECT_EQ(strlen(Buf), Size - 2);

  // Clear the sequence.
  LLVMMutagenResetSequence();
  EXPECT_EQ(LLVMMutagenGetMutationSequence(true, nullptr, 0, nullptr), 0U);
}

static uint8_t FromASCIINybble(char C) {
  if ('0' <= C && C <= '9')
    return static_cast<uint8_t>(C - '0');
  if ('A' <= C && C <= 'F')
    return static_cast<uint8_t>(C - 'A' + 10);
  assert('a' <= C && C <= 'f');
  return static_cast<uint8_t>(C - 'a' + 10);
}

static Word FromASCII(const char *DE) {
  Unit Tmp;
  bool Escape = false;
  size_t Hex = 0;
  uint8_t Nybble = 0;
  for (char C = *DE++; C; C = *DE++) {
    if (Hex == 2) {
      Nybble = FromASCIINybble(C);
      --Hex;
    } else if (Hex == 1) {
      Tmp.push_back(static_cast<uint8_t>(Nybble << 4) | FromASCIINybble(C));
      --Hex;
    } else if (Escape) {
      switch (C) {
      case '\\':
      case '"':
        Tmp.push_back(static_cast<uint8_t>(C));
        break;
      case 'x':
        Hex = 2;
        break;
      default:
        assert(false && "FromASCII failure.");
      }
      Escape = false;
    } else if (C == '\\') {
      Escape = true;
    } else {
      Tmp.push_back(static_cast<uint8_t>(C));
    }
  }
  return Word(Tmp.data(), Tmp.size());
}

TEST_F(MutagenInterface, Dictionaries) {
  LLVMMutagenConfigure(&Config);
  size_t NumItems;
  char Buf[1024];
  std::istringstream ISS;
  std::string Str;

  // Empty sequences
  auto Size =
      LLVMMutagenGetDictionaryEntrySequence(true, Buf, sizeof(Buf), &NumItems);
  EXPECT_STREQ(Buf, "");
  EXPECT_EQ(Size, 0U);
  EXPECT_EQ(NumItems, 0U);

  auto *MD = GetMutationDispatcherForTest();
  while (true) {
    // Can get size without output parameters.
    Size = LLVMMutagenGetDictionaryEntrySequence(true, nullptr, 0, &NumItems);
    if (NumItems > Sequence<DictionaryEntry *>::kMaxBriefItems)
      break;
    // !Verbose has no effect for <= 10 items.
    EXPECT_EQ(LLVMMutagenGetDictionaryEntrySequence(false, nullptr, 0, nullptr),
              Size);
    MD->Mutate_AddWordFromTORC(U.data(), U.size(), U.capacity());
  }

  // All items are valid.
  LLVMMutagenGetDictionaryEntrySequence(true, Buf, sizeof(Buf), nullptr);
  ISS.str(Buf);
  size_t N = 0;
  while (std::getline(ISS, Str, '-')) {
    ASSERT_FALSE(Str.empty());
    EXPECT_EQ(Str[0], '"');
    EXPECT_EQ(Str[Str.size() - 1], '"');
    ++N;
  }
  EXPECT_EQ(N, NumItems);

  // !Verbose truncates, but items are still valid.
  EXPECT_LT(
      LLVMMutagenGetDictionaryEntrySequence(false, Buf, sizeof(Buf), nullptr),
      Size);
  ISS.str(Buf);
  N = 0;
  while (std::getline(ISS, Str, '-')) {
    ASSERT_FALSE(Str.empty());
    EXPECT_EQ(Str[0], '"');
    EXPECT_EQ(Str[Str.size() - 1], '"');
    ++N;
  }
  EXPECT_LT(N, NumItems);

  // Truncated sequence is a prefix of its untruncated equivalent.
  std::string Truncated(Buf);
  LLVMMutagenGetDictionaryEntrySequence(true, Buf, sizeof(Buf), &NumItems);
  Buf[Truncated.size()] = '\0';
  EXPECT_STREQ(Truncated.c_str(), Buf);

  // Stops at the end of |Buf|, and null terminates.
  EXPECT_EQ(LLVMMutagenGetDictionaryEntrySequence(true, Buf, Size - 1, nullptr),
            Size);
  EXPECT_EQ(strlen(Buf), Size - 2);

  // Clear the sequence.
  LLVMMutagenResetSequence();
  EXPECT_EQ(LLVMMutagenGetDictionaryEntrySequence(true, nullptr, 0, nullptr),
            0U);

  // Retuns null if no recommendations.
  size_t UseCount = 0;
  EXPECT_EQ(LLVMMutagenRecommendDictionaryEntry(&UseCount), nullptr);
  EXPECT_EQ(LLVMMutagenRecommendDictionary(), 0U);
  EXPECT_EQ(LLVMMutagenRecommendDictionaryEntry(&UseCount), nullptr);

  // Record sequences.
  for (size_t i = 0; i < 5; ++i) {
    for (size_t i = 0; i < 5; ++i) {
      MD->Mutate_AddWordFromTORC(U.data(), U.size(), U.capacity());
    }
    LLVMMutagenRecordSequence();
  }

  size_t NumDEs = LLVMMutagenRecommendDictionary();
  EXPECT_NE(NumDEs, 0U);
  for (size_t i = 0; i < NumDEs; ++i) {
    auto *DE = LLVMMutagenRecommendDictionaryEntry(&UseCount);
    EXPECT_NE(DE, nullptr);
    EXPECT_EQ(UseCount, 0U);
  }

  // Increment the use counts of entries.
  for (size_t i = 0; i < 100; ++i)
    MD->Mutate_AddWordFromPersistentAutoDictionary(U.data(), U.size(),
                                                   U.capacity());
  NumDEs = LLVMMutagenRecommendDictionary();
  EXPECT_NE(NumDEs, 0U);
  for (size_t i = 0; i < NumDEs; ++i) {
    auto *DE = LLVMMutagenRecommendDictionaryEntry(&UseCount);
    EXPECT_NE(DE, nullptr);
    EXPECT_NE(UseCount, 0U);
  }

  // Add the first few words manually to exclude them from recommendations.
  Vector<Word> ManualAdditions;
  NumDEs = LLVMMutagenRecommendDictionary();
  ASSERT_GT(NumDEs, 3U);
  for (size_t i = 0; i < 3; ++i) {
    auto *DE = LLVMMutagenRecommendDictionaryEntry(nullptr);
    auto W = FromASCII(DE);
    LLVMMutagenAddWordToDictionary(W.data(), W.size());
    ManualAdditions.push_back(W);
  }
  N = NumDEs;

  // Get the recommended dictionary without the manual additions.
  NumDEs = LLVMMutagenRecommendDictionary();
  EXPECT_EQ(NumDEs, N - 3);
  for (size_t i = 0; i < NumDEs; ++i) {
    auto *DE = LLVMMutagenRecommendDictionaryEntry(nullptr);
    ASSERT_NE(DE, nullptr);
    Word W1(reinterpret_cast<const uint8_t *>(DE), strlen(DE));
    for (const auto &W2 : ManualAdditions)
      EXPECT_FALSE(W1 == W2);
  }
}

} // namespace
} // namespace mutagen

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
