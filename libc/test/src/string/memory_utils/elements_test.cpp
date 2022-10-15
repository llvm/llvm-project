//===-- Unittests for memory_utils ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/span.h"
#include "src/string/memory_utils/elements.h"
#include "utils/UnitTest/Test.h"

namespace __llvm_libc {

// Registering Types
using FixedSizeTypes = testing::TypeList<
#if defined(__SSE2__)
    x86::Vector128, //
#endif              // __SSE2__
#if defined(__AVX2__)
    x86::Vector256, //
#endif              // __AVX2__
#if defined(__AVX512F__) and defined(__AVX512BW__)
    x86::Vector512, //
#endif              // defined(__AVX512F__) and defined(__AVX512BW__)
    scalar::UINT8,  //
    scalar::UINT16, //
    scalar::UINT32, //
    scalar::UINT64, //
    Repeated<scalar::UINT64, 2>,                            //
    Repeated<scalar::UINT64, 4>,                            //
    Repeated<scalar::UINT64, 8>,                            //
    Repeated<scalar::UINT64, 16>,                           //
    Repeated<scalar::UINT64, 32>,                           //
    Chained<scalar::UINT16, scalar::UINT8>,                 //
    Chained<scalar::UINT32, scalar::UINT16, scalar::UINT8>, //
    builtin::_1,                                            //
    builtin::_2,                                            //
    builtin::_3,                                            //
    builtin::_4,                                            //
    builtin::_8                                             //
    >;

char GetRandomChar() {
  static constexpr const uint64_t a = 1103515245;
  static constexpr const uint64_t c = 12345;
  static constexpr const uint64_t m = 1ULL << 31;
  static uint64_t seed = 123456789;
  seed = (a * seed + c) % m;
  return seed;
}

void Randomize(cpp::span<char> buffer) {
  for (auto &current : buffer)
    current = GetRandomChar();
}

template <typename Element> using Buffer = cpp::array<char, Element::SIZE>;

template <typename Element> Buffer<Element> GetRandomBuffer() {
  Buffer<Element> buffer;
  Randomize(buffer);
  return buffer;
}

TYPED_TEST(LlvmLibcMemoryElements, copy, FixedSizeTypes) {
  Buffer<ParamType> Dst;
  const auto buffer = GetRandomBuffer<ParamType>();
  copy<ParamType>(Dst.data(), buffer.data());
  for (size_t i = 0; i < ParamType::SIZE; ++i)
    EXPECT_EQ(Dst[i], buffer[i]);
}

template <typename T> T copy(const T &Input) {
  T Output;
  for (size_t I = 0; I < Input.size(); ++I)
    Output[I] = Input[I];
  return Output;
}

TYPED_TEST(LlvmLibcMemoryElements, Move, FixedSizeTypes) {
  constexpr size_t SIZE = ParamType::SIZE;
  using LargeBuffer = cpp::array<char, SIZE * 2>;
  LargeBuffer GroundTruth;
  Randomize(GroundTruth);
  // Forward, we move the SIZE first bytes from offset 0 to SIZE.
  for (size_t Offset = 0; Offset < SIZE; ++Offset) {
    LargeBuffer Buffer = copy(GroundTruth);
    move<ParamType>(&Buffer[Offset], &Buffer[0]);
    for (size_t I = 0; I < SIZE; ++I)
      EXPECT_EQ(Buffer[I + Offset], GroundTruth[I]);
  }
  // Backward, we move the SIZE last bytes from offset 0 to SIZE.
  for (size_t Offset = 0; Offset < SIZE; ++Offset) {
    LargeBuffer Buffer = copy(GroundTruth);
    move<ParamType>(&Buffer[Offset], &Buffer[SIZE]);
    for (size_t I = 0; I < SIZE; ++I)
      EXPECT_EQ(Buffer[I + Offset], GroundTruth[SIZE + I]);
  }
}

TYPED_TEST(LlvmLibcMemoryElements, Equals, FixedSizeTypes) {
  const auto buffer = GetRandomBuffer<ParamType>();
  EXPECT_TRUE(equals<ParamType>(buffer.data(), buffer.data()));
}

TYPED_TEST(LlvmLibcMemoryElements, three_way_compare, FixedSizeTypes) {
  Buffer<ParamType> initial;
  for (auto &c : initial)
    c = 5;

  // Testing equality
  EXPECT_EQ(three_way_compare<ParamType>(initial.data(), initial.data()), 0);

  // Testing all mismatching positions
  for (size_t i = 0; i < ParamType::SIZE; ++i) {
    auto copy = initial;
    ++copy[i]; // copy is now lexicographycally greated than initial
    const auto *less = initial.data();
    const auto *greater = copy.data();
    EXPECT_LT(three_way_compare<ParamType>(less, greater), 0);
    EXPECT_GT(three_way_compare<ParamType>(greater, less), 0);
  }
}

TYPED_TEST(LlvmLibcMemoryElements, Splat, FixedSizeTypes) {
  Buffer<ParamType> Dst;
  const cpp::array<char, 3> values = {char(0x00), char(0x7F), char(0xFF)};
  for (char value : values) {
    splat_set<ParamType>(Dst.data(), value);
    for (size_t i = 0; i < ParamType::SIZE; ++i)
      EXPECT_EQ(Dst[i], value);
  }
}

} // namespace __llvm_libc
