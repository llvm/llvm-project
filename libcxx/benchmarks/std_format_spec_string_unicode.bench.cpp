// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_HAS_NO_UNICODE

#include <array>
#include <format>

#include "benchmark/benchmark.h"

#include "test_macros.h"

template <class CharT, size_t N>
class tester {
  static constexpr size_t size_ = N - 1;
  std::array<CharT, 100 * size_> data_;

public:
  explicit constexpr tester(const CharT (&input)[N]) {
    auto it = data_.begin();
    for (int i = 0; i < 100; ++i)
      it = std::copy_n(input, size_, it);
  }

  constexpr size_t size() const noexcept { return data_.size(); }
  constexpr const CharT* begin() const noexcept { return data_.begin(); }
  constexpr const CharT* end() const noexcept { return data_.end(); }

  void test(benchmark::State& state) const {
    for (auto _ : state)
      benchmark::DoNotOptimize(std::__format_spec::__get_string_alignment(
          begin(), end(), 1'000'000, 1'000'000));
    state.SetItemsProcessed(state.iterations() * size());
  }
};

#define TEST(u8)                                                               \
  if constexpr (std::same_as<CharT, char>) {                                   \
    constexpr auto p = tester{u8};                                             \
    p.test(state);                                                             \
  } else if constexpr (std::same_as<CharT, char16_t>) {                        \
    constexpr auto p = tester{TEST_CONCAT(u, u8)};                             \
    p.test(state);                                                             \
  } else {                                                                     \
    constexpr auto p = tester{TEST_CONCAT(U, u8)};                             \
    p.test(state);                                                             \
  }

template <class CharT>
static void BM_EstimateLengthNoMultiByte(benchmark::State& state) {
  TEST("The quick brown fox jumps over the lazy dog");
}

template <class CharT>
static void BM_EstimateLengthTwoByteDE(benchmark::State& state) {
  static_assert(sizeof("Victor jagt zwölf Boxkämpfer quer über den großen Sylter Deich") == 67);

  // https://en.wikipedia.org/wiki/Pangram
  TEST("Victor jagt zwölf Boxkämpfer quer über den großen Sylter Deich");
}

template <class CharT>
static void BM_EstimateLengthTwoBytePL(benchmark::State& state) {
  static_assert(sizeof("Stróż pchnął kość w quiz gędźb vel fax myjń") == 53);

  // https://en.wikipedia.org/wiki/Pangram
  TEST("Stróż pchnął kość w quiz gędźb vel fax myjń");
}

// All values below are 1100, which is is the first multi column sequence.
template <class CharT>
static void BM_EstimateLengthThreeByteSingleColumnLow(benchmark::State& state) {
  static_assert(sizeof("\u0800\u0801\u0802\u0803\u0804\u0805\u0806\u0807"
                       "\u0808\u0809\u080a\u080b\u080c\u080d\u080e\u080f") ==
                49);

  TEST("\u0800\u0801\u0802\u0803\u0804\u0805\u0806\u0807"
       "\u0808\u0809\u080a\u080b\u080c\u080d\u080e\u080f");
}

template <class CharT>
static void
BM_EstimateLengthThreeByteSingleColumnHigh(benchmark::State& state) {
  static_assert(sizeof("\u1800\u1801\u1802\u1803\u1804\u1805\u1806\u1807"
                       "\u1808\u1809\u180a\u180b\u180c\u180d\u180e\u180f") ==
                49);

  TEST("\u1800\u1801\u1802\u1803\u1804\u1805\u1806\u1807"
       "\u1808\u1809\u180a\u180b\u180c\u180d\u180e\u180f");
}

template <class CharT>
static void BM_EstimateLengthThreeByteDoubleColumn(benchmark::State& state) {
  static_assert(sizeof("\u1100\u0801\u0802\u0803\u0804\u0805\u0806\u0807"
                       "\u1108\u0809\u080a\u080b\u080c\u080d\u080e\u080f") ==
                49);

  TEST("\u1100\u0801\u0802\u0803\u0804\u0805\u0806\u0807"
       "\u1108\u0809\u080a\u080b\u080c\u080d\u080e\u080f");
}

template <class CharT>
static void BM_EstimateLengthThreeByte(benchmark::State& state) {
  static_assert(sizeof("\u1400\u1501\ubbbb\uff00\u0800\u4099\uabcd\u4000"
                       "\u8ead\ubeef\u1111\u4987\u4321\uffff\u357a\ud50e") ==
                49);

  TEST("\u1400\u1501\ubbbb\uff00\u0800\u4099\uabcd\u4000"
       "\u8ead\ubeef\u1111\u4987\u4321\uffff\u357a\ud50e");
}

template <class CharT>
static void BM_EstimateLengthFourByteSingleColumn(benchmark::State& state) {
  static_assert(sizeof("\U00010000\U00010001\U00010002\U00010003"
                       "\U00010004\U00010005\U00010006\U00010007"
                       "\U00010008\U00010009\U0001000a\U0001000b"
                       "\U0001000c\U0001000d\U0001000e\U0001000f") == 65);

  TEST("\U00010000\U00010001\U00010002\U00010003"
       "\U00010004\U00010005\U00010006\U00010007"
       "\U00010008\U00010009\U0001000a\U0001000b"
       "\U0001000c\U0001000d\U0001000e\U0001000f");
}

template <class CharT>
static void BM_EstimateLengthFourByteDoubleColumn(benchmark::State& state) {
  static_assert(sizeof("\U00020000\U00020002\U00020002\U00020003"
                       "\U00020004\U00020005\U00020006\U00020007"
                       "\U00020008\U00020009\U0002000a\U0002000b"
                       "\U0002000c\U0002000d\U0002000e\U0002000f") == 65);

  TEST("\U00020000\U00020002\U00020002\U00020003"
       "\U00020004\U00020005\U00020006\U00020007"
       "\U00020008\U00020009\U0002000a\U0002000b"
       "\U0002000c\U0002000d\U0002000e\U0002000f");
}

template <class CharT>
static void BM_EstimateLengthFourByte(benchmark::State& state) {
  static_assert(sizeof("\U00010000\U00010001\U00010002\U00010003"
                       "\U00020004\U00020005\U00020006\U00020007"
                       "\U00010008\U00010009\U0001000a\U0001000b"
                       "\U0002000c\U0002000d\U0002000e\U0002000f") == 65);

  TEST("\U00010000\U00010001\U00010002\U00010003"
       "\U00020004\U00020005\U00020006\U00020007"
       "\U00010008\U00010009\U0001000a\U0001000b"
       "\U0002000c\U0002000d\U0002000e\U0002000f");
}

BENCHMARK_TEMPLATE(BM_EstimateLengthNoMultiByte, char);
BENCHMARK_TEMPLATE(BM_EstimateLengthTwoByteDE, char);
BENCHMARK_TEMPLATE(BM_EstimateLengthTwoBytePL, char);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByteSingleColumnLow, char);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByteSingleColumnHigh, char);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByteDoubleColumn, char);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByte, char);
BENCHMARK_TEMPLATE(BM_EstimateLengthFourByteSingleColumn, char);
BENCHMARK_TEMPLATE(BM_EstimateLengthFourByteDoubleColumn, char);
BENCHMARK_TEMPLATE(BM_EstimateLengthFourByte, char);

BENCHMARK_TEMPLATE(BM_EstimateLengthNoMultiByte, char16_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthTwoByteDE, char16_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthTwoBytePL, char16_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByteSingleColumnLow, char16_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByteSingleColumnHigh, char16_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByteDoubleColumn, char16_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByte, char16_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthFourByteSingleColumn, char16_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthFourByteDoubleColumn, char16_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthFourByte, char16_t);

BENCHMARK_TEMPLATE(BM_EstimateLengthNoMultiByte, char32_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthTwoByteDE, char32_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthTwoBytePL, char32_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByteSingleColumnLow, char32_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByteSingleColumnHigh, char32_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByteDoubleColumn, char32_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthThreeByte, char32_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthFourByteSingleColumn, char32_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthFourByteDoubleColumn, char32_t);
BENCHMARK_TEMPLATE(BM_EstimateLengthFourByte, char32_t);

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  benchmark::RunSpecifiedBenchmarks();
}
#else
int main(int, char**) { return 0; }
#endif
