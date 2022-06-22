//===-- Unittests for backends --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Array.h"
#include "src/__support/CPP/ArrayRef.h"
#include "src/__support/CPP/Bit.h"
#include "src/__support/architectures.h"
#include "src/string/memory_utils/backends.h"
#include "utils/UnitTest/Test.h"
#include <string.h>

namespace __llvm_libc {

template <size_t Size> using Buffer = cpp::Array<char, Size>;

static char GetRandomChar() {
  // Implementation of C++ minstd_rand seeded with 123456789.
  // https://en.cppreference.com/w/cpp/numeric/random
  // "Minimum standard", recommended by Park, Miller, and Stockmeyer in 1993
  static constexpr const uint64_t a = 48271;
  static constexpr const uint64_t c = 0;
  static constexpr const uint64_t m = 2147483647;
  static uint64_t seed = 123456789;
  seed = (a * seed + c) % m;
  return seed;
}

static void Randomize(cpp::MutableArrayRef<char> buffer) {
  for (auto &current : buffer)
    current = GetRandomChar();
}

template <size_t Size> static Buffer<Size> GetRandomBuffer() {
  Buffer<Size> buffer;
  Randomize(buffer);
  return buffer;
}

template <typename Backend, size_t Size> struct Conf {
  static_assert(Backend::IS_BACKEND_TYPE);
  using BufferT = Buffer<Size>;
  using T = typename Backend::template getNextType<Size>;
  static_assert(sizeof(T) == Size);
  static constexpr size_t SIZE = Size;

  static BufferT splat(ubyte value) {
    return bit_cast<BufferT>(Backend::template splat<T>(value));
  }

  static uint64_t notEquals(const BufferT &v1, const BufferT &v2) {
    return Backend::template notEquals<T>(bit_cast<T>(v1), bit_cast<T>(v2));
  }

  static int32_t threeWayCmp(const BufferT &v1, const BufferT &v2) {
    return Backend::template threeWayCmp<T>(bit_cast<T>(v1), bit_cast<T>(v2));
  }
};

using FunctionTypes = testing::TypeList< //
#if defined(LLVM_LIBC_ARCH_X86)          //
    Conf<X86Backend, 1>,                 //
    Conf<X86Backend, 2>,                 //
    Conf<X86Backend, 4>,                 //
    Conf<X86Backend, 8>,                 //
#if HAS_M128
    Conf<X86Backend, 16>, //
#endif
#if HAS_M256
    Conf<X86Backend, 32>, //
#endif
#if HAS_M512
    Conf<X86Backend, 64>, //
#endif
#endif                           // defined(LLVM_LIBC_ARCH_X86)
    Conf<Scalar64BitBackend, 1>, //
    Conf<Scalar64BitBackend, 2>, //
    Conf<Scalar64BitBackend, 4>, //
    Conf<Scalar64BitBackend, 8>  //
    >;

TYPED_TEST(LlvmLibcMemoryBackend, splat, FunctionTypes) {
  for (auto value : cpp::Array<uint8_t, 3>{0u, 1u, 255u}) {
    alignas(64) const auto stored = ParamType::splat(bit_cast<ubyte>(value));
    for (size_t i = 0; i < ParamType::SIZE; ++i)
      EXPECT_EQ(bit_cast<uint8_t>(stored[i]), value);
  }
}

TYPED_TEST(LlvmLibcMemoryBackend, notEquals, FunctionTypes) {
  alignas(64) const auto a = GetRandomBuffer<ParamType::SIZE>();
  EXPECT_EQ(ParamType::notEquals(a, a), 0UL);
  for (size_t i = 0; i < a.size(); ++i) {
    alignas(64) auto b = a;
    ++b[i];
    EXPECT_NE(ParamType::notEquals(a, b), 0UL);
    EXPECT_NE(ParamType::notEquals(b, a), 0UL);
  }
}

TYPED_TEST(LlvmLibcMemoryBackend, threeWayCmp, FunctionTypes) {
  alignas(64) const auto a = GetRandomBuffer<ParamType::SIZE>();
  EXPECT_EQ(ParamType::threeWayCmp(a, a), 0);
  for (size_t i = 0; i < a.size(); ++i) {
    alignas(64) auto b = a;
    ++b[i];
    const auto cmp = memcmp(&a, &b, sizeof(a));
    ASSERT_NE(cmp, 0);
    if (cmp > 0) {
      EXPECT_GT(ParamType::threeWayCmp(a, b), 0);
      EXPECT_LT(ParamType::threeWayCmp(b, a), 0);
    } else {
      EXPECT_LT(ParamType::threeWayCmp(a, b), 0);
      EXPECT_GT(ParamType::threeWayCmp(b, a), 0);
    }
  }
}

template <typename Backend, size_t Size, Temporality TS, Aligned AS>
struct LoadStoreConf {
  static_assert(Backend::IS_BACKEND_TYPE);
  using BufferT = Buffer<Size>;
  using T = typename Backend::template getNextType<Size>;
  static_assert(sizeof(T) == Size);
  static constexpr size_t SIZE = Size;

  static BufferT load(const BufferT &ref) {
    const auto *ptr = bit_cast<const T *>(ref.data());
    const T value = Backend::template load<T, TS, AS>(ptr);
    return bit_cast<BufferT>(value);
  }

  static void store(BufferT &ref, const BufferT value) {
    auto *ptr = bit_cast<T *>(ref.data());
    Backend::template store<T, TS, AS>(ptr, bit_cast<T>(value));
  }
};

using LoadStoreTypes = testing::TypeList<                              //
#if defined(LLVM_LIBC_ARCH_X86)                                        //
    LoadStoreConf<X86Backend, 1, Temporality::TEMPORAL, Aligned::NO>,  //
    LoadStoreConf<X86Backend, 1, Temporality::TEMPORAL, Aligned::YES>, //
    LoadStoreConf<X86Backend, 2, Temporality::TEMPORAL, Aligned::NO>,  //
    LoadStoreConf<X86Backend, 2, Temporality::TEMPORAL, Aligned::YES>, //
    LoadStoreConf<X86Backend, 4, Temporality::TEMPORAL, Aligned::NO>,  //
    LoadStoreConf<X86Backend, 4, Temporality::TEMPORAL, Aligned::YES>, //
    LoadStoreConf<X86Backend, 8, Temporality::TEMPORAL, Aligned::NO>,  //
    LoadStoreConf<X86Backend, 8, Temporality::TEMPORAL, Aligned::YES>, //
#if HAS_M128
    LoadStoreConf<X86Backend, 16, Temporality::TEMPORAL, Aligned::NO>,      //
    LoadStoreConf<X86Backend, 16, Temporality::TEMPORAL, Aligned::YES>,     //
    LoadStoreConf<X86Backend, 16, Temporality::NON_TEMPORAL, Aligned::YES>, //
#endif
#if HAS_M256
    LoadStoreConf<X86Backend, 32, Temporality::TEMPORAL, Aligned::NO>,      //
    LoadStoreConf<X86Backend, 32, Temporality::TEMPORAL, Aligned::YES>,     //
    LoadStoreConf<X86Backend, 32, Temporality::NON_TEMPORAL, Aligned::YES>, //
#endif
#if HAS_M512
    LoadStoreConf<X86Backend, 64, Temporality::TEMPORAL, Aligned::NO>,      //
    LoadStoreConf<X86Backend, 64, Temporality::TEMPORAL, Aligned::YES>,     //
    LoadStoreConf<X86Backend, 64, Temporality::NON_TEMPORAL, Aligned::YES>, //
#endif
#endif // defined(LLVM_LIBC_ARCH_X86)
    LoadStoreConf<Scalar64BitBackend, 1, Temporality::TEMPORAL, Aligned::NO>, //
    LoadStoreConf<Scalar64BitBackend, 1, Temporality::TEMPORAL,
                  Aligned::YES>,                                              //
    LoadStoreConf<Scalar64BitBackend, 2, Temporality::TEMPORAL, Aligned::NO>, //
    LoadStoreConf<Scalar64BitBackend, 2, Temporality::TEMPORAL,
                  Aligned::YES>,                                              //
    LoadStoreConf<Scalar64BitBackend, 4, Temporality::TEMPORAL, Aligned::NO>, //
    LoadStoreConf<Scalar64BitBackend, 4, Temporality::TEMPORAL,
                  Aligned::YES>,                                              //
    LoadStoreConf<Scalar64BitBackend, 8, Temporality::TEMPORAL, Aligned::NO>, //
    LoadStoreConf<Scalar64BitBackend, 8, Temporality::TEMPORAL, Aligned::YES> //
    >;

TYPED_TEST(LlvmLibcMemoryBackend, load, LoadStoreTypes) {
  alignas(64) const auto expected = GetRandomBuffer<ParamType::SIZE>();
  const auto loaded = ParamType::load(expected);
  for (size_t i = 0; i < ParamType::SIZE; ++i)
    EXPECT_EQ(loaded[i], expected[i]);
}

TYPED_TEST(LlvmLibcMemoryBackend, store, LoadStoreTypes) {
  alignas(64) const auto expected = GetRandomBuffer<ParamType::SIZE>();
  alignas(64) typename ParamType::BufferT stored;
  ParamType::store(stored, expected);
  for (size_t i = 0; i < ParamType::SIZE; ++i)
    EXPECT_EQ(stored[i], expected[i]);
}

} // namespace __llvm_libc
