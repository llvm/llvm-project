//===-- Unittests for op_ files -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_check_utils.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/os.h"
#include "src/__support/macros/properties/types.h" // LIBC_TYPES_HAS_INT64
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_riscv.h"
#include "src/string/memory_utils/op_x86.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

template <typename T> struct has_head_tail {
  template <typename C> static char sfinae(decltype(&C::head_tail));
  template <typename C> static uint16_t sfinae(...);
  static constexpr bool value = sizeof(sfinae<T>(0)) == sizeof(char);
};

template <typename T> struct has_loop_and_tail {
  template <typename C> static char sfinae(decltype(&C::loop_and_tail));
  template <typename C> static uint16_t sfinae(...);
  static constexpr bool value = sizeof(sfinae<T>(0)) == sizeof(char);
};

// Allocates two Buffer and extracts two spans out of them, one
// aligned and one misaligned. Tests are run on both spans.
struct Buffers {
  Buffers(size_t size)
      : aligned_buffer(size, Aligned::YES),
        misaligned_buffer(size, Aligned::NO) {}

  // Returns two spans of 'size' bytes. The first is aligned on
  // Buffer::kAlign and the second one is unaligned.
  cpp::array<cpp::span<char>, 2> spans() {
    return {aligned_buffer.span(), misaligned_buffer.span()};
  }

  Buffer aligned_buffer;
  Buffer misaligned_buffer;
};

using MemcpyImplementations = testing::TypeList<
#ifdef LLVM_LIBC_HAS_BUILTIN_MEMCPY_INLINE
    builtin::Memcpy<1>,  //
    builtin::Memcpy<2>,  //
    builtin::Memcpy<3>,  //
    builtin::Memcpy<4>,  //
    builtin::Memcpy<8>,  //
    builtin::Memcpy<16>, //
    builtin::Memcpy<32>, //
    builtin::Memcpy<64>
#endif // LLVM_LIBC_HAS_BUILTIN_MEMCPY_INLINE
    >;

// Convenient helper to turn a span into cpp::byte *.
static inline cpp::byte *as_byte(cpp::span<char> span) {
  return reinterpret_cast<cpp::byte *>(span.data());
}

// Adapt CheckMemcpy signature to op implementation signatures.
template <auto FnImpl>
void CopyAdaptor(cpp::span<char> dst, cpp::span<char> src, size_t size) {
  FnImpl(as_byte(dst), as_byte(src), size);
}
template <size_t Size, auto FnImpl>
void CopyBlockAdaptor(cpp::span<char> dst, cpp::span<char> src,
                      [[maybe_unused]] size_t size) {
  FnImpl(as_byte(dst), as_byte(src));
}

TYPED_TEST(LlvmLibcOpTest, Memcpy, MemcpyImplementations) {
  using Impl = ParamType;
  constexpr size_t kSize = Impl::SIZE;
  { // Test block operation
    static constexpr auto BlockImpl = CopyBlockAdaptor<kSize, Impl::block>;
    Buffers SrcBuffer(kSize);
    Buffers DstBuffer(kSize);
    for (auto src : SrcBuffer.spans()) {
      Randomize(src);
      for (auto dst : DstBuffer.spans()) {
        ASSERT_TRUE(CheckMemcpy<BlockImpl>(dst, src, kSize));
      }
    }
  }
  { // Test head tail operations from kSize to 2 * kSize.
    static constexpr auto HeadTailImpl = CopyAdaptor<Impl::head_tail>;
    Buffer SrcBuffer(2 * kSize);
    Buffer DstBuffer(2 * kSize);
    Randomize(SrcBuffer.span());
    for (size_t size = kSize; size < 2 * kSize; ++size) {
      auto src = SrcBuffer.span().subspan(0, size);
      auto dst = DstBuffer.span().subspan(0, size);
      ASSERT_TRUE(CheckMemcpy<HeadTailImpl>(dst, src, size));
    }
  }
  { // Test loop operations from kSize to 3 * kSize.
    if constexpr (kSize > 1) {
      static constexpr auto LoopImpl = CopyAdaptor<Impl::loop_and_tail>;
      Buffer SrcBuffer(3 * kSize);
      Buffer DstBuffer(3 * kSize);
      Randomize(SrcBuffer.span());
      for (size_t size = kSize; size < 3 * kSize; ++size) {
        auto src = SrcBuffer.span().subspan(0, size);
        auto dst = DstBuffer.span().subspan(0, size);
        ASSERT_TRUE(CheckMemcpy<LoopImpl>(dst, src, size));
      }
    }
  }
}

using MemsetImplementations = testing::TypeList<
#ifdef LLVM_LIBC_HAS_BUILTIN_MEMSET_INLINE
    builtin::Memset<1>,  //
    builtin::Memset<2>,  //
    builtin::Memset<3>,  //
    builtin::Memset<4>,  //
    builtin::Memset<8>,  //
    builtin::Memset<16>, //
    builtin::Memset<32>, //
    builtin::Memset<64>,
#endif
#ifdef LIBC_TYPES_HAS_INT64
    generic::Memset<uint64_t>, generic::Memset<cpp::array<uint64_t, 2>>,
#endif // LIBC_TYPES_HAS_INT64
#ifdef __AVX512F__
    generic::Memset<generic_v512>, generic::Memset<cpp::array<generic_v512, 2>>,
#endif
#ifdef __AVX__
    generic::Memset<generic_v256>, generic::Memset<cpp::array<generic_v256, 2>>,
#endif
#ifdef __SSE2__
    generic::Memset<generic_v128>, generic::Memset<cpp::array<generic_v128, 2>>,
#endif
    generic::Memset<uint32_t>, generic::Memset<cpp::array<uint32_t, 2>>, //
    generic::Memset<uint16_t>, generic::Memset<cpp::array<uint16_t, 2>>, //
    generic::Memset<uint8_t>, generic::Memset<cpp::array<uint8_t, 2>>,   //
    generic::MemsetSequence<uint8_t, uint8_t>,                           //
    generic::MemsetSequence<uint16_t, uint8_t>,                          //
    generic::MemsetSequence<uint32_t, uint16_t, uint8_t>                 //
    >;

// Adapt CheckMemset signature to op implementation signatures.
template <auto FnImpl>
void SetAdaptor(cpp::span<char> dst, uint8_t value, size_t size) {
  FnImpl(as_byte(dst), value, size);
}
template <size_t Size, auto FnImpl>
void SetBlockAdaptor(cpp::span<char> dst, uint8_t value,
                     [[maybe_unused]] size_t size) {
  FnImpl(as_byte(dst), value);
}

TYPED_TEST(LlvmLibcOpTest, Memset, MemsetImplementations) {
  using Impl = ParamType;
  constexpr size_t kSize = Impl::SIZE;
  { // Test block operation
    static constexpr auto BlockImpl = SetBlockAdaptor<kSize, Impl::block>;
    Buffers DstBuffer(kSize);
    for (uint8_t value : cpp::array<uint8_t, 3>{0, 1, 255}) {
      for (auto dst : DstBuffer.spans()) {
        ASSERT_TRUE(CheckMemset<BlockImpl>(dst, value, kSize));
      }
    }
  }
  if constexpr (has_head_tail<Impl>::value) {
    // Test head tail operations from kSize to 2 * kSize.
    static constexpr auto HeadTailImpl = SetAdaptor<Impl::head_tail>;
    Buffer DstBuffer(2 * kSize);
    for (size_t size = kSize; size < 2 * kSize; ++size) {
      const uint8_t value = size % 10;
      auto dst = DstBuffer.span().subspan(0, size);
      ASSERT_TRUE(CheckMemset<HeadTailImpl>(dst, value, size));
    }
  }
  if constexpr (has_loop_and_tail<Impl>::value) {
    // Test loop operations from kSize to 3 * kSize.
    if constexpr (kSize > 1) {
      static constexpr auto LoopImpl = SetAdaptor<Impl::loop_and_tail>;
      Buffer DstBuffer(3 * kSize);
      for (size_t size = kSize; size < 3 * kSize; ++size) {
        const uint8_t value = size % 10;
        auto dst = DstBuffer.span().subspan(0, size);
        ASSERT_TRUE((CheckMemset<LoopImpl>(dst, value, size)));
      }
    }
  }
}

#ifdef LIBC_TARGET_ARCH_IS_X86_64
// Prevent GCC warning due to ignored __aligned__ attributes when passing x86
// SIMD types as template arguments.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif // LIBC_TARGET_ARCH_IS_X86_64

using BcmpImplementations = testing::TypeList<
#ifdef LIBC_TARGET_ARCH_IS_X86_64
#ifdef __SSE4_1__
    generic::Bcmp<__m128i>,
#endif // __SSE4_1__
#ifdef __AVX2__
    generic::Bcmp<__m256i>,
#endif // __AVX2__
#ifdef __AVX512BW__
    generic::Bcmp<__m512i>,
#endif // __AVX512BW__

#endif // LIBC_TARGET_ARCH_IS_X86_64
#ifdef LIBC_TARGET_ARCH_IS_AARCH64
    aarch64::Bcmp<16>, //
    aarch64::Bcmp<32>,
#endif
#ifndef LIBC_TARGET_ARCH_IS_ARM // Removing non uint8_t types for ARM
    generic::Bcmp<uint16_t>,
    generic::Bcmp<uint32_t>, //
#ifdef LIBC_TYPES_HAS_INT64
    generic::Bcmp<uint64_t>,
#endif // LIBC_TYPES_HAS_INT64
    generic::BcmpSequence<uint16_t, uint8_t>,
    generic::BcmpSequence<uint32_t, uint8_t>,  //
    generic::BcmpSequence<uint32_t, uint16_t>, //
    generic::BcmpSequence<uint32_t, uint16_t, uint8_t>,
#endif // LIBC_TARGET_ARCH_IS_ARM
    generic::BcmpSequence<uint8_t, uint8_t>,
    generic::BcmpSequence<uint8_t, uint8_t, uint8_t>, //
    generic::Bcmp<uint8_t>>;

#ifdef LIBC_TARGET_ARCH_IS_X86_64
#pragma GCC diagnostic pop
#endif // LIBC_TARGET_ARCH_IS_X86_64

// Adapt CheckBcmp signature to op implementation signatures.
template <auto FnImpl>
int CmpAdaptor(cpp::span<char> p1, cpp::span<char> p2, size_t size) {
  return (int)FnImpl(as_byte(p1), as_byte(p2), size);
}
template <size_t Size, auto FnImpl>
int CmpBlockAdaptor(cpp::span<char> p1, cpp::span<char> p2,
                    [[maybe_unused]] size_t size) {
  return (int)FnImpl(as_byte(p1), as_byte(p2));
}

TYPED_TEST(LlvmLibcOpTest, Bcmp, BcmpImplementations) {
  using Impl = ParamType;
  constexpr size_t kSize = Impl::SIZE;
  { // Test block operation
    static constexpr auto BlockImpl = CmpBlockAdaptor<kSize, Impl::block>;
    Buffers Buffer1(kSize);
    Buffers Buffer2(kSize);
    for (auto span1 : Buffer1.spans()) {
      Randomize(span1);
      for (auto span2 : Buffer2.spans())
        ASSERT_TRUE((CheckBcmp<BlockImpl>(span1, span2, kSize)));
    }
  }
  if constexpr (has_head_tail<Impl>::value) {
    // Test head tail operations from kSize to 2 * kSize.
    static constexpr auto HeadTailImpl = CmpAdaptor<Impl::head_tail>;
    Buffer Buffer1(2 * kSize);
    Buffer Buffer2(2 * kSize);
    Randomize(Buffer1.span());
    for (size_t size = kSize; size < 2 * kSize; ++size) {
      auto span1 = Buffer1.span().subspan(0, size);
      auto span2 = Buffer2.span().subspan(0, size);
      ASSERT_TRUE((CheckBcmp<HeadTailImpl>(span1, span2, size)));
    }
  }
  if constexpr (has_loop_and_tail<Impl>::value) {
    // Test loop operations from kSize to 3 * kSize.
    if constexpr (kSize > 1) {
      static constexpr auto LoopImpl = CmpAdaptor<Impl::loop_and_tail>;
      Buffer Buffer1(3 * kSize);
      Buffer Buffer2(3 * kSize);
      Randomize(Buffer1.span());
      for (size_t size = kSize; size < 3 * kSize; ++size) {
        auto span1 = Buffer1.span().subspan(0, size);
        auto span2 = Buffer2.span().subspan(0, size);
        ASSERT_TRUE((CheckBcmp<LoopImpl>(span1, span2, size)));
      }
    }
  }
}

#ifdef LIBC_TARGET_ARCH_IS_X86_64
// Prevent GCC warning due to ignored __aligned__ attributes when passing x86
// SIMD types as template arguments.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif // LIBC_TARGET_ARCH_IS_X86_64

using MemcmpImplementations = testing::TypeList<
#if defined(LIBC_TARGET_ARCH_IS_X86_64) && !defined(LIBC_TARGET_OS_IS_WINDOWS)
#ifdef __SSE2__
    generic::Memcmp<__m128i>, //
#endif
#ifdef __AVX2__
    generic::Memcmp<__m256i>, //
#endif
#ifdef __AVX512BW__
    generic::Memcmp<__m512i>, //
#endif
#endif // LIBC_TARGET_ARCH_IS_X86_64
#ifdef LIBC_TARGET_ARCH_IS_AARCH64
    generic::Memcmp<uint8x16_t>, //
    generic::Memcmp<uint8x16x2_t>,
#endif
#ifndef LIBC_TARGET_ARCH_IS_ARM // Removing non uint8_t types for ARM
    generic::Memcmp<uint16_t>,
    generic::Memcmp<uint32_t>, //
#ifdef LIBC_TYPES_HAS_INT64
    generic::Memcmp<uint64_t>,
#endif // LIBC_TYPES_HAS_INT64
    generic::MemcmpSequence<uint16_t, uint8_t>,
    generic::MemcmpSequence<uint32_t, uint16_t, uint8_t>, //
#endif // LIBC_TARGET_ARCH_IS_ARM
    generic::MemcmpSequence<uint8_t, uint8_t>,
    generic::MemcmpSequence<uint8_t, uint8_t, uint8_t>,
    generic::Memcmp<uint8_t>>;

#ifdef LIBC_TARGET_ARCH_IS_X86_64
#pragma GCC diagnostic pop
#endif // LIBC_TARGET_ARCH_IS_X86_64

TYPED_TEST(LlvmLibcOpTest, Memcmp, MemcmpImplementations) {
  using Impl = ParamType;
  constexpr size_t kSize = Impl::SIZE;
  { // Test block operation
    static constexpr auto BlockImpl = CmpBlockAdaptor<kSize, Impl::block>;
    Buffers Buffer1(kSize);
    Buffers Buffer2(kSize);
    for (auto span1 : Buffer1.spans()) {
      Randomize(span1);
      for (auto span2 : Buffer2.spans())
        ASSERT_TRUE((CheckMemcmp<BlockImpl>(span1, span2, kSize)));
    }
  }
  if constexpr (has_head_tail<Impl>::value) {
    // Test head tail operations from kSize to 2 * kSize.
    static constexpr auto HeadTailImpl = CmpAdaptor<Impl::head_tail>;
    Buffer Buffer1(2 * kSize);
    Buffer Buffer2(2 * kSize);
    Randomize(Buffer1.span());
    for (size_t size = kSize; size < 2 * kSize; ++size) {
      auto span1 = Buffer1.span().subspan(0, size);
      auto span2 = Buffer2.span().subspan(0, size);
      ASSERT_TRUE((CheckMemcmp<HeadTailImpl>(span1, span2, size)));
    }
  }
  if constexpr (has_loop_and_tail<Impl>::value) {
    // Test loop operations from kSize to 3 * kSize.
    if constexpr (kSize > 1) {
      static constexpr auto LoopImpl = CmpAdaptor<Impl::loop_and_tail>;
      Buffer Buffer1(3 * kSize);
      Buffer Buffer2(3 * kSize);
      Randomize(Buffer1.span());
      for (size_t size = kSize; size < 3 * kSize; ++size) {
        auto span1 = Buffer1.span().subspan(0, size);
        auto span2 = Buffer2.span().subspan(0, size);
        ASSERT_TRUE((CheckMemcmp<LoopImpl>(span1, span2, size)));
      }
    }
  }
}

} // namespace LIBC_NAMESPACE_DECL
