//===-- Unittests for op_ files -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_check_utils.h"
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"
#include "utils/UnitTest/Test.h"

#include <assert.h>

#if defined(LLVM_LIBC_ARCH_X86_64) || defined(LLVM_LIBC_ARCH_AARCH64)
#define LLVM_LIBC_HAS_UINT64
#endif

namespace __llvm_libc {

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
void CopyBlockAdaptor(cpp::span<char> dst, cpp::span<char> src, size_t size) {
  assert(size == Size);
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
#ifdef LLVM_LIBC_HAS_UINT64
    generic::Memset<8, 8>,  //
    generic::Memset<16, 8>, //
    generic::Memset<32, 8>, //
    generic::Memset<64, 8>, //
#endif
#ifdef __AVX512F__
    generic::Memset<64, 64>, // prevents warning about avx512f
#endif
    generic::Memset<1, 1>,   //
    generic::Memset<2, 1>,   //
    generic::Memset<2, 2>,   //
    generic::Memset<4, 2>,   //
    generic::Memset<4, 4>,   //
    generic::Memset<16, 16>, //
    generic::Memset<32, 16>, //
    generic::Memset<64, 16>, //
    generic::Memset<32, 32>, //
    generic::Memset<64, 32>  //
    >;

// Adapt CheckMemset signature to op implementation signatures.
template <auto FnImpl>
void SetAdaptor(cpp::span<char> dst, uint8_t value, size_t size) {
  FnImpl(as_byte(dst), value, size);
}
template <size_t Size, auto FnImpl>
void SetBlockAdaptor(cpp::span<char> dst, uint8_t value, size_t size) {
  assert(size == Size);
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
  { // Test head tail operations from kSize to 2 * kSize.
    static constexpr auto HeadTailImpl = SetAdaptor<Impl::head_tail>;
    Buffer DstBuffer(2 * kSize);
    for (size_t size = kSize; size < 2 * kSize; ++size) {
      const char value = size % 10;
      auto dst = DstBuffer.span().subspan(0, size);
      ASSERT_TRUE(CheckMemset<HeadTailImpl>(dst, value, size));
    }
  }
  { // Test loop operations from kSize to 3 * kSize.
    if constexpr (kSize > 1) {
      static constexpr auto LoopImpl = SetAdaptor<Impl::loop_and_tail>;
      Buffer DstBuffer(3 * kSize);
      for (size_t size = kSize; size < 3 * kSize; ++size) {
        const char value = size % 10;
        auto dst = DstBuffer.span().subspan(0, size);
        ASSERT_TRUE((CheckMemset<LoopImpl>(dst, value, size)));
      }
    }
  }
}

using BcmpImplementations = testing::TypeList<
#ifdef __SSE2__
    x86::sse2::Bcmp<16>,  //
    x86::sse2::Bcmp<32>,  //
    x86::sse2::Bcmp<64>,  //
    x86::sse2::Bcmp<128>, //
#endif
#ifdef __AVX2__
    x86::avx2::Bcmp<32>,  //
    x86::avx2::Bcmp<64>,  //
    x86::avx2::Bcmp<128>, //
#endif
#ifdef __AVX512BW__
    x86::avx512bw::Bcmp<64>,  //
    x86::avx512bw::Bcmp<128>, //
#endif
#ifdef LLVM_LIBC_ARCH_AARCH64
    aarch64::Bcmp<16>, //
    aarch64::Bcmp<32>, //
#endif
#ifdef LLVM_LIBC_HAS_UINT64
    generic::Bcmp<8>, //
#endif
    generic::Bcmp<1>,  //
    generic::Bcmp<2>,  //
    generic::Bcmp<4>,  //
    generic::Bcmp<16>, //
    generic::Bcmp<32>, //
    generic::Bcmp<64>  //
    >;

// Adapt CheckBcmp signature to op implementation signatures.
template <auto FnImpl>
int CmpAdaptor(cpp::span<char> p1, cpp::span<char> p2, size_t size) {
  return (int)FnImpl(as_byte(p1), as_byte(p2), size);
}
template <size_t Size, auto FnImpl>
int CmpBlockAdaptor(cpp::span<char> p1, cpp::span<char> p2, size_t size) {
  assert(size == Size);
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
  { // Test head tail operations from kSize to 2 * kSize.
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
  { // Test loop operations from kSize to 3 * kSize.
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

using MemcmpImplementations = testing::TypeList<
#ifdef __SSE2__
    x86::sse2::Memcmp<16>,  //
    x86::sse2::Memcmp<32>,  //
    x86::sse2::Memcmp<64>,  //
    x86::sse2::Memcmp<128>, //
#endif
#ifdef __AVX2__
    x86::avx2::Memcmp<32>,  //
    x86::avx2::Memcmp<64>,  //
    x86::avx2::Memcmp<128>, //
#endif
#ifdef __AVX512BW__
    x86::avx512bw::Memcmp<64>,  //
    x86::avx512bw::Memcmp<128>, //
#endif
#ifdef LLVM_LIBC_HAS_UINT64
    generic::Memcmp<8>, //
#endif
    generic::Memcmp<1>,  //
    generic::Memcmp<2>,  //
    generic::Memcmp<3>,  //
    generic::Memcmp<4>,  //
    generic::Memcmp<16>, //
    generic::Memcmp<32>, //
    generic::Memcmp<64>  //
    >;

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
  { // Test head tail operations from kSize to 2 * kSize.
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
  { // Test loop operations from kSize to 3 * kSize.
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

} // namespace __llvm_libc
