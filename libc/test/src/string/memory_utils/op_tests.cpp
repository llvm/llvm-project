//===-- Unittests for op_ files -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/span.h"
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"
#include "src/string/memory_utils/utils.h"
#include "utils/UnitTest/Test.h"

#include <assert.h>
#include <stdlib.h>

// User code should use macros instead of functions.
#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>
#define ASAN_POISON_MEMORY_REGION(addr, size)                                  \
  __asan_poison_memory_region((addr), (size))
#define ASAN_UNPOISON_MEMORY_REGION(addr, size)                                \
  __asan_unpoison_memory_region((addr), (size))
#else
#define ASAN_POISON_MEMORY_REGION(addr, size) ((void)(addr), (void)(size))
#define ASAN_UNPOISON_MEMORY_REGION(addr, size) ((void)(addr), (void)(size))
#endif

#if defined(LLVM_LIBC_ARCH_X86_64) || defined(LLVM_LIBC_ARCH_AARCH64)
#define LLVM_LIBC_HAS_UINT64
#endif

namespace __llvm_libc {

static char GetRandomChar() {
  static constexpr const uint64_t a = 1103515245;
  static constexpr const uint64_t c = 12345;
  static constexpr const uint64_t m = 1ULL << 31;
  static uint64_t seed = 123456789;
  seed = (a * seed + c) % m;
  return seed;
}

// Randomize the content of the buffer.
static void Randomize(cpp::span<char> buffer) {
  for (auto &current : buffer)
    current = GetRandomChar();
}

// Copy one span to another.
static void Copy(cpp::span<char> dst, const cpp::span<char> src) {
  assert(dst.size() == src.size());
  for (size_t i = 0; i < dst.size(); ++i)
    dst[i] = src[i];
}

cpp::byte *as_byte(cpp::span<char> span) {
  return reinterpret_cast<cpp::byte *>(span.data());
}

// Simple structure to allocate a buffer of a particular size.
struct PoisonedBuffer {
  PoisonedBuffer(size_t size) : ptr((char *)malloc(size)) {
    assert(ptr);
    ASAN_POISON_MEMORY_REGION(ptr, size);
  }
  ~PoisonedBuffer() { free(ptr); }

protected:
  char *ptr = nullptr;
};

// Simple structure to allocate a buffer (aligned or not) of a particular size.
// It is backed by a wider buffer that is marked poisoned when ASAN is present.
// The requested region is unpoisoned, this allows catching out of bounds
// accesses.
enum class Aligned : bool { NO = false, YES = true };
struct Buffer : private PoisonedBuffer {
  static constexpr size_t kAlign = 64;
  static constexpr size_t kLeeway = 2 * kAlign;
  Buffer(size_t size, Aligned aligned = Aligned::YES)
      : PoisonedBuffer(size + kLeeway), size(size) {
    offset_ptr = ptr;
    offset_ptr += distance_to_next_aligned<kAlign>(ptr);
    assert((uintptr_t)(offset_ptr) % kAlign == 0);
    if (aligned == Aligned::NO)
      ++offset_ptr;
    assert(offset_ptr > ptr);
    assert((offset_ptr + size) < (ptr + size + kLeeway));
    ASAN_UNPOISON_MEMORY_REGION(offset_ptr, size);
  }
  cpp::span<char> span() { return cpp::span<char>(offset_ptr, size); }

private:
  size_t size = 0;
  char *offset_ptr = nullptr;
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

template <auto FnImpl>
bool CheckMemcpy(cpp::span<char> dst, cpp::span<char> src, size_t size) {
  assert(dst.size() == src.size());
  assert(dst.size() == size);
  Randomize(dst);
  FnImpl(as_byte(dst), as_byte(src), size);
  for (size_t i = 0; i < size; ++i)
    if (dst[i] != src[i])
      return false;
  return true;
}

template <typename T>
static void MemcpyAdaptor(Ptr dst, CPtr src, size_t size) {
  assert(size == T::SIZE);
  return T::block(dst, src);
}

TYPED_TEST(LlvmLibcOpTest, Memcpy, MemcpyImplementations) {
  using Impl = ParamType;
  constexpr size_t kSize = Impl::SIZE;
  { // Test block operation
    Buffers SrcBuffer(kSize);
    Buffers DstBuffer(kSize);
    for (auto src : SrcBuffer.spans()) {
      Randomize(src);
      for (auto dst : DstBuffer.spans()) {
        ASSERT_TRUE(CheckMemcpy<MemcpyAdaptor<Impl>>(dst, src, kSize));
      }
    }
  }
  { // Test head tail operations from kSize to 2 * kSize.
    Buffer SrcBuffer(2 * kSize);
    Buffer DstBuffer(2 * kSize);
    Randomize(SrcBuffer.span());
    for (size_t size = kSize; size < 2 * kSize; ++size) {
      auto src = SrcBuffer.span().subspan(0, size);
      auto dst = DstBuffer.span().subspan(0, size);
      ASSERT_TRUE(CheckMemcpy<Impl::head_tail>(dst, src, size));
    }
  }
  { // Test loop operations from kSize to 3 * kSize.
    if constexpr (kSize > 1) {
      Buffer SrcBuffer(3 * kSize);
      Buffer DstBuffer(3 * kSize);
      Randomize(SrcBuffer.span());
      for (size_t size = kSize; size < 3 * kSize; ++size) {
        auto src = SrcBuffer.span().subspan(0, size);
        auto dst = DstBuffer.span().subspan(0, size);
        ASSERT_TRUE(CheckMemcpy<Impl::loop_and_tail>(dst, src, size));
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

template <auto FnImpl>
bool CheckMemset(cpp::span<char> dst, uint8_t value, size_t size) {
  Randomize(dst);
  FnImpl(as_byte(dst), value, size);
  for (char c : dst)
    if (c != (char)value)
      return false;
  return true;
}

template <typename T>
static void MemsetAdaptor(Ptr dst, uint8_t value, size_t size) {
  assert(size == T::SIZE);
  return T::block(dst, value);
}

TYPED_TEST(LlvmLibcOpTest, Memset, MemsetImplementations) {
  using Impl = ParamType;
  constexpr size_t kSize = Impl::SIZE;
  { // Test block operation
    Buffers DstBuffer(kSize);
    for (uint8_t value : cpp::array<uint8_t, 3>{0, 1, 255}) {
      for (auto dst : DstBuffer.spans()) {
        ASSERT_TRUE(CheckMemset<MemsetAdaptor<Impl>>(dst, value, kSize));
      }
    }
  }
  { // Test head tail operations from kSize to 2 * kSize.
    Buffer DstBuffer(2 * kSize);
    for (size_t size = kSize; size < 2 * kSize; ++size) {
      const char value = size % 10;
      auto dst = DstBuffer.span().subspan(0, size);
      ASSERT_TRUE(CheckMemset<Impl::head_tail>(dst, value, size));
    }
  }
  { // Test loop operations from kSize to 3 * kSize.
    if constexpr (kSize > 1) {
      Buffer DstBuffer(3 * kSize);
      for (size_t size = kSize; size < 3 * kSize; ++size) {
        const char value = size % 10;
        auto dst = DstBuffer.span().subspan(0, size);
        ASSERT_TRUE((CheckMemset<Impl::loop_and_tail>(dst, value, size)));
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

template <auto FnImpl>
bool CheckBcmp(cpp::span<char> span1, cpp::span<char> span2, size_t size) {
  assert(span1.size() == span2.size());
  Copy(span2, span1);
  // Compare equal
  if (int cmp = (int)FnImpl(as_byte(span1), as_byte(span2), size); cmp != 0)
    return false;
  // Compare not equal if any byte differs
  for (size_t i = 0; i < size; ++i) {
    ++span2[i];
    if (int cmp = (int)FnImpl(as_byte(span1), as_byte(span2), size); cmp == 0)
      return false;
    if (int cmp = (int)FnImpl(as_byte(span2), as_byte(span1), size); cmp == 0)
      return false;
    --span2[i];
  }
  return true;
}

template <typename T>
static BcmpReturnType BcmpAdaptor(CPtr p1, CPtr p2, size_t size) {
  assert(size == T::SIZE);
  return T::block(p1, p2);
}

TYPED_TEST(LlvmLibcOpTest, Bcmp, BcmpImplementations) {
  using Impl = ParamType;
  constexpr size_t kSize = Impl::SIZE;
  { // Test block operation
    Buffers Buffer1(kSize);
    Buffers Buffer2(kSize);
    for (auto span1 : Buffer1.spans()) {
      Randomize(span1);
      for (auto span2 : Buffer2.spans())
        ASSERT_TRUE((CheckBcmp<BcmpAdaptor<Impl>>(span1, span2, kSize)));
    }
  }
  { // Test head tail operations from kSize to 2 * kSize.
    Buffer Buffer1(2 * kSize);
    Buffer Buffer2(2 * kSize);
    Randomize(Buffer1.span());
    for (size_t size = kSize; size < 2 * kSize; ++size) {
      auto span1 = Buffer1.span().subspan(0, size);
      auto span2 = Buffer2.span().subspan(0, size);
      ASSERT_TRUE((CheckBcmp<Impl::head_tail>(span1, span2, size)));
    }
  }
  { // Test loop operations from kSize to 3 * kSize.
    if constexpr (kSize > 1) {
      Buffer Buffer1(3 * kSize);
      Buffer Buffer2(3 * kSize);
      Randomize(Buffer1.span());
      for (size_t size = kSize; size < 3 * kSize; ++size) {
        auto span1 = Buffer1.span().subspan(0, size);
        auto span2 = Buffer2.span().subspan(0, size);
        ASSERT_TRUE((CheckBcmp<Impl::loop_and_tail>(span1, span2, size)));
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

template <auto FnImpl>
bool CheckMemcmp(cpp::span<char> span1, cpp::span<char> span2, size_t size) {
  assert(span1.size() == span2.size());
  Copy(span2, span1);
  // Compare equal
  if (int cmp = (int)FnImpl(as_byte(span1), as_byte(span2), size); cmp != 0)
    return false;
  // Compare not equal if any byte differs
  for (size_t i = 0; i < size; ++i) {
    ++span2[i];
    int ground_truth = __builtin_memcmp(span1.data(), span2.data(), size);
    if (ground_truth > 0) {
      if (int cmp = (int)FnImpl(as_byte(span1), as_byte(span2), size); cmp <= 0)
        return false;
      if (int cmp = (int)FnImpl(as_byte(span2), as_byte(span1), size); cmp >= 0)
        return false;
    } else {
      if (int cmp = (int)FnImpl(as_byte(span1), as_byte(span2), size); cmp >= 0)
        return false;
      if (int cmp = (int)FnImpl(as_byte(span2), as_byte(span1), size); cmp <= 0)
        return false;
    }
    --span2[i];
  }
  return true;
}

template <typename T>
static MemcmpReturnType MemcmpAdaptor(CPtr p1, CPtr p2, size_t size) {
  assert(size == T::SIZE);
  return T::block(p1, p2);
}

TYPED_TEST(LlvmLibcOpTest, Memcmp, MemcmpImplementations) {
  using Impl = ParamType;
  constexpr size_t kSize = Impl::SIZE;
  { // Test block operation
    Buffers Buffer1(kSize);
    Buffers Buffer2(kSize);
    for (auto span1 : Buffer1.spans()) {
      Randomize(span1);
      for (auto span2 : Buffer2.spans())
        ASSERT_TRUE((CheckMemcmp<MemcmpAdaptor<Impl>>(span1, span2, kSize)));
    }
  }
  { // Test head tail operations from kSize to 2 * kSize.
    Buffer Buffer1(2 * kSize);
    Buffer Buffer2(2 * kSize);
    Randomize(Buffer1.span());
    for (size_t size = kSize; size < 2 * kSize; ++size) {
      auto span1 = Buffer1.span().subspan(0, size);
      auto span2 = Buffer2.span().subspan(0, size);
      ASSERT_TRUE((CheckMemcmp<Impl::head_tail>(span1, span2, size)));
    }
  }
  { // Test loop operations from kSize to 3 * kSize.
    if constexpr (kSize > 1) {
      Buffer Buffer1(3 * kSize);
      Buffer Buffer2(3 * kSize);
      Randomize(Buffer1.span());
      for (size_t size = kSize; size < 3 * kSize; ++size) {
        auto span1 = Buffer1.span().subspan(0, size);
        auto span2 = Buffer2.span().subspan(0, size);
        ASSERT_TRUE((CheckMemcmp<Impl::loop_and_tail>(span1, span2, size)));
      }
    }
  }
}

} // namespace __llvm_libc
