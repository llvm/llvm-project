//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test libc++'s implementation of align_val_t, and the relevant new/delete
// overloads in all dialects when -faligned-allocation is present.

// Some dylibs do not contain the aligned allocation functions, so trying to force
// using those with -faligned-allocation results in a link error.
// XFAIL: availability-aligned_allocation-missing

// Libc++ when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

// XFAIL: sanitizer-new-delete && !hwasan

// It fails with clang-14 or clang-16, but passes with clang-15.
// UNSUPPORTED: ubsan

// GCC doesn't support the aligned-allocation flags.
// XFAIL: gcc

// RUN: %{build} -faligned-allocation -fsized-deallocation
// RUN: %{run}
// RUN: %{build} -faligned-allocation -fno-sized-deallocation -DNO_SIZE
// RUN: %{run}
// RUN: %{build} -fno-aligned-allocation -fsized-deallocation -DNO_ALIGN
// RUN: %{run}
// RUN: %{build} -fno-aligned-allocation -fno-sized-deallocation -DNO_ALIGN -DNO_SIZE
// RUN: %{run}

#include <new>
#include <typeinfo>
#include <string>
#include <cassert>

#include "test_macros.h"

TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wprivate-header")
#include <__memory/aligned_alloc.h>
TEST_DIAGNOSTIC_POP

struct alloc_stats {
  alloc_stats() { reset(); }

  int aligned_sized_called;
  int aligned_called;
  int sized_called;
  int plain_called;
  int last_size;
  int last_align;

  void reset() {
    aligned_sized_called = aligned_called = sized_called = plain_called = 0;
    last_align = last_size = -1;
  }

  bool expect_plain() const {
    assert(aligned_sized_called == 0);
    assert(aligned_called == 0);
    assert(sized_called == 0);
    assert(last_size == -1);
    assert(last_align == -1);
    return plain_called == 1;
  }

  bool expect_size(int n) const {
    assert(plain_called == 0);
    assert(aligned_sized_called == 0);
    assert(aligned_called == 0);
    assert(last_size == n);
    assert(last_align == -1);
    return sized_called == 1;
  }

  bool expect_align(int a) const {
    assert(plain_called == 0);
    assert(aligned_sized_called == 0);
    assert(sized_called == 0);
    assert(last_size == -1);
    assert(last_align == a);
    return aligned_called == 1;
  }

  bool expect_size_align(int n, int a) const {
    assert(plain_called == 0);
    assert(sized_called == 0);
    assert(aligned_called == 0);
    assert(last_size == n);
    assert(last_align == a);
    return aligned_sized_called == 1;
  }
};
alloc_stats stats;

void operator delete(void* p)TEST_NOEXCEPT {
  ::free(p);
  stats.plain_called++;
  stats.last_size = stats.last_align = -1;
}

#ifndef NO_SIZE
void operator delete(void* p, std::size_t n)TEST_NOEXCEPT {
  ::free(p);
  stats.sized_called++;
  stats.last_size = n;
  stats.last_align = -1;
}
#endif

#ifndef NO_ALIGN
void operator delete(void* p, std::align_val_t a)TEST_NOEXCEPT {
  std::__libcpp_aligned_free(p);
  stats.aligned_called++;
  stats.last_align = static_cast<int>(a);
  stats.last_size = -1;
}

void operator delete(void* p, std::size_t n, std::align_val_t a)TEST_NOEXCEPT {
  std::__libcpp_aligned_free(p);
  stats.aligned_sized_called++;
  stats.last_align = static_cast<int>(a);
  stats.last_size = n;
}
#endif

void test_libcpp_dealloc() {
  void* p = nullptr;
#ifdef __STDCPP_DEFAULT_NEW_ALIGNMENT__
  std::size_t over_align_val = __STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2;
#else
  std::size_t over_align_val = TEST_ALIGNOF(std::max_align_t) * 2;
#endif
  std::size_t under_align_val = TEST_ALIGNOF(int);
  std::size_t with_size_val = 2;

  {
    std::__libcpp_deallocate_unsized(p, under_align_val);
    assert(stats.expect_plain());
  }
  stats.reset();

#if defined(NO_SIZE) && defined(NO_ALIGN)
  {
    std::__libcpp_deallocate(p, with_size_val, over_align_val);
    assert(stats.expect_plain());
  }
  stats.reset();
#elif defined(NO_SIZE)
  {
    std::__libcpp_deallocate(p, with_size_val, over_align_val);
    assert(stats.expect_align(over_align_val));
  }
  stats.reset();
#elif defined(NO_ALIGN)
  {
    std::__libcpp_deallocate(p, with_size_val, over_align_val);
    assert(stats.expect_size(with_size_val));
  }
  stats.reset();
#else
  {
    std::__libcpp_deallocate(p, with_size_val, over_align_val);
    assert(stats.expect_size_align(with_size_val, over_align_val));
  }
  stats.reset();
  {
    std::__libcpp_deallocate_unsized(p, over_align_val);
    assert(stats.expect_align(over_align_val));
  }
  stats.reset();
  {
    std::__libcpp_deallocate(p, with_size_val, under_align_val);
    assert(stats.expect_size(with_size_val));
  }
  stats.reset();
#endif
}

struct TEST_ALIGNAS(128) AlignedType {
  AlignedType() : elem(0) {}
  TEST_ALIGNAS(128) char elem;
};

void test_allocator_and_new_match() {
  stats.reset();
#if defined(NO_SIZE) && defined(NO_ALIGN)
  {
    int* x = new int(42);
    delete x;
    assert(stats.expect_plain());
  }
  stats.reset();
  {
    AlignedType* a = new AlignedType();
    delete a;
    assert(stats.expect_plain());
  }
  stats.reset();
#elif defined(NO_SIZE)
  stats.reset();
#if TEST_STD_VER >= 11
  {
    int* x = new int(42);
    delete x;
    assert(stats.expect_plain());
  }
#endif
  stats.reset();
  {
    AlignedType* a = new AlignedType();
    delete a;
    assert(stats.expect_align(TEST_ALIGNOF(AlignedType)));
  }
  stats.reset();
#elif defined(NO_ALIGN)
  stats.reset();
  {
    int* x = new int(42);
    delete x;
    assert(stats.expect_size(sizeof(int)));
  }
  stats.reset();
  {
    AlignedType* a = new AlignedType();
    delete a;
    assert(stats.expect_size(sizeof(AlignedType)));
  }
  stats.reset();
#else
  stats.reset();
  {
    int* x = new int(42);
    delete x;
    assert(stats.expect_size(sizeof(int)));
  }
  stats.reset();
  {
    AlignedType* a = new AlignedType();
    delete a;
    assert(stats.expect_size_align(sizeof(AlignedType),
                                   TEST_ALIGNOF(AlignedType)));
  }
  stats.reset();
#endif
}

int main(int, char**) {
  test_libcpp_dealloc();
  test_allocator_and_new_match();

  return 0;
}
