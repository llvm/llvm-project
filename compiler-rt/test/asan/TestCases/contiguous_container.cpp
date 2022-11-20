// RUN: %clangxx_asan -fexceptions -O %s -o %t && %env_asan_opts=detect_stack_use_after_return=0 %run %t
//
// Test __sanitizer_annotate_contiguous_container.

#include <algorithm>
#include <assert.h>
#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static constexpr size_t kGranularity = 8;

template <class T> static constexpr T RoundDown(T x) {
  return reinterpret_cast<T>(reinterpret_cast<uintptr_t>(x) &
                             ~(kGranularity - 1));
}

void TestContainer(size_t capacity, size_t off_begin, bool poison_buffer) {
  size_t buffer_size = capacity + off_begin + kGranularity * 2;
  char *buffer = new char[buffer_size];
  if (poison_buffer)
    __asan_poison_memory_region(buffer, buffer_size);
  char *st_beg = buffer + off_begin;
  char *st_end = st_beg + capacity;
  char *end = poison_buffer ? st_beg : st_end;
  char *old_end;

  for (int i = 0; i < 1000; i++) {
    size_t size = rand() % (capacity + 1);
    assert(size <= capacity);
    old_end = end;
    end = st_beg + size;
    __sanitizer_annotate_contiguous_container(st_beg, st_end, old_end, end);

    char *cur = buffer;
    for (; cur < buffer + RoundDown(off_begin); ++cur)
      assert(__asan_address_is_poisoned(cur) == poison_buffer);
    // The prefix of the first incomplete granule can switch from poisoned to
    // unpoisoned but not otherwise.
    for (; cur < buffer + off_begin; ++cur)
      assert(poison_buffer || !__asan_address_is_poisoned(cur));
    for (; cur < end; ++cur)
      assert(!__asan_address_is_poisoned(cur));
    for (; cur < RoundDown(st_end); ++cur)
      assert(__asan_address_is_poisoned(cur));
    // The suffix of the last incomplete granule must be poisoned the same as
    // bytes after the end.
    for (; cur != st_end + kGranularity; ++cur)
      assert(__asan_address_is_poisoned(cur) == poison_buffer);
  }

  for (int i = 0; i <= capacity; i++) {
    old_end = end;
    end = st_beg + i;
    __sanitizer_annotate_contiguous_container(st_beg, st_end, old_end, end);

    for (char *cur = std::max(st_beg, st_end - 2 * kGranularity);
         cur <= std::min(st_end, end + 2 * kGranularity); ++cur) {
      if (cur == end ||
          // Any end in the last unaligned granule is OK, if bytes after the
          // storage are not poisoned.
          (!poison_buffer && RoundDown(st_end) <= std::min(cur, end))) {
        assert(__sanitizer_verify_contiguous_container(st_beg, cur, st_end));
        assert(NULL == __sanitizer_contiguous_container_find_bad_address(
                           st_beg, cur, st_end));
      } else if (cur < end) {
        assert(!__sanitizer_verify_contiguous_container(st_beg, cur, st_end));
        assert(cur == __sanitizer_contiguous_container_find_bad_address(
                          st_beg, cur, st_end));
      } else {
        assert(!__sanitizer_verify_contiguous_container(st_beg, cur, st_end));
        assert(end == __sanitizer_contiguous_container_find_bad_address(
                          st_beg, cur, st_end));
      }
    }
  }

  __asan_unpoison_memory_region(buffer, buffer_size);
  delete[] buffer;
}

__attribute__((noinline)) void Throw() { throw 1; }

__attribute__((noinline)) void ThrowAndCatch() {
  try {
    Throw();
  } catch (...) {
  }
}

void TestThrow() {
  char x[32];
  __sanitizer_annotate_contiguous_container(x, x + 32, x + 32, x + 14);
  assert(!__asan_address_is_poisoned(x + 13));
  assert(__asan_address_is_poisoned(x + 14));
  ThrowAndCatch();
  assert(!__asan_address_is_poisoned(x + 13));
  assert(!__asan_address_is_poisoned(x + 14));
  __sanitizer_annotate_contiguous_container(x, x + 32, x + 14, x + 32);
  assert(!__asan_address_is_poisoned(x + 13));
  assert(!__asan_address_is_poisoned(x + 14));
}

int main(int argc, char **argv) {
  int n = argc == 1 ? 64 : atoi(argv[1]);
  for (int i = 0; i <= n; i++)
    for (int j = 0; j < kGranularity * 2; j++)
      for (int poison = 0; poison < 2; ++poison)
        TestContainer(i, j, poison);
  TestThrow();
}
