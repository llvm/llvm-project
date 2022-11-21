// RUN: %clangxx_asan -fexceptions -O %s -o %t && %env_asan_opts=detect_stack_use_after_return=0 %run %t
//
// Test __sanitizer_annotate_contiguous_container.

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

void TestContainer(size_t capacity, size_t off_begin, size_t off_end,
                   bool poison_buffer) {
  char *buffer = new char[capacity + off_begin + off_end];
  char *buffer_end = buffer + capacity + off_begin + off_end;
  if (poison_buffer)
    __asan_poison_memory_region(buffer, buffer_end - buffer);
  else
    __asan_unpoison_memory_region(buffer, buffer_end - buffer);
  char *beg = buffer + off_begin;
  char *end = beg + capacity;
  char *mid = poison_buffer ? beg : beg + capacity;
  char *old_mid = 0;
  // If after the container, there is another object, last granule
  // cannot be poisoned.
  char *cannot_poison = (off_end == 0) ? end : RoundDown(end);

  for (int i = 0; i < 1000; i++) {
    size_t size = rand() % (capacity + 1);
    assert(size <= capacity);
    old_mid = mid;
    mid = beg + size;
    __sanitizer_annotate_contiguous_container(beg, end, old_mid, mid);

    // If off buffer before the container was poisoned and we had to
    // unpoison it, we won't poison it again as we don't have information,
    // if it was poisoned.
    if (!poison_buffer)
      for (size_t idx = 0; idx < off_begin; idx++)
        assert(!__asan_address_is_poisoned(buffer + idx));
    for (size_t idx = 0; idx < size; idx++)
      assert(!__asan_address_is_poisoned(beg + idx));
    for (size_t idx = size; beg + idx < cannot_poison; idx++)
      assert(__asan_address_is_poisoned(beg + idx));
    for (size_t idx = 0; idx < off_end; idx++)
      assert(__asan_address_is_poisoned(end + idx) == poison_buffer);

    assert(__sanitizer_verify_contiguous_container(beg, mid, end));
    assert(NULL ==
           __sanitizer_contiguous_container_find_bad_address(beg, mid, end));
    size_t distance = (off_end > 0) ? kGranularity + 1 : 1;
    if (mid >= beg + distance) {
      assert(
          !__sanitizer_verify_contiguous_container(beg, mid - distance, end));
      assert(mid - distance ==
             __sanitizer_contiguous_container_find_bad_address(
                 beg, mid - distance, end));
    }

    if (mid + distance <= end) {
      assert(
          !__sanitizer_verify_contiguous_container(beg, mid + distance, end));
      assert(mid == __sanitizer_contiguous_container_find_bad_address(
                        beg, mid + distance, end));
    }
  }

  __asan_unpoison_memory_region(buffer, buffer_end - buffer);
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
    for (int j = 0; j < 8; j++)
      for (int k = 0; k < 8; k++)
        for (int poison = 0; poison < 2; ++poison)
          TestContainer(i, j, k, poison);
  TestThrow();
}
