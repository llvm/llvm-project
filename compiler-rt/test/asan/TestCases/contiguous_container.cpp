// RUN: %clangxx_asan -fexceptions -O %s -o %t && %env_asan_opts=detect_stack_use_after_return=0 %run %t
//
// Test __sanitizer_annotate_contiguous_container.

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sanitizer/asan_interface.h>

void TestContainer(size_t capacity) {
  char *beg = new char[capacity];
  char *end = beg + capacity;
  char *mid = beg + capacity;
  char *old_mid = 0;

  for (int i = 0; i < 10000; i++) {
    size_t size = rand() % (capacity + 1);
    assert(size <= capacity);
    old_mid = mid;
    mid = beg + size;
    __sanitizer_annotate_contiguous_container(beg, end, old_mid, mid);

    for (size_t idx = 0; idx < size; idx++)
        assert(!__asan_address_is_poisoned(beg + idx));
    for (size_t idx = size; idx < capacity; idx++)
        assert(__asan_address_is_poisoned(beg + idx));
    assert(__sanitizer_verify_contiguous_container(beg, mid, end));
    assert(NULL ==
           __sanitizer_contiguous_container_find_bad_address(beg, mid, end));
    if (mid != beg) {
      assert(!__sanitizer_verify_contiguous_container(beg, mid - 1, end));
      assert(mid - 1 == __sanitizer_contiguous_container_find_bad_address(
                            beg, mid - 1, end));
    }
    if (mid != end) {
      assert(!__sanitizer_verify_contiguous_container(beg, mid + 1, end));
      assert(mid == __sanitizer_contiguous_container_find_bad_address(
                        beg, mid + 1, end));
    }
  }

  // Don't forget to unpoison the whole thing before destroying/reallocating.
  __sanitizer_annotate_contiguous_container(beg, end, mid, end);
  for (size_t idx = 0; idx < capacity; idx++)
    assert(!__asan_address_is_poisoned(beg + idx));
  delete[] beg;
}

__attribute__((noinline))
void Throw() { throw 1; }

__attribute__((noinline))
void ThrowAndCatch() {
  try {
    Throw();
  } catch(...) {
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
  int n = argc == 1 ? 128 : atoi(argv[1]);
  for (int i = 0; i <= n; i++)
    TestContainer(i);
  TestThrow();
}
