// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

struct alignas(16) Large {
  char data[16];
};

struct Huge {
  char data[32];
};

extern "C" bool __atomic_is_lock_free(size_t size, const volatile void *ptr);

int main() {
  std::atomic<Huge> b;
  assert(!b.is_lock_free());

  // Use volatile to ensure runtime function invocation (avoid compiler constant folding).
  volatile size_t s0 = 0;
  volatile size_t s1 = 1;
  volatile size_t s2 = 2;
  volatile size_t s3 = 3;
  volatile size_t s4 = 4;
  volatile size_t s8 = 8;
  volatile size_t s16 = 16;
  volatile size_t s32 = 32;

  assert(!__atomic_is_lock_free(s0, nullptr));
  assert(__atomic_is_lock_free(s1, nullptr));
  assert(__atomic_is_lock_free(s2, nullptr));
  assert(!__atomic_is_lock_free(s3, nullptr));
  assert(__atomic_is_lock_free(s4, nullptr));
  assert(__atomic_is_lock_free(s8, nullptr));
  assert(!__atomic_is_lock_free(s32, nullptr));

  alignas(16) char buffer[32];
  // Size 1 is always lock-free regardless of alignment.
  assert(__atomic_is_lock_free(s1, buffer));
  assert(__atomic_is_lock_free(s1, buffer + 1));

  // Size 2
  assert(__atomic_is_lock_free(s2, buffer));
  assert(!__atomic_is_lock_free(s2, buffer + 1));

  // Size 4
  assert(__atomic_is_lock_free(s4, buffer));
  assert(!__atomic_is_lock_free(s4, buffer + 1));

  // Size 8
  assert(__atomic_is_lock_free(s8, buffer));
  assert(!__atomic_is_lock_free(s8, buffer + 1));

#if defined(__SIZEOF_INT128__)
  std::atomic<Large> a;
  bool is_16_lock_free = a.is_lock_free();
  assert(__atomic_is_lock_free(s16, nullptr) == is_16_lock_free);

  if (is_16_lock_free) {
    assert(__atomic_is_lock_free(s16, buffer));
    assert(!__atomic_is_lock_free(s16, buffer + 1));
  } else {
    assert(!__atomic_is_lock_free(s16, buffer));
  }
#endif

  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK: PASS
