// __NO_INLINE__ is defined so bsearch needs interceptor.
// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
// RUN: %clangxx_msan -DPOISON_DATA -O0 -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_msan -DPOISON_KEY -O0 -g %s -o %t && not %run %t 2>&1 | FileCheck %s

// __NO_INLINE__ is undefined so bsearch should be inlined and instrumented and still work as expected.
// RUN: %clangxx_msan -O2 -g %s -o %t && %run %t
// RUN: %clangxx_msan -DPOISON_DATA -O2 -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_msan -DPOISON_KEY -O2 -g %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>

#include <sanitizer/msan_interface.h>

long z;

__attribute__((noinline, optnone)) void
poison_msan_param_tls(long a, long b, long c, long d, long e, long f) {
  z = a + b + c + d + e + f;
}

static int compar(const void *a, const void *b) {
  int r = *(const long *)a - *(const long *)b;
  long x;
  __msan_poison(&x, sizeof(x));
  poison_msan_param_tls(x, x, x, x, x, x);
  return r;
}

int main(int argc, char *argv[]) {
  constexpr size_t SZ = 27;
  long p[SZ + 1];
  for (int i = 0; i < SZ + 1; ++i)
    p[i] = i;
  p[SZ] = SZ / 3;
#if defined(POISON_DATA)
  __msan_poison(p, sizeof(long) * SZ / 2);
#elif defined(POISON_KEY)
  __msan_poison(p + SZ, sizeof(long));
#endif
  const long *r = (const long *)bsearch(p + SZ, p, SZ, sizeof(long), compar);
  // CHECK: MemorySanitizer: use-of-uninitialized-value

  assert(r == p + SZ / 3);

  return 0;
}
