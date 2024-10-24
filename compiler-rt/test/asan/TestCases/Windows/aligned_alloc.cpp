// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: %run %t

#include <windows.h>

#define CHECK_ALIGNED(ptr, alignment)                                          \
  do {                                                                         \
    if (((uintptr_t)(ptr) % (alignment)) != 0)                                 \
      return __LINE__;                                                         \
  } while (0)

int main(void) {
  char *p = reinterpret_cast<char *>(_aligned_malloc(16, 8);
  CHECK_ALIGNED(p, 8);
  char *n = reinterpret_cast<char *>(_aligned_realloc(32, 16));
  CHECK_ALIGNED(n, 16);
  _aligned_free(n);

  return 0;
}
