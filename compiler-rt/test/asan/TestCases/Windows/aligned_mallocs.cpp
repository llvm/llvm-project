// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <windows.h>

#ifdef __MINGW32__
// FIXME: remove after mingw-w64 adds this declaration.
extern "C" size_t __cdecl _aligned_msize(void *_Memory, size_t _Alignment,
                                         size_t _Offset);
#endif

#define CHECK_ALIGNED(ptr,alignment) \
  do { \
    if (((uintptr_t)(ptr) % (alignment)) != 0) \
      return __LINE__; \
    } \
  while(0)

int main(void) {
  int *p = (int*)_aligned_malloc(1024 * sizeof(int), 32);
  CHECK_ALIGNED(p, 32);
  p[512] = 0;
  _aligned_free(p);

  p = (int*)_aligned_malloc(128, 128);
  CHECK_ALIGNED(p, 128);
  p = (int*)_aligned_realloc(p, 2048 * sizeof(int), 128);
  CHECK_ALIGNED(p, 128);
  p[1024] = 0;
  if (_aligned_msize(p, 128, 0) != 2048 * sizeof(int))
    return __LINE__;
  _aligned_free(p);

  char *y = (char *)malloc(1024);
  char *u = (char *)realloc(y, 2048);
  u[0] = 'a';
  _aligned_free(u);
  u = (char *)_aligned_offset_malloc(1024, 8, 0);
  _aligned_free(u);

  char *t = (char *)_aligned_malloc(128, 8);
  t[-153] = 'a';

  return 0;
}

// CHECK: AddressSanitizer: access-violation on unknown address
// CHECK: The signal is caused by a WRITE memory access.
