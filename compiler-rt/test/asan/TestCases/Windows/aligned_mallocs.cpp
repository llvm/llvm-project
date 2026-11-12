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

  _aligned_free(nullptr);

  // Size need not be a multiple of the alignment.
  char *c = (char *)_aligned_malloc(100, 32);
  CHECK_ALIGNED(c, 32);
  c[99] = 0;
  _aligned_free(c);

  // _aligned_offset_malloc aligns ptr + offset, not ptr.
  c = (char *)_aligned_offset_malloc(100, 64, 8);
  CHECK_ALIGNED(c + 8, 64);
  if (_aligned_msize(c, 64, 8) != 100)
    return __LINE__;
  c[99] = 0;
  c = (char *)_aligned_offset_realloc(c, 200, 64, 8);
  CHECK_ALIGNED(c + 8, 64);
  c[199] = 0;
  _aligned_free(c);

  // _aligned_recalloc keeps the old contents and zeroes the grown tail.
  c = (char *)_aligned_recalloc(nullptr, 4, 4, 16);
  CHECK_ALIGNED(c, 16);
  c[0] = 'a';
  c = (char *)_aligned_recalloc(c, 8, 4, 16);
  CHECK_ALIGNED(c, 16);
  if (c[0] != 'a' || c[31] != 0)
    return __LINE__;
  _aligned_free(c);

  char *t = (char *)_aligned_malloc(128, 8);
  t[128] = 'a';
  // CHECK: AddressSanitizer: heap-buffer-overflow
  // CHECK: WRITE of size 1
  // CHECK: 0 bytes after 128-byte region

  return 0;
}
