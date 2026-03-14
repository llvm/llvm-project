// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: %run %t

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

  return 0;
}
