#ifndef __ALIGNED_ALLOC_H__
#define __ALIGNED_ALLOC_H__

#include <cstdlib>

inline void *my_aligned_alloc(size_t alignment, size_t size) {
#if defined(_ISOC11_SOURCE)
  return aligned_alloc(alignment, size);
#else
  void *ptr;
  posix_memalign(&ptr, alignment, size);
  return ptr;
#endif
}

#endif //__ALIGNED_ALLOC_H__
