#ifndef LIB_B_HEADER
#define LIB_B_HEADER

#include "LibAHeader.h"

template <typename T, size_t SlabSize, size_t SizeThreshold>
void *operator new(size_t, BumpPtrAllocatorImpl<T, SlabSize, SizeThreshold> &) {
  struct S {};
  return (void*)0xdead;
}

#endif // LIB_B_HEADER
