#ifndef LIB_A_HEADER
#define LIB_A_HEADER

typedef __SIZE_TYPE__ size_t;

template <typename = int, size_t SlabSize = 4096, size_t = SlabSize>
class BumpPtrAllocatorImpl;

template <typename T, size_t SlabSize, size_t SizeThreshold>
void * operator new(size_t, BumpPtrAllocatorImpl<T, SlabSize, SizeThreshold> &);

#endif // LIB_A_HEADER
