#include "src/string/memory_utils/memcpy_utils.h"

namespace __llvm_libc {

static void CopyRepMovsb(char *__restrict dst, const char *__restrict src,
                         size_t count) {
  // FIXME: Add MSVC suppport with
  // #include <intrin.h>
  // __movsb(reinterpret_cast<unsigned char *>(dst),
  //         reinterpret_cast<const unsigned char *>(src), count);
  asm volatile("rep movsb" : "+D"(dst), "+S"(src), "+c"(count) : : "memory");
}

#if defined(__AVX__)
#define BEST_SIZE 64
#else
#define BEST_SIZE 32
#endif

static void CopyGE128(char *__restrict dst, const char *__restrict src,
                      size_t count) {
#if defined(__AVX__)
  if (count < 256)
    return CopyOverlap<128>(dst, src, count);
#endif
  // kRepMovsBSize == -1 : Only CopyAligned is used.
  // kRepMovsBSize ==  0 : Only RepMovsb is used.
  // else CopyAligned is used to to kRepMovsBSize and then RepMovsb.
  constexpr size_t kRepMovsBSize = -1;
  if (count <= kRepMovsBSize)
    return CopyAligned<BEST_SIZE>(dst, src, count);
  CopyRepMovsb(dst, src, count);
}

} // namespace __llvm_libc
