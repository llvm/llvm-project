#include <stddef.h>

// WARNING: When building the scalar versions of these functions you need to
// use the compiler flag "-mllvm -disable-loop-idiom-all" to prevent clang
// from recognising a loop idiom and planting calls to memcpy!

static void *__arm_sc_memcpy_fwd(void *dest, const void *src,
                                 size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  const unsigned char *srcp = (const unsigned char *)src;
  for (size_t i = 0; i < n; ++i)
    destp[i] = srcp[i];

  return dest;
}

// If dest and src overlap then behaviour is undefined, hence we can add the
// restrict keywords here. This also matches the definition of the libc memcpy
// according to the man page.
void *__arm_sc_memcpy(void *__restrict__ dest, const void *__restrict__ src,
                      size_t n) __arm_streaming_compatible {
  return __arm_sc_memcpy_fwd(dest, src, n);
}

void *__arm_sc_memset(void *dest, int c, size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  unsigned char c8 = (unsigned char)c;
  for (size_t i = 0; i < n; ++i)
    destp[i] = c8;

  return dest;
}

static void *__arm_sc_memcpy_rev(void *dest, const void *src,
                                 size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  const unsigned char *srcp = (const unsigned char *)src;
  // TODO: Improve performance by copying larger chunks in reverse, or by
  // using SVE.
  while (n > 0) {
    --n;
    destp[n] = srcp[n];
  }
  return dest;
}

// Semantically a memmove is equivalent to the following:
//   1. Copy the entire contents of src to a temporary array that does not
//      overlap with src or dest.
//   2. Copy the contents of the temporary array into dest.
void *__arm_sc_memmove(void *dest, const void *src,
                       size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  const unsigned char *srcp = (const unsigned char *)src;

  // If src and dest don't overlap then just invoke memcpy
  if ((srcp > (destp + n)) || (destp > (srcp + n)))
    return __arm_sc_memcpy_fwd(dest, src, n);

  // Overlap case 1:
  //     src: Low     |   ->   |     High
  //    dest: Low  |   ->   |        High
  // Here src is always ahead of dest at a higher addres. If we first read a
  // chunk of data from src we can safely write the same chunk to dest without
  // corrupting future reads of src.
  if (srcp > destp)
    return __arm_sc_memcpy_fwd(dest, src, n);

  // Overlap case 2:
  //     src: Low  |   ->   |        High
  //    dest: Low     |   ->   |     High
  // While we're in the overlap region we're always corrupting future reads of
  // src when writing to dest. An efficient way to do this is to copy the data
  // in reverse by starting at the highest address.
  return __arm_sc_memcpy_rev(dest, src, n);
}

const void *__arm_sc_memchr(const void *src, int c,
                            size_t n) __arm_streaming_compatible {
  const unsigned char *srcp = (const unsigned char *)src;
  unsigned char c8 = (unsigned char)c;
  for (size_t i = 0; i < n; ++i)
    if (srcp[i] == c8)
      return &srcp[i];

  return NULL;
}
