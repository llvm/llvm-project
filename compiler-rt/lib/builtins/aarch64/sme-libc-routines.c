#include <stddef.h>

// The asm version uses FP registers and unaligned memory accesses. Use this on
// targets without them.
#if __ARM_FP == 0 || !defined(__ARM_FEATURE_UNALIGNED)
void *__arm_sc_memset(void *dest, int c, size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  unsigned char c8 = (unsigned char)c;
  for (size_t i = 0; i < n; ++i)
    destp[i] = c8;

  return dest;
}
#endif

const void *__arm_sc_memchr(const void *src, int c,
                            size_t n) __arm_streaming_compatible {
  const unsigned char *srcp = (const unsigned char *)src;
  unsigned char c8 = (unsigned char)c;
  for (size_t i = 0; i < n; ++i)
    if (srcp[i] == c8)
      return &srcp[i];

  return NULL;
}

#ifndef __ARM_FEATURE_UNALIGNED

static void *memcpy_fwd(void *dest, const void *src,
                        size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  const unsigned char *srcp = (const unsigned char *)src;

  for (size_t i = 0; i < n; ++i)
    destp[i] = srcp[i];
  return dest;
}

static void *memcpy_rev(void *dest, const void *src,
                        size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  const unsigned char *srcp = (const unsigned char *)src;

  while (n > 0) {
    --n;
    destp[n] = srcp[n];
  }
  return dest;
}

void *__arm_sc_memcpy(void *__restrict dest, const void *__restrict src,
                      size_t n) __arm_streaming_compatible {
  return memcpy_fwd(dest, src, n);
}

void *__arm_sc_memmove(void *dest, const void *src,
                       size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  const unsigned char *srcp = (const unsigned char *)src;

  if ((srcp > (destp + n)) || (destp > (srcp + n)))
    return __arm_sc_memcpy(dest, src, n);
  if (srcp > destp)
    return memcpy_fwd(dest, src, n);
  return memcpy_rev(dest, src, n);
}

#endif /* !defined(__ARM_FEATURE_UNALIGNED) */
