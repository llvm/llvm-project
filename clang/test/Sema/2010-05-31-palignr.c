// RUN: not %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o /dev/null %s

#include <tmmintrin.h>

extern int i;

int main ()
{
#if defined( __SSSE3__ )

  typedef int16_t     vSInt16         __attribute__ ((__vector_size__ (16)));

  short   dtbl[] = {1,2,3,4,5,6,7,8};
  vSInt16 *vdtbl = (vSInt16*) dtbl;

  vSInt16 v0;
  v0 = *vdtbl;
  v0 = _mm_alignr_epi8(v0, v0, i); // expected-error {{argument to '__builtin_ia32_palignr128' must be a constant integer}}

  return 0;
#endif
}
