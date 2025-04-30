#ifndef MATH_PRIVATE_PPC64LE_MA
#define MATH_PRIVATE_PPC64LE_MA 1

#include_next <math_private.h>

#if defined (_F128_ENABLE_IFUNC)

/* math_private.h redeclares many float128_private.h renamed functions, but
   we can't include float128_private.h as this header is used beyond
   private float128 files.  */
#include <float128-ifunc-redirects-mp.h>

#endif

#endif /* MATH_PRIVATE_PPC64LE_MA */
