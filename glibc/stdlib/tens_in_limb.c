#include <gmp.h>


/* Definitions according to limb size used.  */
#if	BITS_PER_MP_LIMB == 32
# define MAX_DIG_PER_LIMB	9
# define MAX_FAC_PER_LIMB	1000000000UL
#elif	BITS_PER_MP_LIMB == 64
# define MAX_DIG_PER_LIMB	19
# define MAX_FAC_PER_LIMB	10000000000000000000ULL
#else
# error "mp_limb_t size " BITS_PER_MP_LIMB "not accounted for"
#endif


/* Local data structure.  */
const mp_limb_t _tens_in_limb[MAX_DIG_PER_LIMB + 1] =
{    0,                   10,                   100,
     1000,                10000,                100000L,
     1000000L,            10000000L,            100000000L,
     1000000000L
#if BITS_PER_MP_LIMB > 32
	        ,	  10000000000ULL,       100000000000ULL,
     1000000000000ULL,    10000000000000ULL,    100000000000000ULL,
     1000000000000000ULL, 10000000000000000ULL, 100000000000000000ULL,
     1000000000000000000ULL, 10000000000000000000ULL
#endif
#if BITS_PER_MP_LIMB > 64
  #error "Need to expand tens_in_limb table to" MAX_DIG_PER_LIMB
#endif
};
