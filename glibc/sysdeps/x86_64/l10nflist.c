#ifdef __POPCNT__
# include <popcntintrin.h>

static inline unsigned int
pop (unsigned int x)
{
  return _mm_popcnt_u32 (x);
}
# define ARCH_POP 1

#endif

#include <intl/l10nflist.c>
