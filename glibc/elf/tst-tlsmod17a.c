#include <stdio.h>

#ifndef N
#define N 0
#endif
#define CONCAT1(s, n) s##n
#define CONCAT(s, n) CONCAT1(s, n)

__thread int CONCAT (v, N) = 4;

int
CONCAT (tlsmod17a, N) (void)
{
  int *p = &CONCAT (v, N);
  /* GCC assumes &var is never NULL, add optimization barrier.  */
  asm volatile ("" : "+r" (p));
  if (p == NULL || *p != 4)
    {
      printf ("fail %d %p\n", N, p);
      return 1;
    }
  return 0;
}
