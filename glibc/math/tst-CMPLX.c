#include <complex.h>
#include <stdio.h>


static int
do_test (void)
{
  int result = 0;

#ifdef CMPLX
  size_t s;

#define T(f, r, i, t)							      \
  do {									      \
    s = sizeof (f (r, i));						      \
    if (s != sizeof (complex t))					      \
      {									      \
	printf ("\
CMPLX (" #r ", " #i ") does not produce complex " #t ": %zu\n", s);	      \
	result = 1;							      \
      }									      \
  } while (0)

#define C(f, t)								      \
  do {									      \
    T (f, 0.0f, 0.0f, t);						      \
    T (f, 0.0f, 0.0, t);						      \
    T (f, 0.0f, 0.0L, t);						      \
    T (f, 0.0f, 0.0f, t);						      \
    T (f, 0.0, 0.0f, t);						      \
    T (f, 0.0L, 0.0f, t);						      \
    T (f, 0.0, 0.0f, t);						      \
    T (f, 0.0, 0.0, t);							      \
    T (f, 0.0, 0.0L, t);						      \
    T (f, 0.0f, 0.0, t);						      \
    T (f, 0.0, 0.0, t);							      \
    T (f, 0.0L, 0.0, t);						      \
    T (f, 0.0L, 0.0f, t);						      \
    T (f, 0.0L, 0.0, t);						      \
    T (f, 0.0L, 0.0L, t);						      \
    T (f, 0.0f, 0.0L, t);						      \
    T (f, 0.0, 0.0L, t);						      \
    T (f, 0.0L, 0.0L, t);						      \
  } while (0)

  C (CMPLXF, float);
  C (CMPLX, double);
  C (CMPLXL, long double);
#endif

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
