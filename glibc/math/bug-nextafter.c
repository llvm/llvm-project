#include <fenv.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math-tests.h>

#if !defined(FE_OVERFLOW) && !defined(FE_UNDERFLOW)
/* If there's no support for the exceptions this test is checking,
   then just return success and allow the test to be compiled.  */
# define fetestexcept(e) 1
#endif

float zero = 0.0;
float inf = INFINITY;

int
main (void)
{
  int result = 0;

  float i = INFINITY;
  float m = FLT_MAX;
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafterf (m, i) != i)
    {
      puts ("nextafterf+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nextafterf+ did not overflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafterf (-m, -i) != -i)
    {
      puts ("nextafterf- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nextafterf- did not overflow");
      ++result;
    }

  i = 0;
  m = FLT_MIN;
  feclearexcept (FE_ALL_EXCEPT);
  i = nextafterf (m, i);
  if (i < 0 || i >= FLT_MIN)
    {
      puts ("nextafterf+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterf+ did not underflow");
      ++result;
    }
  i = 0;
  feclearexcept (FE_ALL_EXCEPT);
  i = nextafterf (-m, -i);
  if (i > 0 || i <= -FLT_MIN)
    {
      puts ("nextafterf- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterf- did not underflow");
      ++result;
    }
  i = -INFINITY;
  feclearexcept (FE_ALL_EXCEPT);
  m = nextafterf (zero, inf);
  if (m < 0.0 || m >= FLT_MIN)
    {
      puts ("nextafterf+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterf+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafterf (m, i) != 0.0)
    {
      puts ("nextafterf+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterf+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  m = nextafterf (copysignf (zero, -1.0), -inf);
  if (m > 0.0 || m <= -FLT_MIN)
    {
      puts ("nextafterf- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterf- did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafterf (m, -i) != 0.0)
    {
      puts ("nextafterf- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterf- did not underflow");
      ++result;
    }

  double di = INFINITY;
  double dm = DBL_MAX;
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafter (dm, di) != di)
    {
      puts ("nextafter+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nextafter+ did not overflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafter (-dm, -di) != -di)
    {
      puts ("nextafter failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nextafter- did not overflow");
      ++result;
    }

  di = 0;
  dm = DBL_MIN;
  feclearexcept (FE_ALL_EXCEPT);
  di = nextafter (dm, di);
  if (di < 0 || di >= DBL_MIN)
    {
      puts ("nextafter+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafter+ did not underflow");
      ++result;
    }
  di = 0;
  feclearexcept (FE_ALL_EXCEPT);
  di = nextafter (-dm, -di);
  if (di > 0 || di <= -DBL_MIN)
    {
      puts ("nextafter- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafter- did not underflow");
      ++result;
    }
  di = -INFINITY;
  feclearexcept (FE_ALL_EXCEPT);
  dm = nextafter (zero, inf);
  if (dm < 0.0 || dm >= DBL_MIN)
    {
      puts ("nextafter+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafter+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafter (dm, di) != 0.0)
    {
      puts ("nextafter+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafter+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  dm = nextafter (copysign (zero, -1.0), -inf);
  if (dm > 0.0 || dm <= -DBL_MIN)
    {
      puts ("nextafter- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafter- did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafter (dm, -di) != 0.0)
    {
      puts ("nextafter- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafter- did not underflow");
      ++result;
    }

  long double li = INFINITY;
  long double lm = LDBL_MAX;
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafterl (lm, li) != li)
    {
      puts ("nextafterl+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nextafterl+ did not overflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafterl (-lm, -li) != -li)
    {
      puts ("nextafterl failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nextafterl- did not overflow");
      ++result;
    }

  li = 0;
  lm = LDBL_MIN;
  feclearexcept (FE_ALL_EXCEPT);
  li = nextafterl (lm, li);
  if (li < 0 || li >= LDBL_MIN)
    {
      puts ("nextafterl+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterl+ did not underflow");
      ++result;
    }
  li = 0;
  feclearexcept (FE_ALL_EXCEPT);
  li = nextafterl (-lm, -li);
  if (li > 0 || li <= -LDBL_MIN)
    {
      puts ("nextafterl- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterl- did not underflow");
      ++result;
    }
  li = -INFINITY;
  feclearexcept (FE_ALL_EXCEPT);
  lm = nextafterl (zero, inf);
  if (lm < 0.0 || lm >= LDBL_MIN)
    {
      puts ("nextafterl+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterl+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafterl (lm, li) != 0.0)
    {
      puts ("nextafterl+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterl+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  lm = nextafterl (copysign (zero, -1.0), -inf);
  if (lm > 0.0 || lm <= -LDBL_MIN)
    {
      puts ("nextafterl- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterl- did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nextafterl (lm, -li) != 0.0)
    {
      puts ("nextafterl- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nextafterl- did not underflow");
      ++result;
    }

  return result;
}
