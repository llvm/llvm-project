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

  long double tl = (long double) FLT_MAX + 0x1.0p128L;
  float fi = INFINITY;
  float m = FLT_MAX;
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttowardf (m, tl) != fi)
    {
      puts ("nexttowardf+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nexttowardf+ did not overflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttowardf (-m, -tl) != -fi)
    {
      puts ("nexttowardf- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nexttowardf- did not overflow");
      ++result;
    }

  fi = 0;
  m = FLT_MIN;
  feclearexcept (FE_ALL_EXCEPT);
  fi = nexttowardf (m, fi);
  if (fi < 0 || fi >= FLT_MIN)
    {
      puts ("nexttowardf+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardf+ did not underflow");
      ++result;
    }
  fi = 0;
  feclearexcept (FE_ALL_EXCEPT);
  fi = nexttowardf (-m, -fi);
  if (fi > 0 || fi <= -FLT_MIN)
    {
      puts ("nexttowardf- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardf- did not underflow");
      ++result;
    }
  fi = -INFINITY;
  feclearexcept (FE_ALL_EXCEPT);
  m = nexttowardf (zero, inf);
  if (m < 0.0 || m >= FLT_MIN)
    {
      puts ("nexttowardf+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardf+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttowardf (m, fi) != 0.0)
    {
      puts ("nexttowardf+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardf+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  m = nexttowardf (copysignf (zero, -1.0), -inf);
  if (m > 0.0 || m <= -FLT_MIN)
    {
      puts ("nexttowardf- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardf- did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttowardf (m, -fi) != 0.0)
    {
      puts ("nexttowardf- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (float) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardf- did not underflow");
      ++result;
    }

  tl = (long double) DBL_MAX + 1.0e305L;
  double di = INFINITY;
  double dm = DBL_MAX;
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttoward (dm, tl) != di)
    {
      puts ("nexttoward+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nexttoward+ did not overflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttoward (-dm, -tl) != -di)
    {
      puts ("nexttoward- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nexttoward- did not overflow");
      ++result;
    }

  di = 0;
  dm = DBL_MIN;
  feclearexcept (FE_ALL_EXCEPT);
  di = nexttoward (dm, di);
  if (di < 0 || di >= DBL_MIN)
    {
      puts ("nexttoward+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttoward+ did not underflow");
      ++result;
    }
  di = 0;
  feclearexcept (FE_ALL_EXCEPT);
  di = nexttoward (-dm, -di);
  if (di > 0 || di <= -DBL_MIN)
    {
      puts ("nexttoward- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttoward- did not underflow");
      ++result;
    }
  di = -INFINITY;
  feclearexcept (FE_ALL_EXCEPT);
  dm = nexttoward (zero, inf);
  if (dm < 0.0 || dm >= DBL_MIN)
    {
      puts ("nexttoward+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttoward+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttoward (dm, di) != 0.0)
    {
      puts ("nexttoward+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttoward+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  dm = nexttoward (copysign (zero, -1.0), -inf);
  if (dm > 0.0 || dm <= -DBL_MIN)
    {
      puts ("nexttoward- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttoward- did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttoward (dm, -di) != 0.0)
    {
      puts ("nexttoward- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttoward- did not underflow");
      ++result;
    }

  long double li = INFINITY;
  long double lm = LDBL_MAX;
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttowardl (lm, li) != li)
    {
      puts ("nexttowardl+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nexttowardl+ did not overflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttowardl (-lm, -li) != -li)
    {
      puts ("nexttowardl failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_OVERFLOW) == 0)
    {
      puts ("nexttowardl- did not overflow");
      ++result;
    }

  li = 0;
  lm = LDBL_MIN;
  feclearexcept (FE_ALL_EXCEPT);
  li = nexttowardl (lm, li);
  if (li < 0 || li >= LDBL_MIN)
    {
      puts ("nexttowardl+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardl+ did not underflow");
      ++result;
    }
  li = 0;
  feclearexcept (FE_ALL_EXCEPT);
  li = nexttowardl (-lm, -li);
  if (li > 0 || li <= -LDBL_MIN)
    {
      puts ("nexttowardl- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardl- did not underflow");
      ++result;
    }
  li = -INFINITY;
  feclearexcept (FE_ALL_EXCEPT);
  lm = nexttowardl (zero, inf);
  if (lm < 0.0 || lm >= LDBL_MIN)
    {
      puts ("nexttowardl+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardl+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttowardl (lm, li) != 0.0)
    {
      puts ("nexttowardl+ failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardl+ did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  lm = nexttowardl (copysign (zero, -1.0), -inf);
  if (lm > 0.0 || lm <= -LDBL_MIN)
    {
      puts ("nexttowardl- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardl- did not underflow");
      ++result;
    }
  feclearexcept (FE_ALL_EXCEPT);
  if (nexttowardl (lm, -li) != 0.0)
    {
      puts ("nexttowardl- failed");
      ++result;
    }
  if (EXCEPTION_TESTS (long double) && fetestexcept (FE_UNDERFLOW) == 0)
    {
      puts ("nexttowardl- did not underflow");
      ++result;
    }

  return result;
}
