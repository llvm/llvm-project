#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int
do_test (void)
{
#if LDBL_MANT_DIG >= 106
  volatile union { long double l; long long x[2]; } u, v;
  char buf[64];
#endif
  int result = 0;

#if LDBL_MANT_DIG == 106 || LDBL_MANT_DIG == 113
# define COMPARE_LDBL(u, v) \
  ((u).l == (v).l && (u).x[0] == (v).x[0] && (u).x[1] == (v).x[1])
#else
# define COMPARE_LDBL(u, v) ((u).l == (v).l)
#endif

#define TEST(val) \
  do									   \
    {									   \
      u.l = (val);							   \
      snprintf (buf, sizeof buf, "%LaL", u.l);				   \
      if (strcmp (buf, #val) != 0)					   \
	{								   \
	  printf ("Error on line %d: %s != %s\n", __LINE__, buf, #val);	   \
	  result = 1;							   \
	}								   \
      if (sscanf (#val, "%La", &v.l) != 1 || !COMPARE_LDBL (u, v))	   \
	{								   \
	  printf ("Error sscanf on line %d: %La != %La\n", __LINE__,	   \
		  u.l, v.l);						   \
	  result = 1;							   \
	}								   \
      /* printf ("%s %La %016Lx %016Lx\n", #val, u.l, u.x[0], u.x[1]); */  \
    }									   \
  while (0)

#if LDBL_MANT_DIG >= 106
# if LDBL_MANT_DIG == 106
  TEST (0x0.ffffffffffffp-1022L);
  TEST (0x0.ffffffffffff1p-1022L);
  TEST (0x0.fffffffffffffp-1022L);
# endif
  TEST (0x1p-1022L);
  TEST (0x1.0000000000001p-1022L);
  TEST (0x1.00000000001e7p-1022L);
  TEST (0x1.fffffffffffffp-1022L);
  TEST (0x1p-1021L);
  TEST (0x1.00000000000008p-1021L);
  TEST (0x1.0000000000001p-1021L);
  TEST (0x1.00000000000018p-1021L);
  TEST (0x1.0000000000000f8p-1017L);
  TEST (0x1.0000000000001p-1017L);
  TEST (0x1.000000000000108p-1017L);
  TEST (0x1.000000000000dcf8p-1013L);
  TEST (0x1.000000000000ddp-1013L);
  TEST (0x1.000000000000dd08p-1013L);
  TEST (0x1.ffffffffffffffffffffffffffp-1L);
  TEST (0x1.ffffffffffffffffffffffffff8p-1L);
  TEST (0x1p+0L);
  TEST (0x1.000000000000000000000000008p+0L);
  TEST (0x1.00000000000000000000000001p+0L);
  TEST (0x1.000000000000000000000000018p+0L);
  TEST (0x1.23456789abcdef123456789abc8p+0L);
  TEST (0x1.23456789abcde7123456789abc8p+0L);
  TEST (0x1.23456789abcdef123456789abc8p+64L);
  TEST (0x1.23456789abcde7123456789abc8p+64L);
  TEST (0x1.123456789abcdef123456789p-969L);
# if LDBL_MANT_DIG == 106
  TEST (-0x1.2d71957cc1263bbbeb1d365f1e8p-969L);
  TEST (0x1.23456789abcdef0123456789abp-970L);
  TEST (0x1.579bde02468acp-1001L);
  TEST (0x0.abcdef0123456p-1022L);
  TEST (0x1.abcdef0123456p-1022L);
  TEST (0x1.abcdef012345678p-1014L);
  TEST (0x1.abcdef0123456f8p-1014L);
# endif
#endif
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
