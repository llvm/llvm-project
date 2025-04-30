#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "soft-fp.h"
#include "single.h"
#include "double.h"

#include <fpu_control.h>

/*======================================================================*/
/* declarations for the functions we are testing */

double __adddf3(double, double);
double __subdf3(double, double);
double __muldf3(double, double);
double __divdf3(double, double);
double __negdf2(double);
double __sqrtdf2(double);
double __negdf3(double a, double dummy) { return __negdf2(a); }
double __sqrtdf3(double a, double dummy) { return __sqrtdf2(a); }

float __addsf3(float, float);
float __subsf3(float, float);
float __mulsf3(float, float);
float __divsf3(float, float);
float __negsf2(float);
float __sqrtsf2(float);
float __negsf3(float a, float dummy) { return __negsf2(a); }
float __sqrtsf3(float a, float dummy) { return __sqrtsf2(a); }

int __fixdfsi(double);
int __fixsfsi(float);
double __floatsidf(int);
float __floatsisf(int);
double __extendsfdf2(float);
float __truncdfsf2(double);

int __eqdf2(double, double);
int __nedf2(double, double);
int __gtdf2(double, double);
int __gedf2(double, double);
int __ltdf2(double, double);
int __ledf2(double, double);

int __eqsf2(float, float);
int __nesf2(float, float);
int __gtsf2(float, float);
int __gesf2(float, float);
int __ltsf2(float, float);
int __lesf2(float, float);

/*======================================================================*/
/* definitions for functions we are checking against */

double r_adddf3(double a, double b) { return a + b; }
double r_subdf3(double a, double b) { return a - b; }
double r_muldf3(double a, double b) { return a * b; }
double r_divdf3(double a, double b) { return a / b; }
double r_negdf3(double a, double b) { return -a; }
double sqrt(double x);
double r_sqrtdf3(double a, double b) { return sqrt(a); }

float r_addsf3(float a, float b) { return a + b; }
float r_subsf3(float a, float b) { return a - b; }
float r_mulsf3(float a, float b) { return a * b; }
float r_divsf3(float a, float b) { return a / b; }
float r_negsf3(float a, float b) { return -a; }
float sqrtf(float x);
float r_sqrtsf3(float a, float b) { return sqrtf(a); }

int r_fixdfsi(double a) { return (int)a; }
int r_fixsfsi(float a) { return (int)a; }
double r_floatsidf(int a) { return (double)a; }
float r_floatsisf(int a) { return (float)a; }
double r_extendsfdf2(float a) { return (double)a; }
float r_truncdfsf2(double a) { return (float)a; }

int r_eqdf2(double a, double b) { return !(a == b); }
int r_nedf2(double a, double b) { return a != b; }
int r_gtdf2(double a, double b) { return a > b; }
int r_gedf2(double a, double b) { return (a >= b) - 1; }
int r_ltdf2(double a, double b) { return -(a < b); }
int r_ledf2(double a, double b) { return 1 - (a <= b); }

int r_eqsf2(float a, float b) { return !(a == b); }
int r_nesf2(float a, float b) { return a != b; }
int r_gtsf2(float a, float b) { return a > b; }
int r_gesf2(float a, float b) { return (a >= b) - 1; }
int r_ltsf2(float a, float b) { return -(a < b); }
int r_lesf2(float a, float b) { return 1 - (a <= b); }

/*======================================================================*/

void print_float(float x)
{
    union _FP_UNION_S ux;
    ux.flt = x;
    printf("%-20.8e %X %02X %06lX",
	   x, ux.bits.sign, ux.bits.exp, (unsigned long)ux.bits.frac);
}

void print_double(double x)
{
    union _FP_UNION_D ux;
    ux.flt = x;
#if _FP_W_TYPE_SIZE < _FP_FRACBITS_D
    printf("%-30.18e %X %04X %06lX%08lX",
	   x, ux.bits.sign, ux.bits.exp,
	   (unsigned long)ux.bits.frac1, (unsigned long)ux.bits.frac0);
#else
    printf("%-30.18e %X %04X %014lX",
	   x, ux.bits.sign, ux.bits.exp,
	   (unsigned long)ux.bits.frac);
#endif
}

float rand_float(void)
{
    union {
	union _FP_UNION_S u;
	int i;
    } u;

    u.i = lrand48() << 1;

    if (u.u.bits.exp == _FP_EXPMAX_S)
	u.u.bits.exp--;
    else if (u.u.bits.exp == 0 && u.u.bits.frac != 0)
	u.u.bits.exp++;

    return u.u.flt;
}


double rand_double(void)
{
    union {
	union _FP_UNION_D u;
	int i[2];
    } u;

    u.i[0] = lrand48() << 1;
    u.i[1] = lrand48() << 1;

    if (u.u.bits.exp == _FP_EXPMAX_D)
	u.u.bits.exp--;
#if _FP_W_TYPE_SIZE < _FP_FRACBITS_D
    else if (u.u.bits.exp == 0 && !(u.u.bits.frac0 == 0 && u.u.bits.frac1 == 0))
	u.u.bits.exp++;
#else
    else if (u.u.bits.exp == 0 && u.u.bits.frac != 0)
	u.u.bits.exp++;
#endif

    return u.u.flt;
}

#define NSPECIALS  10

float gen_special_float(int i)
{
    FP_DECL_EX;
    FP_DECL_S(X);
    float x;

    switch (i & ~1)
    {
      case 0:
	X_c = FP_CLS_NAN; X_f = 0x1234;
	break;
      case 2:
	X_c = FP_CLS_NAN; X_f = 0x1;
	break;
      case 4:
	X_c = FP_CLS_INF;
	break;
      case 6:
	X_c = FP_CLS_ZERO;
	break;
      case 8:
	X_c = FP_CLS_NORMAL; X_e = 0;
	X_f = 0x4321;
	break;
    }
    X_s = (i & 1);

    FP_PACK_S(x, X);
    return x;
}

double gen_special_double(int i)
{
    FP_DECL_EX;
    FP_DECL_D(X);
    double x;

    switch (i & ~1)
    {
      case 0:
	X_c = FP_CLS_NAN;
#if _FP_W_TYPE_SIZE < _FP_FRACBITS_D
	__FP_FRAC_SET_2(X, _FP_QNANNEGATEDP ? 0 : _FP_QNANBIT_D, 0x1234);
#else
	_FP_FRAC_SET_1(X, (_FP_QNANNEGATEDP ? 0 : _FP_QNANBIT_D) | 0x1234);
#endif
	break;
      case 2:
	X_c = FP_CLS_NAN;
#if _FP_W_TYPE_SIZE < _FP_FRACBITS_D
	__FP_FRAC_SET_2(X, _FP_QNANNEGATEDP ? 0 : _FP_QNANBIT_D, 0x1);
#else
	_FP_FRAC_SET_1(X, (_FP_QNANNEGATEDP ? 0 : _FP_QNANBIT_D) | 0x1);
#endif
	break;
      case 4:
	X_c = FP_CLS_INF;
	break;
      case 6:
	X_c = FP_CLS_ZERO;
	break;
      case 8:
	X_c = FP_CLS_NORMAL; X_e = 0;
#if _FP_W_TYPE_SIZE < _FP_FRACBITS_D
	__FP_FRAC_SET_2(X, 0, 0x87654321);
#else
	_FP_FRAC_SET_1(X, 0x87654321);
#endif
	break;
    }
    X_s = (i & 1);

    FP_PACK_D(x, X);
    return x;
}

float build_float(const char *s, const char *e, const char *f)
{
    union _FP_UNION_S u;

    u.bits.sign = strtoul(s, 0, 16);
    u.bits.exp = strtoul(e, 0, 16);
    u.bits.frac = strtoul(f, 0, 16);

    return u.flt;
}

double build_double(const char *s, const char *e, const char *f)
{
    union _FP_UNION_D u;

    u.bits.sign = strtoul(s, 0, 16);
    u.bits.exp = strtoul(e, 0, 16);
#if _FP_W_TYPE_SIZE < _FP_FRACBITS_D
    {
	size_t len = strlen(f)+1;
	char *dup = memcpy(alloca(len), f, len);
	char *low = dup + len - _FP_W_TYPE_SIZE/4 - 1;

	u.bits.frac0 = strtoul(low, 0, 16);
	*low = 0;
	u.bits.frac1 = strtoul(dup, 0, 16);
    }
#else
    u.bits.frac = strtoul(f, 0, 16);
#endif

    return u.flt;
}

/*======================================================================*/

fpu_control_t fcw0, fcw1;

void test_float_arith(float (*tf)(float, float),
		      float (*rf)(float, float),
		      float x, float y)
{
    float tr, rr;
    rr = (*rf)(x, y);
    tr = (*tf)(x, y);
    if (memcmp(&tr, &rr, sizeof(float)) != 0)
    {
	fputs("error:\n\tx     = ", stdout); print_float(x);
	fputs("\n\ty     = ", stdout); print_float(y);
	fputs("\n\ttrue  = ", stdout); print_float(rr);
	fputs("\n\tfalse = ", stdout); print_float(tr);
	putchar('\n');
    }
}

void test_double_arith(double (*tf)(double, double),
		       double (*rf)(double, double),
		       double x, double y)
{
    double tr, rr;
#ifdef __i386__
    /* Don't worry.  Even this does not make it error free
       on ia32.  If the result is denormal,  it will not
       honour the double precision and generate bad results
       anyway.  On the other side,  who wants to use ia32
       for IEEE math?  I don't.  */
    _FPU_GETCW(fcw0);
    fcw1 = ((fcw0 & ~_FPU_EXTENDED) | _FPU_DOUBLE);
    _FPU_SETCW(fcw1);
#endif
    rr = (*rf)(x, y);
#ifdef __i386__
    _FPU_SETCW(fcw0);
#endif
    tr = (*tf)(x, y);
    if (memcmp(&tr, &rr, sizeof(double)) != 0)
    {
	fputs("error:\n\tx     = ", stdout); print_double(x);
	fputs("\n\ty     = ", stdout); print_double(y);
	fputs("\n\ttrue  = ", stdout); print_double(rr);
	fputs("\n\tfalse = ", stdout); print_double(tr);
	putchar('\n');
    }
}

void test_float_double_conv(float x)
{
    double tr, rr;
    rr = r_extendsfdf2(x);
    tr = __extendsfdf2(x);
    if (memcmp(&tr, &rr, sizeof(double)) != 0)
    {
	fputs("error:\n\tx     = ", stdout); print_float(x);
	fputs("\n\ttrue  = ", stdout); print_double(rr);
	fputs("\n\tfalse = ", stdout); print_double(tr);
	putchar('\n');
    }
}

void test_double_float_conv(double x)
{
    float tr, rr;
    rr = r_truncdfsf2(x);
    tr = __truncdfsf2(x);
    if (memcmp(&tr, &rr, sizeof(float)) != 0)
    {
	fputs("error:\n\tx     = ", stdout); print_double(x);
	fputs("\n\ttrue  = ", stdout); print_float(rr);
	fputs("\n\tfalse = ", stdout); print_float(tr);
	putchar('\n');
    }
}

void test_int_float_conv(int x)
{
    float tr, rr;
    rr = r_floatsisf(x);
    tr = __floatsisf(x);
    if (memcmp(&tr, &rr, sizeof(float)) != 0)
    {
	printf("error\n\tx     = %d", x);
	fputs("\n\ttrue  = ", stdout); print_float(rr);
	fputs("\n\tfalse = ", stdout); print_float(tr);
	putchar('\n');
    }
}

void test_int_double_conv(int x)
{
    double tr, rr;
    rr = r_floatsidf(x);
    tr = __floatsidf(x);
    if (memcmp(&tr, &rr, sizeof(double)) != 0)
    {
	printf("error\n\tx     = %d", x);
	fputs("\n\ttrue  = ", stdout); print_double(rr);
	fputs("\n\tfalse = ", stdout); print_double(tr);
	putchar('\n');
    }
}

void test_float_int_conv(float x)
{
    int tr, rr;
    rr = r_fixsfsi(x);
    tr = __fixsfsi(x);
    if (rr != tr)
    {
	fputs("error:\n\tx     = ", stdout); print_float(x);
	printf("\n\ttrue  = %d\n\tfalse = %d\n", rr, tr);
    }
}

void test_double_int_conv(double x)
{
    int tr, rr;
    rr = r_fixsfsi(x);
    tr = __fixsfsi(x);
    if (rr != tr)
    {
	fputs("error:\n\tx     = ", stdout); print_double(x);
	printf("\n\ttrue  = %d\n\tfalse = %d\n", rr, tr);
    }
}

int eq0(int x) { return x == 0; }
int ne0(int x) { return x != 0; }
int le0(int x) { return x <= 0; }
int lt0(int x) { return x < 0; }
int ge0(int x) { return x >= 0; }
int gt0(int x) { return x > 0; }

void test_float_cmp(int (*tf)(float, float),
		    int (*rf)(float, float),
		    int (*cmp0)(int),
		    float x, float y)
{
    int tr, rr;
    rr = (*rf)(x, y);
    tr = (*tf)(x, y);
    if (cmp0(rr) != cmp0(tr))
    {
	fputs("error:\n\tx     = ", stdout); print_float(x);
	fputs("\n\ty     = ", stdout); print_float(y);
	printf("\n\ttrue  = %d\n\tfalse = %d\n", rr, tr);
    }
}

void test_double_cmp(int (*tf)(double, double),
		     int (*rf)(double, double),
		     int (*cmp0)(int),
		     double x, double y)
{
    int tr, rr;
    rr = (*rf)(x, y);
    tr = (*tf)(x, y);
    if (cmp0(rr) != cmp0(tr))
    {
	fputs("error:\n\tx     = ", stdout); print_double(x);
	fputs("\n\ty     = ", stdout); print_double(y);
	printf("\n\ttrue  = %d\n\tfalse = %d\n", rr, tr);
    }
}


/*======================================================================*/


int main(int ac, char **av)
{
#ifdef __alpha__
    __ieee_set_fp_control(0);
#endif
    av++, ac--;
    switch (*(*av)++)
    {
	{
	    float (*r)(float, float);
	    float (*t)(float, float);

	    do {
	      case 'a': r = r_addsf3; t = __addsf3; break;
	      case 's': r = r_subsf3; t = __subsf3; break;
	      case 'm': r = r_mulsf3; t = __mulsf3; break;
	      case 'd': r = r_divsf3; t = __divsf3; break;
	      case 'r': r = r_sqrtsf3; t = __sqrtsf3; break;
	      case 'j': r = r_negsf3; t = __negsf3; break;
	    } while (0);

	    switch (*(*av)++)
	    {
	      case 'n':
		{
		    int count = (ac > 1 ? atoi(av[1]) : 100);
		    while (count--)
			test_float_arith(t, r, rand_float(), rand_float());
		}
		break;

	      case 's':
		{
		    int i, j;
		    for (i = 0; i < NSPECIALS; i++)
			for (j = 0; j < NSPECIALS; j++)
			    test_float_arith(t, r, gen_special_float(i),
					      gen_special_float(j));
		}
		break;

	      case 0:
		if (ac < 7) abort();
		test_float_arith(t, r, build_float(av[1], av[2], av[3]),
				 build_float(av[4], av[5], av[6]));
		break;
	    }
	}
	break;

	{
	    double (*r)(double, double);
	    double (*t)(double, double);

	    do {
	      case 'A': r = r_adddf3; t = __adddf3; break;
	      case 'S': r = r_subdf3; t = __subdf3; break;
	      case 'M': r = r_muldf3; t = __muldf3; break;
	      case 'D': r = r_divdf3; t = __divdf3; break;
	      case 'R': r = r_sqrtdf3; t = __sqrtdf3; break;
	      case 'J': r = r_negdf3; t = __negdf3; break;
	    } while (0);

	    switch (*(*av)++)
	    {
	      case 'n':
		{
		    int count = (ac > 1 ? atoi(av[1]) : 100);
		    while (count--)
			test_double_arith(t, r, rand_double(), rand_double());
		}
		break;

	      case 's':
		{
		    int i, j;
		    for (i = 0; i < NSPECIALS; i++)
			for (j = 0; j < NSPECIALS; j++)
			    test_double_arith(t, r, gen_special_double(i),
					      gen_special_double(j));
		}
		break;

	      case 0:
		if (ac < 7) abort();
		test_double_arith(t, r, build_double(av[1], av[2], av[3]),
				  build_double(av[4], av[5], av[6]));
		break;
	    }
	}
	break;

      case 'c':
	switch (*(*av)++)
	{
	  case 'n':
	    {
		int count = (ac > 1 ? atoi(av[1]) : 100);
		while (count--)
		    test_float_double_conv(rand_float());
	    }
	    break;

	  case 's':
	    {
		int i;
		for (i = 0; i < NSPECIALS; i++)
		    test_float_double_conv(gen_special_float(i));
	    }
	    break;

	  case 0:
	    if (ac < 4) abort();
	    test_float_double_conv(build_float(av[1], av[2], av[3]));
	    break;
	}
	break;

      case 'C':
	switch (*(*av)++)
	{
	  case 'n':
	    {
		int count = (ac > 1 ? atoi(av[1]) : 100);
		while (count--)
		    test_double_float_conv(rand_double());
	    }
	    break;

	  case 's':
	    {
		int i;
		for (i = 0; i < NSPECIALS; i++)
		    test_double_float_conv(gen_special_double(i));
	    }
	    break;

	  case 0:
	    if (ac < 4) abort();
	    test_double_float_conv(build_double(av[1], av[2], av[3]));
	    break;
	}
	break;

      case 'i':
	switch (*(*av)++)
	{
	  case 'n':
	    {
		int count = (ac > 1 ? atoi(av[1]) : 100);
		while (count--)
		    test_int_float_conv(lrand48() << 1);
	    }
	    break;

	  case 0:
	    if (ac < 2) abort();
	    test_int_float_conv(strtol(av[1], 0, 0));
	    break;
	}
	break;

      case 'I':
	switch (*(*av)++)
	{
	  case 'n':
	    {
		int count = (ac > 1 ? atoi(av[1]) : 100);
		while (count--)
		    test_int_double_conv(lrand48() << 1);
	    }
	    break;

	  case 0:
	    if (ac < 2) abort();
	    test_int_double_conv(strtol(av[1], 0, 0));
	    break;
	}
	break;

      case 'f':
	switch (*(*av)++)
	{
	  case 'n':
	    {
		int count = (ac > 1 ? atoi(av[1]) : 100);
		while (count--)
		    test_float_int_conv(rand_float());
	    }
	    break;

	  case 's':
	    {
		int i;
		for (i = 0; i < NSPECIALS; i++)
		    test_float_int_conv(gen_special_float(i));
	    }
	    break;

	  case 0:
	    if (ac < 4) abort();
	    test_float_int_conv(build_float(av[1], av[2], av[3]));
	    break;
	}
	break;

      case 'F':
	switch (*(*av)++)
	{
	  case 'n':
	    {
		int count = (ac > 1 ? atoi(av[1]) : 100);
		while (count--)
		    test_double_int_conv(rand_double());
	    }
	    break;

	  case 's':
	    {
		int i;
		for (i = 0; i < NSPECIALS; i++)
		    test_double_int_conv(gen_special_double(i));
	    }
	    break;

	  case 0:
	    if (ac < 4) abort();
	    test_double_int_conv(build_double(av[1], av[2], av[3]));
	    break;
	}
	break;

	{
	    int (*r)(float, float);
	    int (*t)(float, float);
	    int (*c)(int);

	    do {
	      case 'e': r = r_eqsf2; t = __eqsf2; c = eq0; break;
	      case 'n': r = r_nesf2; t = __nesf2; c = ne0; break;
	      case 'l':
		switch (*(*av)++)
		{
		  case 'e': r = r_lesf2; t = __lesf2; c = le0; break;
		  case 't': r = r_ltsf2; t = __ltsf2; c = lt0; break;
		}
		break;
	      case 'g':
		switch (*(*av)++)
		{
		  case 'e': r = r_gesf2; t = __gesf2; c = ge0; break;
		  case 't': r = r_gtsf2; t = __gtsf2; c = gt0; break;
		}
		break;
	    } while (0);

	    switch (*(*av)++)
	    {
	      case 'n':
		{
		    int count = (ac > 1 ? atoi(av[1]) : 100);
		    while (count--)
			test_float_cmp(t, r, c, rand_float(), rand_float());
		}
		break;

	      case 's':
		{
		    int i, j;
		    for (i = 0; i < NSPECIALS; i++)
			for (j = 0; j < NSPECIALS; j++)
			    test_float_cmp(t, r, c, gen_special_float(i),
					   gen_special_float(j));
		}
		break;

	      case 0:
		if (ac < 7) abort();
		test_float_cmp(t, r, c, build_float(av[1], av[2], av[3]),
				build_float(av[4], av[5], av[6]));
		break;
	    }
	}
	break;

	{
	    int (*r)(double, double);
	    int (*t)(double, double);
	    int (*c)(int);

	    do {
	      case 'E': r = r_eqdf2; t = __eqdf2; c = eq0; break;
	      case 'N': r = r_nedf2; t = __nedf2; c = ne0; break;
	      case 'L':
		switch (*(*av)++)
		{
		  case 'E': r = r_ledf2; t = __ledf2; c = le0; break;
		  case 'T': r = r_ltdf2; t = __ltdf2; c = lt0; break;
		}
		break;
	      case 'G':
		switch (*(*av)++)
		{
		  case 'E': r = r_gedf2; t = __gedf2; c = ge0; break;
		  case 'T': r = r_gtdf2; t = __gtdf2; c = gt0; break;
		}
		break;
	    } while (0);

	    switch (*(*av)++)
	    {
	      case 'n':
		{
		    int count = (ac > 1 ? atoi(av[1]) : 100);
		    while (count--)
			test_double_cmp(t, r, c, rand_double(), rand_double());
		}
		break;

	      case 's':
		{
		    int i, j;
		    for (i = 0; i < NSPECIALS; i++)
			for (j = 0; j < NSPECIALS; j++)
			    test_double_cmp(t, r, c, gen_special_double(i),
					    gen_special_double(j));
		}
		break;

	      case 0:
		if (ac < 7) abort();
		test_double_cmp(t, r, c, build_double(av[1], av[2], av[3]),
				build_double(av[4], av[5], av[6]));
		break;
	    }
	}
	break;

      default:
	abort();
    }

    return 0;
}
