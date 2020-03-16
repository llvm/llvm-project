/*
 * ULP error checking tool for math functions.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <ctype.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mathlib.h"

/* Don't depend on mpfr by default.  */
#ifndef USE_MPFR
# define USE_MPFR 0
#endif
#if USE_MPFR
# include <mpfr.h>
#endif

#ifndef WANT_VMATH
/* Enable the build of vector math code.  */
# define WANT_VMATH 1
#endif

static inline uint64_t
asuint64 (double f)
{
  union
  {
    double f;
    uint64_t i;
  } u = {f};
  return u.i;
}

static inline double
asdouble (uint64_t i)
{
  union
  {
    uint64_t i;
    double f;
  } u = {i};
  return u.f;
}

static inline uint32_t
asuint (float f)
{
  union
  {
    float f;
    uint32_t i;
  } u = {f};
  return u.i;
}

static inline float
asfloat (uint32_t i)
{
  union
  {
    uint32_t i;
    float f;
  } u = {i};
  return u.f;
}

static uint64_t seed = 0x0123456789abcdef;
static uint64_t
rand64 (void)
{
  seed = 6364136223846793005ull * seed + 1;
  return seed ^ (seed >> 32);
}

/* Uniform random in [0,n].  */
static uint64_t
randn (uint64_t n)
{
  uint64_t r, m;

  if (n == 0)
    return 0;
  n++;
  if (n == 0)
    return rand64 ();
  for (;;)
    {
      r = rand64 ();
      m = r % n;
      if (r - m <= -n)
	return m;
    }
}

struct gen
{
  uint64_t start;
  uint64_t len;
  uint64_t start2;
  uint64_t len2;
  uint64_t off;
  uint64_t step;
  uint64_t cnt;
};

struct args_f1
{
  float x;
};

struct args_f2
{
  float x;
  float x2;
};

struct args_d1
{
  double x;
};

struct args_d2
{
  double x;
  double x2;
};

/* result = y + tail*2^ulpexp.  */
struct ret_f
{
  float y;
  double tail;
  int ulpexp;
  int ex;
  int ex_may;
};

struct ret_d
{
  double y;
  double tail;
  int ulpexp;
  int ex;
  int ex_may;
};

static inline uint64_t
next1 (struct gen *g)
{
  /* For single argument use randomized incremental steps,
     that produce dense sampling without collisions and allow
     testing all inputs in a range.  */
  uint64_t r = g->start + g->off;
  g->off += g->step + randn (g->step / 2);
  if (g->off > g->len)
    g->off -= g->len; /* hack.  */
  return r;
}

static inline uint64_t
next2 (uint64_t *x2, struct gen *g)
{
  /* For two arguments use uniform random sampling.  */
  uint64_t r = g->start + randn (g->len);
  *x2 = g->start2 + randn (g->len2);
  return r;
}

static struct args_f1
next_f1 (void *g)
{
  return (struct args_f1){asfloat (next1 (g))};
}

static struct args_f2
next_f2 (void *g)
{
  uint64_t x2;
  uint64_t x = next2 (&x2, g);
  return (struct args_f2){asfloat (x), asfloat (x2)};
}

static struct args_d1
next_d1 (void *g)
{
  return (struct args_d1){asdouble (next1 (g))};
}

static struct args_d2
next_d2 (void *g)
{
  uint64_t x2;
  uint64_t x = next2 (&x2, g);
  return (struct args_d2){asdouble (x), asdouble (x2)};
}

struct conf
{
  int r;
  int rc;
  int quiet;
  int mpfr;
  int fenv;
  unsigned long long n;
  double softlim;
  double errlim;
};

/* Wrappers for sincos.  */
static float sincosf_sinf(float x) {(void)cosf(x); return sinf(x);}
static float sincosf_cosf(float x) {(void)sinf(x); return cosf(x);}
static double sincos_sin(double x) {(void)cos(x); return sin(x);}
static double sincos_cos(double x) {(void)sin(x); return cos(x);}
#if USE_MPFR
static int sincos_mpfr_sin(mpfr_t y, const mpfr_t x, mpfr_rnd_t r) { mpfr_cos(y,x,r); return mpfr_sin(y,x,r); }
static int sincos_mpfr_cos(mpfr_t y, const mpfr_t x, mpfr_rnd_t r) { mpfr_sin(y,x,r); return mpfr_cos(y,x,r); }
#endif

/* A bit of a hack: call vector functions twice with the same
   input in lane 0 but a different value in other lanes: once
   with an in-range value and then with a special case value.  */
static int secondcall;

/* Wrappers for vector functions.  */
#if __aarch64__ && WANT_VMATH
typedef __f32x4_t v_float;
typedef __f64x2_t v_double;
static const float fv[2] = {1.0f, -INFINITY};
static const double dv[2] = {1.0, -INFINITY};
static inline v_float argf(float x) { return (v_float){x,x,x,fv[secondcall]}; }
static inline v_double argd(double x) { return (v_double){x,dv[secondcall]}; }

static float v_sinf(float x) { return __v_sinf(argf(x))[0]; }
static float v_cosf(float x) { return __v_cosf(argf(x))[0]; }
static float v_expf_1u(float x) { return __v_expf_1u(argf(x))[0]; }
static float v_expf(float x) { return __v_expf(argf(x))[0]; }
static float v_exp2f_1u(float x) { return __v_exp2f_1u(argf(x))[0]; }
static float v_exp2f(float x) { return __v_exp2f(argf(x))[0]; }
static float v_logf(float x) { return __v_logf(argf(x))[0]; }
static float v_powf(float x, float y) { return __v_powf(argf(x),argf(y))[0]; }
static double v_sin(double x) { return __v_sin(argd(x))[0]; }
static double v_cos(double x) { return __v_cos(argd(x))[0]; }
static double v_exp(double x) { return __v_exp(argd(x))[0]; }
static double v_log(double x) { return __v_log(argd(x))[0]; }
static double v_pow(double x, double y) { return __v_pow(argd(x),argd(y))[0]; }
#ifdef __vpcs
static float vn_sinf(float x) { return __vn_sinf(argf(x))[0]; }
static float vn_cosf(float x) { return __vn_cosf(argf(x))[0]; }
static float vn_expf_1u(float x) { return __vn_expf_1u(argf(x))[0]; }
static float vn_expf(float x) { return __vn_expf(argf(x))[0]; }
static float vn_exp2f_1u(float x) { return __vn_exp2f_1u(argf(x))[0]; }
static float vn_exp2f(float x) { return __vn_exp2f(argf(x))[0]; }
static float vn_logf(float x) { return __vn_logf(argf(x))[0]; }
static float vn_powf(float x, float y) { return __vn_powf(argf(x),argf(y))[0]; }
static double vn_sin(double x) { return __vn_sin(argd(x))[0]; }
static double vn_cos(double x) { return __vn_cos(argd(x))[0]; }
static double vn_exp(double x) { return __vn_exp(argd(x))[0]; }
static double vn_log(double x) { return __vn_log(argd(x))[0]; }
static double vn_pow(double x, double y) { return __vn_pow(argd(x),argd(y))[0]; }
static float Z_sinf(float x) { return _ZGVnN4v_sinf(argf(x))[0]; }
static float Z_cosf(float x) { return _ZGVnN4v_cosf(argf(x))[0]; }
static float Z_expf(float x) { return _ZGVnN4v_expf(argf(x))[0]; }
static float Z_exp2f(float x) { return _ZGVnN4v_exp2f(argf(x))[0]; }
static float Z_logf(float x) { return _ZGVnN4v_logf(argf(x))[0]; }
static float Z_powf(float x, float y) { return _ZGVnN4vv_powf(argf(x),argf(y))[0]; }
static double Z_sin(double x) { return _ZGVnN2v_sin(argd(x))[0]; }
static double Z_cos(double x) { return _ZGVnN2v_cos(argd(x))[0]; }
static double Z_exp(double x) { return _ZGVnN2v_exp(argd(x))[0]; }
static double Z_log(double x) { return _ZGVnN2v_log(argd(x))[0]; }
static double Z_pow(double x, double y) { return _ZGVnN2vv_pow(argd(x),argd(y))[0]; }
#endif
#endif

struct fun
{
  const char *name;
  int arity;
  int singleprec;
  int twice;
  union
  {
    float (*f1) (float);
    float (*f2) (float, float);
    double (*d1) (double);
    double (*d2) (double, double);
  } fun;
  union
  {
    double (*f1) (double);
    double (*f2) (double, double);
    long double (*d1) (long double);
    long double (*d2) (long double, long double);
  } fun_long;
#if USE_MPFR
  union
  {
    int (*f1) (mpfr_t, const mpfr_t, mpfr_rnd_t);
    int (*f2) (mpfr_t, const mpfr_t, const mpfr_t, mpfr_rnd_t);
    int (*d1) (mpfr_t, const mpfr_t, mpfr_rnd_t);
    int (*d2) (mpfr_t, const mpfr_t, const mpfr_t, mpfr_rnd_t);
  } fun_mpfr;
#endif
};

static const struct fun fun[] = {
#if USE_MPFR
# define F(x, x_wrap, x_long, x_mpfr, a, s, t, twice) \
  {#x, a, s, twice, {.t = x_wrap}, {.t = x_long}, {.t = x_mpfr}},
#else
# define F(x, x_wrap, x_long, x_mpfr, a, s, t, twice) \
  {#x, a, s, twice, {.t = x_wrap}, {.t = x_long}},
#endif
#define F1(x) F (x##f, x##f, x, mpfr_##x, 1, 1, f1, 0)
#define F2(x) F (x##f, x##f, x, mpfr_##x, 2, 1, f2, 0)
#define D1(x) F (x, x, x##l, mpfr_##x, 1, 0, d1, 0)
#define D2(x) F (x, x, x##l, mpfr_##x, 2, 0, d2, 0)
 F1 (sin)
 F1 (cos)
 F (sincosf_sinf, sincosf_sinf, sincos_sin, sincos_mpfr_sin, 1, 1, f1, 0)
 F (sincosf_cosf, sincosf_cosf, sincos_cos, sincos_mpfr_cos, 1, 1, f1, 0)
 F1 (exp)
 F1 (exp2)
 F1 (log)
 F1 (log2)
 F2 (pow)
 D1 (exp)
 D1 (exp2)
 D1 (log)
 D1 (log2)
 D2 (pow)
#if WANT_VMATH
 F (__s_sinf, __s_sinf, sin, mpfr_sin, 1, 1, f1, 0)
 F (__s_cosf, __s_cosf, cos, mpfr_cos, 1, 1, f1, 0)
 F (__s_expf_1u, __s_expf_1u, exp, mpfr_exp, 1, 1, f1, 0)
 F (__s_expf, __s_expf, exp, mpfr_exp, 1, 1, f1, 0)
 F (__s_exp2f_1u, __s_exp2f_1u, exp2, mpfr_exp2, 1, 1, f1, 0)
 F (__s_exp2f, __s_exp2f, exp2, mpfr_exp2, 1, 1, f1, 0)
 F (__s_powf, __s_powf, pow, mpfr_pow, 2, 1, f2, 0)
 F (__s_logf, __s_logf, log, mpfr_log, 1, 1, f1, 0)
 F (__s_sin, __s_sin, sinl, mpfr_sin, 1, 0, d1, 0)
 F (__s_cos, __s_cos, cosl, mpfr_cos, 1, 0, d1, 0)
 F (__s_exp, __s_exp, expl, mpfr_exp, 1, 0, d1, 0)
 F (__s_log, __s_log, logl, mpfr_log, 1, 0, d1, 0)
 F (__s_pow, __s_pow, powl, mpfr_pow, 2, 0, d2, 0)
#if __aarch64__
 F (__v_sinf, v_sinf, sin, mpfr_sin, 1, 1, f1, 1)
 F (__v_cosf, v_cosf, cos, mpfr_cos, 1, 1, f1, 1)
 F (__v_expf_1u, v_expf_1u, exp, mpfr_exp, 1, 1, f1, 1)
 F (__v_expf, v_expf, exp, mpfr_exp, 1, 1, f1, 1)
 F (__v_exp2f_1u, v_exp2f_1u, exp2, mpfr_exp2, 1, 1, f1, 1)
 F (__v_exp2f, v_exp2f, exp2, mpfr_exp2, 1, 1, f1, 1)
 F (__v_logf, v_logf, log, mpfr_log, 1, 1, f1, 1)
 F (__v_powf, v_powf, pow, mpfr_pow, 2, 1, f2, 1)
 F (__v_sin, v_sin, sinl, mpfr_sin, 1, 0, d1, 1)
 F (__v_cos, v_cos, cosl, mpfr_cos, 1, 0, d1, 1)
 F (__v_exp, v_exp, expl, mpfr_exp, 1, 0, d1, 1)
 F (__v_log, v_log, logl, mpfr_log, 1, 0, d1, 1)
 F (__v_pow, v_pow, powl, mpfr_pow, 2, 0, d2, 1)
#ifdef __vpcs
 F (__vn_sinf, vn_sinf, sin, mpfr_sin, 1, 1, f1, 1)
 F (__vn_cosf, vn_cosf, cos, mpfr_cos, 1, 1, f1, 1)
 F (__vn_expf_1u, vn_expf_1u, exp, mpfr_exp, 1, 1, f1, 1)
 F (__vn_expf, vn_expf, exp, mpfr_exp, 1, 1, f1, 1)
 F (__vn_exp2f_1u, vn_exp2f_1u, exp2, mpfr_exp2, 1, 1, f1, 1)
 F (__vn_exp2f, vn_exp2f, exp2, mpfr_exp2, 1, 1, f1, 1)
 F (__vn_logf, vn_logf, log, mpfr_log, 1, 1, f1, 1)
 F (__vn_powf, vn_powf, pow, mpfr_pow, 2, 1, f2, 1)
 F (__vn_sin, vn_sin, sinl, mpfr_sin, 1, 0, d1, 1)
 F (__vn_cos, vn_cos, cosl, mpfr_cos, 1, 0, d1, 1)
 F (__vn_exp, vn_exp, expl, mpfr_exp, 1, 0, d1, 1)
 F (__vn_log, vn_log, logl, mpfr_log, 1, 0, d1, 1)
 F (__vn_pow, vn_pow, powl, mpfr_pow, 2, 0, d2, 1)
 F (_ZGVnN4v_sinf, Z_sinf, sin, mpfr_sin, 1, 1, f1, 1)
 F (_ZGVnN4v_cosf, Z_cosf, cos, mpfr_cos, 1, 1, f1, 1)
 F (_ZGVnN4v_expf, Z_expf, exp, mpfr_exp, 1, 1, f1, 1)
 F (_ZGVnN4v_exp2f, Z_exp2f, exp2, mpfr_exp2, 1, 1, f1, 1)
 F (_ZGVnN4v_logf, Z_logf, log, mpfr_log, 1, 1, f1, 1)
 F (_ZGVnN4vv_powf, Z_powf, pow, mpfr_pow, 2, 1, f2, 1)
 F (_ZGVnN2v_sin, Z_sin, sinl, mpfr_sin, 1, 0, d1, 1)
 F (_ZGVnN2v_cos, Z_cos, cosl, mpfr_cos, 1, 0, d1, 1)
 F (_ZGVnN2v_exp, Z_exp, expl, mpfr_exp, 1, 0, d1, 1)
 F (_ZGVnN2v_log, Z_log, logl, mpfr_log, 1, 0, d1, 1)
 F (_ZGVnN2vv_pow, Z_pow, powl, mpfr_pow, 2, 0, d2, 1)
#endif
#endif
#endif
#undef F
#undef F1
#undef F2
#undef D1
#undef D2
 {0}};

/* Boilerplate for generic calls.  */

static inline int
ulpscale_f (float x)
{
  int e = asuint (x) >> 23 & 0xff;
  if (!e)
    e++;
  return e - 0x7f - 23;
}
static inline int
ulpscale_d (double x)
{
  int e = asuint64 (x) >> 52 & 0x7ff;
  if (!e)
    e++;
  return e - 0x3ff - 52;
}
static inline float
call_f1 (const struct fun *f, struct args_f1 a)
{
  return f->fun.f1 (a.x);
}
static inline float
call_f2 (const struct fun *f, struct args_f2 a)
{
  return f->fun.f2 (a.x, a.x2);
}

static inline double
call_d1 (const struct fun *f, struct args_d1 a)
{
  return f->fun.d1 (a.x);
}
static inline double
call_d2 (const struct fun *f, struct args_d2 a)
{
  return f->fun.d2 (a.x, a.x2);
}
static inline double
call_long_f1 (const struct fun *f, struct args_f1 a)
{
  return f->fun_long.f1 (a.x);
}
static inline double
call_long_f2 (const struct fun *f, struct args_f2 a)
{
  return f->fun_long.f2 (a.x, a.x2);
}
static inline long double
call_long_d1 (const struct fun *f, struct args_d1 a)
{
  return f->fun_long.d1 (a.x);
}
static inline long double
call_long_d2 (const struct fun *f, struct args_d2 a)
{
  return f->fun_long.d2 (a.x, a.x2);
}
static inline void
printcall_f1 (const struct fun *f, struct args_f1 a)
{
  printf ("%s(%a)", f->name, a.x);
}
static inline void
printcall_f2 (const struct fun *f, struct args_f2 a)
{
  printf ("%s(%a, %a)", f->name, a.x, a.x2);
}
static inline void
printcall_d1 (const struct fun *f, struct args_d1 a)
{
  printf ("%s(%a)", f->name, a.x);
}
static inline void
printcall_d2 (const struct fun *f, struct args_d2 a)
{
  printf ("%s(%a, %a)", f->name, a.x, a.x2);
}
static inline void
printgen_f1 (const struct fun *f, struct gen *gen)
{
  printf ("%s in [%a;%a]", f->name, asfloat (gen->start),
	  asfloat (gen->start + gen->len));
}
static inline void
printgen_f2 (const struct fun *f, struct gen *gen)
{
  printf ("%s in [%a;%a] x [%a;%a]", f->name, asfloat (gen->start),
	  asfloat (gen->start + gen->len), asfloat (gen->start2),
	  asfloat (gen->start2 + gen->len2));
}
static inline void
printgen_d1 (const struct fun *f, struct gen *gen)
{
  printf ("%s in [%a;%a]", f->name, asdouble (gen->start),
	  asdouble (gen->start + gen->len));
}
static inline void
printgen_d2 (const struct fun *f, struct gen *gen)
{
  printf ("%s in [%a;%a] x [%a;%a]", f->name, asdouble (gen->start),
	  asdouble (gen->start + gen->len), asdouble (gen->start2),
	  asdouble (gen->start2 + gen->len2));
}

#define reduce_f1(a, f, op) (f (a.x))
#define reduce_f2(a, f, op) (f (a.x) op f (a.x2))
#define reduce_d1(a, f, op) (f (a.x))
#define reduce_d2(a, f, op) (f (a.x) op f (a.x2))

#ifndef IEEE_754_2008_SNAN
# define IEEE_754_2008_SNAN 1
#endif
static inline int
issignaling_f (float x)
{
  uint32_t ix = asuint (x);
  if (!IEEE_754_2008_SNAN)
    return (ix & 0x7fc00000) == 0x7fc00000;
  return 2 * (ix ^ 0x00400000) > 2u * 0x7fc00000;
}
static inline int
issignaling_d (double x)
{
  uint64_t ix = asuint64 (x);
  if (!IEEE_754_2008_SNAN)
    return (ix & 0x7ff8000000000000) == 0x7ff8000000000000;
  return 2 * (ix ^ 0x0008000000000000) > 2 * 0x7ff8000000000000ULL;
}

#if USE_MPFR
static mpfr_rnd_t
rmap (int r)
{
  switch (r)
    {
    case FE_TONEAREST:
      return MPFR_RNDN;
    case FE_TOWARDZERO:
      return MPFR_RNDZ;
    case FE_UPWARD:
      return MPFR_RNDU;
    case FE_DOWNWARD:
      return MPFR_RNDD;
    }
  return -1;
}

#define prec_mpfr_f 50
#define prec_mpfr_d 80
#define prec_f 24
#define prec_d 53
#define emin_f -148
#define emin_d -1073
#define emax_f 128
#define emax_d 1024
static inline int
call_mpfr_f1 (mpfr_t y, const struct fun *f, struct args_f1 a, mpfr_rnd_t r)
{
  MPFR_DECL_INIT (x, prec_f);
  mpfr_set_flt (x, a.x, MPFR_RNDN);
  return f->fun_mpfr.f1 (y, x, r);
}
static inline int
call_mpfr_f2 (mpfr_t y, const struct fun *f, struct args_f2 a, mpfr_rnd_t r)
{
  MPFR_DECL_INIT (x, prec_f);
  MPFR_DECL_INIT (x2, prec_f);
  mpfr_set_flt (x, a.x, MPFR_RNDN);
  mpfr_set_flt (x2, a.x2, MPFR_RNDN);
  return f->fun_mpfr.f2 (y, x, x2, r);
}
static inline int
call_mpfr_d1 (mpfr_t y, const struct fun *f, struct args_d1 a, mpfr_rnd_t r)
{
  MPFR_DECL_INIT (x, prec_d);
  mpfr_set_d (x, a.x, MPFR_RNDN);
  return f->fun_mpfr.d1 (y, x, r);
}
static inline int
call_mpfr_d2 (mpfr_t y, const struct fun *f, struct args_d2 a, mpfr_rnd_t r)
{
  MPFR_DECL_INIT (x, prec_d);
  MPFR_DECL_INIT (x2, prec_d);
  mpfr_set_d (x, a.x, MPFR_RNDN);
  mpfr_set_d (x2, a.x2, MPFR_RNDN);
  return f->fun_mpfr.d2 (y, x, x2, r);
}
#endif

#define float_f float
#define double_f double
#define copysign_f copysignf
#define nextafter_f nextafterf
#define fabs_f fabsf
#define asuint_f asuint
#define asfloat_f asfloat
#define scalbn_f scalbnf
#define lscalbn_f scalbn
#define halfinf_f 0x1p127f
#define min_normal_f 0x1p-126f

#define float_d double
#define double_d long double
#define copysign_d copysign
#define nextafter_d nextafter
#define fabs_d fabs
#define asuint_d asuint64
#define asfloat_d asdouble
#define scalbn_d scalbn
#define lscalbn_d scalbnl
#define halfinf_d 0x1p1023
#define min_normal_d 0x1p-1022

#define NEW_RT
#define RT(x) x##_f
#define T(x) x##_f1
#include "ulp.h"
#undef T
#define T(x) x##_f2
#include "ulp.h"
#undef T
#undef RT

#define NEW_RT
#define RT(x) x##_d
#define T(x) x##_d1
#include "ulp.h"
#undef T
#define T(x) x##_d2
#include "ulp.h"
#undef T
#undef RT

static void
usage (void)
{
  puts ("./ulp [-q] [-m] [-f] [-r nudz] [-l soft-ulplimit] [-e ulplimit] func "
	"lo [hi [x lo2 hi2] [count]]");
  puts ("Compares func against a higher precision implementation in [lo; hi].");
  puts ("-q: quiet.");
  puts ("-m: use mpfr even if faster method is available.");
  puts ("-f: disable fenv testing (rounding modes and exceptions).");
  puts ("Supported func:");
  for (const struct fun *f = fun; f->name; f++)
    printf ("\t%s\n", f->name);
  exit (1);
}

static int
cmp (const struct fun *f, struct gen *gen, const struct conf *conf)
{
  int r = 1;
  if (f->arity == 1 && f->singleprec)
    r = cmp_f1 (f, gen, conf);
  else if (f->arity == 2 && f->singleprec)
    r = cmp_f2 (f, gen, conf);
  else if (f->arity == 1 && !f->singleprec)
    r = cmp_d1 (f, gen, conf);
  else if (f->arity == 2 && !f->singleprec)
    r = cmp_d2 (f, gen, conf);
  else
    usage ();
  return r;
}

static uint64_t
getnum (const char *s, int singleprec)
{
  //	int i;
  uint64_t sign = 0;
  //	char buf[12];

  if (s[0] == '+')
    s++;
  else if (s[0] == '-')
    {
      sign = singleprec ? 1ULL << 31 : 1ULL << 63;
      s++;
    }
  /* 0xXXXX is treated as bit representation, '-' flips the sign bit.  */
  if (s[0] == '0' && tolower (s[1]) == 'x' && strchr (s, 'p') == 0)
    return sign ^ strtoull (s, 0, 0);
  //	/* SNaN, QNaN, NaN, Inf.  */
  //	for (i=0; s[i] && i < sizeof buf; i++)
  //		buf[i] = tolower(s[i]);
  //	buf[i] = 0;
  //	if (strcmp(buf, "snan") == 0)
  //		return sign | (singleprec ? 0x7fa00000 : 0x7ff4000000000000);
  //	if (strcmp(buf, "qnan") == 0 || strcmp(buf, "nan") == 0)
  //		return sign | (singleprec ? 0x7fc00000 : 0x7ff8000000000000);
  //	if (strcmp(buf, "inf") == 0 || strcmp(buf, "infinity") == 0)
  //		return sign | (singleprec ? 0x7f800000 : 0x7ff0000000000000);
  /* Otherwise assume it's a floating-point literal.  */
  return sign
	 | (singleprec ? asuint (strtof (s, 0)) : asuint64 (strtod (s, 0)));
}

static void
parsegen (struct gen *g, int argc, char *argv[], const struct fun *f)
{
  int singleprec = f->singleprec;
  int arity = f->arity;
  uint64_t a, b, a2, b2, n;
  if (argc < 1)
    usage ();
  b = a = getnum (argv[0], singleprec);
  n = 0;
  if (argc > 1 && strcmp (argv[1], "x") == 0)
    {
      argc -= 2;
      argv += 2;
    }
  else if (argc > 1)
    {
      b = getnum (argv[1], singleprec);
      if (argc > 2 && strcmp (argv[2], "x") == 0)
	{
	  argc -= 3;
	  argv += 3;
	}
    }
  b2 = a2 = getnum (argv[0], singleprec);
  if (argc > 1)
    b2 = getnum (argv[1], singleprec);
  if (argc > 2)
    n = strtoull (argv[2], 0, 0);
  if (argc > 3)
    usage ();
  //printf("ab %lx %lx ab2 %lx %lx n %lu\n", a, b, a2, b2, n);
  if (arity == 1)
    {
      g->start = a;
      g->len = b - a;
      if (n - 1 > b - a)
	n = b - a + 1;
      g->off = 0;
      g->step = n ? (g->len + 1) / n : 1;
      g->start2 = g->len2 = 0;
      g->cnt = n;
    }
  else if (arity == 2)
    {
      g->start = a;
      g->len = b - a;
      g->off = g->step = 0;
      g->start2 = a2;
      g->len2 = b2 - a2;
      g->cnt = n;
    }
  else
    usage ();
}

int
main (int argc, char *argv[])
{
  const struct fun *f;
  struct gen gen;
  struct conf conf;
  conf.rc = 'n';
  conf.quiet = 0;
  conf.mpfr = 0;
  conf.fenv = 1;
  conf.softlim = 0;
  conf.errlim = INFINITY;
  for (;;)
    {
      argc--;
      argv++;
      if (argc < 1)
	usage ();
      if (argv[0][0] != '-')
	break;
      switch (argv[0][1])
	{
	case 'e':
	  argc--;
	  argv++;
	  if (argc < 1)
	    usage ();
	  conf.errlim = strtod (argv[0], 0);
	  break;
	case 'f':
	  conf.fenv = 0;
	  break;
	case 'l':
	  argc--;
	  argv++;
	  if (argc < 1)
	    usage ();
	  conf.softlim = strtod (argv[0], 0);
	  break;
	case 'm':
	  conf.mpfr = 1;
	  break;
	case 'q':
	  conf.quiet = 1;
	  break;
	case 'r':
	  conf.rc = argv[0][2];
	  if (!conf.rc)
	    {
	      argc--;
	      argv++;
	      if (argc < 1)
		usage ();
	      conf.rc = argv[0][0];
	    }
	  break;
	default:
	  usage ();
	}
    }
  switch (conf.rc)
    {
    case 'n':
      conf.r = FE_TONEAREST;
      break;
    case 'u':
      conf.r = FE_UPWARD;
      break;
    case 'd':
      conf.r = FE_DOWNWARD;
      break;
    case 'z':
      conf.r = FE_TOWARDZERO;
      break;
    default:
      usage ();
    }
  for (f = fun; f->name; f++)
    if (strcmp (argv[0], f->name) == 0)
      break;
  if (!f->name)
    usage ();
  if (!f->singleprec && LDBL_MANT_DIG == DBL_MANT_DIG)
    conf.mpfr = 1; /* Use mpfr if long double has no extra precision.  */
  if (!USE_MPFR && conf.mpfr)
    {
      puts ("mpfr is not available.");
      return 0;
    }
  argc--;
  argv++;
  parsegen (&gen, argc, argv, f);
  conf.n = gen.cnt;
  return cmp (f, &gen, &conf);
}
