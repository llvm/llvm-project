/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief  Set ieee floating point environment.
 */

#if defined(TARGET_LINUX)
# define _GNU_SOURCE
#endif

#include <stdint.h>

#if (defined(TARGET_X8664) || defined(TARGET_X86) || defined(X86)) 

/* These routines are included in linux and osx.
   Plus, we can standardize our support of F2003 ieee_exceptions
   and ieee_arithmetic modules across all platforms
*/

typedef struct {
  unsigned int mxcsr;
  unsigned int x87cw;
  unsigned int x87sw;
} fenv_t;

typedef unsigned int fexcept_t;

#define IS_SSE_ENABLED 1
#define IS_SSE2_ENABLED 1

typedef union {
  int w;
  struct {
    unsigned int ie : 1;
    unsigned int de : 1;
    unsigned int ze : 1;
    unsigned int oe : 1;
    unsigned int ue : 1;
    unsigned int pe : 1;
    unsigned int rs2 : 2;
    unsigned int pc : 2; /* 0 - 32
                          * 1 - reserved
                          * 2 - 64
                          * 3 - 80
                          */
    unsigned int rc : 2; /* 0 - round to nearest
                          * 1 - round down
                          * 2 - round up
                          * 3 - chop
                          */
    unsigned int ic : 1;
    unsigned int rs3 : 3;
  } xbits;
} FCW;

typedef union {
  int w;
  struct {
    unsigned int ie : 1;
    unsigned int de : 1;
    unsigned int ze : 1;
    unsigned int oe : 1;
    unsigned int ue : 1;
    unsigned int pe : 1;
    unsigned int daz : 1;
    unsigned int iem : 1;
    unsigned int dem : 1;
    unsigned int zem : 1;
    unsigned int oem : 1;
    unsigned int uem : 1;
    unsigned int pem : 1;
    unsigned int rc : 2; /* 0 - round to nearest
                          * 1 - round down
                          * 2 - round up
                          * 3 - chop
                          */
    unsigned int fz : 1;
  } mbits;
} MXCSR;

int
__fenv_fegetround(void)
{
  FCW x87;
  MXCSR sse;

  int rmode_x87;
  int rmode_sse;
  rmode_sse = 0;

  asm("\tfnstcw %0" : "=m"(x87) :);
  rmode_x87 = x87.xbits.rc;
  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    rmode_sse = sse.mbits.rc;
  }
  return ((rmode_x87 | rmode_sse) << 10);
}

int
__fenv_fesetround(int rmode)
{
  FCW x87;
  MXCSR sse;

  asm("\tfnstcw %0" : "=m"(x87) :);
  x87.xbits.rc = (rmode >> 10);
  asm("\tfldcw %0" ::"m"(x87));
  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse.mbits.rc = (rmode >> 10);
    asm("\tldmxcsr %0" ::"m"(sse));
  }
  return 0;
}

int
__fenv_fegetexceptflag(fexcept_t *flagp, int exc)
{
  int x87;
  int sse;
  sse = 0;

  asm("\tfnstsw %0" : "=m"(x87) :);
  x87 &= exc;
  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse &= exc;
    x87 = 0;
  }
  *flagp = ((x87 | sse) & 63);
  return 0;
}

int
__fenv_fesetexceptflag(fexcept_t *flagp, int exc)
{
  unsigned int x87[7];
  unsigned int sse;
  unsigned int uexc;

  uexc = exc & 63;
  asm("\tfnstenv %0" : "=m"(x87[0]) :);
  x87[1] &= ~uexc;
  x87[1] |= (uexc & *flagp);
  asm("\tfldenv %0\n\tfwait" ::"m"(x87[0]));
  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse &= ~uexc;
    sse |= (uexc & *flagp);
    asm("\tldmxcsr %0" ::"m"(sse));
  }
  return 0;
}

int
__fenv_fetestexcept(int exc)
{
  int x87;
  int sse;
  sse = 0;

/* Windows doesn't seem to preserve x87 exception bits across context
 * switches, so this info is unreliable.
 */
#if defined(_WIN64)
  x87 = 0;
#else
  asm("\tfnstsw %0" : "=m"(x87) :);
  x87 &= exc;
#endif
  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse &= exc;
  }
  return ((x87 | sse) & 63);
}

int
__fenv_feclearexcept(int exc)
{
  unsigned int x87[7];
  int sse;
  unsigned int uexc;

  uexc = exc & 63;
  asm("\tfnstenv %0" : "=m"(x87[0]) :);
  x87[1] &= ~uexc;
  asm("\tfldenv %0\n\tfwait" ::"m"(x87[0]));

  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse &= ~uexc;
    asm("\tldmxcsr %0" ::"m"(sse));
  }
  return 0;
}

int
__fenv_feclearx87except(int exc)
{
  unsigned int x87[7];
  int sse;
  unsigned int uexc;

  uexc = exc & 63;
  asm("\tfnstenv %0" : "=m"(x87[0]) :);
  x87[1] &= ~uexc;
  asm("\tfldenv %0\n\tfwait" ::"m"(x87[0]));
  return 0;
}

int
__fenv_feraiseexcept(int exc)
{
  unsigned int x87[7];
  int sse;

  exc &= 63;
  asm("\tfnstenv %0" : "=m"(x87[0]) :);
  x87[1] |= exc;
  asm("\tfldenv %0\n\tfwait" ::"m"(x87[0]));

  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse |= exc;
    asm("\tldmxcsr %0" ::"m"(sse));
  }
  return 0;
}

int
__fenv_feenableexcept(int exc)
{
  unsigned int x87;
  unsigned int sse;
  unsigned int uexc;

  uexc = exc & 63;
  asm("\tfnstcw %0" : "=m"(x87) :);
  x87 &= ~uexc;
  asm("\tfldcw %0" ::"m"(x87));
  if (IS_SSE_ENABLED) {
    uexc = ((exc & 63) << 7);
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse &= ~uexc;
    asm("\tldmxcsr %0" ::"m"(sse));
  }
  return 0;
}

int
__fenv_fedisableexcept(int exc)
{
  int x87;
  int sse;

  asm("\tfnstcw %0" : "=m"(x87) :);
  x87 |= (exc & 63);
  asm("\tfldcw %0" ::"m"(x87));
  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse |= ((exc & 63) << 7);
    asm("\tldmxcsr %0" ::"m"(sse));
  }
  return 0;
}

int
__fenv_fegetexcept(void)
{
  int x87;
  int sse;

  sse = 0;
  asm("\tfnstcw %0" : "=m"(x87) :);
  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse = sse >> 7;
    x87 = 0;
  }
  return (63 - ((x87 | sse) & 63));
}

int
__fenv_fegetenv(fenv_t *env)
{
  unsigned int fcw;
  unsigned int x87[7];
  unsigned int sse;

  asm("\tfnstcw %0" : "=m"(fcw) :);
  env->x87cw = fcw;
  asm("\tfnstenv %0" : "=m"(x87[0]) :);
  env->x87sw = x87[1];
  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    env->mxcsr = sse;
  }
  return 0;
}

int
__fenv_feholdexcept(fenv_t *env)
{
  unsigned int fcw;
  unsigned int x87[7];
  unsigned int sse;
  unsigned int uexc;

  asm("\tfnstcw %0" : "=m"(fcw) :);
  env->x87cw = fcw;
  asm("\tfnstenv %0" : "=m"(x87[0]) :);
  env->x87sw = x87[1];
  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    env->mxcsr = sse;
  }
  uexc = 63;
  fcw |= uexc;
  asm("\tfldcw %0" ::"m"(fcw));
  x87[1] &= ~uexc;
  asm("\tfldenv %0" ::"m"(x87[0]));

  if (IS_SSE_ENABLED) {
    sse &= ~uexc;
    sse |= (uexc << 7);
    asm("\tldmxcsr %0" ::"m"(sse));
  }
  return 0;
}

int
__fenv_fesetenv(fenv_t *env)
{
  unsigned int fcw;
  unsigned int x87[7];
  unsigned int sse;
  unsigned int uexc;

  asm("\tfnstcw %0" : "=m"(fcw) :);
  fcw = fcw & 0xFFFFF0C0;
  fcw = fcw | (env->x87cw & 0x00000F3F);
  asm("\tfldcw %0" ::"m"(fcw));

  asm("\tfnstenv %0" : "=m"(x87[0]) :);
  x87[1] = x87[1] & 0xFFFFFFC0;
  x87[1] = x87[1] | (env->x87sw & 0x0000003F);
  asm("\tfldenv %0" ::"m"(x87[0]));

  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse = sse & 0xFFFF8040;
    sse = sse | (env->mxcsr & 0x00007FBF);
    asm("\tldmxcsr %0" ::"m"(sse));
  }
  return 0;
}

int
__fenv_feupdateenv(fenv_t *env)
{
  unsigned int fcw;
  unsigned int x87[7];
  unsigned int sse;
  unsigned int uexc;

  asm("\tfnstcw %0" : "=m"(fcw) :);
  fcw = fcw & 0xFFFFF0C0;
  fcw = fcw | (env->x87cw & 0x00000F3F);
  asm("\tfldcw %0" ::"m"(fcw));

  asm("\tfnstenv %0" : "=m"(x87[0]) :);
  x87[1] = x87[1] | (env->x87sw & 0x0000003F);
  asm("\tfldenv %0" ::"m"(x87[0]));

  if (IS_SSE_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse = sse & 0xFFFF807F;
    sse = sse | (env->mxcsr & 0x00007FBF);
    asm("\tldmxcsr %0" ::"m"(sse));
  }
  return 0;
}

/** \brief Set (flush to zero) underflow mode
 *
 * \param uflow zero to allow denorm numbers,
 *              non-zero integer to flush to zero
 *
 * \return zero (?)
 */
int
__fenv_fesetzerodenorm(int uflow)
{
  unsigned int sse;
  unsigned int uexc;

  if (IS_SSE2_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    uexc = (1 << 15) | (1 << 6);
    sse = sse & ~uexc;
    uexc = uflow ? 1 : 0;
    sse = sse | (uexc << 15);
    sse = sse | (uexc << 6);
    asm("\tldmxcsr %0" ::"m"(sse));
  }
  return 0;
}

/** \brief Get (flush to zero) underflow mode
 *
 * \return 1 if flush to zero is set, 0 otherwise
 */
int
__fenv_fegetzerodenorm(void)
{
  unsigned int sse;
  sse = 0;
  if (IS_SSE2_ENABLED) {
    asm("\tstmxcsr %0" : "=m"(sse) :);
    sse = ((sse >> 15) | (sse >> 6)) & 1;
  }
  return sse;
}

/** \brief
 * Mask mxcsr, e.g., a value of 0xffff7fbf says to clear FZ and DAZ
 * (i.e., enable 'full' denorm support).
 *
 * Save the current value of the mxcsr if requested.
 * Note this routine will only be called by the compiler for SSE2 or
 * better targets.
 */
void
__fenv_mask_mxcsr(int mask, int *psv)
{
  int tmp;
  asm("\tstmxcsr %0" : "=m"(tmp) :);
  if (psv)
    *psv = tmp;
  tmp &= mask;
  asm("\tldmxcsr %0" ::"m"(tmp));
  return;
}

/** \brief
 * Restore the current value of the mxcsr.
 */
void
__fenv_restore_mxcsr(int sv)
{
  asm("\tldmxcsr %0" ::"m"(sv));
  return;
}

#else   /* #if (defined(TARGET_X8664) || defined(TARGET_X86) || defined(X86)) */

/*
 * aarch64 and POWER (not X86-64).
 *
 * Without loss of generality, use libc's implemenations of floating point
 * control/status get/set operations.
 */

#include <fenv.h>

int
__fenv_fegetround(void)
{
  return fegetround();
}

int
__fenv_fesetround(int rmode)
{
  return fesetround(rmode);
}

int
__fenv_fegetexceptflag(fexcept_t *flagp, int exc)
{
  return fegetexceptflag(flagp, exc);
}

int
__fenv_fesetexceptflag(fexcept_t *flagp, int exc)
{
  return fesetexceptflag(flagp, exc);
}

int
__fenv_fetestexcept(int exc)
{
  return fetestexcept(exc);
}

int
__fenv_feclearexcept(int exc)
{
  return feclearexcept(exc);
}

int
__fenv_feraiseexcept(int exc)
{
  return feraiseexcept(exc);
}
#if defined(TARGET_WIN_ARM64)
/* TODO: Implement and test these functions */
int
__fenv_feenableexcept(int exc)
{
  // implement function.
  return 0;
}

int
__fenv_fedisableexcept(int exc)
{
  // implement function.
  return 0;
}

int
__fenv_fegetexcept(void)
{
  // implement function.
  return 0;
}

#else
int
__fenv_feenableexcept(int exc)
{
  return feenableexcept(exc);
}

int
__fenv_fedisableexcept(int exc)
{
  return fedisableexcept(exc);
}

int
__fenv_fegetexcept(void)
{
  return fegetexcept();
}
#endif
int
__fenv_fegetenv(fenv_t *env)
{
  return fegetenv(env);
}

int
__fenv_feholdexcept(fenv_t *env)
{
  return feholdexcept(env);
}

int
__fenv_fesetenv(fenv_t *env)
{
  return fesetenv(env);
}

int
__fenv_feupdateenv(fenv_t *env)
{
  return feupdateenv(env);
}

#if     defined(TARGET_LINUX_ARM)

/*
 * ARM aarch64.
 * Does implement __fenv_fesetzerodenorm() and __fenv_fegetzerodenorm.
 *
 * Additional compiler support routines:
 * __fenv_mask_fz() and __fenv_restore_fz().
 */

#include <fpu_control.h>

/** \brief Set (flush to zero) underflow mode
 *
 * \param uflow zero to allow denorm numbers,
 *              non-zero integer to flush to zero
 */
int
__fenv_fesetzerodenorm(int uflow)
{
  uint64_t cw;

  _FPU_GETCW(cw);
  if (uflow)
    cw |= (1ULL << 24);
  else
    cw &= ~(1ULL << 24);
  _FPU_SETCW(cw);
  return 0;
}

/** \brief Get (flush to zero) underflow mode
 *
 * \return 1 if flush to zero is set, 0 otherwise
 */
int
__fenv_fegetzerodenorm(void)
{
  uint64_t cw;

  _FPU_GETCW(cw);
  return (cw & (1ULL << 24)) ? 1 : 0;
}

/** \brief
 * Mask fz bit of fpcr, e.g., a value of 0x0 says to clear FZ
 * (i.e., enable 'full' denorm support).
 *
 * Save the current value of the fpcr.fz if requested.
 * Note this routine will only be called by the compiler for
 * better targets.
 */
void
__fenv_mask_fz(int mask, int *psv)
{
  uint64_t tmp;

  _FPU_GETCW(tmp);
  if (psv)
    *psv = ((tmp & (1ULL << 24)) ? 1 : 0);
  if (mask)
    tmp |= (1ULL << 24);
  else
    tmp &= ~(1ULL << 24);
  _FPU_SETCW(tmp);
}

/** \brief
 * Restore the current value of the fpcr.fz.
 */
void
__fenv_restore_fz(int sv)
{
  uint64_t tmp;

  _FPU_GETCW(tmp);
  if (sv)
    tmp |= (1ULL << 24);
  else
    tmp &= ~(1ULL << 24);
  _FPU_SETCW(tmp);
}

#else

/*
 * Other architectures - currently only POWER.
 * Stub out __fenv_fesetzerodenorm() and __fenv_fegetzerodenorm().
 */

/** \brief Unimplemented: Set (flush to zero) underflow mode
 *
 * \param uflow zero to allow denorm numbers,
 *              non-zero integer to flush to zero
 */
int
__fenv_fesetzerodenorm(int uflow)
{
  return 0;
}

/** \brief Unimplemented: Get (flush to zero) underflow mode
 *
 * \return 1 if flush to zero is set, 0 otherwise
 */
int
__fenv_fegetzerodenorm(void)
{
  return 0;
}

#endif  /* #if     ! defined(TARGET_LINUX_ARM) */

#endif  /* #if (defined(TARGET_X8664) || defined(TARGET_X86) || defined(X86)) */
