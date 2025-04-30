/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Definitions and declarations global to Hammer Code Generator
 */

#define GENERATE_AVX    TEST_FEATURE(FEATURE_AVX)
#define GENERATE_AVX2   TEST_FEATURE(FEATURE_AVX2)
#define GENERATE_AVX3   TEST_FEATURE(FEATURE_AVX512F)
#define GENERATE_AVX32  TEST_FEATURE(FEATURE_AVX512VL)
#define HAS_FMA3        TEST_FEATURE(FEATURE_FMA3)
#define HAS_FMA4        TEST_FEATURE(FEATURE_FMA4)
#define HAS_FMA         (HAS_FMA3 || HAS_FMA4)

#define XBIT_GENERATE_SCALAR_FMA  (! XBIT(164, 0x40000000) && HAS_FMA)

/*-----------------------------------------------
 * Define an additional assert macro, 'asrt()'
 *----------------------------------------------*/

#if DEBUG
#define asrt(c) \
  if (c)        \
    ;           \
  else          \
  fprintf(stderr, "asrt failed. line %d, file %s\n", __LINE__, __FILE__)
#else  /* if ! DEBUG */
#define asrt(c)
#endif  /* if DEBUG */

/*------------------------------------------------------
 * Define the 3 getitem areas used by the Code Generator
 *----------------------------------------------------*/

/* 'CG_LONGTERM_AREA' is freed just once, at the end of the code
 * generation for each user function.
 */
#define CG_LONGTERM_AREA  10

/* 'CG_MEDTERM_AREA' is freed:
 * -- immediately before the cgoptim2 phase;
 * -- at the end of cgoptim2 phase, before cgassem.
 */
#define CG_MEDTERM_AREA  6

/* 'CG_SHORTTERM_AREA' is freed:
 * -- at the end of the linearize/cgoptim1/genaili phase for each block;
 * -- at certain points during OPT2 register allocation.
 */
#define CG_SHORTTERM_AREA  11

/* Timing statistics.
 */
typedef enum {
  CG_TIMING_START = 0,
  ALIAS_BUILD,
  COLLECT_DEF_USE,
  RD_LOCAL,
  RD_GLOBAL,
  RD_UD,
  CG_TIMING_FINISH
} CGTIMING;

/* Debug traces.
 */
#undef TRACE0
#undef TRACE1
#undef TRACE2

#if DEBUG
#define TRACE0(t, s)              if (t) fprintf(gbl.dbgfil, s)
#define TRACE1(t, s, a1)          if (t) fprintf(gbl.dbgfil, s, a1)
#define TRACE2(t, s, a1, a2)      if (t) fprintf(gbl.dbgfil, s, a1, a2)
#define TRACE3(t, s, a1, a2, a3)  if (t) fprintf(gbl.dbgfil, s, a1, a2, a3)

#define CLOCK_START
#define CLOCK_FINISH
#define CLOCK_DURATION

#else  /* if ! DEBUG */
#define TRACE0(t, s)
#define TRACE1(t, s, a1)
#define TRACE2(t, s, a1, a2)
#define TRACE3(t, s, a1, a2, a3)
#define CLOCK_START
#define CLOCK_FINISH
#define CLOCK_DURATION
#endif  /* if DEBUG */

/*===========================================
 * External functions local to the CG
 *=========================================*/

/*--------
 * error.c
 *------*/
void  asrt_failed(const char *filename, int line);

/*---------
 * cgmain.c
 *-------*/
void  schedule(void);
void  reset_expr_id(void);

