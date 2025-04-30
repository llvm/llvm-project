/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

#include <time.h>
#include "stdioInterf.h"
#include "fioMacros.h"
#include "llcrit.h"

/*
 * ========================================================================
 * Common definitions.
 * ========================================================================
 */

/*
 * Construct constant factors.
 */

#define T23 (8388608.0)

#define T46 (T23 * T23)

#define R23 (1.0 / T23)
#define R46 (1.0 / T46)

/*
 * 46 or 23 bit mask.
 */

#define MASK23 ((unsigned)0x7fffff)

static __INT_T last_i = 0;

#ifdef DEBUG

static void
rnum_abort(const char *file, int line, const char *mesg)
{
  char temp[256];

  (void)sprintf(temp, "internal error in file \"%s\", line %d:  %s\n", file,
                line, mesg);
  __fort_abort(temp);
}

#endif

/*
 * ========================================================================
 * NAS parallel benchmarks multiplicative congruential pseudo-random number
 * generator code.
 * ========================================================================
 */

#define DEFAULT_SEED_HI (R23 * 32.0)
#define DEFAULT_SEED_LO (R46 * 3392727.0)

static __BIGREAL_T seed_hi = DEFAULT_SEED_HI;
static __BIGREAL_T seed_lo = DEFAULT_SEED_LO;

static __BIGREAL_T table[32][2] = {
    {4354965.0, T23 * 145.0},     {210105.0, T23 * 6909540.0},
    {3255729.0, T23 * 1196310.0}, {1750113.0, T23 * 3474515.0},
    {5016769.0, T23 * 2330923.0}, {8104321.0, T23 * 1946261.0},
    {3248897.0, T23 * 6040782.0}, {4990465.0, T23 * 1843409.0},
    {3951617.0, T23 * 3906254.0}, {563201.0, T23 * 912650.0},
    {5320705.0, T23 * 3689928.0}, {2252801.0, T23 * 6232675.0},
    {4505601.0, T23 * 6139918.0}, {622593.0, T23 * 6638909.0},
    {1245185.0, T23 * 5394170.0}, {2490369.0, T23 * 4419572.0},
    {4980737.0, T23 * 141288.0},  {1572865.0, T23 * 7434193.0},
    {3145729.0, T23 * 1531810.0}, {6291457.0, T23 * 48964.0},
    {4194305.0, T23 * 4816521.0}, {1.0, T23 * 3341587.0},
    {1.0, T23 * 6683174.0},       {1.0, T23 * 4977740.0},
    {1.0, T23 * 1566872.0},       {1.0, T23 * 3133744.0},
    {1.0, T23 * 6267488.0},       {1.0, T23 * 4146368.0},
    {1.0, T23 * 8292736.0},       {1.0, T23 * 8196864.0},
    {1.0, T23 * 8005120.0},       {1.0, T23 * 7621632.0}};

MP_SEMAPHORE(static, sem);

static __BIGREAL_T
advance_seed_npb(__INT_T n)
{
  int itmp;
  __BIGREAL_T tmp1, tmp2;
  __BIGREAL_T(*tp)[2];

#ifdef DEBUG
  /*
   * Check for zero advance.
   */
  if (n < 1)
    rnum_abort(__FILE__, __LINE__,
               "random_number:  internal error in advance_seed_npb:  n < 1");
#endif
  tp = table;
  while (n > 0) {
    if (n & 1) {
      tmp1 = seed_lo * tp[0][0];
      itmp = T23 * tmp1;
      tmp2 = R23 * itmp;
      seed_hi = tmp2 + seed_lo * tp[0][1] + seed_hi * tp[0][0];
      seed_lo = tmp1 - tmp2;
      itmp = seed_hi;
      seed_hi -= itmp;
    }
    ++tp;
    n >>= 1;
  }
  return seed_lo + seed_hi;
}

static void I8(prng_loop_d_npb)(__REAL8_T *hb, F90_Desc *harvest, __INT_T li,
                                int dim, __INT_T section_offset, __INT_T limit)
{
  DECL_DIM_PTRS(hdd);
  DECL_DIM_PTRS(tdd);
  __INT_T cl, clof, cn, current, i, il, iu, lo, n;
  __INT_T hi, tcl, tcn, tclof;
  int itmp;
  __BIGREAL_T tmp1, tmp2;

  SET_DIM_PTRS(hdd, harvest, dim - 1);
  cl = DIST_DPTR_CL_G(hdd);
  cn = DIST_DPTR_CN_G(hdd);
  clof = DIST_DPTR_CLOF_G(hdd);

  if (dim > (limit + 1))
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      for (i = 0; i < n; ++i) {
        I8(prng_loop_d_npb)(hb, harvest, lo, dim - 1, current + i, limit);
        lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      }
    }
  /*
   * Optimization collapsing non-distributed leading dimensions.
   */
  else if (limit > 0) {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      /*
       * Find first current and low value of fill range.
       */
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      hi = lo + (n - 1) * F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      for (i = dim - 1; i > 0; --i) {
        SET_DIM_PTRS(tdd, harvest, i - 1);
        tcl = DIST_DPTR_CL_G(tdd);
        tcn = DIST_DPTR_CN_G(tdd);
        tclof = DIST_DPTR_CLOF_G(tdd);
        (void)I8(__fort_block_bounds)(harvest, i, tcl, &il, &iu);
        lo = lo +
             (F90_DPTR_SSTRIDE_G(tdd) * il + F90_DPTR_SOFFSET_G(tdd) - tclof) *
                 F90_DPTR_LSTRIDE_G(tdd);
        current =
            F90_DPTR_EXTENT_G(tdd) * current + (il - F90_DPTR_LBOUND_G(tdd));
        n = I8(__fort_block_bounds)(
            harvest, i, tcl + (tcn - 1) * DIST_DPTR_CS_G(tdd), &il, &iu);
        hi = hi +
             (F90_DPTR_SSTRIDE_G(tdd) * (il + n - 1) + F90_DPTR_SOFFSET_G(tdd) -
              tclof) *
                 F90_DPTR_LSTRIDE_G(tdd);
      }
      /*
       * Fill the array with random numbers.
       */
      hb[lo] = advance_seed_npb(current - last_i);
      last_i = current + hi - lo;
      for (i = lo + 1; i <= hi; ++i) {
        tmp1 = seed_lo * table[0][0];
        itmp = T23 * tmp1;
        tmp2 = R23 * itmp;
        seed_hi = tmp2 + seed_lo * table[0][1] + seed_hi * table[0][0];
        seed_lo = tmp1 - tmp2;
        itmp = seed_hi;
        seed_hi -= itmp;
        hb[i] = seed_lo + seed_hi;
      }
    }
  } else {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      if (n > 0) {
        lo = li +
             (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
                 F90_DPTR_LSTRIDE_G(hdd);
        current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                  (il - F90_DPTR_LBOUND_G(hdd));
        hb[lo] = advance_seed_npb(current - last_i);
        for (i = 1; i < n; ++i) {
          lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
          tmp1 = seed_lo * table[0][0];
          itmp = T23 * tmp1;
          tmp2 = R23 * itmp;
          seed_hi = tmp2 + seed_lo * table[0][1] + seed_hi * table[0][0];
          seed_lo = tmp1 - tmp2;
          itmp = seed_hi;
          seed_hi -= itmp;
          hb[lo] = seed_lo + seed_hi;
        }
        last_i = current + n - 1;
      }
    }
  }
}

#ifdef TARGET_SUPPORTS_QUADFP
static void I8(prng_loop_q_npb)(__REAL16_T *hb, F90_Desc *harvest, __INT_T li,
                                int dim, __INT_T section_offset, __INT_T limit)
{
  DECL_DIM_PTRS(hdd);
  DECL_DIM_PTRS(tdd);
  __INT_T cl, clof, cn, current, i, il, iu, lo, n;
  __INT_T hi, tcl, tcn, tclof;
  int itmp;
  __BIGREAL_T tmp1, tmp2;

  SET_DIM_PTRS(hdd, harvest, dim - 1);
  cl = DIST_DPTR_CL_G(hdd);
  cn = DIST_DPTR_CN_G(hdd);
  clof = DIST_DPTR_CLOF_G(hdd);

  if (dim > (limit + 1))
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      for (i = 0; i < n; ++i) {
        I8(prng_loop_q_npb)(hb, harvest, lo, dim - 1, current + i, limit);
        lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      }
    }
  /*
   * Optimization collapsing non-distributed leading dimensions.
   */
  else if (limit > 0) {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      /*
       * Find first current and low value of fill range.
       */
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      hi = lo + (n - 1) * F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      for (i = dim - 1; i > 0; --i) {
        SET_DIM_PTRS(tdd, harvest, i - 1);
        tcl = DIST_DPTR_CL_G(tdd);
        tcn = DIST_DPTR_CN_G(tdd);
        tclof = DIST_DPTR_CLOF_G(tdd);
        (void)I8(__fort_block_bounds)(harvest, i, tcl, &il, &iu);
        lo = lo +
             (F90_DPTR_SSTRIDE_G(tdd) * il + F90_DPTR_SOFFSET_G(tdd) - tclof) *
                 F90_DPTR_LSTRIDE_G(tdd);
        current =
            F90_DPTR_EXTENT_G(tdd) * current + (il - F90_DPTR_LBOUND_G(tdd));
        n = I8(__fort_block_bounds)(
            harvest, i, tcl + (tcn - 1) * DIST_DPTR_CS_G(tdd), &il, &iu);
        hi = hi +
             (F90_DPTR_SSTRIDE_G(tdd) * (il + n - 1) + F90_DPTR_SOFFSET_G(tdd) -
              tclof) *
                 F90_DPTR_LSTRIDE_G(tdd);
      }
      /*
       * Fill the array with random numbers.
       */
      hb[lo] = advance_seed_npb(current - last_i);
      last_i = current + hi - lo;
      for (i = lo + 1; i <= hi; ++i) {
        tmp1 = seed_lo * table[0][0];
        itmp = T23 * tmp1;
        tmp2 = R23 * itmp;
        seed_hi = tmp2 + seed_lo * table[0][1] + seed_hi * table[0][0];
        seed_lo = tmp1 - tmp2;
        itmp = seed_hi;
        seed_hi -= itmp;
        hb[i] = seed_lo + seed_hi;
      }
    }
  } else {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      if (n > 0) {
        lo = li +
             (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
                 F90_DPTR_LSTRIDE_G(hdd);
        current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                  (il - F90_DPTR_LBOUND_G(hdd));
        hb[lo] = advance_seed_npb(current - last_i);
        for (i = 1; i < n; ++i) {
          lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
          tmp1 = seed_lo * table[0][0];
          itmp = T23 * tmp1;
          tmp2 = R23 * itmp;
          seed_hi = tmp2 + seed_lo * table[0][1] + seed_hi * table[0][0];
          seed_lo = tmp1 - tmp2;
          itmp = seed_hi;
          seed_hi -= itmp;
          hb[lo] = seed_lo + seed_hi;
        }
        last_i = current + n - 1;
      }
    }
  }
}
#endif

static void I8(prng_loop_r_npb)(__REAL4_T *hb, F90_Desc *harvest, __INT_T li,
                                int dim, __INT_T section_offset, __INT_T limit)
{
  DECL_DIM_PTRS(hdd);
  DECL_DIM_PTRS(tdd);
  __INT_T cl, cn, current, i, il, iu, lo, clof, n;
  __INT_T hi, tcl, tcn, tclof;
  int itmp;
  __BIGREAL_T tmp1, tmp2;

  SET_DIM_PTRS(hdd, harvest, dim - 1);
  cl = DIST_DPTR_CL_G(hdd);
  cn = DIST_DPTR_CN_G(hdd);
  clof = DIST_DPTR_CLOF_G(hdd);

  if (dim > (limit + 1))
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      for (i = 0; i < n; ++i) {
        I8(prng_loop_r_npb)(hb, harvest, lo, dim - 1, current + i, limit);
        lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      }
    }
  /*
   * Optimization collapsing non-distributed leading dimensions.
   */
  else if (limit > 0) {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      /*
       * Find first current and low value of fill range.
       */
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      hi = lo + (n - 1) * F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      for (i = dim - 1; i > 0; --i) {
        SET_DIM_PTRS(tdd, harvest, i - 1);
        tcl = DIST_DPTR_CL_G(tdd);
        tcn = DIST_DPTR_CN_G(tdd);
        tclof = DIST_DPTR_CLOF_G(tdd);
        (void)I8(__fort_block_bounds)(harvest, i, tcl, &il, &iu);
        lo = lo +
             (F90_DPTR_SSTRIDE_G(tdd) * il + F90_DPTR_SOFFSET_G(tdd) - tclof) *
                 F90_DPTR_LSTRIDE_G(hdd);
        current =
            F90_DPTR_EXTENT_G(tdd) * current + (il - F90_DPTR_LBOUND_G(tdd));
        n = I8(__fort_block_bounds)(
            harvest, i, tcl + (tcn - 1) * DIST_DPTR_CS_G(tdd), &il, &iu);
        hi = hi +
             (F90_DPTR_SSTRIDE_G(tdd) * (il + n - 1) + F90_DPTR_SOFFSET_G(tdd) -
              tclof) *
                 F90_DPTR_LSTRIDE_G(hdd);
      }
      /*
       * Fill the array with random numbers.
       */
      hb[lo] = advance_seed_npb(current - last_i);
      last_i = current + hi - lo;
      for (i = lo + 1; i <= hi; ++i) {
        tmp1 = seed_lo * table[0][0];
        itmp = T23 * tmp1;
        tmp2 = R23 * itmp;
        seed_hi = tmp2 + seed_lo * table[0][1] + seed_hi * table[0][0];
        seed_lo = tmp1 - tmp2;
        itmp = seed_hi;
        seed_hi -= itmp;
        hb[i] = seed_lo + seed_hi;
      }
    }
  } else {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      if (n > 0) {
        lo = li +
             (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
                 F90_DPTR_LSTRIDE_G(hdd);
        current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                  (il - F90_DPTR_LBOUND_G(hdd));
        hb[lo] = advance_seed_npb(current - last_i);
        for (i = 1; i < n; ++i) {
          lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
          tmp1 = seed_lo * table[0][0];
          itmp = T23 * tmp1;
          tmp2 = R23 * itmp;
          seed_hi = tmp2 + seed_lo * table[0][1] + seed_hi * table[0][0];
          seed_lo = tmp1 - tmp2;
          itmp = seed_hi;
          seed_hi -= itmp;
          hb[lo] = seed_lo + seed_hi;
        }
        last_i = current + n - 1;
      }
    }
  }
}

/*
 * ========================================================================
 * Lagged fibonacci pseudo-random number generator code.
 * ========================================================================
 */

typedef struct {
  __BIGREAL_T lo, hi;
} Seed;

#define NBITS 2
#define DIGIT ((1 << NBITS) - 1)
#define NDIGITS ((32 + NBITS - 1) / NBITS)

#define LONG_LAG 17
#define SHORT_LAG 5
#define L2CYCLE 6
#define L2CUTOFF 8

#define CYCLE (1 << L2CYCLE)
#define MASK (CYCLE - 1)
#define TOGGLE (CYCLE >> 1)

#define CUTOFF (1 << L2CUTOFF)
#define CUTMASK (CUTOFF - 1)

/*
 * Implement modulo 2^46 multiplication.
 */

static __BIGREAL_T
mul46(const Seed *xp, __BIGREAL_T ylo, __BIGREAL_T yhi)
{
  int i;
  __BIGREAL_T z;

  z = xp->hi * ylo + xp->lo * yhi;
  i = z;
  z -= i;
  z += xp->lo * ylo;
  return z;
}

/* used to create a distinct default random number each
   time the program is run - Mat Colgrove 1/27/05
 */
static int start_time_is_set = 0;
static time_t start_time;

/*
 * These are used as the default seeds.  They must be identical to the
 * initial values of seed_lf[].
 */

static const __BIGREAL_T default_seed_lf[LONG_LAG] = {
    21443106311501.0, 5197437683097.0,  3622043880426.0,  53312694480426.0,
    54665542338115.0, 51292272760733.0, 28013141389639.0, 6466909594288.0,
    36631377956900.0, 45800305729322.0, 1486199964658.0,  1320339397524.0,
    42446291962239.0, 8221323655096.0,  1104293620992.0,  2988247604277.0,
    1440485417884.0,
};

static __BIGREAL_T seed_lf[CYCLE] = {
    21443106311501.0 / T46, 5197437683097.0 / T46,  3622043880426.0 / T46,
    53312694480426.0 / T46, 54665542338115.0 / T46, 51292272760733.0 / T46,
    28013141389639.0 / T46, 6466909594288.0 / T46,  36631377956900.0 / T46,
    45800305729322.0 / T46, 1486199964658.0 / T46,  1320339397524.0 / T46,
    42446291962239.0 / T46, 8221323655096.0 / T46,  1104293620992.0 / T46,
    2988247604277.0 / T46,  1440485417884.0 / T46,
};

static int offset = LONG_LAG - 1;

#define SEED(x, y)                                                             \
  {                                                                            \
    (__BIGREAL_T)x, T23 * (__BIGREAL_T)y                                         \
  }

static const Seed table_lf[NDIGITS][DIGIT][LONG_LAG][LONG_LAG] = {
    {
        {{SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)}},
        {{SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)}},
        {{SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)}},
    },
    {
        {{SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)}},
        {{SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)}},
        {{SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)}},
    },
    {
        {{SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)}},
        {{SEED(4, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(3, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(2, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(4, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(3, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(2, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(4, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(3, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(2, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(1, 0), SEED(0, 0), SEED(4, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(3, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(2, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(1, 0), SEED(0, 0), SEED(4, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(3, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(2, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(2, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(2, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(2, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(2, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(2, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(3, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(2, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(3, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(2, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(0, 0), SEED(3, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(2, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(3, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(2, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(3, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(2, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(1, 0)}},
        {{SEED(0, 0), SEED(7, 0), SEED(0, 0), SEED(1, 0), SEED(6, 0),
          SEED(0, 0), SEED(6, 0), SEED(0, 0), SEED(1, 0), SEED(3, 0),
          SEED(0, 0), SEED(5, 0), SEED(0, 0), SEED(1, 0), SEED(1, 0),
          SEED(0, 0), SEED(4, 0)},
         {SEED(10, 0), SEED(0, 0), SEED(7, 0), SEED(0, 0), SEED(1, 0),
          SEED(6, 0), SEED(0, 0), SEED(6, 0), SEED(0, 0), SEED(1, 0),
          SEED(3, 0), SEED(0, 0), SEED(5, 0), SEED(0, 0), SEED(1, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(10, 0), SEED(0, 0), SEED(7, 0), SEED(0, 0),
          SEED(1, 0), SEED(6, 0), SEED(0, 0), SEED(6, 0), SEED(0, 0),
          SEED(1, 0), SEED(3, 0), SEED(0, 0), SEED(5, 0), SEED(0, 0),
          SEED(1, 0), SEED(1, 0)},
         {SEED(1, 0), SEED(1, 0), SEED(10, 0), SEED(0, 0), SEED(7, 0),
          SEED(0, 0), SEED(1, 0), SEED(6, 0), SEED(0, 0), SEED(6, 0),
          SEED(0, 0), SEED(1, 0), SEED(3, 0), SEED(0, 0), SEED(5, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(8, 0), SEED(1, 0), SEED(1, 0), SEED(10, 0), SEED(0, 0),
          SEED(7, 0), SEED(0, 0), SEED(1, 0), SEED(6, 0), SEED(0, 0),
          SEED(6, 0), SEED(0, 0), SEED(1, 0), SEED(3, 0), SEED(0, 0),
          SEED(5, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(1, 0), SEED(0, 0), SEED(4, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(3, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(2, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(5, 0), SEED(0, 0), SEED(1, 0), SEED(1, 0), SEED(0, 0),
          SEED(4, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(3, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(2, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(5, 0), SEED(0, 0), SEED(1, 0), SEED(1, 0),
          SEED(0, 0), SEED(4, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(3, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(2, 0)},
         {SEED(3, 0), SEED(0, 0), SEED(5, 0), SEED(0, 0), SEED(1, 0),
          SEED(1, 0), SEED(0, 0), SEED(4, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0), SEED(3, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(3, 0), SEED(0, 0), SEED(5, 0), SEED(0, 0),
          SEED(1, 0), SEED(1, 0), SEED(0, 0), SEED(4, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(3, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(3, 0), SEED(0, 0), SEED(5, 0),
          SEED(0, 0), SEED(1, 0), SEED(1, 0), SEED(0, 0), SEED(4, 0),
          SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0), SEED(3, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(6, 0), SEED(0, 0), SEED(1, 0), SEED(3, 0), SEED(0, 0),
          SEED(5, 0), SEED(0, 0), SEED(1, 0), SEED(1, 0), SEED(0, 0),
          SEED(4, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0), SEED(0, 0),
          SEED(3, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(6, 0), SEED(0, 0), SEED(1, 0), SEED(3, 0),
          SEED(0, 0), SEED(5, 0), SEED(0, 0), SEED(1, 0), SEED(1, 0),
          SEED(0, 0), SEED(4, 0), SEED(0, 0), SEED(1, 0), SEED(0, 0),
          SEED(0, 0), SEED(3, 0)},
         {SEED(6, 0), SEED(0, 0), SEED(6, 0), SEED(0, 0), SEED(1, 0),
          SEED(3, 0), SEED(0, 0), SEED(5, 0), SEED(0, 0), SEED(1, 0),
          SEED(1, 0), SEED(0, 0), SEED(4, 0), SEED(0, 0), SEED(1, 0),
          SEED(0, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(6, 0), SEED(0, 0), SEED(6, 0), SEED(0, 0),
          SEED(1, 0), SEED(3, 0), SEED(0, 0), SEED(5, 0), SEED(0, 0),
          SEED(1, 0), SEED(1, 0), SEED(0, 0), SEED(4, 0), SEED(0, 0),
          SEED(1, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(1, 0), SEED(6, 0), SEED(0, 0), SEED(6, 0),
          SEED(0, 0), SEED(1, 0), SEED(3, 0), SEED(0, 0), SEED(5, 0),
          SEED(0, 0), SEED(1, 0), SEED(1, 0), SEED(0, 0), SEED(4, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(7, 0), SEED(0, 0), SEED(1, 0), SEED(6, 0), SEED(0, 0),
          SEED(6, 0), SEED(0, 0), SEED(1, 0), SEED(3, 0), SEED(0, 0),
          SEED(5, 0), SEED(0, 0), SEED(1, 0), SEED(1, 0), SEED(0, 0),
          SEED(4, 0), SEED(0, 0)}},
    },
    {
        {{SEED(28, 0), SEED(0, 0), SEED(10, 0), SEED(10, 0), SEED(1, 0),
          SEED(21, 0), SEED(0, 0), SEED(9, 0), SEED(4, 0), SEED(1, 0),
          SEED(15, 0), SEED(0, 0), SEED(8, 0), SEED(1, 0), SEED(1, 0),
          SEED(10, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(28, 0), SEED(0, 0), SEED(10, 0), SEED(10, 0),
          SEED(1, 0), SEED(21, 0), SEED(0, 0), SEED(9, 0), SEED(4, 0),
          SEED(1, 0), SEED(15, 0), SEED(0, 0), SEED(8, 0), SEED(1, 0),
          SEED(1, 0), SEED(10, 0)},
         {SEED(20, 0), SEED(1, 0), SEED(28, 0), SEED(0, 0), SEED(10, 0),
          SEED(10, 0), SEED(1, 0), SEED(21, 0), SEED(0, 0), SEED(9, 0),
          SEED(4, 0), SEED(1, 0), SEED(15, 0), SEED(0, 0), SEED(8, 0),
          SEED(1, 0), SEED(1, 0)},
         {SEED(11, 0), SEED(20, 0), SEED(1, 0), SEED(28, 0), SEED(0, 0),
          SEED(10, 0), SEED(10, 0), SEED(1, 0), SEED(21, 0), SEED(0, 0),
          SEED(9, 0), SEED(4, 0), SEED(1, 0), SEED(15, 0), SEED(0, 0),
          SEED(8, 0), SEED(1, 0)},
         {SEED(1, 0), SEED(11, 0), SEED(20, 0), SEED(1, 0), SEED(28, 0),
          SEED(0, 0), SEED(10, 0), SEED(10, 0), SEED(1, 0), SEED(21, 0),
          SEED(0, 0), SEED(9, 0), SEED(4, 0), SEED(1, 0), SEED(15, 0),
          SEED(0, 0), SEED(8, 0)},
         {SEED(8, 0), SEED(1, 0), SEED(1, 0), SEED(10, 0), SEED(0, 0),
          SEED(7, 0), SEED(0, 0), SEED(1, 0), SEED(6, 0), SEED(0, 0),
          SEED(6, 0), SEED(0, 0), SEED(1, 0), SEED(3, 0), SEED(0, 0),
          SEED(5, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(8, 0), SEED(1, 0), SEED(1, 0), SEED(10, 0),
          SEED(0, 0), SEED(7, 0), SEED(0, 0), SEED(1, 0), SEED(6, 0),
          SEED(0, 0), SEED(6, 0), SEED(0, 0), SEED(1, 0), SEED(3, 0),
          SEED(0, 0), SEED(5, 0)},
         {SEED(15, 0), SEED(0, 0), SEED(8, 0), SEED(1, 0), SEED(1, 0),
          SEED(10, 0), SEED(0, 0), SEED(7, 0), SEED(0, 0), SEED(1, 0),
          SEED(6, 0), SEED(0, 0), SEED(6, 0), SEED(0, 0), SEED(1, 0),
          SEED(3, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(15, 0), SEED(0, 0), SEED(8, 0), SEED(1, 0),
          SEED(1, 0), SEED(10, 0), SEED(0, 0), SEED(7, 0), SEED(0, 0),
          SEED(1, 0), SEED(6, 0), SEED(0, 0), SEED(6, 0), SEED(0, 0),
          SEED(1, 0), SEED(3, 0)},
         {SEED(4, 0), SEED(1, 0), SEED(15, 0), SEED(0, 0), SEED(8, 0),
          SEED(1, 0), SEED(1, 0), SEED(10, 0), SEED(0, 0), SEED(7, 0),
          SEED(0, 0), SEED(1, 0), SEED(6, 0), SEED(0, 0), SEED(6, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(9, 0), SEED(4, 0), SEED(1, 0), SEED(15, 0), SEED(0, 0),
          SEED(8, 0), SEED(1, 0), SEED(1, 0), SEED(10, 0), SEED(0, 0),
          SEED(7, 0), SEED(0, 0), SEED(1, 0), SEED(6, 0), SEED(0, 0),
          SEED(6, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(9, 0), SEED(4, 0), SEED(1, 0), SEED(15, 0),
          SEED(0, 0), SEED(8, 0), SEED(1, 0), SEED(1, 0), SEED(10, 0),
          SEED(0, 0), SEED(7, 0), SEED(0, 0), SEED(1, 0), SEED(6, 0),
          SEED(0, 0), SEED(6, 0)},
         {SEED(21, 0), SEED(0, 0), SEED(9, 0), SEED(4, 0), SEED(1, 0),
          SEED(15, 0), SEED(0, 0), SEED(8, 0), SEED(1, 0), SEED(1, 0),
          SEED(10, 0), SEED(0, 0), SEED(7, 0), SEED(0, 0), SEED(1, 0),
          SEED(6, 0), SEED(0, 0)},
         {SEED(1, 0), SEED(21, 0), SEED(0, 0), SEED(9, 0), SEED(4, 0),
          SEED(1, 0), SEED(15, 0), SEED(0, 0), SEED(8, 0), SEED(1, 0),
          SEED(1, 0), SEED(10, 0), SEED(0, 0), SEED(7, 0), SEED(0, 0),
          SEED(1, 0), SEED(6, 0)},
         {SEED(10, 0), SEED(1, 0), SEED(21, 0), SEED(0, 0), SEED(9, 0),
          SEED(4, 0), SEED(1, 0), SEED(15, 0), SEED(0, 0), SEED(8, 0),
          SEED(1, 0), SEED(1, 0), SEED(10, 0), SEED(0, 0), SEED(7, 0),
          SEED(0, 0), SEED(1, 0)},
         {SEED(10, 0), SEED(10, 0), SEED(1, 0), SEED(21, 0), SEED(0, 0),
          SEED(9, 0), SEED(4, 0), SEED(1, 0), SEED(15, 0), SEED(0, 0),
          SEED(8, 0), SEED(1, 0), SEED(1, 0), SEED(10, 0), SEED(0, 0),
          SEED(7, 0), SEED(0, 0)},
         {SEED(0, 0), SEED(10, 0), SEED(10, 0), SEED(1, 0), SEED(21, 0),
          SEED(0, 0), SEED(9, 0), SEED(4, 0), SEED(1, 0), SEED(15, 0),
          SEED(0, 0), SEED(8, 0), SEED(1, 0), SEED(1, 0), SEED(10, 0),
          SEED(0, 0), SEED(7, 0)}},
        {{SEED(1820, 0), SEED(485, 0), SEED(816, 0), SEED(1288, 0),
          SEED(198, 0), SEED(1365, 0), SEED(232, 0), SEED(680, 0), SEED(793, 0),
          SEED(172, 0), SEED(1001, 0), SEED(105, 0), SEED(560, 0), SEED(463, 0),
          SEED(153, 0), SEED(715, 0), SEED(48, 0)},
         {SEED(246, 0), SEED(1820, 0), SEED(485, 0), SEED(816, 0),
          SEED(1288, 0), SEED(198, 0), SEED(1365, 0), SEED(232, 0),
          SEED(680, 0), SEED(793, 0), SEED(172, 0), SEED(1001, 0), SEED(105, 0),
          SEED(560, 0), SEED(463, 0), SEED(153, 0), SEED(715, 0)},
         {SEED(2003, 0), SEED(246, 0), SEED(1820, 0), SEED(485, 0),
          SEED(816, 0), SEED(1288, 0), SEED(198, 0), SEED(1365, 0),
          SEED(232, 0), SEED(680, 0), SEED(793, 0), SEED(172, 0), SEED(1001, 0),
          SEED(105, 0), SEED(560, 0), SEED(463, 0), SEED(153, 0)},
         {SEED(969, 0), SEED(2003, 0), SEED(246, 0), SEED(1820, 0),
          SEED(485, 0), SEED(816, 0), SEED(1288, 0), SEED(198, 0),
          SEED(1365, 0), SEED(232, 0), SEED(680, 0), SEED(793, 0), SEED(172, 0),
          SEED(1001, 0), SEED(105, 0), SEED(560, 0), SEED(463, 0)},
         {SEED(948, 0), SEED(969, 0), SEED(2003, 0), SEED(246, 0),
          SEED(1820, 0), SEED(485, 0), SEED(816, 0), SEED(1288, 0),
          SEED(198, 0), SEED(1365, 0), SEED(232, 0), SEED(680, 0), SEED(793, 0),
          SEED(172, 0), SEED(1001, 0), SEED(105, 0), SEED(560, 0)},
         {SEED(560, 0), SEED(463, 0), SEED(153, 0), SEED(715, 0), SEED(48, 0),
          SEED(455, 0), SEED(253, 0), SEED(136, 0), SEED(495, 0), SEED(26, 0),
          SEED(364, 0), SEED(127, 0), SEED(120, 0), SEED(330, 0), SEED(19, 0),
          SEED(286, 0), SEED(57, 0)},
         {SEED(105, 0), SEED(560, 0), SEED(463, 0), SEED(153, 0), SEED(715, 0),
          SEED(48, 0), SEED(455, 0), SEED(253, 0), SEED(136, 0), SEED(495, 0),
          SEED(26, 0), SEED(364, 0), SEED(127, 0), SEED(120, 0), SEED(330, 0),
          SEED(19, 0), SEED(286, 0)},
         {SEED(1001, 0), SEED(105, 0), SEED(560, 0), SEED(463, 0), SEED(153, 0),
          SEED(715, 0), SEED(48, 0), SEED(455, 0), SEED(253, 0), SEED(136, 0),
          SEED(495, 0), SEED(26, 0), SEED(364, 0), SEED(127, 0), SEED(120, 0),
          SEED(330, 0), SEED(19, 0)},
         {SEED(172, 0), SEED(1001, 0), SEED(105, 0), SEED(560, 0), SEED(463, 0),
          SEED(153, 0), SEED(715, 0), SEED(48, 0), SEED(455, 0), SEED(253, 0),
          SEED(136, 0), SEED(495, 0), SEED(26, 0), SEED(364, 0), SEED(127, 0),
          SEED(120, 0), SEED(330, 0)},
         {SEED(793, 0), SEED(172, 0), SEED(1001, 0), SEED(105, 0), SEED(560, 0),
          SEED(463, 0), SEED(153, 0), SEED(715, 0), SEED(48, 0), SEED(455, 0),
          SEED(253, 0), SEED(136, 0), SEED(495, 0), SEED(26, 0), SEED(364, 0),
          SEED(127, 0), SEED(120, 0)},
         {SEED(680, 0), SEED(793, 0), SEED(172, 0), SEED(1001, 0), SEED(105, 0),
          SEED(560, 0), SEED(463, 0), SEED(153, 0), SEED(715, 0), SEED(48, 0),
          SEED(455, 0), SEED(253, 0), SEED(136, 0), SEED(495, 0), SEED(26, 0),
          SEED(364, 0), SEED(127, 0)},
         {SEED(232, 0), SEED(680, 0), SEED(793, 0), SEED(172, 0), SEED(1001, 0),
          SEED(105, 0), SEED(560, 0), SEED(463, 0), SEED(153, 0), SEED(715, 0),
          SEED(48, 0), SEED(455, 0), SEED(253, 0), SEED(136, 0), SEED(495, 0),
          SEED(26, 0), SEED(364, 0)},
         {SEED(1365, 0), SEED(232, 0), SEED(680, 0), SEED(793, 0), SEED(172, 0),
          SEED(1001, 0), SEED(105, 0), SEED(560, 0), SEED(463, 0), SEED(153, 0),
          SEED(715, 0), SEED(48, 0), SEED(455, 0), SEED(253, 0), SEED(136, 0),
          SEED(495, 0), SEED(26, 0)},
         {SEED(198, 0), SEED(1365, 0), SEED(232, 0), SEED(680, 0), SEED(793, 0),
          SEED(172, 0), SEED(1001, 0), SEED(105, 0), SEED(560, 0), SEED(463, 0),
          SEED(153, 0), SEED(715, 0), SEED(48, 0), SEED(455, 0), SEED(253, 0),
          SEED(136, 0), SEED(495, 0)},
         {SEED(1288, 0), SEED(198, 0), SEED(1365, 0), SEED(232, 0),
          SEED(680, 0), SEED(793, 0), SEED(172, 0), SEED(1001, 0), SEED(105, 0),
          SEED(560, 0), SEED(463, 0), SEED(153, 0), SEED(715, 0), SEED(48, 0),
          SEED(455, 0), SEED(253, 0), SEED(136, 0)},
         {SEED(816, 0), SEED(1288, 0), SEED(198, 0), SEED(1365, 0),
          SEED(232, 0), SEED(680, 0), SEED(793, 0), SEED(172, 0), SEED(1001, 0),
          SEED(105, 0), SEED(560, 0), SEED(463, 0), SEED(153, 0), SEED(715, 0),
          SEED(48, 0), SEED(455, 0), SEED(253, 0)},
         {SEED(485, 0), SEED(816, 0), SEED(1288, 0), SEED(198, 0),
          SEED(1365, 0), SEED(232, 0), SEED(680, 0), SEED(793, 0), SEED(172, 0),
          SEED(1001, 0), SEED(105, 0), SEED(560, 0), SEED(463, 0), SEED(153, 0),
          SEED(715, 0), SEED(48, 0), SEED(455, 0)}},
        {{SEED(134644, 0), SEED(80077, 0), SEED(66782, 0), SEED(116808, 0),
          SEED(31915, 0), SEED(100983, 0), SEED(47818, 0), SEED(53417, 0),
          SEED(78016, 0), SEED(22555, 0), SEED(74647, 0), SEED(27964, 0),
          SEED(42571, 0), SEED(50853, 0), SEED(16952, 0), SEED(54297, 0),
          SEED(16146, 0)},
         {SEED(48061, 0), SEED(134644, 0), SEED(80077, 0), SEED(66782, 0),
          SEED(116808, 0), SEED(31915, 0), SEED(100983, 0), SEED(47818, 0),
          SEED(53417, 0), SEED(78016, 0), SEED(22555, 0), SEED(74647, 0),
          SEED(27964, 0), SEED(42571, 0), SEED(50853, 0), SEED(16952, 0),
          SEED(54297, 0)},
         {SEED(171105, 0), SEED(48061, 0), SEED(134644, 0), SEED(80077, 0),
          SEED(66782, 0), SEED(116808, 0), SEED(31915, 0), SEED(100983, 0),
          SEED(47818, 0), SEED(53417, 0), SEED(78016, 0), SEED(22555, 0),
          SEED(74647, 0), SEED(27964, 0), SEED(42571, 0), SEED(50853, 0),
          SEED(16952, 0)},
         {SEED(83734, 0), SEED(171105, 0), SEED(48061, 0), SEED(134644, 0),
          SEED(80077, 0), SEED(66782, 0), SEED(116808, 0), SEED(31915, 0),
          SEED(100983, 0), SEED(47818, 0), SEED(53417, 0), SEED(78016, 0),
          SEED(22555, 0), SEED(74647, 0), SEED(27964, 0), SEED(42571, 0),
          SEED(50853, 0)},
         {SEED(130930, 0), SEED(83734, 0), SEED(171105, 0), SEED(48061, 0),
          SEED(134644, 0), SEED(80077, 0), SEED(66782, 0), SEED(116808, 0),
          SEED(31915, 0), SEED(100983, 0), SEED(47818, 0), SEED(53417, 0),
          SEED(78016, 0), SEED(22555, 0), SEED(74647, 0), SEED(27964, 0),
          SEED(42571, 0)},
         {SEED(42571, 0), SEED(50853, 0), SEED(16952, 0), SEED(54297, 0),
          SEED(16146, 0), SEED(33661, 0), SEED(32259, 0), SEED(13365, 0),
          SEED(38792, 0), SEED(9360, 0), SEED(26336, 0), SEED(19854, 0),
          SEED(10846, 0), SEED(27163, 0), SEED(5603, 0), SEED(20350, 0),
          SEED(11818, 0)},
         {SEED(27964, 0), SEED(42571, 0), SEED(50853, 0), SEED(16952, 0),
          SEED(54297, 0), SEED(16146, 0), SEED(33661, 0), SEED(32259, 0),
          SEED(13365, 0), SEED(38792, 0), SEED(9360, 0), SEED(26336, 0),
          SEED(19854, 0), SEED(10846, 0), SEED(27163, 0), SEED(5603, 0),
          SEED(20350, 0)},
         {SEED(74647, 0), SEED(27964, 0), SEED(42571, 0), SEED(50853, 0),
          SEED(16952, 0), SEED(54297, 0), SEED(16146, 0), SEED(33661, 0),
          SEED(32259, 0), SEED(13365, 0), SEED(38792, 0), SEED(9360, 0),
          SEED(26336, 0), SEED(19854, 0), SEED(10846, 0), SEED(27163, 0),
          SEED(5603, 0)},
         {SEED(22555, 0), SEED(74647, 0), SEED(27964, 0), SEED(42571, 0),
          SEED(50853, 0), SEED(16952, 0), SEED(54297, 0), SEED(16146, 0),
          SEED(33661, 0), SEED(32259, 0), SEED(13365, 0), SEED(38792, 0),
          SEED(9360, 0), SEED(26336, 0), SEED(19854, 0), SEED(10846, 0),
          SEED(27163, 0)},
         {SEED(78016, 0), SEED(22555, 0), SEED(74647, 0), SEED(27964, 0),
          SEED(42571, 0), SEED(50853, 0), SEED(16952, 0), SEED(54297, 0),
          SEED(16146, 0), SEED(33661, 0), SEED(32259, 0), SEED(13365, 0),
          SEED(38792, 0), SEED(9360, 0), SEED(26336, 0), SEED(19854, 0),
          SEED(10846, 0)},
         {SEED(53417, 0), SEED(78016, 0), SEED(22555, 0), SEED(74647, 0),
          SEED(27964, 0), SEED(42571, 0), SEED(50853, 0), SEED(16952, 0),
          SEED(54297, 0), SEED(16146, 0), SEED(33661, 0), SEED(32259, 0),
          SEED(13365, 0), SEED(38792, 0), SEED(9360, 0), SEED(26336, 0),
          SEED(19854, 0)},
         {SEED(47818, 0), SEED(53417, 0), SEED(78016, 0), SEED(22555, 0),
          SEED(74647, 0), SEED(27964, 0), SEED(42571, 0), SEED(50853, 0),
          SEED(16952, 0), SEED(54297, 0), SEED(16146, 0), SEED(33661, 0),
          SEED(32259, 0), SEED(13365, 0), SEED(38792, 0), SEED(9360, 0),
          SEED(26336, 0)},
         {SEED(100983, 0), SEED(47818, 0), SEED(53417, 0), SEED(78016, 0),
          SEED(22555, 0), SEED(74647, 0), SEED(27964, 0), SEED(42571, 0),
          SEED(50853, 0), SEED(16952, 0), SEED(54297, 0), SEED(16146, 0),
          SEED(33661, 0), SEED(32259, 0), SEED(13365, 0), SEED(38792, 0),
          SEED(9360, 0)},
         {SEED(31915, 0), SEED(100983, 0), SEED(47818, 0), SEED(53417, 0),
          SEED(78016, 0), SEED(22555, 0), SEED(74647, 0), SEED(27964, 0),
          SEED(42571, 0), SEED(50853, 0), SEED(16952, 0), SEED(54297, 0),
          SEED(16146, 0), SEED(33661, 0), SEED(32259, 0), SEED(13365, 0),
          SEED(38792, 0)},
         {SEED(116808, 0), SEED(31915, 0), SEED(100983, 0), SEED(47818, 0),
          SEED(53417, 0), SEED(78016, 0), SEED(22555, 0), SEED(74647, 0),
          SEED(27964, 0), SEED(42571, 0), SEED(50853, 0), SEED(16952, 0),
          SEED(54297, 0), SEED(16146, 0), SEED(33661, 0), SEED(32259, 0),
          SEED(13365, 0)},
         {SEED(66782, 0), SEED(116808, 0), SEED(31915, 0), SEED(100983, 0),
          SEED(47818, 0), SEED(53417, 0), SEED(78016, 0), SEED(22555, 0),
          SEED(74647, 0), SEED(27964, 0), SEED(42571, 0), SEED(50853, 0),
          SEED(16952, 0), SEED(54297, 0), SEED(16146, 0), SEED(33661, 0),
          SEED(32259, 0)},
         {SEED(80077, 0), SEED(66782, 0), SEED(116808, 0), SEED(31915, 0),
          SEED(100983, 0), SEED(47818, 0), SEED(53417, 0), SEED(78016, 0),
          SEED(22555, 0), SEED(74647, 0), SEED(27964, 0), SEED(42571, 0),
          SEED(50853, 0), SEED(16952, 0), SEED(54297, 0), SEED(16146, 0),
          SEED(33661, 0)}},
    },
    {
        {{SEED(2220456, 1), SEED(623436, 1), SEED(6027297, 0), SEED(1728347, 1),
          SEED(4443984, 0), SEED(7928198, 0), SEED(5813678, 0),
          SEED(4566968, 0), SEED(6998410, 0), SEED(2975285, 0),
          SEED(5872973, 0), SEED(3704658, 0), SEED(3492772, 0),
          SEED(4769091, 0), SEED(2050382, 0), SEED(4305185, 0),
          SEED(2338249, 0)},
         {SEED(6782233, 0), SEED(2220456, 1), SEED(623436, 1), SEED(6027297, 0),
          SEED(1728347, 1), SEED(4443984, 0), SEED(7928198, 0),
          SEED(5813678, 0), SEED(4566968, 0), SEED(6998410, 0),
          SEED(2975285, 0), SEED(5872973, 0), SEED(3704658, 0),
          SEED(3492772, 0), SEED(4769091, 0), SEED(2050382, 0),
          SEED(4305185, 0)},
         {SEED(6033532, 1), SEED(6782233, 0), SEED(2220456, 1), SEED(623436, 1),
          SEED(6027297, 0), SEED(1728347, 1), SEED(4443984, 0),
          SEED(7928198, 0), SEED(5813678, 0), SEED(4566968, 0),
          SEED(6998410, 0), SEED(2975285, 0), SEED(5872973, 0),
          SEED(3704658, 0), SEED(3492772, 0), SEED(4769091, 0),
          SEED(2050382, 0)},
         {SEED(8077679, 0), SEED(6033532, 1), SEED(6782233, 0),
          SEED(2220456, 1), SEED(623436, 1), SEED(6027297, 0), SEED(1728347, 1),
          SEED(4443984, 0), SEED(7928198, 0), SEED(5813678, 0),
          SEED(4566968, 0), SEED(6998410, 0), SEED(2975285, 0),
          SEED(5872973, 0), SEED(3704658, 0), SEED(3492772, 0),
          SEED(4769091, 0)},
         {SEED(5392527, 1), SEED(8077679, 0), SEED(6033532, 1),
          SEED(6782233, 0), SEED(2220456, 1), SEED(623436, 1), SEED(6027297, 0),
          SEED(1728347, 1), SEED(4443984, 0), SEED(7928198, 0),
          SEED(5813678, 0), SEED(4566968, 0), SEED(6998410, 0),
          SEED(2975285, 0), SEED(5872973, 0), SEED(3704658, 0),
          SEED(3492772, 0)},
         {SEED(3492772, 0), SEED(4769091, 0), SEED(2050382, 0),
          SEED(4305185, 0), SEED(2338249, 0), SEED(2680866, 0),
          SEED(3198366, 0), SEED(1460329, 0), SEED(3118545, 0),
          SEED(1468699, 0), SEED(2055225, 0), SEED(2109020, 0),
          SEED(1074196, 0), SEED(2229319, 0), SEED(924903, 0), SEED(1567788, 0),
          SEED(1366409, 0)},
         {SEED(3704658, 0), SEED(3492772, 0), SEED(4769091, 0),
          SEED(2050382, 0), SEED(4305185, 0), SEED(2338249, 0),
          SEED(2680866, 0), SEED(3198366, 0), SEED(1460329, 0),
          SEED(3118545, 0), SEED(1468699, 0), SEED(2055225, 0),
          SEED(2109020, 0), SEED(1074196, 0), SEED(2229319, 0), SEED(924903, 0),
          SEED(1567788, 0)},
         {SEED(5872973, 0), SEED(3704658, 0), SEED(3492772, 0),
          SEED(4769091, 0), SEED(2050382, 0), SEED(4305185, 0),
          SEED(2338249, 0), SEED(2680866, 0), SEED(3198366, 0),
          SEED(1460329, 0), SEED(3118545, 0), SEED(1468699, 0),
          SEED(2055225, 0), SEED(2109020, 0), SEED(1074196, 0),
          SEED(2229319, 0), SEED(924903, 0)},
         {SEED(2975285, 0), SEED(5872973, 0), SEED(3704658, 0),
          SEED(3492772, 0), SEED(4769091, 0), SEED(2050382, 0),
          SEED(4305185, 0), SEED(2338249, 0), SEED(2680866, 0),
          SEED(3198366, 0), SEED(1460329, 0), SEED(3118545, 0),
          SEED(1468699, 0), SEED(2055225, 0), SEED(2109020, 0),
          SEED(1074196, 0), SEED(2229319, 0)},
         {SEED(6998410, 0), SEED(2975285, 0), SEED(5872973, 0),
          SEED(3704658, 0), SEED(3492772, 0), SEED(4769091, 0),
          SEED(2050382, 0), SEED(4305185, 0), SEED(2338249, 0),
          SEED(2680866, 0), SEED(3198366, 0), SEED(1460329, 0),
          SEED(3118545, 0), SEED(1468699, 0), SEED(2055225, 0),
          SEED(2109020, 0), SEED(1074196, 0)},
         {SEED(4566968, 0), SEED(6998410, 0), SEED(2975285, 0),
          SEED(5872973, 0), SEED(3704658, 0), SEED(3492772, 0),
          SEED(4769091, 0), SEED(2050382, 0), SEED(4305185, 0),
          SEED(2338249, 0), SEED(2680866, 0), SEED(3198366, 0),
          SEED(1460329, 0), SEED(3118545, 0), SEED(1468699, 0),
          SEED(2055225, 0), SEED(2109020, 0)},
         {SEED(5813678, 0), SEED(4566968, 0), SEED(6998410, 0),
          SEED(2975285, 0), SEED(5872973, 0), SEED(3704658, 0),
          SEED(3492772, 0), SEED(4769091, 0), SEED(2050382, 0),
          SEED(4305185, 0), SEED(2338249, 0), SEED(2680866, 0),
          SEED(3198366, 0), SEED(1460329, 0), SEED(3118545, 0),
          SEED(1468699, 0), SEED(2055225, 0)},
         {SEED(7928198, 0), SEED(5813678, 0), SEED(4566968, 0),
          SEED(6998410, 0), SEED(2975285, 0), SEED(5872973, 0),
          SEED(3704658, 0), SEED(3492772, 0), SEED(4769091, 0),
          SEED(2050382, 0), SEED(4305185, 0), SEED(2338249, 0),
          SEED(2680866, 0), SEED(3198366, 0), SEED(1460329, 0),
          SEED(3118545, 0), SEED(1468699, 0)},
         {SEED(4443984, 0), SEED(7928198, 0), SEED(5813678, 0),
          SEED(4566968, 0), SEED(6998410, 0), SEED(2975285, 0),
          SEED(5872973, 0), SEED(3704658, 0), SEED(3492772, 0),
          SEED(4769091, 0), SEED(2050382, 0), SEED(4305185, 0),
          SEED(2338249, 0), SEED(2680866, 0), SEED(3198366, 0),
          SEED(1460329, 0), SEED(3118545, 0)},
         {SEED(1728347, 1), SEED(4443984, 0), SEED(7928198, 0),
          SEED(5813678, 0), SEED(4566968, 0), SEED(6998410, 0),
          SEED(2975285, 0), SEED(5872973, 0), SEED(3704658, 0),
          SEED(3492772, 0), SEED(4769091, 0), SEED(2050382, 0),
          SEED(4305185, 0), SEED(2338249, 0), SEED(2680866, 0),
          SEED(3198366, 0), SEED(1460329, 0)},
         {SEED(6027297, 0), SEED(1728347, 1), SEED(4443984, 0),
          SEED(7928198, 0), SEED(5813678, 0), SEED(4566968, 0),
          SEED(6998410, 0), SEED(2975285, 0), SEED(5872973, 0),
          SEED(3704658, 0), SEED(3492772, 0), SEED(4769091, 0),
          SEED(2050382, 0), SEED(4305185, 0), SEED(2338249, 0),
          SEED(2680866, 0), SEED(3198366, 0)},
         {SEED(623436, 1), SEED(6027297, 0), SEED(1728347, 1), SEED(4443984, 0),
          SEED(7928198, 0), SEED(5813678, 0), SEED(4566968, 0),
          SEED(6998410, 0), SEED(2975285, 0), SEED(5872973, 0),
          SEED(3704658, 0), SEED(3492772, 0), SEED(4769091, 0),
          SEED(2050382, 0), SEED(4305185, 0), SEED(2338249, 0),
          SEED(2680866, 0)}},
        {{SEED(5353312, 6312246), SEED(3318706, 2157835), SEED(890124, 3375732),
          SEED(3348089, 5003315), SEED(6861865, 6830929),
          SEED(5373984, 7575303), SEED(3291887, 1564706),
          SEED(8144542, 7043611), SEED(2812876, 925395), SEED(6937239, 3290337),
          SEED(638120, 7555237), SEED(7012800, 152758), SEED(671474, 600033),
          SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586), SEED(4183518, 4128466)},
         {SEED(2656775, 2570788), SEED(5353312, 6312246),
          SEED(3318706, 2157835), SEED(890124, 3375732), SEED(3348089, 5003315),
          SEED(6861865, 6830929), SEED(5373984, 7575303),
          SEED(3291887, 1564706), SEED(8144542, 7043611), SEED(2812876, 925395),
          SEED(6937239, 3290337), SEED(638120, 7555237), SEED(7012800, 152758),
          SEED(671474, 600033), SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586)},
         {SEED(2068287, 704294), SEED(2656775, 2570788), SEED(5353312, 6312246),
          SEED(3318706, 2157835), SEED(890124, 3375732), SEED(3348089, 5003315),
          SEED(6861865, 6830929), SEED(5373984, 7575303),
          SEED(3291887, 1564706), SEED(8144542, 7043611), SEED(2812876, 925395),
          SEED(6937239, 3290337), SEED(638120, 7555237), SEED(7012800, 152758),
          SEED(671474, 600033), SEED(4486326, 2822244), SEED(7322428, 6019280)},
         {SEED(8212552, 1006404), SEED(2068287, 704294), SEED(2656775, 2570788),
          SEED(5353312, 6312246), SEED(3318706, 2157835), SEED(890124, 3375732),
          SEED(3348089, 5003315), SEED(6861865, 6830929),
          SEED(5373984, 7575303), SEED(3291887, 1564706),
          SEED(8144542, 7043611), SEED(2812876, 925395), SEED(6937239, 3290337),
          SEED(638120, 7555237), SEED(7012800, 152758), SEED(671474, 600033),
          SEED(4486326, 2822244)},
         {SEED(7805032, 4980079), SEED(8212552, 1006404), SEED(2068287, 704294),
          SEED(2656775, 2570788), SEED(5353312, 6312246),
          SEED(3318706, 2157835), SEED(890124, 3375732), SEED(3348089, 5003315),
          SEED(6861865, 6830929), SEED(5373984, 7575303),
          SEED(3291887, 1564706), SEED(8144542, 7043611), SEED(2812876, 925395),
          SEED(6937239, 3290337), SEED(638120, 7555237), SEED(7012800, 152758),
          SEED(671474, 600033)},
         {SEED(671474, 600033), SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586), SEED(4183518, 4128466),
          SEED(8367936, 7125550), SEED(26819, 593129), SEED(1134190, 4720728),
          SEED(535213, 4077920), SEED(8313234, 3540591), SEED(4735864, 20066),
          SEED(4667695, 1411947), SEED(7473068, 6443578),
          SEED(6715158, 6491758), SEED(8003419, 5659664),
          SEED(1917922, 3465650), SEED(2829282, 4412900)},
         {SEED(7012800, 152758), SEED(671474, 600033), SEED(4486326, 2822244),
          SEED(7322428, 6019280), SEED(7108806, 4089586),
          SEED(4183518, 4128466), SEED(8367936, 7125550), SEED(26819, 593129),
          SEED(1134190, 4720728), SEED(535213, 4077920), SEED(8313234, 3540591),
          SEED(4735864, 20066), SEED(4667695, 1411947), SEED(7473068, 6443578),
          SEED(6715158, 6491758), SEED(8003419, 5659664),
          SEED(1917922, 3465650)},
         {SEED(638120, 7555237), SEED(7012800, 152758), SEED(671474, 600033),
          SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586), SEED(4183518, 4128466),
          SEED(8367936, 7125550), SEED(26819, 593129), SEED(1134190, 4720728),
          SEED(535213, 4077920), SEED(8313234, 3540591), SEED(4735864, 20066),
          SEED(4667695, 1411947), SEED(7473068, 6443578),
          SEED(6715158, 6491758), SEED(8003419, 5659664)},
         {SEED(6937239, 3290337), SEED(638120, 7555237), SEED(7012800, 152758),
          SEED(671474, 600033), SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586), SEED(4183518, 4128466),
          SEED(8367936, 7125550), SEED(26819, 593129), SEED(1134190, 4720728),
          SEED(535213, 4077920), SEED(8313234, 3540591), SEED(4735864, 20066),
          SEED(4667695, 1411947), SEED(7473068, 6443578),
          SEED(6715158, 6491758)},
         {SEED(2812876, 925395), SEED(6937239, 3290337), SEED(638120, 7555237),
          SEED(7012800, 152758), SEED(671474, 600033), SEED(4486326, 2822244),
          SEED(7322428, 6019280), SEED(7108806, 4089586),
          SEED(4183518, 4128466), SEED(8367936, 7125550), SEED(26819, 593129),
          SEED(1134190, 4720728), SEED(535213, 4077920), SEED(8313234, 3540591),
          SEED(4735864, 20066), SEED(4667695, 1411947), SEED(7473068, 6443578)},
         {SEED(8144542, 7043611), SEED(2812876, 925395), SEED(6937239, 3290337),
          SEED(638120, 7555237), SEED(7012800, 152758), SEED(671474, 600033),
          SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586), SEED(4183518, 4128466),
          SEED(8367936, 7125550), SEED(26819, 593129), SEED(1134190, 4720728),
          SEED(535213, 4077920), SEED(8313234, 3540591), SEED(4735864, 20066),
          SEED(4667695, 1411947)},
         {SEED(3291887, 1564706), SEED(8144542, 7043611), SEED(2812876, 925395),
          SEED(6937239, 3290337), SEED(638120, 7555237), SEED(7012800, 152758),
          SEED(671474, 600033), SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586), SEED(4183518, 4128466),
          SEED(8367936, 7125550), SEED(26819, 593129), SEED(1134190, 4720728),
          SEED(535213, 4077920), SEED(8313234, 3540591), SEED(4735864, 20066)},
         {SEED(5373984, 7575303), SEED(3291887, 1564706),
          SEED(8144542, 7043611), SEED(2812876, 925395), SEED(6937239, 3290337),
          SEED(638120, 7555237), SEED(7012800, 152758), SEED(671474, 600033),
          SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586), SEED(4183518, 4128466),
          SEED(8367936, 7125550), SEED(26819, 593129), SEED(1134190, 4720728),
          SEED(535213, 4077920), SEED(8313234, 3540591)},
         {SEED(6861865, 6830929), SEED(5373984, 7575303),
          SEED(3291887, 1564706), SEED(8144542, 7043611), SEED(2812876, 925395),
          SEED(6937239, 3290337), SEED(638120, 7555237), SEED(7012800, 152758),
          SEED(671474, 600033), SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586), SEED(4183518, 4128466),
          SEED(8367936, 7125550), SEED(26819, 593129), SEED(1134190, 4720728),
          SEED(535213, 4077920)},
         {SEED(3348089, 5003315), SEED(6861865, 6830929),
          SEED(5373984, 7575303), SEED(3291887, 1564706),
          SEED(8144542, 7043611), SEED(2812876, 925395), SEED(6937239, 3290337),
          SEED(638120, 7555237), SEED(7012800, 152758), SEED(671474, 600033),
          SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586), SEED(4183518, 4128466),
          SEED(8367936, 7125550), SEED(26819, 593129), SEED(1134190, 4720728)},
         {SEED(890124, 3375732), SEED(3348089, 5003315), SEED(6861865, 6830929),
          SEED(5373984, 7575303), SEED(3291887, 1564706),
          SEED(8144542, 7043611), SEED(2812876, 925395), SEED(6937239, 3290337),
          SEED(638120, 7555237), SEED(7012800, 152758), SEED(671474, 600033),
          SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586), SEED(4183518, 4128466),
          SEED(8367936, 7125550), SEED(26819, 593129)},
         {SEED(3318706, 2157835), SEED(890124, 3375732), SEED(3348089, 5003315),
          SEED(6861865, 6830929), SEED(5373984, 7575303),
          SEED(3291887, 1564706), SEED(8144542, 7043611), SEED(2812876, 925395),
          SEED(6937239, 3290337), SEED(638120, 7555237), SEED(7012800, 152758),
          SEED(671474, 600033), SEED(4486326, 2822244), SEED(7322428, 6019280),
          SEED(7108806, 4089586), SEED(4183518, 4128466),
          SEED(8367936, 7125550)}},
        {{SEED(4298068, 4760805), SEED(5927590, 6212710),
          SEED(6044203, 7489378), SEED(202685, 5289387), SEED(2659719, 6316716),
          SEED(3688191, 8099882), SEED(6008900, 5883602), SEED(283746, 6948643),
          SEED(6860322, 3072886), SEED(3207187, 4294697),
          SEED(5391255, 6139027), SEED(4464462, 948575), SEED(2202688, 3028464),
          SEED(3906885, 2454036), SEED(445479, 7776334), SEED(7298327, 777386),
          SEED(7676823, 77188)},
         {SEED(1947934, 6393905), SEED(4298068, 4760805),
          SEED(5927590, 6212710), SEED(6044203, 7489378), SEED(202685, 5289387),
          SEED(2659719, 6316716), SEED(3688191, 8099882),
          SEED(6008900, 5883602), SEED(283746, 6948643), SEED(6860322, 3072886),
          SEED(3207187, 4294697), SEED(5391255, 6139027), SEED(4464462, 948575),
          SEED(2202688, 3028464), SEED(3906885, 2454036), SEED(445479, 7776334),
          SEED(7298327, 777386)},
         {SEED(7501012, 6066773), SEED(1947934, 6393905),
          SEED(4298068, 4760805), SEED(5927590, 6212710),
          SEED(6044203, 7489378), SEED(202685, 5289387), SEED(2659719, 6316716),
          SEED(3688191, 8099882), SEED(6008900, 5883602), SEED(283746, 6948643),
          SEED(6860322, 3072886), SEED(3207187, 4294697),
          SEED(5391255, 6139027), SEED(4464462, 948575), SEED(2202688, 3028464),
          SEED(3906885, 2454036), SEED(445479, 7776334)},
         {SEED(6489682, 6877104), SEED(7501012, 6066773),
          SEED(1947934, 6393905), SEED(4298068, 4760805),
          SEED(5927590, 6212710), SEED(6044203, 7489378), SEED(202685, 5289387),
          SEED(2659719, 6316716), SEED(3688191, 8099882),
          SEED(6008900, 5883602), SEED(283746, 6948643), SEED(6860322, 3072886),
          SEED(3207187, 4294697), SEED(5391255, 6139027), SEED(4464462, 948575),
          SEED(2202688, 3028464), SEED(3906885, 2454036)},
         {SEED(1445867, 278139), SEED(6489682, 6877104), SEED(7501012, 6066773),
          SEED(1947934, 6393905), SEED(4298068, 4760805),
          SEED(5927590, 6212710), SEED(6044203, 7489378), SEED(202685, 5289387),
          SEED(2659719, 6316716), SEED(3688191, 8099882),
          SEED(6008900, 5883602), SEED(283746, 6948643), SEED(6860322, 3072886),
          SEED(3207187, 4294697), SEED(5391255, 6139027), SEED(4464462, 948575),
          SEED(2202688, 3028464)},
         {SEED(2202688, 3028464), SEED(3906885, 2454036), SEED(445479, 7776334),
          SEED(7298327, 777386), SEED(7676823, 77188), SEED(609877, 5049531),
          SEED(8307298, 329107), SEED(5760457, 540735), SEED(1730971, 2216500),
          SEED(7841140, 2022018), SEED(6685544, 1960854),
          SEED(1544438, 4935027), SEED(6469666, 3920178), SEED(2953437, 618850),
          SEED(2761708, 4906971), SEED(6481536, 5361640),
          SEED(5176247, 871386)},
         {SEED(4464462, 948575), SEED(2202688, 3028464), SEED(3906885, 2454036),
          SEED(445479, 7776334), SEED(7298327, 777386), SEED(7676823, 77188),
          SEED(609877, 5049531), SEED(8307298, 329107), SEED(5760457, 540735),
          SEED(1730971, 2216500), SEED(7841140, 2022018),
          SEED(6685544, 1960854), SEED(1544438, 4935027),
          SEED(6469666, 3920178), SEED(2953437, 618850), SEED(2761708, 4906971),
          SEED(6481536, 5361640)},
         {SEED(5391255, 6139027), SEED(4464462, 948575), SEED(2202688, 3028464),
          SEED(3906885, 2454036), SEED(445479, 7776334), SEED(7298327, 777386),
          SEED(7676823, 77188), SEED(609877, 5049531), SEED(8307298, 329107),
          SEED(5760457, 540735), SEED(1730971, 2216500), SEED(7841140, 2022018),
          SEED(6685544, 1960854), SEED(1544438, 4935027),
          SEED(6469666, 3920178), SEED(2953437, 618850),
          SEED(2761708, 4906971)},
         {SEED(3207187, 4294697), SEED(5391255, 6139027), SEED(4464462, 948575),
          SEED(2202688, 3028464), SEED(3906885, 2454036), SEED(445479, 7776334),
          SEED(7298327, 777386), SEED(7676823, 77188), SEED(609877, 5049531),
          SEED(8307298, 329107), SEED(5760457, 540735), SEED(1730971, 2216500),
          SEED(7841140, 2022018), SEED(6685544, 1960854),
          SEED(1544438, 4935027), SEED(6469666, 3920178),
          SEED(2953437, 618850)},
         {SEED(6860322, 3072886), SEED(3207187, 4294697),
          SEED(5391255, 6139027), SEED(4464462, 948575), SEED(2202688, 3028464),
          SEED(3906885, 2454036), SEED(445479, 7776334), SEED(7298327, 777386),
          SEED(7676823, 77188), SEED(609877, 5049531), SEED(8307298, 329107),
          SEED(5760457, 540735), SEED(1730971, 2216500), SEED(7841140, 2022018),
          SEED(6685544, 1960854), SEED(1544438, 4935027),
          SEED(6469666, 3920178)},
         {SEED(283746, 6948643), SEED(6860322, 3072886), SEED(3207187, 4294697),
          SEED(5391255, 6139027), SEED(4464462, 948575), SEED(2202688, 3028464),
          SEED(3906885, 2454036), SEED(445479, 7776334), SEED(7298327, 777386),
          SEED(7676823, 77188), SEED(609877, 5049531), SEED(8307298, 329107),
          SEED(5760457, 540735), SEED(1730971, 2216500), SEED(7841140, 2022018),
          SEED(6685544, 1960854), SEED(1544438, 4935027)},
         {SEED(6008900, 5883602), SEED(283746, 6948643), SEED(6860322, 3072886),
          SEED(3207187, 4294697), SEED(5391255, 6139027), SEED(4464462, 948575),
          SEED(2202688, 3028464), SEED(3906885, 2454036), SEED(445479, 7776334),
          SEED(7298327, 777386), SEED(7676823, 77188), SEED(609877, 5049531),
          SEED(8307298, 329107), SEED(5760457, 540735), SEED(1730971, 2216500),
          SEED(7841140, 2022018), SEED(6685544, 1960854)},
         {SEED(3688191, 8099882), SEED(6008900, 5883602), SEED(283746, 6948643),
          SEED(6860322, 3072886), SEED(3207187, 4294697),
          SEED(5391255, 6139027), SEED(4464462, 948575), SEED(2202688, 3028464),
          SEED(3906885, 2454036), SEED(445479, 7776334), SEED(7298327, 777386),
          SEED(7676823, 77188), SEED(609877, 5049531), SEED(8307298, 329107),
          SEED(5760457, 540735), SEED(1730971, 2216500),
          SEED(7841140, 2022018)},
         {SEED(2659719, 6316716), SEED(3688191, 8099882),
          SEED(6008900, 5883602), SEED(283746, 6948643), SEED(6860322, 3072886),
          SEED(3207187, 4294697), SEED(5391255, 6139027), SEED(4464462, 948575),
          SEED(2202688, 3028464), SEED(3906885, 2454036), SEED(445479, 7776334),
          SEED(7298327, 777386), SEED(7676823, 77188), SEED(609877, 5049531),
          SEED(8307298, 329107), SEED(5760457, 540735), SEED(1730971, 2216500)},
         {SEED(202685, 5289387), SEED(2659719, 6316716), SEED(3688191, 8099882),
          SEED(6008900, 5883602), SEED(283746, 6948643), SEED(6860322, 3072886),
          SEED(3207187, 4294697), SEED(5391255, 6139027), SEED(4464462, 948575),
          SEED(2202688, 3028464), SEED(3906885, 2454036), SEED(445479, 7776334),
          SEED(7298327, 777386), SEED(7676823, 77188), SEED(609877, 5049531),
          SEED(8307298, 329107), SEED(5760457, 540735)},
         {SEED(6044203, 7489378), SEED(202685, 5289387), SEED(2659719, 6316716),
          SEED(3688191, 8099882), SEED(6008900, 5883602), SEED(283746, 6948643),
          SEED(6860322, 3072886), SEED(3207187, 4294697),
          SEED(5391255, 6139027), SEED(4464462, 948575), SEED(2202688, 3028464),
          SEED(3906885, 2454036), SEED(445479, 7776334), SEED(7298327, 777386),
          SEED(7676823, 77188), SEED(609877, 5049531), SEED(8307298, 329107)},
         {SEED(5927590, 6212710), SEED(6044203, 7489378), SEED(202685, 5289387),
          SEED(2659719, 6316716), SEED(3688191, 8099882),
          SEED(6008900, 5883602), SEED(283746, 6948643), SEED(6860322, 3072886),
          SEED(3207187, 4294697), SEED(5391255, 6139027), SEED(4464462, 948575),
          SEED(2202688, 3028464), SEED(3906885, 2454036), SEED(445479, 7776334),
          SEED(7298327, 777386), SEED(7676823, 77188), SEED(609877, 5049531)}},
    },
    {
        {{SEED(3473328, 1205396), SEED(4518122, 3539530),
          SEED(1465482, 1886922), SEED(304573, 7625340), SEED(5830244, 6753948),
          SEED(664892, 284126), SEED(7081289, 6124457), SEED(7318501, 8311500),
          SEED(2467273, 8032645), SEED(6832886, 1285130),
          SEED(3600364, 2146209), SEED(1981352, 2537891),
          SEED(7003751, 2611978), SEED(4872769, 5046389),
          SEED(4807152, 6749348), SEED(5283050, 3031476),
          SEED(8217820, 255619)},
         {SEED(5659456, 7009568), SEED(3473328, 1205396),
          SEED(4518122, 3539530), SEED(1465482, 1886922), SEED(304573, 7625340),
          SEED(5830244, 6753948), SEED(664892, 284126), SEED(7081289, 6124457),
          SEED(7318501, 8311500), SEED(2467273, 8032645),
          SEED(6832886, 1285130), SEED(3600364, 2146209),
          SEED(1981352, 2537891), SEED(7003751, 2611978),
          SEED(4872769, 5046389), SEED(4807152, 6749348),
          SEED(5283050, 3031476)},
         {SEED(5587623, 2268208), SEED(5659456, 7009568),
          SEED(3473328, 1205396), SEED(4518122, 3539530),
          SEED(1465482, 1886922), SEED(304573, 7625340), SEED(5830244, 6753948),
          SEED(664892, 284126), SEED(7081289, 6124457), SEED(7318501, 8311500),
          SEED(2467273, 8032645), SEED(6832886, 1285130),
          SEED(3600364, 2146209), SEED(1981352, 2537891),
          SEED(7003751, 2611978), SEED(4872769, 5046389),
          SEED(4807152, 6749348)},
         {SEED(6272634, 247662), SEED(5587623, 2268208), SEED(5659456, 7009568),
          SEED(3473328, 1205396), SEED(4518122, 3539530),
          SEED(1465482, 1886922), SEED(304573, 7625340), SEED(5830244, 6753948),
          SEED(664892, 284126), SEED(7081289, 6124457), SEED(7318501, 8311500),
          SEED(2467273, 8032645), SEED(6832886, 1285130),
          SEED(3600364, 2146209), SEED(1981352, 2537891),
          SEED(7003751, 2611978), SEED(4872769, 5046389)},
         {SEED(1002283, 197312), SEED(6272634, 247662), SEED(5587623, 2268208),
          SEED(5659456, 7009568), SEED(3473328, 1205396),
          SEED(4518122, 3539530), SEED(1465482, 1886922), SEED(304573, 7625340),
          SEED(5830244, 6753948), SEED(664892, 284126), SEED(7081289, 6124457),
          SEED(7318501, 8311500), SEED(2467273, 8032645),
          SEED(6832886, 1285130), SEED(3600364, 2146209),
          SEED(1981352, 2537891), SEED(7003751, 2611978)},
         {SEED(7003751, 2611978), SEED(4872769, 5046389),
          SEED(4807152, 6749348), SEED(5283050, 3031476), SEED(8217820, 255619),
          SEED(2808436, 921270), SEED(5825441, 5803680), SEED(2535589, 1964029),
          SEED(6225908, 7981302), SEED(7385966, 5468817),
          SEED(5453136, 6526524), SEED(5099937, 3586566), SEED(314750, 5699522),
          SEED(5983112, 2986255), SEED(2025734, 2924390),
          SEED(6705922, 7503340), SEED(2152140, 2282271)},
         {SEED(1981352, 2537891), SEED(7003751, 2611978),
          SEED(4872769, 5046389), SEED(4807152, 6749348),
          SEED(5283050, 3031476), SEED(8217820, 255619), SEED(2808436, 921270),
          SEED(5825441, 5803680), SEED(2535589, 1964029),
          SEED(6225908, 7981302), SEED(7385966, 5468817),
          SEED(5453136, 6526524), SEED(5099937, 3586566), SEED(314750, 5699522),
          SEED(5983112, 2986255), SEED(2025734, 2924390),
          SEED(6705922, 7503340)},
         {SEED(3600364, 2146209), SEED(1981352, 2537891),
          SEED(7003751, 2611978), SEED(4872769, 5046389),
          SEED(4807152, 6749348), SEED(5283050, 3031476), SEED(8217820, 255619),
          SEED(2808436, 921270), SEED(5825441, 5803680), SEED(2535589, 1964029),
          SEED(6225908, 7981302), SEED(7385966, 5468817),
          SEED(5453136, 6526524), SEED(5099937, 3586566), SEED(314750, 5699522),
          SEED(5983112, 2986255), SEED(2025734, 2924390)},
         {SEED(6832886, 1285130), SEED(3600364, 2146209),
          SEED(1981352, 2537891), SEED(7003751, 2611978),
          SEED(4872769, 5046389), SEED(4807152, 6749348),
          SEED(5283050, 3031476), SEED(8217820, 255619), SEED(2808436, 921270),
          SEED(5825441, 5803680), SEED(2535589, 1964029),
          SEED(6225908, 7981302), SEED(7385966, 5468817),
          SEED(5453136, 6526524), SEED(5099937, 3586566), SEED(314750, 5699522),
          SEED(5983112, 2986255)},
         {SEED(2467273, 8032645), SEED(6832886, 1285130),
          SEED(3600364, 2146209), SEED(1981352, 2537891),
          SEED(7003751, 2611978), SEED(4872769, 5046389),
          SEED(4807152, 6749348), SEED(5283050, 3031476), SEED(8217820, 255619),
          SEED(2808436, 921270), SEED(5825441, 5803680), SEED(2535589, 1964029),
          SEED(6225908, 7981302), SEED(7385966, 5468817),
          SEED(5453136, 6526524), SEED(5099937, 3586566),
          SEED(314750, 5699522)},
         {SEED(7318501, 8311500), SEED(2467273, 8032645),
          SEED(6832886, 1285130), SEED(3600364, 2146209),
          SEED(1981352, 2537891), SEED(7003751, 2611978),
          SEED(4872769, 5046389), SEED(4807152, 6749348),
          SEED(5283050, 3031476), SEED(8217820, 255619), SEED(2808436, 921270),
          SEED(5825441, 5803680), SEED(2535589, 1964029),
          SEED(6225908, 7981302), SEED(7385966, 5468817),
          SEED(5453136, 6526524), SEED(5099937, 3586566)},
         {SEED(7081289, 6124457), SEED(7318501, 8311500),
          SEED(2467273, 8032645), SEED(6832886, 1285130),
          SEED(3600364, 2146209), SEED(1981352, 2537891),
          SEED(7003751, 2611978), SEED(4872769, 5046389),
          SEED(4807152, 6749348), SEED(5283050, 3031476), SEED(8217820, 255619),
          SEED(2808436, 921270), SEED(5825441, 5803680), SEED(2535589, 1964029),
          SEED(6225908, 7981302), SEED(7385966, 5468817),
          SEED(5453136, 6526524)},
         {SEED(664892, 284126), SEED(7081289, 6124457), SEED(7318501, 8311500),
          SEED(2467273, 8032645), SEED(6832886, 1285130),
          SEED(3600364, 2146209), SEED(1981352, 2537891),
          SEED(7003751, 2611978), SEED(4872769, 5046389),
          SEED(4807152, 6749348), SEED(5283050, 3031476), SEED(8217820, 255619),
          SEED(2808436, 921270), SEED(5825441, 5803680), SEED(2535589, 1964029),
          SEED(6225908, 7981302), SEED(7385966, 5468817)},
         {SEED(5830244, 6753948), SEED(664892, 284126), SEED(7081289, 6124457),
          SEED(7318501, 8311500), SEED(2467273, 8032645),
          SEED(6832886, 1285130), SEED(3600364, 2146209),
          SEED(1981352, 2537891), SEED(7003751, 2611978),
          SEED(4872769, 5046389), SEED(4807152, 6749348),
          SEED(5283050, 3031476), SEED(8217820, 255619), SEED(2808436, 921270),
          SEED(5825441, 5803680), SEED(2535589, 1964029),
          SEED(6225908, 7981302)},
         {SEED(304573, 7625340), SEED(5830244, 6753948), SEED(664892, 284126),
          SEED(7081289, 6124457), SEED(7318501, 8311500),
          SEED(2467273, 8032645), SEED(6832886, 1285130),
          SEED(3600364, 2146209), SEED(1981352, 2537891),
          SEED(7003751, 2611978), SEED(4872769, 5046389),
          SEED(4807152, 6749348), SEED(5283050, 3031476), SEED(8217820, 255619),
          SEED(2808436, 921270), SEED(5825441, 5803680),
          SEED(2535589, 1964029)},
         {SEED(1465482, 1886922), SEED(304573, 7625340), SEED(5830244, 6753948),
          SEED(664892, 284126), SEED(7081289, 6124457), SEED(7318501, 8311500),
          SEED(2467273, 8032645), SEED(6832886, 1285130),
          SEED(3600364, 2146209), SEED(1981352, 2537891),
          SEED(7003751, 2611978), SEED(4872769, 5046389),
          SEED(4807152, 6749348), SEED(5283050, 3031476), SEED(8217820, 255619),
          SEED(2808436, 921270), SEED(5825441, 5803680)},
         {SEED(4518122, 3539530), SEED(1465482, 1886922), SEED(304573, 7625340),
          SEED(5830244, 6753948), SEED(664892, 284126), SEED(7081289, 6124457),
          SEED(7318501, 8311500), SEED(2467273, 8032645),
          SEED(6832886, 1285130), SEED(3600364, 2146209),
          SEED(1981352, 2537891), SEED(7003751, 2611978),
          SEED(4872769, 5046389), SEED(4807152, 6749348),
          SEED(5283050, 3031476), SEED(8217820, 255619),
          SEED(2808436, 921270)}},
        {{SEED(7359440, 7298396), SEED(926223, 4960921), SEED(6908972, 769753),
          SEED(7028768, 3533453), SEED(4341026, 4805348), SEED(3199252, 484205),
          SEED(4966021, 2738119), SEED(8114420, 7945245),
          SEED(4318236, 7478423), SEED(4186840, 4979102), SEED(808588, 5262887),
          SEED(5876009, 2894197), SEED(4020601, 7190508),
          SEED(1890280, 7871143), SEED(6795283, 2964757),
          SEED(1418360, 1409347), SEED(123653, 3636026)},
         {SEED(4464679, 52766), SEED(7359440, 7298396), SEED(926223, 4960921),
          SEED(6908972, 769753), SEED(7028768, 3533453), SEED(4341026, 4805348),
          SEED(3199252, 484205), SEED(4966021, 2738119), SEED(8114420, 7945245),
          SEED(4318236, 7478423), SEED(4186840, 4979102), SEED(808588, 5262887),
          SEED(5876009, 2894197), SEED(4020601, 7190508),
          SEED(1890280, 7871143), SEED(6795283, 2964757),
          SEED(1418360, 1409347)},
         {SEED(58520, 4942801), SEED(4464679, 52766), SEED(7359440, 7298396),
          SEED(926223, 4960921), SEED(6908972, 769753), SEED(7028768, 3533453),
          SEED(4341026, 4805348), SEED(3199252, 484205), SEED(4966021, 2738119),
          SEED(8114420, 7945245), SEED(4318236, 7478423),
          SEED(4186840, 4979102), SEED(808588, 5262887), SEED(5876009, 2894197),
          SEED(4020601, 7190508), SEED(1890280, 7871143),
          SEED(6795283, 2964757)},
         {SEED(5315647, 3734511), SEED(58520, 4942801), SEED(4464679, 52766),
          SEED(7359440, 7298396), SEED(926223, 4960921), SEED(6908972, 769753),
          SEED(7028768, 3533453), SEED(4341026, 4805348), SEED(3199252, 484205),
          SEED(4966021, 2738119), SEED(8114420, 7945245),
          SEED(4318236, 7478423), SEED(4186840, 4979102), SEED(808588, 5262887),
          SEED(5876009, 2894197), SEED(4020601, 7190508),
          SEED(1890280, 7871143)},
         {SEED(2816503, 4443456), SEED(5315647, 3734511), SEED(58520, 4942801),
          SEED(4464679, 52766), SEED(7359440, 7298396), SEED(926223, 4960921),
          SEED(6908972, 769753), SEED(7028768, 3533453), SEED(4341026, 4805348),
          SEED(3199252, 484205), SEED(4966021, 2738119), SEED(8114420, 7945245),
          SEED(4318236, 7478423), SEED(4186840, 4979102), SEED(808588, 5262887),
          SEED(5876009, 2894197), SEED(4020601, 7190508)},
         {SEED(4020601, 7190508), SEED(1890280, 7871143),
          SEED(6795283, 2964757), SEED(1418360, 1409347), SEED(123653, 3636026),
          SEED(4160188, 6814191), SEED(4348810, 2222801),
          SEED(7183160, 1213115), SEED(2710532, 4443638), SEED(154186, 8214854),
          SEED(2390664, 3609926), SEED(7478620, 8232529), SEED(4093819, 754737),
          SEED(2427956, 7995888), SEED(5780165, 2014344),
          SEED(7778836, 3853539), SEED(5752356, 7646779)},
         {SEED(5876009, 2894197), SEED(4020601, 7190508),
          SEED(1890280, 7871143), SEED(6795283, 2964757),
          SEED(1418360, 1409347), SEED(123653, 3636026), SEED(4160188, 6814191),
          SEED(4348810, 2222801), SEED(7183160, 1213115),
          SEED(2710532, 4443638), SEED(154186, 8214854), SEED(2390664, 3609926),
          SEED(7478620, 8232529), SEED(4093819, 754737), SEED(2427956, 7995888),
          SEED(5780165, 2014344), SEED(7778836, 3853539)},
         {SEED(808588, 5262887), SEED(5876009, 2894197), SEED(4020601, 7190508),
          SEED(1890280, 7871143), SEED(6795283, 2964757),
          SEED(1418360, 1409347), SEED(123653, 3636026), SEED(4160188, 6814191),
          SEED(4348810, 2222801), SEED(7183160, 1213115),
          SEED(2710532, 4443638), SEED(154186, 8214854), SEED(2390664, 3609926),
          SEED(7478620, 8232529), SEED(4093819, 754737), SEED(2427956, 7995888),
          SEED(5780165, 2014344)},
         {SEED(4186840, 4979102), SEED(808588, 5262887), SEED(5876009, 2894197),
          SEED(4020601, 7190508), SEED(1890280, 7871143),
          SEED(6795283, 2964757), SEED(1418360, 1409347), SEED(123653, 3636026),
          SEED(4160188, 6814191), SEED(4348810, 2222801),
          SEED(7183160, 1213115), SEED(2710532, 4443638), SEED(154186, 8214854),
          SEED(2390664, 3609926), SEED(7478620, 8232529), SEED(4093819, 754737),
          SEED(2427956, 7995888)},
         {SEED(4318236, 7478423), SEED(4186840, 4979102), SEED(808588, 5262887),
          SEED(5876009, 2894197), SEED(4020601, 7190508),
          SEED(1890280, 7871143), SEED(6795283, 2964757),
          SEED(1418360, 1409347), SEED(123653, 3636026), SEED(4160188, 6814191),
          SEED(4348810, 2222801), SEED(7183160, 1213115),
          SEED(2710532, 4443638), SEED(154186, 8214854), SEED(2390664, 3609926),
          SEED(7478620, 8232529), SEED(4093819, 754737)},
         {SEED(8114420, 7945245), SEED(4318236, 7478423),
          SEED(4186840, 4979102), SEED(808588, 5262887), SEED(5876009, 2894197),
          SEED(4020601, 7190508), SEED(1890280, 7871143),
          SEED(6795283, 2964757), SEED(1418360, 1409347), SEED(123653, 3636026),
          SEED(4160188, 6814191), SEED(4348810, 2222801),
          SEED(7183160, 1213115), SEED(2710532, 4443638), SEED(154186, 8214854),
          SEED(2390664, 3609926), SEED(7478620, 8232529)},
         {SEED(4966021, 2738119), SEED(8114420, 7945245),
          SEED(4318236, 7478423), SEED(4186840, 4979102), SEED(808588, 5262887),
          SEED(5876009, 2894197), SEED(4020601, 7190508),
          SEED(1890280, 7871143), SEED(6795283, 2964757),
          SEED(1418360, 1409347), SEED(123653, 3636026), SEED(4160188, 6814191),
          SEED(4348810, 2222801), SEED(7183160, 1213115),
          SEED(2710532, 4443638), SEED(154186, 8214854),
          SEED(2390664, 3609926)},
         {SEED(3199252, 484205), SEED(4966021, 2738119), SEED(8114420, 7945245),
          SEED(4318236, 7478423), SEED(4186840, 4979102), SEED(808588, 5262887),
          SEED(5876009, 2894197), SEED(4020601, 7190508),
          SEED(1890280, 7871143), SEED(6795283, 2964757),
          SEED(1418360, 1409347), SEED(123653, 3636026), SEED(4160188, 6814191),
          SEED(4348810, 2222801), SEED(7183160, 1213115),
          SEED(2710532, 4443638), SEED(154186, 8214854)},
         {SEED(4341026, 4805348), SEED(3199252, 484205), SEED(4966021, 2738119),
          SEED(8114420, 7945245), SEED(4318236, 7478423),
          SEED(4186840, 4979102), SEED(808588, 5262887), SEED(5876009, 2894197),
          SEED(4020601, 7190508), SEED(1890280, 7871143),
          SEED(6795283, 2964757), SEED(1418360, 1409347), SEED(123653, 3636026),
          SEED(4160188, 6814191), SEED(4348810, 2222801),
          SEED(7183160, 1213115), SEED(2710532, 4443638)},
         {SEED(7028768, 3533453), SEED(4341026, 4805348), SEED(3199252, 484205),
          SEED(4966021, 2738119), SEED(8114420, 7945245),
          SEED(4318236, 7478423), SEED(4186840, 4979102), SEED(808588, 5262887),
          SEED(5876009, 2894197), SEED(4020601, 7190508),
          SEED(1890280, 7871143), SEED(6795283, 2964757),
          SEED(1418360, 1409347), SEED(123653, 3636026), SEED(4160188, 6814191),
          SEED(4348810, 2222801), SEED(7183160, 1213115)},
         {SEED(6908972, 769753), SEED(7028768, 3533453), SEED(4341026, 4805348),
          SEED(3199252, 484205), SEED(4966021, 2738119), SEED(8114420, 7945245),
          SEED(4318236, 7478423), SEED(4186840, 4979102), SEED(808588, 5262887),
          SEED(5876009, 2894197), SEED(4020601, 7190508),
          SEED(1890280, 7871143), SEED(6795283, 2964757),
          SEED(1418360, 1409347), SEED(123653, 3636026), SEED(4160188, 6814191),
          SEED(4348810, 2222801)},
         {SEED(926223, 4960921), SEED(6908972, 769753), SEED(7028768, 3533453),
          SEED(4341026, 4805348), SEED(3199252, 484205), SEED(4966021, 2738119),
          SEED(8114420, 7945245), SEED(4318236, 7478423),
          SEED(4186840, 4979102), SEED(808588, 5262887), SEED(5876009, 2894197),
          SEED(4020601, 7190508), SEED(1890280, 7871143),
          SEED(6795283, 2964757), SEED(1418360, 1409347), SEED(123653, 3636026),
          SEED(4160188, 6814191)}},
        {{SEED(7233632, 7129919), SEED(994507, 7355154), SEED(6480792, 1589481),
          SEED(5172940, 1105352), SEED(1154990, 3089046), SEED(4187984, 516000),
          SEED(45022, 6215161), SEED(5521459, 2118224), SEED(4118267, 3505010),
          SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361), SEED(2456222, 4988990),
          SEED(2151915, 2559567), SEED(3571547, 7107933)},
         {SEED(4726537, 1808371), SEED(7233632, 7129919), SEED(994507, 7355154),
          SEED(6480792, 1589481), SEED(5172940, 1105352),
          SEED(1154990, 3089046), SEED(4187984, 516000), SEED(45022, 6215161),
          SEED(5521459, 2118224), SEED(4118267, 3505010),
          SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361), SEED(2456222, 4988990),
          SEED(2151915, 2559567)},
         {SEED(7324855, 3664919), SEED(4726537, 1808371),
          SEED(7233632, 7129919), SEED(994507, 7355154), SEED(6480792, 1589481),
          SEED(5172940, 1105352), SEED(1154990, 3089046), SEED(4187984, 516000),
          SEED(45022, 6215161), SEED(5521459, 2118224), SEED(4118267, 3505010),
          SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361), SEED(2456222, 4988990)},
         {SEED(548406, 6578472), SEED(7324855, 3664919), SEED(4726537, 1808371),
          SEED(7233632, 7129919), SEED(994507, 7355154), SEED(6480792, 1589481),
          SEED(5172940, 1105352), SEED(1154990, 3089046), SEED(4187984, 516000),
          SEED(45022, 6215161), SEED(5521459, 2118224), SEED(4118267, 3505010),
          SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361)},
         {SEED(925076, 1509908), SEED(548406, 6578472), SEED(7324855, 3664919),
          SEED(4726537, 1808371), SEED(7233632, 7129919), SEED(994507, 7355154),
          SEED(6480792, 1589481), SEED(5172940, 1105352),
          SEED(1154990, 3089046), SEED(4187984, 516000), SEED(45022, 6215161),
          SEED(5521459, 2118224), SEED(4118267, 3505010),
          SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941)},
         {SEED(8109963, 2358941), SEED(8319177, 2543361),
          SEED(2456222, 4988990), SEED(2151915, 2559567),
          SEED(3571547, 7107933), SEED(3045648, 6613919), SEED(949485, 1139993),
          SEED(959333, 7859865), SEED(1054673, 5988950), SEED(3775828, 3970816),
          SEED(7827013, 3081529), SEED(4837853, 8309618),
          SEED(5800104, 8147890), SEED(4187698, 961648), SEED(3311548, 2517847),
          SEED(2597664, 3263511), SEED(24230, 7574825)},
         {SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361), SEED(2456222, 4988990),
          SEED(2151915, 2559567), SEED(3571547, 7107933),
          SEED(3045648, 6613919), SEED(949485, 1139993), SEED(959333, 7859865),
          SEED(1054673, 5988950), SEED(3775828, 3970816),
          SEED(7827013, 3081529), SEED(4837853, 8309618),
          SEED(5800104, 8147890), SEED(4187698, 961648), SEED(3311548, 2517847),
          SEED(2597664, 3263511)},
         {SEED(4749579, 5823078), SEED(3595777, 6294150),
          SEED(8109963, 2358941), SEED(8319177, 2543361),
          SEED(2456222, 4988990), SEED(2151915, 2559567),
          SEED(3571547, 7107933), SEED(3045648, 6613919), SEED(949485, 1139993),
          SEED(959333, 7859865), SEED(1054673, 5988950), SEED(3775828, 3970816),
          SEED(7827013, 3081529), SEED(4837853, 8309618),
          SEED(5800104, 8147890), SEED(4187698, 961648),
          SEED(3311548, 2517847)},
         {SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361), SEED(2456222, 4988990),
          SEED(2151915, 2559567), SEED(3571547, 7107933),
          SEED(3045648, 6613919), SEED(949485, 1139993), SEED(959333, 7859865),
          SEED(1054673, 5988950), SEED(3775828, 3970816),
          SEED(7827013, 3081529), SEED(4837853, 8309618),
          SEED(5800104, 8147890), SEED(4187698, 961648)},
         {SEED(4118267, 3505010), SEED(5767770, 7506837),
          SEED(4749579, 5823078), SEED(3595777, 6294150),
          SEED(8109963, 2358941), SEED(8319177, 2543361),
          SEED(2456222, 4988990), SEED(2151915, 2559567),
          SEED(3571547, 7107933), SEED(3045648, 6613919), SEED(949485, 1139993),
          SEED(959333, 7859865), SEED(1054673, 5988950), SEED(3775828, 3970816),
          SEED(7827013, 3081529), SEED(4837853, 8309618),
          SEED(5800104, 8147890)},
         {SEED(5521459, 2118224), SEED(4118267, 3505010),
          SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361), SEED(2456222, 4988990),
          SEED(2151915, 2559567), SEED(3571547, 7107933),
          SEED(3045648, 6613919), SEED(949485, 1139993), SEED(959333, 7859865),
          SEED(1054673, 5988950), SEED(3775828, 3970816),
          SEED(7827013, 3081529), SEED(4837853, 8309618)},
         {SEED(45022, 6215161), SEED(5521459, 2118224), SEED(4118267, 3505010),
          SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361), SEED(2456222, 4988990),
          SEED(2151915, 2559567), SEED(3571547, 7107933),
          SEED(3045648, 6613919), SEED(949485, 1139993), SEED(959333, 7859865),
          SEED(1054673, 5988950), SEED(3775828, 3970816),
          SEED(7827013, 3081529)},
         {SEED(4187984, 516000), SEED(45022, 6215161), SEED(5521459, 2118224),
          SEED(4118267, 3505010), SEED(5767770, 7506837),
          SEED(4749579, 5823078), SEED(3595777, 6294150),
          SEED(8109963, 2358941), SEED(8319177, 2543361),
          SEED(2456222, 4988990), SEED(2151915, 2559567),
          SEED(3571547, 7107933), SEED(3045648, 6613919), SEED(949485, 1139993),
          SEED(959333, 7859865), SEED(1054673, 5988950),
          SEED(3775828, 3970816)},
         {SEED(1154990, 3089046), SEED(4187984, 516000), SEED(45022, 6215161),
          SEED(5521459, 2118224), SEED(4118267, 3505010),
          SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361), SEED(2456222, 4988990),
          SEED(2151915, 2559567), SEED(3571547, 7107933),
          SEED(3045648, 6613919), SEED(949485, 1139993), SEED(959333, 7859865),
          SEED(1054673, 5988950)},
         {SEED(5172940, 1105352), SEED(1154990, 3089046), SEED(4187984, 516000),
          SEED(45022, 6215161), SEED(5521459, 2118224), SEED(4118267, 3505010),
          SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361), SEED(2456222, 4988990),
          SEED(2151915, 2559567), SEED(3571547, 7107933),
          SEED(3045648, 6613919), SEED(949485, 1139993), SEED(959333, 7859865)},
         {SEED(6480792, 1589481), SEED(5172940, 1105352),
          SEED(1154990, 3089046), SEED(4187984, 516000), SEED(45022, 6215161),
          SEED(5521459, 2118224), SEED(4118267, 3505010),
          SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361), SEED(2456222, 4988990),
          SEED(2151915, 2559567), SEED(3571547, 7107933),
          SEED(3045648, 6613919), SEED(949485, 1139993)},
         {SEED(994507, 7355154), SEED(6480792, 1589481), SEED(5172940, 1105352),
          SEED(1154990, 3089046), SEED(4187984, 516000), SEED(45022, 6215161),
          SEED(5521459, 2118224), SEED(4118267, 3505010),
          SEED(5767770, 7506837), SEED(4749579, 5823078),
          SEED(3595777, 6294150), SEED(8109963, 2358941),
          SEED(8319177, 2543361), SEED(2456222, 4988990),
          SEED(2151915, 2559567), SEED(3571547, 7107933),
          SEED(3045648, 6613919)}},
    },
    {
        {{SEED(7701460, 503640), SEED(2832342, 1324494), SEED(3156457, 6777221),
          SEED(6100470, 6016141), SEED(574984, 993945), SEED(4838511, 1429881),
          SEED(6526906, 3121787), SEED(6372382, 1091327),
          SEED(5357698, 3104569), SEED(3822446, 7405818),
          SEED(1158686, 2093503), SEED(7350483, 5518948),
          SEED(7751541, 4429235), SEED(7292298, 6310499),
          SEED(3043144, 3822470), SEED(2545905, 4852446),
          SEED(6146068, 6238172)},
         {SEED(6721052, 7232117), SEED(7701460, 503640), SEED(2832342, 1324494),
          SEED(3156457, 6777221), SEED(6100470, 6016141), SEED(574984, 993945),
          SEED(4838511, 1429881), SEED(6526906, 3121787),
          SEED(6372382, 1091327), SEED(5357698, 3104569),
          SEED(3822446, 7405818), SEED(1158686, 2093503),
          SEED(7350483, 5518948), SEED(7751541, 4429235),
          SEED(7292298, 6310499), SEED(3043144, 3822470),
          SEED(2545905, 4852446)},
         {SEED(257767, 2479980), SEED(6721052, 7232117), SEED(7701460, 503640),
          SEED(2832342, 1324494), SEED(3156457, 6777221),
          SEED(6100470, 6016141), SEED(574984, 993945), SEED(4838511, 1429881),
          SEED(6526906, 3121787), SEED(6372382, 1091327),
          SEED(5357698, 3104569), SEED(3822446, 7405818),
          SEED(1158686, 2093503), SEED(7350483, 5518948),
          SEED(7751541, 4429235), SEED(7292298, 6310499),
          SEED(3043144, 3822470)},
         {SEED(6199601, 2211083), SEED(257767, 2479980), SEED(6721052, 7232117),
          SEED(7701460, 503640), SEED(2832342, 1324494), SEED(3156457, 6777221),
          SEED(6100470, 6016141), SEED(574984, 993945), SEED(4838511, 1429881),
          SEED(6526906, 3121787), SEED(6372382, 1091327),
          SEED(5357698, 3104569), SEED(3822446, 7405818),
          SEED(1158686, 2093503), SEED(7350483, 5518948),
          SEED(7751541, 4429235), SEED(7292298, 6310499)},
         {SEED(1736032, 7634994), SEED(6199601, 2211083), SEED(257767, 2479980),
          SEED(6721052, 7232117), SEED(7701460, 503640), SEED(2832342, 1324494),
          SEED(3156457, 6777221), SEED(6100470, 6016141), SEED(574984, 993945),
          SEED(4838511, 1429881), SEED(6526906, 3121787),
          SEED(6372382, 1091327), SEED(5357698, 3104569),
          SEED(3822446, 7405818), SEED(1158686, 2093503),
          SEED(7350483, 5518948), SEED(7751541, 4429235)},
         {SEED(7751541, 4429235), SEED(7292298, 6310499),
          SEED(3043144, 3822470), SEED(2545905, 4852446),
          SEED(6146068, 6238172), SEED(2862949, 7462367),
          SEED(4694044, 6591314), SEED(5172683, 5685893), SEED(742772, 2911572),
          SEED(5141146, 1976734), SEED(3679825, 7724986),
          SEED(7565031, 5991446), SEED(7009449, 5050699),
          SEED(6454008, 5182677), SEED(779302, 3583348), SEED(7001389, 5629664),
          SEED(1204415, 7669384)},
         {SEED(7350483, 5518948), SEED(7751541, 4429235),
          SEED(7292298, 6310499), SEED(3043144, 3822470),
          SEED(2545905, 4852446), SEED(6146068, 6238172),
          SEED(2862949, 7462367), SEED(4694044, 6591314),
          SEED(5172683, 5685893), SEED(742772, 2911572), SEED(5141146, 1976734),
          SEED(3679825, 7724986), SEED(7565031, 5991446),
          SEED(7009449, 5050699), SEED(6454008, 5182677), SEED(779302, 3583348),
          SEED(7001389, 5629664)},
         {SEED(1158686, 2093503), SEED(7350483, 5518948),
          SEED(7751541, 4429235), SEED(7292298, 6310499),
          SEED(3043144, 3822470), SEED(2545905, 4852446),
          SEED(6146068, 6238172), SEED(2862949, 7462367),
          SEED(4694044, 6591314), SEED(5172683, 5685893), SEED(742772, 2911572),
          SEED(5141146, 1976734), SEED(3679825, 7724986),
          SEED(7565031, 5991446), SEED(7009449, 5050699),
          SEED(6454008, 5182677), SEED(779302, 3583348)},
         {SEED(3822446, 7405818), SEED(1158686, 2093503),
          SEED(7350483, 5518948), SEED(7751541, 4429235),
          SEED(7292298, 6310499), SEED(3043144, 3822470),
          SEED(2545905, 4852446), SEED(6146068, 6238172),
          SEED(2862949, 7462367), SEED(4694044, 6591314),
          SEED(5172683, 5685893), SEED(742772, 2911572), SEED(5141146, 1976734),
          SEED(3679825, 7724986), SEED(7565031, 5991446),
          SEED(7009449, 5050699), SEED(6454008, 5182677)},
         {SEED(5357698, 3104569), SEED(3822446, 7405818),
          SEED(1158686, 2093503), SEED(7350483, 5518948),
          SEED(7751541, 4429235), SEED(7292298, 6310499),
          SEED(3043144, 3822470), SEED(2545905, 4852446),
          SEED(6146068, 6238172), SEED(2862949, 7462367),
          SEED(4694044, 6591314), SEED(5172683, 5685893), SEED(742772, 2911572),
          SEED(5141146, 1976734), SEED(3679825, 7724986),
          SEED(7565031, 5991446), SEED(7009449, 5050699)},
         {SEED(6372382, 1091327), SEED(5357698, 3104569),
          SEED(3822446, 7405818), SEED(1158686, 2093503),
          SEED(7350483, 5518948), SEED(7751541, 4429235),
          SEED(7292298, 6310499), SEED(3043144, 3822470),
          SEED(2545905, 4852446), SEED(6146068, 6238172),
          SEED(2862949, 7462367), SEED(4694044, 6591314),
          SEED(5172683, 5685893), SEED(742772, 2911572), SEED(5141146, 1976734),
          SEED(3679825, 7724986), SEED(7565031, 5991446)},
         {SEED(6526906, 3121787), SEED(6372382, 1091327),
          SEED(5357698, 3104569), SEED(3822446, 7405818),
          SEED(1158686, 2093503), SEED(7350483, 5518948),
          SEED(7751541, 4429235), SEED(7292298, 6310499),
          SEED(3043144, 3822470), SEED(2545905, 4852446),
          SEED(6146068, 6238172), SEED(2862949, 7462367),
          SEED(4694044, 6591314), SEED(5172683, 5685893), SEED(742772, 2911572),
          SEED(5141146, 1976734), SEED(3679825, 7724986)},
         {SEED(4838511, 1429881), SEED(6526906, 3121787),
          SEED(6372382, 1091327), SEED(5357698, 3104569),
          SEED(3822446, 7405818), SEED(1158686, 2093503),
          SEED(7350483, 5518948), SEED(7751541, 4429235),
          SEED(7292298, 6310499), SEED(3043144, 3822470),
          SEED(2545905, 4852446), SEED(6146068, 6238172),
          SEED(2862949, 7462367), SEED(4694044, 6591314),
          SEED(5172683, 5685893), SEED(742772, 2911572),
          SEED(5141146, 1976734)},
         {SEED(574984, 993945), SEED(4838511, 1429881), SEED(6526906, 3121787),
          SEED(6372382, 1091327), SEED(5357698, 3104569),
          SEED(3822446, 7405818), SEED(1158686, 2093503),
          SEED(7350483, 5518948), SEED(7751541, 4429235),
          SEED(7292298, 6310499), SEED(3043144, 3822470),
          SEED(2545905, 4852446), SEED(6146068, 6238172),
          SEED(2862949, 7462367), SEED(4694044, 6591314),
          SEED(5172683, 5685893), SEED(742772, 2911572)},
         {SEED(6100470, 6016141), SEED(574984, 993945), SEED(4838511, 1429881),
          SEED(6526906, 3121787), SEED(6372382, 1091327),
          SEED(5357698, 3104569), SEED(3822446, 7405818),
          SEED(1158686, 2093503), SEED(7350483, 5518948),
          SEED(7751541, 4429235), SEED(7292298, 6310499),
          SEED(3043144, 3822470), SEED(2545905, 4852446),
          SEED(6146068, 6238172), SEED(2862949, 7462367),
          SEED(4694044, 6591314), SEED(5172683, 5685893)},
         {SEED(3156457, 6777221), SEED(6100470, 6016141), SEED(574984, 993945),
          SEED(4838511, 1429881), SEED(6526906, 3121787),
          SEED(6372382, 1091327), SEED(5357698, 3104569),
          SEED(3822446, 7405818), SEED(1158686, 2093503),
          SEED(7350483, 5518948), SEED(7751541, 4429235),
          SEED(7292298, 6310499), SEED(3043144, 3822470),
          SEED(2545905, 4852446), SEED(6146068, 6238172),
          SEED(2862949, 7462367), SEED(4694044, 6591314)},
         {SEED(2832342, 1324494), SEED(3156457, 6777221),
          SEED(6100470, 6016141), SEED(574984, 993945), SEED(4838511, 1429881),
          SEED(6526906, 3121787), SEED(6372382, 1091327),
          SEED(5357698, 3104569), SEED(3822446, 7405818),
          SEED(1158686, 2093503), SEED(7350483, 5518948),
          SEED(7751541, 4429235), SEED(7292298, 6310499),
          SEED(3043144, 3822470), SEED(2545905, 4852446),
          SEED(6146068, 6238172), SEED(2862949, 7462367)}},
        {{SEED(5585584, 5172198), SEED(903256, 4120546), SEED(6411300, 6225589),
          SEED(7984158, 3571873), SEED(6708525, 8306850),
          SEED(6260604, 3684040), SEED(2285014, 5472630),
          SEED(5569071, 8326460), SEED(6443980, 3672170),
          SEED(5848294, 2095352), SEED(4108339, 4645542),
          SEED(6383838, 7452588), SEED(968230, 4557888), SEED(4448863, 4287846),
          SEED(6907734, 2521296), SEED(4882698, 2179163),
          SEED(2871150, 6404008)},
         {SEED(1191067, 6322251), SEED(5585584, 5172198), SEED(903256, 4120546),
          SEED(6411300, 6225589), SEED(7984158, 3571873),
          SEED(6708525, 8306850), SEED(6260604, 3684040),
          SEED(2285014, 5472630), SEED(5569071, 8326460),
          SEED(6443980, 3672170), SEED(5848294, 2095352),
          SEED(4108339, 4645542), SEED(6383838, 7452588), SEED(968230, 4557888),
          SEED(4448863, 4287846), SEED(6907734, 2521296),
          SEED(4882698, 2179163)},
         {SEED(4478248, 5751037), SEED(1191067, 6322251),
          SEED(5585584, 5172198), SEED(903256, 4120546), SEED(6411300, 6225589),
          SEED(7984158, 3571873), SEED(6708525, 8306850),
          SEED(6260604, 3684040), SEED(2285014, 5472630),
          SEED(5569071, 8326460), SEED(6443980, 3672170),
          SEED(5848294, 2095352), SEED(4108339, 4645542),
          SEED(6383838, 7452588), SEED(968230, 4557888), SEED(4448863, 4287846),
          SEED(6907734, 2521296)},
         {SEED(4930426, 358278), SEED(4478248, 5751037), SEED(1191067, 6322251),
          SEED(5585584, 5172198), SEED(903256, 4120546), SEED(6411300, 6225589),
          SEED(7984158, 3571873), SEED(6708525, 8306850),
          SEED(6260604, 3684040), SEED(2285014, 5472630),
          SEED(5569071, 8326460), SEED(6443980, 3672170),
          SEED(5848294, 2095352), SEED(4108339, 4645542),
          SEED(6383838, 7452588), SEED(968230, 4557888),
          SEED(4448863, 4287846)},
         {SEED(5352119, 19784), SEED(4930426, 358278), SEED(4478248, 5751037),
          SEED(1191067, 6322251), SEED(5585584, 5172198), SEED(903256, 4120546),
          SEED(6411300, 6225589), SEED(7984158, 3571873),
          SEED(6708525, 8306850), SEED(6260604, 3684040),
          SEED(2285014, 5472630), SEED(5569071, 8326460),
          SEED(6443980, 3672170), SEED(5848294, 2095352),
          SEED(4108339, 4645542), SEED(6383838, 7452588),
          SEED(968230, 4557888)},
         {SEED(968230, 4557888), SEED(4448863, 4287846), SEED(6907734, 2521296),
          SEED(4882698, 2179163), SEED(2871150, 6404008),
          SEED(7713588, 1488157), SEED(7006850, 7036523), SEED(842229, 6287737),
          SEED(1540178, 8288311), SEED(860231, 6211498), SEED(2152265, 7427106),
          SEED(4289784, 6408649), SEED(4600841, 3768572),
          SEED(1995117, 7772932), SEED(7329168, 7962663),
          SEED(7614249, 2466378), SEED(3512688, 1048580)},
         {SEED(6383838, 7452588), SEED(968230, 4557888), SEED(4448863, 4287846),
          SEED(6907734, 2521296), SEED(4882698, 2179163),
          SEED(2871150, 6404008), SEED(7713588, 1488157),
          SEED(7006850, 7036523), SEED(842229, 6287737), SEED(1540178, 8288311),
          SEED(860231, 6211498), SEED(2152265, 7427106), SEED(4289784, 6408649),
          SEED(4600841, 3768572), SEED(1995117, 7772932),
          SEED(7329168, 7962663), SEED(7614249, 2466378)},
         {SEED(4108339, 4645542), SEED(6383838, 7452588), SEED(968230, 4557888),
          SEED(4448863, 4287846), SEED(6907734, 2521296),
          SEED(4882698, 2179163), SEED(2871150, 6404008),
          SEED(7713588, 1488157), SEED(7006850, 7036523), SEED(842229, 6287737),
          SEED(1540178, 8288311), SEED(860231, 6211498), SEED(2152265, 7427106),
          SEED(4289784, 6408649), SEED(4600841, 3768572),
          SEED(1995117, 7772932), SEED(7329168, 7962663)},
         {SEED(5848294, 2095352), SEED(4108339, 4645542),
          SEED(6383838, 7452588), SEED(968230, 4557888), SEED(4448863, 4287846),
          SEED(6907734, 2521296), SEED(4882698, 2179163),
          SEED(2871150, 6404008), SEED(7713588, 1488157),
          SEED(7006850, 7036523), SEED(842229, 6287737), SEED(1540178, 8288311),
          SEED(860231, 6211498), SEED(2152265, 7427106), SEED(4289784, 6408649),
          SEED(4600841, 3768572), SEED(1995117, 7772932)},
         {SEED(6443980, 3672170), SEED(5848294, 2095352),
          SEED(4108339, 4645542), SEED(6383838, 7452588), SEED(968230, 4557888),
          SEED(4448863, 4287846), SEED(6907734, 2521296),
          SEED(4882698, 2179163), SEED(2871150, 6404008),
          SEED(7713588, 1488157), SEED(7006850, 7036523), SEED(842229, 6287737),
          SEED(1540178, 8288311), SEED(860231, 6211498), SEED(2152265, 7427106),
          SEED(4289784, 6408649), SEED(4600841, 3768572)},
         {SEED(5569071, 8326460), SEED(6443980, 3672170),
          SEED(5848294, 2095352), SEED(4108339, 4645542),
          SEED(6383838, 7452588), SEED(968230, 4557888), SEED(4448863, 4287846),
          SEED(6907734, 2521296), SEED(4882698, 2179163),
          SEED(2871150, 6404008), SEED(7713588, 1488157),
          SEED(7006850, 7036523), SEED(842229, 6287737), SEED(1540178, 8288311),
          SEED(860231, 6211498), SEED(2152265, 7427106),
          SEED(4289784, 6408649)},
         {SEED(2285014, 5472630), SEED(5569071, 8326460),
          SEED(6443980, 3672170), SEED(5848294, 2095352),
          SEED(4108339, 4645542), SEED(6383838, 7452588), SEED(968230, 4557888),
          SEED(4448863, 4287846), SEED(6907734, 2521296),
          SEED(4882698, 2179163), SEED(2871150, 6404008),
          SEED(7713588, 1488157), SEED(7006850, 7036523), SEED(842229, 6287737),
          SEED(1540178, 8288311), SEED(860231, 6211498),
          SEED(2152265, 7427106)},
         {SEED(6260604, 3684040), SEED(2285014, 5472630),
          SEED(5569071, 8326460), SEED(6443980, 3672170),
          SEED(5848294, 2095352), SEED(4108339, 4645542),
          SEED(6383838, 7452588), SEED(968230, 4557888), SEED(4448863, 4287846),
          SEED(6907734, 2521296), SEED(4882698, 2179163),
          SEED(2871150, 6404008), SEED(7713588, 1488157),
          SEED(7006850, 7036523), SEED(842229, 6287737), SEED(1540178, 8288311),
          SEED(860231, 6211498)},
         {SEED(6708525, 8306850), SEED(6260604, 3684040),
          SEED(2285014, 5472630), SEED(5569071, 8326460),
          SEED(6443980, 3672170), SEED(5848294, 2095352),
          SEED(4108339, 4645542), SEED(6383838, 7452588), SEED(968230, 4557888),
          SEED(4448863, 4287846), SEED(6907734, 2521296),
          SEED(4882698, 2179163), SEED(2871150, 6404008),
          SEED(7713588, 1488157), SEED(7006850, 7036523), SEED(842229, 6287737),
          SEED(1540178, 8288311)},
         {SEED(7984158, 3571873), SEED(6708525, 8306850),
          SEED(6260604, 3684040), SEED(2285014, 5472630),
          SEED(5569071, 8326460), SEED(6443980, 3672170),
          SEED(5848294, 2095352), SEED(4108339, 4645542),
          SEED(6383838, 7452588), SEED(968230, 4557888), SEED(4448863, 4287846),
          SEED(6907734, 2521296), SEED(4882698, 2179163),
          SEED(2871150, 6404008), SEED(7713588, 1488157),
          SEED(7006850, 7036523), SEED(842229, 6287737)},
         {SEED(6411300, 6225589), SEED(7984158, 3571873),
          SEED(6708525, 8306850), SEED(6260604, 3684040),
          SEED(2285014, 5472630), SEED(5569071, 8326460),
          SEED(6443980, 3672170), SEED(5848294, 2095352),
          SEED(4108339, 4645542), SEED(6383838, 7452588), SEED(968230, 4557888),
          SEED(4448863, 4287846), SEED(6907734, 2521296),
          SEED(4882698, 2179163), SEED(2871150, 6404008),
          SEED(7713588, 1488157), SEED(7006850, 7036523)},
         {SEED(903256, 4120546), SEED(6411300, 6225589), SEED(7984158, 3571873),
          SEED(6708525, 8306850), SEED(6260604, 3684040),
          SEED(2285014, 5472630), SEED(5569071, 8326460),
          SEED(6443980, 3672170), SEED(5848294, 2095352),
          SEED(4108339, 4645542), SEED(6383838, 7452588), SEED(968230, 4557888),
          SEED(4448863, 4287846), SEED(6907734, 2521296),
          SEED(4882698, 2179163), SEED(2871150, 6404008),
          SEED(7713588, 1488157)}},
        {{SEED(2507240, 3227798), SEED(5583071, 7543554),
          SEED(6981552, 5071571), SEED(1655346, 2990506),
          SEED(8052297, 2281241), SEED(7107622, 2425670),
          SEED(5977771, 2894868), SEED(883970, 5553476), SEED(6745960, 8347316),
          SEED(816650, 408993), SEED(389291, 685033), SEED(3286135, 3063019),
          SEED(1312744, 1249646), SEED(2312752, 1709223), SEED(949261, 880414),
          SEED(4278049, 4535397), SEED(2909154, 1745999)},
         {SEED(2572843, 4027241), SEED(2507240, 3227798),
          SEED(5583071, 7543554), SEED(6981552, 5071571),
          SEED(1655346, 2990506), SEED(8052297, 2281241),
          SEED(7107622, 2425670), SEED(5977771, 2894868), SEED(883970, 5553476),
          SEED(6745960, 8347316), SEED(816650, 408993), SEED(389291, 685033),
          SEED(3286135, 3063019), SEED(1312744, 1249646),
          SEED(2312752, 1709223), SEED(949261, 880414), SEED(4278049, 4535397)},
         {SEED(5933395, 7525903), SEED(2572843, 4027241),
          SEED(2507240, 3227798), SEED(5583071, 7543554),
          SEED(6981552, 5071571), SEED(1655346, 2990506),
          SEED(8052297, 2281241), SEED(7107622, 2425670),
          SEED(5977771, 2894868), SEED(883970, 5553476), SEED(6745960, 8347316),
          SEED(816650, 408993), SEED(389291, 685033), SEED(3286135, 3063019),
          SEED(1312744, 1249646), SEED(2312752, 1709223), SEED(949261, 880414)},
         {SEED(7930813, 5951985), SEED(5933395, 7525903),
          SEED(2572843, 4027241), SEED(2507240, 3227798),
          SEED(5583071, 7543554), SEED(6981552, 5071571),
          SEED(1655346, 2990506), SEED(8052297, 2281241),
          SEED(7107622, 2425670), SEED(5977771, 2894868), SEED(883970, 5553476),
          SEED(6745960, 8347316), SEED(816650, 408993), SEED(389291, 685033),
          SEED(3286135, 3063019), SEED(1312744, 1249646),
          SEED(2312752, 1709223)},
         {SEED(7895823, 864169), SEED(7930813, 5951985), SEED(5933395, 7525903),
          SEED(2572843, 4027241), SEED(2507240, 3227798),
          SEED(5583071, 7543554), SEED(6981552, 5071571),
          SEED(1655346, 2990506), SEED(8052297, 2281241),
          SEED(7107622, 2425670), SEED(5977771, 2894868), SEED(883970, 5553476),
          SEED(6745960, 8347316), SEED(816650, 408993), SEED(389291, 685033),
          SEED(3286135, 3063019), SEED(1312744, 1249646)},
         {SEED(1312744, 1249646), SEED(2312752, 1709223), SEED(949261, 880414),
          SEED(4278049, 4535397), SEED(2909154, 1745999), SEED(3788226, 802127),
          SEED(7993908, 4648685), SEED(6097582, 7906703),
          SEED(3297994, 3031797), SEED(7235647, 1872248),
          SEED(6718331, 1740637), SEED(2691636, 8220457),
          SEED(7959834, 4303829), SEED(4433208, 6638093),
          SEED(8255997, 7917186), SEED(4499850, 4538243),
          SEED(376981, 1317020)},
         {SEED(3286135, 3063019), SEED(1312744, 1249646),
          SEED(2312752, 1709223), SEED(949261, 880414), SEED(4278049, 4535397),
          SEED(2909154, 1745999), SEED(3788226, 802127), SEED(7993908, 4648685),
          SEED(6097582, 7906703), SEED(3297994, 3031797),
          SEED(7235647, 1872248), SEED(6718331, 1740637),
          SEED(2691636, 8220457), SEED(7959834, 4303829),
          SEED(4433208, 6638093), SEED(8255997, 7917186),
          SEED(4499850, 4538243)},
         {SEED(389291, 685033), SEED(3286135, 3063019), SEED(1312744, 1249646),
          SEED(2312752, 1709223), SEED(949261, 880414), SEED(4278049, 4535397),
          SEED(2909154, 1745999), SEED(3788226, 802127), SEED(7993908, 4648685),
          SEED(6097582, 7906703), SEED(3297994, 3031797),
          SEED(7235647, 1872248), SEED(6718331, 1740637),
          SEED(2691636, 8220457), SEED(7959834, 4303829),
          SEED(4433208, 6638093), SEED(8255997, 7917186)},
         {SEED(816650, 408993), SEED(389291, 685033), SEED(3286135, 3063019),
          SEED(1312744, 1249646), SEED(2312752, 1709223), SEED(949261, 880414),
          SEED(4278049, 4535397), SEED(2909154, 1745999), SEED(3788226, 802127),
          SEED(7993908, 4648685), SEED(6097582, 7906703),
          SEED(3297994, 3031797), SEED(7235647, 1872248),
          SEED(6718331, 1740637), SEED(2691636, 8220457),
          SEED(7959834, 4303829), SEED(4433208, 6638093)},
         {SEED(6745960, 8347316), SEED(816650, 408993), SEED(389291, 685033),
          SEED(3286135, 3063019), SEED(1312744, 1249646),
          SEED(2312752, 1709223), SEED(949261, 880414), SEED(4278049, 4535397),
          SEED(2909154, 1745999), SEED(3788226, 802127), SEED(7993908, 4648685),
          SEED(6097582, 7906703), SEED(3297994, 3031797),
          SEED(7235647, 1872248), SEED(6718331, 1740637),
          SEED(2691636, 8220457), SEED(7959834, 4303829)},
         {SEED(883970, 5553476), SEED(6745960, 8347316), SEED(816650, 408993),
          SEED(389291, 685033), SEED(3286135, 3063019), SEED(1312744, 1249646),
          SEED(2312752, 1709223), SEED(949261, 880414), SEED(4278049, 4535397),
          SEED(2909154, 1745999), SEED(3788226, 802127), SEED(7993908, 4648685),
          SEED(6097582, 7906703), SEED(3297994, 3031797),
          SEED(7235647, 1872248), SEED(6718331, 1740637),
          SEED(2691636, 8220457)},
         {SEED(5977771, 2894868), SEED(883970, 5553476), SEED(6745960, 8347316),
          SEED(816650, 408993), SEED(389291, 685033), SEED(3286135, 3063019),
          SEED(1312744, 1249646), SEED(2312752, 1709223), SEED(949261, 880414),
          SEED(4278049, 4535397), SEED(2909154, 1745999), SEED(3788226, 802127),
          SEED(7993908, 4648685), SEED(6097582, 7906703),
          SEED(3297994, 3031797), SEED(7235647, 1872248),
          SEED(6718331, 1740637)},
         {SEED(7107622, 2425670), SEED(5977771, 2894868), SEED(883970, 5553476),
          SEED(6745960, 8347316), SEED(816650, 408993), SEED(389291, 685033),
          SEED(3286135, 3063019), SEED(1312744, 1249646),
          SEED(2312752, 1709223), SEED(949261, 880414), SEED(4278049, 4535397),
          SEED(2909154, 1745999), SEED(3788226, 802127), SEED(7993908, 4648685),
          SEED(6097582, 7906703), SEED(3297994, 3031797),
          SEED(7235647, 1872248)},
         {SEED(8052297, 2281241), SEED(7107622, 2425670),
          SEED(5977771, 2894868), SEED(883970, 5553476), SEED(6745960, 8347316),
          SEED(816650, 408993), SEED(389291, 685033), SEED(3286135, 3063019),
          SEED(1312744, 1249646), SEED(2312752, 1709223), SEED(949261, 880414),
          SEED(4278049, 4535397), SEED(2909154, 1745999), SEED(3788226, 802127),
          SEED(7993908, 4648685), SEED(6097582, 7906703),
          SEED(3297994, 3031797)},
         {SEED(1655346, 2990506), SEED(8052297, 2281241),
          SEED(7107622, 2425670), SEED(5977771, 2894868), SEED(883970, 5553476),
          SEED(6745960, 8347316), SEED(816650, 408993), SEED(389291, 685033),
          SEED(3286135, 3063019), SEED(1312744, 1249646),
          SEED(2312752, 1709223), SEED(949261, 880414), SEED(4278049, 4535397),
          SEED(2909154, 1745999), SEED(3788226, 802127), SEED(7993908, 4648685),
          SEED(6097582, 7906703)},
         {SEED(6981552, 5071571), SEED(1655346, 2990506),
          SEED(8052297, 2281241), SEED(7107622, 2425670),
          SEED(5977771, 2894868), SEED(883970, 5553476), SEED(6745960, 8347316),
          SEED(816650, 408993), SEED(389291, 685033), SEED(3286135, 3063019),
          SEED(1312744, 1249646), SEED(2312752, 1709223), SEED(949261, 880414),
          SEED(4278049, 4535397), SEED(2909154, 1745999), SEED(3788226, 802127),
          SEED(7993908, 4648685)},
         {SEED(5583071, 7543554), SEED(6981552, 5071571),
          SEED(1655346, 2990506), SEED(8052297, 2281241),
          SEED(7107622, 2425670), SEED(5977771, 2894868), SEED(883970, 5553476),
          SEED(6745960, 8347316), SEED(816650, 408993), SEED(389291, 685033),
          SEED(3286135, 3063019), SEED(1312744, 1249646),
          SEED(2312752, 1709223), SEED(949261, 880414), SEED(4278049, 4535397),
          SEED(2909154, 1745999), SEED(3788226, 802127)}},
    },
    {
        {{SEED(5747616, 3906417), SEED(6089648, 1998081), SEED(2182436, 703809),
          SEED(3440272, 3697899), SEED(1704592, 5788700),
          SEED(3007184, 6405345), SEED(2439418, 4677290),
          SEED(7826100, 7599270), SEED(6929093, 7943752),
          SEED(7363820, 8196423), SEED(2023596, 5902078), SEED(623686, 1824652),
          SEED(4877918, 3434377), SEED(3611520, 7225115),
          SEED(6158275, 2673598), SEED(7277741, 4616964),
          SEED(5457992, 1027409)},
         {SEED(7162584, 6816109), SEED(5747616, 3906417),
          SEED(6089648, 1998081), SEED(2182436, 703809), SEED(3440272, 3697899),
          SEED(1704592, 5788700), SEED(3007184, 6405345),
          SEED(2439418, 4677290), SEED(7826100, 7599270),
          SEED(6929093, 7943752), SEED(7363820, 8196423),
          SEED(2023596, 5902078), SEED(623686, 1824652), SEED(4877918, 3434377),
          SEED(3611520, 7225115), SEED(6158275, 2673598),
          SEED(7277741, 4616964)},
         {SEED(2329405, 8314864), SEED(7162584, 6816109),
          SEED(5747616, 3906417), SEED(6089648, 1998081), SEED(2182436, 703809),
          SEED(3440272, 3697899), SEED(1704592, 5788700),
          SEED(3007184, 6405345), SEED(2439418, 4677290),
          SEED(7826100, 7599270), SEED(6929093, 7943752),
          SEED(7363820, 8196423), SEED(2023596, 5902078), SEED(623686, 1824652),
          SEED(4877918, 3434377), SEED(3611520, 7225115),
          SEED(6158275, 2673598)},
         {SEED(8340711, 3377407), SEED(2329405, 8314864),
          SEED(7162584, 6816109), SEED(5747616, 3906417),
          SEED(6089648, 1998081), SEED(2182436, 703809), SEED(3440272, 3697899),
          SEED(1704592, 5788700), SEED(3007184, 6405345),
          SEED(2439418, 4677290), SEED(7826100, 7599270),
          SEED(6929093, 7943752), SEED(7363820, 8196423),
          SEED(2023596, 5902078), SEED(623686, 1824652), SEED(4877918, 3434377),
          SEED(3611520, 7225115)},
         {SEED(1312560, 834589), SEED(8340711, 3377407), SEED(2329405, 8314864),
          SEED(7162584, 6816109), SEED(5747616, 3906417),
          SEED(6089648, 1998081), SEED(2182436, 703809), SEED(3440272, 3697899),
          SEED(1704592, 5788700), SEED(3007184, 6405345),
          SEED(2439418, 4677290), SEED(7826100, 7599270),
          SEED(6929093, 7943752), SEED(7363820, 8196423),
          SEED(2023596, 5902078), SEED(623686, 1824652),
          SEED(4877918, 3434377)},
         {SEED(4877918, 3434377), SEED(3611520, 7225115),
          SEED(6158275, 2673598), SEED(7277741, 4616964),
          SEED(5457992, 1027409), SEED(2740432, 5889680),
          SEED(3650230, 5709399), SEED(2744944, 1493146),
          SEED(4899787, 4142754), SEED(2729380, 5980884), SEED(983588, 503267),
          SEED(1815732, 2852638), SEED(2948182, 4164893), SEED(3317573, 718637),
          SEED(1205545, 5522825), SEED(3134463, 1285113),
          SEED(3554302, 797242)},
         {SEED(623686, 1824652), SEED(4877918, 3434377), SEED(3611520, 7225115),
          SEED(6158275, 2673598), SEED(7277741, 4616964),
          SEED(5457992, 1027409), SEED(2740432, 5889680),
          SEED(3650230, 5709399), SEED(2744944, 1493146),
          SEED(4899787, 4142754), SEED(2729380, 5980884), SEED(983588, 503267),
          SEED(1815732, 2852638), SEED(2948182, 4164893), SEED(3317573, 718637),
          SEED(1205545, 5522825), SEED(3134463, 1285113)},
         {SEED(2023596, 5902078), SEED(623686, 1824652), SEED(4877918, 3434377),
          SEED(3611520, 7225115), SEED(6158275, 2673598),
          SEED(7277741, 4616964), SEED(5457992, 1027409),
          SEED(2740432, 5889680), SEED(3650230, 5709399),
          SEED(2744944, 1493146), SEED(4899787, 4142754),
          SEED(2729380, 5980884), SEED(983588, 503267), SEED(1815732, 2852638),
          SEED(2948182, 4164893), SEED(3317573, 718637),
          SEED(1205545, 5522825)},
         {SEED(7363820, 8196423), SEED(2023596, 5902078), SEED(623686, 1824652),
          SEED(4877918, 3434377), SEED(3611520, 7225115),
          SEED(6158275, 2673598), SEED(7277741, 4616964),
          SEED(5457992, 1027409), SEED(2740432, 5889680),
          SEED(3650230, 5709399), SEED(2744944, 1493146),
          SEED(4899787, 4142754), SEED(2729380, 5980884), SEED(983588, 503267),
          SEED(1815732, 2852638), SEED(2948182, 4164893),
          SEED(3317573, 718637)},
         {SEED(6929093, 7943752), SEED(7363820, 8196423),
          SEED(2023596, 5902078), SEED(623686, 1824652), SEED(4877918, 3434377),
          SEED(3611520, 7225115), SEED(6158275, 2673598),
          SEED(7277741, 4616964), SEED(5457992, 1027409),
          SEED(2740432, 5889680), SEED(3650230, 5709399),
          SEED(2744944, 1493146), SEED(4899787, 4142754),
          SEED(2729380, 5980884), SEED(983588, 503267), SEED(1815732, 2852638),
          SEED(2948182, 4164893)},
         {SEED(7826100, 7599270), SEED(6929093, 7943752),
          SEED(7363820, 8196423), SEED(2023596, 5902078), SEED(623686, 1824652),
          SEED(4877918, 3434377), SEED(3611520, 7225115),
          SEED(6158275, 2673598), SEED(7277741, 4616964),
          SEED(5457992, 1027409), SEED(2740432, 5889680),
          SEED(3650230, 5709399), SEED(2744944, 1493146),
          SEED(4899787, 4142754), SEED(2729380, 5980884), SEED(983588, 503267),
          SEED(1815732, 2852638)},
         {SEED(2439418, 4677290), SEED(7826100, 7599270),
          SEED(6929093, 7943752), SEED(7363820, 8196423),
          SEED(2023596, 5902078), SEED(623686, 1824652), SEED(4877918, 3434377),
          SEED(3611520, 7225115), SEED(6158275, 2673598),
          SEED(7277741, 4616964), SEED(5457992, 1027409),
          SEED(2740432, 5889680), SEED(3650230, 5709399),
          SEED(2744944, 1493146), SEED(4899787, 4142754),
          SEED(2729380, 5980884), SEED(983588, 503267)},
         {SEED(3007184, 6405345), SEED(2439418, 4677290),
          SEED(7826100, 7599270), SEED(6929093, 7943752),
          SEED(7363820, 8196423), SEED(2023596, 5902078), SEED(623686, 1824652),
          SEED(4877918, 3434377), SEED(3611520, 7225115),
          SEED(6158275, 2673598), SEED(7277741, 4616964),
          SEED(5457992, 1027409), SEED(2740432, 5889680),
          SEED(3650230, 5709399), SEED(2744944, 1493146),
          SEED(4899787, 4142754), SEED(2729380, 5980884)},
         {SEED(1704592, 5788700), SEED(3007184, 6405345),
          SEED(2439418, 4677290), SEED(7826100, 7599270),
          SEED(6929093, 7943752), SEED(7363820, 8196423),
          SEED(2023596, 5902078), SEED(623686, 1824652), SEED(4877918, 3434377),
          SEED(3611520, 7225115), SEED(6158275, 2673598),
          SEED(7277741, 4616964), SEED(5457992, 1027409),
          SEED(2740432, 5889680), SEED(3650230, 5709399),
          SEED(2744944, 1493146), SEED(4899787, 4142754)},
         {SEED(3440272, 3697899), SEED(1704592, 5788700),
          SEED(3007184, 6405345), SEED(2439418, 4677290),
          SEED(7826100, 7599270), SEED(6929093, 7943752),
          SEED(7363820, 8196423), SEED(2023596, 5902078), SEED(623686, 1824652),
          SEED(4877918, 3434377), SEED(3611520, 7225115),
          SEED(6158275, 2673598), SEED(7277741, 4616964),
          SEED(5457992, 1027409), SEED(2740432, 5889680),
          SEED(3650230, 5709399), SEED(2744944, 1493146)},
         {SEED(2182436, 703809), SEED(3440272, 3697899), SEED(1704592, 5788700),
          SEED(3007184, 6405345), SEED(2439418, 4677290),
          SEED(7826100, 7599270), SEED(6929093, 7943752),
          SEED(7363820, 8196423), SEED(2023596, 5902078), SEED(623686, 1824652),
          SEED(4877918, 3434377), SEED(3611520, 7225115),
          SEED(6158275, 2673598), SEED(7277741, 4616964),
          SEED(5457992, 1027409), SEED(2740432, 5889680),
          SEED(3650230, 5709399)},
         {SEED(6089648, 1998081), SEED(2182436, 703809), SEED(3440272, 3697899),
          SEED(1704592, 5788700), SEED(3007184, 6405345),
          SEED(2439418, 4677290), SEED(7826100, 7599270),
          SEED(6929093, 7943752), SEED(7363820, 8196423),
          SEED(2023596, 5902078), SEED(623686, 1824652), SEED(4877918, 3434377),
          SEED(3611520, 7225115), SEED(6158275, 2673598),
          SEED(7277741, 4616964), SEED(5457992, 1027409),
          SEED(2740432, 5889680)}},
        {{SEED(7517720, 7851583), SEED(502800, 6559910), SEED(2096012, 1671926),
          SEED(1886920, 5252350), SEED(1979020, 2545044),
          SEED(6082810, 1627569), SEED(2187314, 5022697),
          SEED(6170864, 5832022), SEED(7742268, 3564897), SEED(891260, 1744566),
          SEED(6550428, 8097019), SEED(8346112, 5351395),
          SEED(5245154, 3857566), SEED(1440233, 5631031),
          SEED(3262632, 2330136), SEED(1679184, 3069943),
          SEED(6466085, 7082952)},
         {SEED(56497, 1239389), SEED(7517720, 7851583), SEED(502800, 6559910),
          SEED(2096012, 1671926), SEED(1886920, 5252350),
          SEED(1979020, 2545044), SEED(6082810, 1627569),
          SEED(2187314, 5022697), SEED(6170864, 5832022),
          SEED(7742268, 3564897), SEED(891260, 1744566), SEED(6550428, 8097019),
          SEED(8346112, 5351395), SEED(5245154, 3857566),
          SEED(1440233, 5631031), SEED(3262632, 2330136),
          SEED(1679184, 3069943)},
         {SEED(3566104, 8322293), SEED(56497, 1239389), SEED(7517720, 7851583),
          SEED(502800, 6559910), SEED(2096012, 1671926), SEED(1886920, 5252350),
          SEED(1979020, 2545044), SEED(6082810, 1627569),
          SEED(2187314, 5022697), SEED(6170864, 5832022),
          SEED(7742268, 3564897), SEED(891260, 1744566), SEED(6550428, 8097019),
          SEED(8346112, 5351395), SEED(5245154, 3857566),
          SEED(1440233, 5631031), SEED(3262632, 2330136)},
         {SEED(5358644, 4002062), SEED(3566104, 8322293), SEED(56497, 1239389),
          SEED(7517720, 7851583), SEED(502800, 6559910), SEED(2096012, 1671926),
          SEED(1886920, 5252350), SEED(1979020, 2545044),
          SEED(6082810, 1627569), SEED(2187314, 5022697),
          SEED(6170864, 5832022), SEED(7742268, 3564897), SEED(891260, 1744566),
          SEED(6550428, 8097019), SEED(8346112, 5351395),
          SEED(5245154, 3857566), SEED(1440233, 5631031)},
         {SEED(1943033, 3802333), SEED(5358644, 4002062),
          SEED(3566104, 8322293), SEED(56497, 1239389), SEED(7517720, 7851583),
          SEED(502800, 6559910), SEED(2096012, 1671926), SEED(1886920, 5252350),
          SEED(1979020, 2545044), SEED(6082810, 1627569),
          SEED(2187314, 5022697), SEED(6170864, 5832022),
          SEED(7742268, 3564897), SEED(891260, 1744566), SEED(6550428, 8097019),
          SEED(8346112, 5351395), SEED(5245154, 3857566)},
         {SEED(5245154, 3857566), SEED(1440233, 5631031),
          SEED(3262632, 2330136), SEED(1679184, 3069943),
          SEED(6466085, 7082952), SEED(1434910, 6224014),
          SEED(6704094, 1537212), SEED(4313756, 4228511),
          SEED(2533260, 1687452), SEED(1087760, 800478), SEED(7920990, 1919157),
          SEED(2229810, 8059909), SEED(925710, 1974456), SEED(6302035, 6322474),
          SEED(6017236, 7803037), SEED(4871244, 5027076),
          SEED(1880027, 6657051)},
         {SEED(8346112, 5351395), SEED(5245154, 3857566),
          SEED(1440233, 5631031), SEED(3262632, 2330136),
          SEED(1679184, 3069943), SEED(6466085, 7082952),
          SEED(1434910, 6224014), SEED(6704094, 1537212),
          SEED(4313756, 4228511), SEED(2533260, 1687452), SEED(1087760, 800478),
          SEED(7920990, 1919157), SEED(2229810, 8059909), SEED(925710, 1974456),
          SEED(6302035, 6322474), SEED(6017236, 7803037),
          SEED(4871244, 5027076)},
         {SEED(6550428, 8097019), SEED(8346112, 5351395),
          SEED(5245154, 3857566), SEED(1440233, 5631031),
          SEED(3262632, 2330136), SEED(1679184, 3069943),
          SEED(6466085, 7082952), SEED(1434910, 6224014),
          SEED(6704094, 1537212), SEED(4313756, 4228511),
          SEED(2533260, 1687452), SEED(1087760, 800478), SEED(7920990, 1919157),
          SEED(2229810, 8059909), SEED(925710, 1974456), SEED(6302035, 6322474),
          SEED(6017236, 7803037)},
         {SEED(891260, 1744566), SEED(6550428, 8097019), SEED(8346112, 5351395),
          SEED(5245154, 3857566), SEED(1440233, 5631031),
          SEED(3262632, 2330136), SEED(1679184, 3069943),
          SEED(6466085, 7082952), SEED(1434910, 6224014),
          SEED(6704094, 1537212), SEED(4313756, 4228511),
          SEED(2533260, 1687452), SEED(1087760, 800478), SEED(7920990, 1919157),
          SEED(2229810, 8059909), SEED(925710, 1974456),
          SEED(6302035, 6322474)},
         {SEED(7742268, 3564897), SEED(891260, 1744566), SEED(6550428, 8097019),
          SEED(8346112, 5351395), SEED(5245154, 3857566),
          SEED(1440233, 5631031), SEED(3262632, 2330136),
          SEED(1679184, 3069943), SEED(6466085, 7082952),
          SEED(1434910, 6224014), SEED(6704094, 1537212),
          SEED(4313756, 4228511), SEED(2533260, 1687452), SEED(1087760, 800478),
          SEED(7920990, 1919157), SEED(2229810, 8059909),
          SEED(925710, 1974456)},
         {SEED(6170864, 5832022), SEED(7742268, 3564897), SEED(891260, 1744566),
          SEED(6550428, 8097019), SEED(8346112, 5351395),
          SEED(5245154, 3857566), SEED(1440233, 5631031),
          SEED(3262632, 2330136), SEED(1679184, 3069943),
          SEED(6466085, 7082952), SEED(1434910, 6224014),
          SEED(6704094, 1537212), SEED(4313756, 4228511),
          SEED(2533260, 1687452), SEED(1087760, 800478), SEED(7920990, 1919157),
          SEED(2229810, 8059909)},
         {SEED(2187314, 5022697), SEED(6170864, 5832022),
          SEED(7742268, 3564897), SEED(891260, 1744566), SEED(6550428, 8097019),
          SEED(8346112, 5351395), SEED(5245154, 3857566),
          SEED(1440233, 5631031), SEED(3262632, 2330136),
          SEED(1679184, 3069943), SEED(6466085, 7082952),
          SEED(1434910, 6224014), SEED(6704094, 1537212),
          SEED(4313756, 4228511), SEED(2533260, 1687452), SEED(1087760, 800478),
          SEED(7920990, 1919157)},
         {SEED(6082810, 1627569), SEED(2187314, 5022697),
          SEED(6170864, 5832022), SEED(7742268, 3564897), SEED(891260, 1744566),
          SEED(6550428, 8097019), SEED(8346112, 5351395),
          SEED(5245154, 3857566), SEED(1440233, 5631031),
          SEED(3262632, 2330136), SEED(1679184, 3069943),
          SEED(6466085, 7082952), SEED(1434910, 6224014),
          SEED(6704094, 1537212), SEED(4313756, 4228511),
          SEED(2533260, 1687452), SEED(1087760, 800478)},
         {SEED(1979020, 2545044), SEED(6082810, 1627569),
          SEED(2187314, 5022697), SEED(6170864, 5832022),
          SEED(7742268, 3564897), SEED(891260, 1744566), SEED(6550428, 8097019),
          SEED(8346112, 5351395), SEED(5245154, 3857566),
          SEED(1440233, 5631031), SEED(3262632, 2330136),
          SEED(1679184, 3069943), SEED(6466085, 7082952),
          SEED(1434910, 6224014), SEED(6704094, 1537212),
          SEED(4313756, 4228511), SEED(2533260, 1687452)},
         {SEED(1886920, 5252350), SEED(1979020, 2545044),
          SEED(6082810, 1627569), SEED(2187314, 5022697),
          SEED(6170864, 5832022), SEED(7742268, 3564897), SEED(891260, 1744566),
          SEED(6550428, 8097019), SEED(8346112, 5351395),
          SEED(5245154, 3857566), SEED(1440233, 5631031),
          SEED(3262632, 2330136), SEED(1679184, 3069943),
          SEED(6466085, 7082952), SEED(1434910, 6224014),
          SEED(6704094, 1537212), SEED(4313756, 4228511)},
         {SEED(2096012, 1671926), SEED(1886920, 5252350),
          SEED(1979020, 2545044), SEED(6082810, 1627569),
          SEED(2187314, 5022697), SEED(6170864, 5832022),
          SEED(7742268, 3564897), SEED(891260, 1744566), SEED(6550428, 8097019),
          SEED(8346112, 5351395), SEED(5245154, 3857566),
          SEED(1440233, 5631031), SEED(3262632, 2330136),
          SEED(1679184, 3069943), SEED(6466085, 7082952),
          SEED(1434910, 6224014), SEED(6704094, 1537212)},
         {SEED(502800, 6559910), SEED(2096012, 1671926), SEED(1886920, 5252350),
          SEED(1979020, 2545044), SEED(6082810, 1627569),
          SEED(2187314, 5022697), SEED(6170864, 5832022),
          SEED(7742268, 3564897), SEED(891260, 1744566), SEED(6550428, 8097019),
          SEED(8346112, 5351395), SEED(5245154, 3857566),
          SEED(1440233, 5631031), SEED(3262632, 2330136),
          SEED(1679184, 3069943), SEED(6466085, 7082952),
          SEED(1434910, 6224014)}},
        {{SEED(6921368, 8122475), SEED(745724, 2654497), SEED(6503892, 7785486),
          SEED(2713734, 6925648), SEED(2305343, 6301881), SEED(4108106, 668951),
          SEED(3981832, 5227342), SEED(1079991, 2520699),
          SEED(2450644, 6641566), SEED(4136472, 5942147),
          SEED(2743119, 5387286), SEED(2602647, 2696170),
          SEED(2324580, 7266413), SEED(5812373, 1773833),
          SEED(2054315, 8150631), SEED(5085470, 5379706),
          SEED(6974821, 803786)},
         {SEED(891556, 7105668), SEED(6921368, 8122475), SEED(745724, 2654497),
          SEED(6503892, 7785486), SEED(2713734, 6925648),
          SEED(2305343, 6301881), SEED(4108106, 668951), SEED(3981832, 5227342),
          SEED(1079991, 2520699), SEED(2450644, 6641566),
          SEED(4136472, 5942147), SEED(2743119, 5387286),
          SEED(2602647, 2696170), SEED(2324580, 7266413),
          SEED(5812373, 1773833), SEED(2054315, 8150631),
          SEED(5085470, 5379706)},
         {SEED(7799204, 3916746), SEED(891556, 7105668), SEED(6921368, 8122475),
          SEED(745724, 2654497), SEED(6503892, 7785486), SEED(2713734, 6925648),
          SEED(2305343, 6301881), SEED(4108106, 668951), SEED(3981832, 5227342),
          SEED(1079991, 2520699), SEED(2450644, 6641566),
          SEED(4136472, 5942147), SEED(2743119, 5387286),
          SEED(2602647, 2696170), SEED(2324580, 7266413),
          SEED(5812373, 1773833), SEED(2054315, 8150631)},
         {SEED(169599, 7547510), SEED(7799204, 3916746), SEED(891556, 7105668),
          SEED(6921368, 8122475), SEED(745724, 2654497), SEED(6503892, 7785486),
          SEED(2713734, 6925648), SEED(2305343, 6301881), SEED(4108106, 668951),
          SEED(3981832, 5227342), SEED(1079991, 2520699),
          SEED(2450644, 6641566), SEED(4136472, 5942147),
          SEED(2743119, 5387286), SEED(2602647, 2696170),
          SEED(2324580, 7266413), SEED(5812373, 1773833)},
         {SEED(6558097, 4428330), SEED(169599, 7547510), SEED(7799204, 3916746),
          SEED(891556, 7105668), SEED(6921368, 8122475), SEED(745724, 2654497),
          SEED(6503892, 7785486), SEED(2713734, 6925648),
          SEED(2305343, 6301881), SEED(4108106, 668951), SEED(3981832, 5227342),
          SEED(1079991, 2520699), SEED(2450644, 6641566),
          SEED(4136472, 5942147), SEED(2743119, 5387286),
          SEED(2602647, 2696170), SEED(2324580, 7266413)},
         {SEED(2324580, 7266413), SEED(5812373, 1773833),
          SEED(2054315, 8150631), SEED(5085470, 5379706), SEED(6974821, 803786),
          SEED(2813262, 7453524), SEED(5152500, 5815762),
          SEED(5423901, 5264787), SEED(263090, 284082), SEED(6557479, 359733),
          SEED(1364987, 3670273), SEED(1379185, 2531172),
          SEED(7144019, 3642893), SEED(5026879, 4867732),
          SEED(2082157, 6180124), SEED(6046257, 7579), SEED(4016434, 1892383)},
         {SEED(2602647, 2696170), SEED(2324580, 7266413),
          SEED(5812373, 1773833), SEED(2054315, 8150631),
          SEED(5085470, 5379706), SEED(6974821, 803786), SEED(2813262, 7453524),
          SEED(5152500, 5815762), SEED(5423901, 5264787), SEED(263090, 284082),
          SEED(6557479, 359733), SEED(1364987, 3670273), SEED(1379185, 2531172),
          SEED(7144019, 3642893), SEED(5026879, 4867732),
          SEED(2082157, 6180124), SEED(6046257, 7579)},
         {SEED(2743119, 5387286), SEED(2602647, 2696170),
          SEED(2324580, 7266413), SEED(5812373, 1773833),
          SEED(2054315, 8150631), SEED(5085470, 5379706), SEED(6974821, 803786),
          SEED(2813262, 7453524), SEED(5152500, 5815762),
          SEED(5423901, 5264787), SEED(263090, 284082), SEED(6557479, 359733),
          SEED(1364987, 3670273), SEED(1379185, 2531172),
          SEED(7144019, 3642893), SEED(5026879, 4867732),
          SEED(2082157, 6180124)},
         {SEED(4136472, 5942147), SEED(2743119, 5387286),
          SEED(2602647, 2696170), SEED(2324580, 7266413),
          SEED(5812373, 1773833), SEED(2054315, 8150631),
          SEED(5085470, 5379706), SEED(6974821, 803786), SEED(2813262, 7453524),
          SEED(5152500, 5815762), SEED(5423901, 5264787), SEED(263090, 284082),
          SEED(6557479, 359733), SEED(1364987, 3670273), SEED(1379185, 2531172),
          SEED(7144019, 3642893), SEED(5026879, 4867732)},
         {SEED(2450644, 6641566), SEED(4136472, 5942147),
          SEED(2743119, 5387286), SEED(2602647, 2696170),
          SEED(2324580, 7266413), SEED(5812373, 1773833),
          SEED(2054315, 8150631), SEED(5085470, 5379706), SEED(6974821, 803786),
          SEED(2813262, 7453524), SEED(5152500, 5815762),
          SEED(5423901, 5264787), SEED(263090, 284082), SEED(6557479, 359733),
          SEED(1364987, 3670273), SEED(1379185, 2531172),
          SEED(7144019, 3642893)},
         {SEED(1079991, 2520699), SEED(2450644, 6641566),
          SEED(4136472, 5942147), SEED(2743119, 5387286),
          SEED(2602647, 2696170), SEED(2324580, 7266413),
          SEED(5812373, 1773833), SEED(2054315, 8150631),
          SEED(5085470, 5379706), SEED(6974821, 803786), SEED(2813262, 7453524),
          SEED(5152500, 5815762), SEED(5423901, 5264787), SEED(263090, 284082),
          SEED(6557479, 359733), SEED(1364987, 3670273),
          SEED(1379185, 2531172)},
         {SEED(3981832, 5227342), SEED(1079991, 2520699),
          SEED(2450644, 6641566), SEED(4136472, 5942147),
          SEED(2743119, 5387286), SEED(2602647, 2696170),
          SEED(2324580, 7266413), SEED(5812373, 1773833),
          SEED(2054315, 8150631), SEED(5085470, 5379706), SEED(6974821, 803786),
          SEED(2813262, 7453524), SEED(5152500, 5815762),
          SEED(5423901, 5264787), SEED(263090, 284082), SEED(6557479, 359733),
          SEED(1364987, 3670273)},
         {SEED(4108106, 668951), SEED(3981832, 5227342), SEED(1079991, 2520699),
          SEED(2450644, 6641566), SEED(4136472, 5942147),
          SEED(2743119, 5387286), SEED(2602647, 2696170),
          SEED(2324580, 7266413), SEED(5812373, 1773833),
          SEED(2054315, 8150631), SEED(5085470, 5379706), SEED(6974821, 803786),
          SEED(2813262, 7453524), SEED(5152500, 5815762),
          SEED(5423901, 5264787), SEED(263090, 284082), SEED(6557479, 359733)},
         {SEED(2305343, 6301881), SEED(4108106, 668951), SEED(3981832, 5227342),
          SEED(1079991, 2520699), SEED(2450644, 6641566),
          SEED(4136472, 5942147), SEED(2743119, 5387286),
          SEED(2602647, 2696170), SEED(2324580, 7266413),
          SEED(5812373, 1773833), SEED(2054315, 8150631),
          SEED(5085470, 5379706), SEED(6974821, 803786), SEED(2813262, 7453524),
          SEED(5152500, 5815762), SEED(5423901, 5264787), SEED(263090, 284082)},
         {SEED(2713734, 6925648), SEED(2305343, 6301881), SEED(4108106, 668951),
          SEED(3981832, 5227342), SEED(1079991, 2520699),
          SEED(2450644, 6641566), SEED(4136472, 5942147),
          SEED(2743119, 5387286), SEED(2602647, 2696170),
          SEED(2324580, 7266413), SEED(5812373, 1773833),
          SEED(2054315, 8150631), SEED(5085470, 5379706), SEED(6974821, 803786),
          SEED(2813262, 7453524), SEED(5152500, 5815762),
          SEED(5423901, 5264787)},
         {SEED(6503892, 7785486), SEED(2713734, 6925648),
          SEED(2305343, 6301881), SEED(4108106, 668951), SEED(3981832, 5227342),
          SEED(1079991, 2520699), SEED(2450644, 6641566),
          SEED(4136472, 5942147), SEED(2743119, 5387286),
          SEED(2602647, 2696170), SEED(2324580, 7266413),
          SEED(5812373, 1773833), SEED(2054315, 8150631),
          SEED(5085470, 5379706), SEED(6974821, 803786), SEED(2813262, 7453524),
          SEED(5152500, 5815762)},
         {SEED(745724, 2654497), SEED(6503892, 7785486), SEED(2713734, 6925648),
          SEED(2305343, 6301881), SEED(4108106, 668951), SEED(3981832, 5227342),
          SEED(1079991, 2520699), SEED(2450644, 6641566),
          SEED(4136472, 5942147), SEED(2743119, 5387286),
          SEED(2602647, 2696170), SEED(2324580, 7266413),
          SEED(5812373, 1773833), SEED(2054315, 8150631),
          SEED(5085470, 5379706), SEED(6974821, 803786),
          SEED(2813262, 7453524)}},
    },
    {
        {{SEED(4508672, 4410178), SEED(7257124, 7646701),
          SEED(5675348, 3384988), SEED(2825160, 646594), SEED(4270204, 2650596),
          SEED(3425624, 2188585), SEED(5915632, 3239254), SEED(92680, 885383),
          SEED(4034284, 47237), SEED(1754105, 4661571), SEED(5207460, 2902413),
          SEED(3061948, 433317), SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001),
          SEED(410308, 3987212)},
         {SEED(4680512, 6637808), SEED(4508672, 4410178),
          SEED(7257124, 7646701), SEED(5675348, 3384988), SEED(2825160, 646594),
          SEED(4270204, 2650596), SEED(3425624, 2188585),
          SEED(5915632, 3239254), SEED(92680, 885383), SEED(4034284, 47237),
          SEED(1754105, 4661571), SEED(5207460, 2902413), SEED(3061948, 433317),
          SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001)},
         {SEED(1008889, 1670596), SEED(4680512, 6637808),
          SEED(4508672, 4410178), SEED(7257124, 7646701),
          SEED(5675348, 3384988), SEED(2825160, 646594), SEED(4270204, 2650596),
          SEED(3425624, 2188585), SEED(5915632, 3239254), SEED(92680, 885383),
          SEED(4034284, 47237), SEED(1754105, 4661571), SEED(5207460, 2902413),
          SEED(3061948, 433317), SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324)},
         {SEED(3888512, 6114313), SEED(1008889, 1670596),
          SEED(4680512, 6637808), SEED(4508672, 4410178),
          SEED(7257124, 7646701), SEED(5675348, 3384988), SEED(2825160, 646594),
          SEED(4270204, 2650596), SEED(3425624, 2188585),
          SEED(5915632, 3239254), SEED(92680, 885383), SEED(4034284, 47237),
          SEED(1754105, 4661571), SEED(5207460, 2902413), SEED(3061948, 433317),
          SEED(5390646, 6538527), SEED(7712580, 5896813)},
         {SEED(6581096, 5154907), SEED(3888512, 6114313),
          SEED(1008889, 1670596), SEED(4680512, 6637808),
          SEED(4508672, 4410178), SEED(7257124, 7646701),
          SEED(5675348, 3384988), SEED(2825160, 646594), SEED(4270204, 2650596),
          SEED(3425624, 2188585), SEED(5915632, 3239254), SEED(92680, 885383),
          SEED(4034284, 47237), SEED(1754105, 4661571), SEED(5207460, 2902413),
          SEED(3061948, 433317), SEED(5390646, 6538527)},
         {SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001), SEED(410308, 3987212),
          SEED(1083048, 2221593), SEED(1341492, 4407447),
          SEED(5582668, 2499605), SEED(7179484, 599356), SEED(2516099, 6377633),
          SEED(6606772, 7674779), SEED(2853684, 2805937),
          SEED(3090642, 2735463), SEED(4710312, 2539031),
          SEED(3540941, 1932246), SEED(7023731, 1878411),
          SEED(2651640, 4834713)},
         {SEED(3061948, 433317), SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001), SEED(410308, 3987212),
          SEED(1083048, 2221593), SEED(1341492, 4407447),
          SEED(5582668, 2499605), SEED(7179484, 599356), SEED(2516099, 6377633),
          SEED(6606772, 7674779), SEED(2853684, 2805937),
          SEED(3090642, 2735463), SEED(4710312, 2539031),
          SEED(3540941, 1932246), SEED(7023731, 1878411)},
         {SEED(5207460, 2902413), SEED(3061948, 433317), SEED(5390646, 6538527),
          SEED(7712580, 5896813), SEED(6601772, 2729324),
          SEED(6572337, 1024001), SEED(410308, 3987212), SEED(1083048, 2221593),
          SEED(1341492, 4407447), SEED(5582668, 2499605), SEED(7179484, 599356),
          SEED(2516099, 6377633), SEED(6606772, 7674779),
          SEED(2853684, 2805937), SEED(3090642, 2735463),
          SEED(4710312, 2539031), SEED(3540941, 1932246)},
         {SEED(1754105, 4661571), SEED(5207460, 2902413), SEED(3061948, 433317),
          SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001), SEED(410308, 3987212),
          SEED(1083048, 2221593), SEED(1341492, 4407447),
          SEED(5582668, 2499605), SEED(7179484, 599356), SEED(2516099, 6377633),
          SEED(6606772, 7674779), SEED(2853684, 2805937),
          SEED(3090642, 2735463), SEED(4710312, 2539031)},
         {SEED(4034284, 47237), SEED(1754105, 4661571), SEED(5207460, 2902413),
          SEED(3061948, 433317), SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001), SEED(410308, 3987212),
          SEED(1083048, 2221593), SEED(1341492, 4407447),
          SEED(5582668, 2499605), SEED(7179484, 599356), SEED(2516099, 6377633),
          SEED(6606772, 7674779), SEED(2853684, 2805937),
          SEED(3090642, 2735463)},
         {SEED(92680, 885383), SEED(4034284, 47237), SEED(1754105, 4661571),
          SEED(5207460, 2902413), SEED(3061948, 433317), SEED(5390646, 6538527),
          SEED(7712580, 5896813), SEED(6601772, 2729324),
          SEED(6572337, 1024001), SEED(410308, 3987212), SEED(1083048, 2221593),
          SEED(1341492, 4407447), SEED(5582668, 2499605), SEED(7179484, 599356),
          SEED(2516099, 6377633), SEED(6606772, 7674779),
          SEED(2853684, 2805937)},
         {SEED(5915632, 3239254), SEED(92680, 885383), SEED(4034284, 47237),
          SEED(1754105, 4661571), SEED(5207460, 2902413), SEED(3061948, 433317),
          SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001), SEED(410308, 3987212),
          SEED(1083048, 2221593), SEED(1341492, 4407447),
          SEED(5582668, 2499605), SEED(7179484, 599356), SEED(2516099, 6377633),
          SEED(6606772, 7674779)},
         {SEED(3425624, 2188585), SEED(5915632, 3239254), SEED(92680, 885383),
          SEED(4034284, 47237), SEED(1754105, 4661571), SEED(5207460, 2902413),
          SEED(3061948, 433317), SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001), SEED(410308, 3987212),
          SEED(1083048, 2221593), SEED(1341492, 4407447),
          SEED(5582668, 2499605), SEED(7179484, 599356),
          SEED(2516099, 6377633)},
         {SEED(4270204, 2650596), SEED(3425624, 2188585),
          SEED(5915632, 3239254), SEED(92680, 885383), SEED(4034284, 47237),
          SEED(1754105, 4661571), SEED(5207460, 2902413), SEED(3061948, 433317),
          SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001), SEED(410308, 3987212),
          SEED(1083048, 2221593), SEED(1341492, 4407447),
          SEED(5582668, 2499605), SEED(7179484, 599356)},
         {SEED(2825160, 646594), SEED(4270204, 2650596), SEED(3425624, 2188585),
          SEED(5915632, 3239254), SEED(92680, 885383), SEED(4034284, 47237),
          SEED(1754105, 4661571), SEED(5207460, 2902413), SEED(3061948, 433317),
          SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001), SEED(410308, 3987212),
          SEED(1083048, 2221593), SEED(1341492, 4407447),
          SEED(5582668, 2499605)},
         {SEED(5675348, 3384988), SEED(2825160, 646594), SEED(4270204, 2650596),
          SEED(3425624, 2188585), SEED(5915632, 3239254), SEED(92680, 885383),
          SEED(4034284, 47237), SEED(1754105, 4661571), SEED(5207460, 2902413),
          SEED(3061948, 433317), SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001), SEED(410308, 3987212),
          SEED(1083048, 2221593), SEED(1341492, 4407447)},
         {SEED(7257124, 7646701), SEED(5675348, 3384988), SEED(2825160, 646594),
          SEED(4270204, 2650596), SEED(3425624, 2188585),
          SEED(5915632, 3239254), SEED(92680, 885383), SEED(4034284, 47237),
          SEED(1754105, 4661571), SEED(5207460, 2902413), SEED(3061948, 433317),
          SEED(5390646, 6538527), SEED(7712580, 5896813),
          SEED(6601772, 2729324), SEED(6572337, 1024001), SEED(410308, 3987212),
          SEED(1083048, 2221593)}},
        {{SEED(2742528, 802197), SEED(906689, 4213848), SEED(7472464, 8156587),
          SEED(4537944, 6073698), SEED(1021052, 6132525),
          SEED(1928472, 6898858), SEED(1359872, 8267029), SEED(672982, 3518078),
          SEED(6535024, 7331245), SEED(6132648, 5914215), SEED(333236, 7020278),
          SEED(2396528, 256865), SEED(140464, 4109753), SEED(5458360, 1254262),
          SEED(6857056, 5260558), SEED(8188200, 538792),
          SEED(2638444, 4936678)},
         {SEED(3659496, 2680595), SEED(2742528, 802197), SEED(906689, 4213848),
          SEED(7472464, 8156587), SEED(4537944, 6073698),
          SEED(1021052, 6132525), SEED(1928472, 6898858),
          SEED(1359872, 8267029), SEED(672982, 3518078), SEED(6535024, 7331245),
          SEED(6132648, 5914215), SEED(333236, 7020278), SEED(2396528, 256865),
          SEED(140464, 4109753), SEED(5458360, 1254262), SEED(6857056, 5260558),
          SEED(8188200, 538792)},
         {SEED(4337536, 6612491), SEED(3659496, 2680595), SEED(2742528, 802197),
          SEED(906689, 4213848), SEED(7472464, 8156587), SEED(4537944, 6073698),
          SEED(1021052, 6132525), SEED(1928472, 6898858),
          SEED(1359872, 8267029), SEED(672982, 3518078), SEED(6535024, 7331245),
          SEED(6132648, 5914215), SEED(333236, 7020278), SEED(2396528, 256865),
          SEED(140464, 4109753), SEED(5458360, 1254262),
          SEED(6857056, 5260558)},
         {SEED(5940912, 5028538), SEED(4337536, 6612491),
          SEED(3659496, 2680595), SEED(2742528, 802197), SEED(906689, 4213848),
          SEED(7472464, 8156587), SEED(4537944, 6073698),
          SEED(1021052, 6132525), SEED(1928472, 6898858),
          SEED(1359872, 8267029), SEED(672982, 3518078), SEED(6535024, 7331245),
          SEED(6132648, 5914215), SEED(333236, 7020278), SEED(2396528, 256865),
          SEED(140464, 4109753), SEED(5458360, 1254262)},
         {SEED(6365049, 5468110), SEED(5940912, 5028538),
          SEED(4337536, 6612491), SEED(3659496, 2680595), SEED(2742528, 802197),
          SEED(906689, 4213848), SEED(7472464, 8156587), SEED(4537944, 6073698),
          SEED(1021052, 6132525), SEED(1928472, 6898858),
          SEED(1359872, 8267029), SEED(672982, 3518078), SEED(6535024, 7331245),
          SEED(6132648, 5914215), SEED(333236, 7020278), SEED(2396528, 256865),
          SEED(140464, 4109753)},
         {SEED(140464, 4109753), SEED(5458360, 1254262), SEED(6857056, 5260558),
          SEED(8188200, 538792), SEED(2638444, 4936678), SEED(814056, 2291947),
          SEED(7935425, 4335426), SEED(6799482, 4638509),
          SEED(6391528, 7131060), SEED(3277012, 218309), SEED(1595236, 8267188),
          SEED(7351952, 8010163), SEED(532518, 7796933), SEED(1076664, 6076983),
          SEED(7664200, 653656), SEED(533644, 6481485), SEED(8146692, 3708794)},
         {SEED(2396528, 256865), SEED(140464, 4109753), SEED(5458360, 1254262),
          SEED(6857056, 5260558), SEED(8188200, 538792), SEED(2638444, 4936678),
          SEED(814056, 2291947), SEED(7935425, 4335426), SEED(6799482, 4638509),
          SEED(6391528, 7131060), SEED(3277012, 218309), SEED(1595236, 8267188),
          SEED(7351952, 8010163), SEED(532518, 7796933), SEED(1076664, 6076983),
          SEED(7664200, 653656), SEED(533644, 6481485)},
         {SEED(333236, 7020278), SEED(2396528, 256865), SEED(140464, 4109753),
          SEED(5458360, 1254262), SEED(6857056, 5260558), SEED(8188200, 538792),
          SEED(2638444, 4936678), SEED(814056, 2291947), SEED(7935425, 4335426),
          SEED(6799482, 4638509), SEED(6391528, 7131060), SEED(3277012, 218309),
          SEED(1595236, 8267188), SEED(7351952, 8010163), SEED(532518, 7796933),
          SEED(1076664, 6076983), SEED(7664200, 653656)},
         {SEED(6132648, 5914215), SEED(333236, 7020278), SEED(2396528, 256865),
          SEED(140464, 4109753), SEED(5458360, 1254262), SEED(6857056, 5260558),
          SEED(8188200, 538792), SEED(2638444, 4936678), SEED(814056, 2291947),
          SEED(7935425, 4335426), SEED(6799482, 4638509),
          SEED(6391528, 7131060), SEED(3277012, 218309), SEED(1595236, 8267188),
          SEED(7351952, 8010163), SEED(532518, 7796933),
          SEED(1076664, 6076983)},
         {SEED(6535024, 7331245), SEED(6132648, 5914215), SEED(333236, 7020278),
          SEED(2396528, 256865), SEED(140464, 4109753), SEED(5458360, 1254262),
          SEED(6857056, 5260558), SEED(8188200, 538792), SEED(2638444, 4936678),
          SEED(814056, 2291947), SEED(7935425, 4335426), SEED(6799482, 4638509),
          SEED(6391528, 7131060), SEED(3277012, 218309), SEED(1595236, 8267188),
          SEED(7351952, 8010163), SEED(532518, 7796933)},
         {SEED(672982, 3518078), SEED(6535024, 7331245), SEED(6132648, 5914215),
          SEED(333236, 7020278), SEED(2396528, 256865), SEED(140464, 4109753),
          SEED(5458360, 1254262), SEED(6857056, 5260558), SEED(8188200, 538792),
          SEED(2638444, 4936678), SEED(814056, 2291947), SEED(7935425, 4335426),
          SEED(6799482, 4638509), SEED(6391528, 7131060), SEED(3277012, 218309),
          SEED(1595236, 8267188), SEED(7351952, 8010163)},
         {SEED(1359872, 8267029), SEED(672982, 3518078), SEED(6535024, 7331245),
          SEED(6132648, 5914215), SEED(333236, 7020278), SEED(2396528, 256865),
          SEED(140464, 4109753), SEED(5458360, 1254262), SEED(6857056, 5260558),
          SEED(8188200, 538792), SEED(2638444, 4936678), SEED(814056, 2291947),
          SEED(7935425, 4335426), SEED(6799482, 4638509),
          SEED(6391528, 7131060), SEED(3277012, 218309),
          SEED(1595236, 8267188)},
         {SEED(1928472, 6898858), SEED(1359872, 8267029), SEED(672982, 3518078),
          SEED(6535024, 7331245), SEED(6132648, 5914215), SEED(333236, 7020278),
          SEED(2396528, 256865), SEED(140464, 4109753), SEED(5458360, 1254262),
          SEED(6857056, 5260558), SEED(8188200, 538792), SEED(2638444, 4936678),
          SEED(814056, 2291947), SEED(7935425, 4335426), SEED(6799482, 4638509),
          SEED(6391528, 7131060), SEED(3277012, 218309)},
         {SEED(1021052, 6132525), SEED(1928472, 6898858),
          SEED(1359872, 8267029), SEED(672982, 3518078), SEED(6535024, 7331245),
          SEED(6132648, 5914215), SEED(333236, 7020278), SEED(2396528, 256865),
          SEED(140464, 4109753), SEED(5458360, 1254262), SEED(6857056, 5260558),
          SEED(8188200, 538792), SEED(2638444, 4936678), SEED(814056, 2291947),
          SEED(7935425, 4335426), SEED(6799482, 4638509),
          SEED(6391528, 7131060)},
         {SEED(4537944, 6073698), SEED(1021052, 6132525),
          SEED(1928472, 6898858), SEED(1359872, 8267029), SEED(672982, 3518078),
          SEED(6535024, 7331245), SEED(6132648, 5914215), SEED(333236, 7020278),
          SEED(2396528, 256865), SEED(140464, 4109753), SEED(5458360, 1254262),
          SEED(6857056, 5260558), SEED(8188200, 538792), SEED(2638444, 4936678),
          SEED(814056, 2291947), SEED(7935425, 4335426),
          SEED(6799482, 4638509)},
         {SEED(7472464, 8156587), SEED(4537944, 6073698),
          SEED(1021052, 6132525), SEED(1928472, 6898858),
          SEED(1359872, 8267029), SEED(672982, 3518078), SEED(6535024, 7331245),
          SEED(6132648, 5914215), SEED(333236, 7020278), SEED(2396528, 256865),
          SEED(140464, 4109753), SEED(5458360, 1254262), SEED(6857056, 5260558),
          SEED(8188200, 538792), SEED(2638444, 4936678), SEED(814056, 2291947),
          SEED(7935425, 4335426)},
         {SEED(906689, 4213848), SEED(7472464, 8156587), SEED(4537944, 6073698),
          SEED(1021052, 6132525), SEED(1928472, 6898858),
          SEED(1359872, 8267029), SEED(672982, 3518078), SEED(6535024, 7331245),
          SEED(6132648, 5914215), SEED(333236, 7020278), SEED(2396528, 256865),
          SEED(140464, 4109753), SEED(5458360, 1254262), SEED(6857056, 5260558),
          SEED(8188200, 538792), SEED(2638444, 4936678),
          SEED(814056, 2291947)}},
        {{SEED(6250744, 1528094), SEED(1332776, 3936681),
          SEED(3092728, 5400929), SEED(3684692, 8008339),
          SEED(8027424, 5027727), SEED(3152594, 3038389),
          SEED(7077056, 4407860), SEED(4451664, 1783769), SEED(704768, 1244184),
          SEED(4313740, 7426803), SEED(4599057, 448442), SEED(5204966, 1913863),
          SEED(2556220, 6248215), SEED(4662154, 945731), SEED(7249696, 1451407),
          SEED(6850164, 3176694), SEED(4412199, 5295043)},
         {SEED(4051015, 1934163), SEED(6250744, 1528094),
          SEED(1332776, 3936681), SEED(3092728, 5400929),
          SEED(3684692, 8008339), SEED(8027424, 5027727),
          SEED(3152594, 3038389), SEED(7077056, 4407860),
          SEED(4451664, 1783769), SEED(704768, 1244184), SEED(4313740, 7426803),
          SEED(4599057, 448442), SEED(5204966, 1913863), SEED(2556220, 6248215),
          SEED(4662154, 945731), SEED(7249696, 1451407),
          SEED(6850164, 3176694)},
         {SEED(2146248, 2796426), SEED(4051015, 1934163),
          SEED(6250744, 1528094), SEED(1332776, 3936681),
          SEED(3092728, 5400929), SEED(3684692, 8008339),
          SEED(8027424, 5027727), SEED(3152594, 3038389),
          SEED(7077056, 4407860), SEED(4451664, 1783769), SEED(704768, 1244184),
          SEED(4313740, 7426803), SEED(4599057, 448442), SEED(5204966, 1913863),
          SEED(2556220, 6248215), SEED(4662154, 945731),
          SEED(7249696, 1451407)},
         {SEED(1953816, 6852337), SEED(2146248, 2796426),
          SEED(4051015, 1934163), SEED(6250744, 1528094),
          SEED(1332776, 3936681), SEED(3092728, 5400929),
          SEED(3684692, 8008339), SEED(8027424, 5027727),
          SEED(3152594, 3038389), SEED(7077056, 4407860),
          SEED(4451664, 1783769), SEED(704768, 1244184), SEED(4313740, 7426803),
          SEED(4599057, 448442), SEED(5204966, 1913863), SEED(2556220, 6248215),
          SEED(4662154, 945731)},
         {SEED(5994930, 4882412), SEED(1953816, 6852337),
          SEED(2146248, 2796426), SEED(4051015, 1934163),
          SEED(6250744, 1528094), SEED(1332776, 3936681),
          SEED(3092728, 5400929), SEED(3684692, 8008339),
          SEED(8027424, 5027727), SEED(3152594, 3038389),
          SEED(7077056, 4407860), SEED(4451664, 1783769), SEED(704768, 1244184),
          SEED(4313740, 7426803), SEED(4599057, 448442), SEED(5204966, 1913863),
          SEED(2556220, 6248215)},
         {SEED(2556220, 6248215), SEED(4662154, 945731), SEED(7249696, 1451407),
          SEED(6850164, 3176694), SEED(4412199, 5295043),
          SEED(3098150, 6878313), SEED(2644328, 7917428),
          SEED(7029672, 3617159), SEED(2979924, 6764155),
          SEED(3713684, 5989532), SEED(6942145, 2589946),
          SEED(1872090, 2493997), SEED(1895444, 3924162), SEED(4431222, 298452),
          SEED(5452652, 5975395), SEED(6137501, 5660355),
          SEED(792767, 5007428)},
         {SEED(5204966, 1913863), SEED(2556220, 6248215), SEED(4662154, 945731),
          SEED(7249696, 1451407), SEED(6850164, 3176694),
          SEED(4412199, 5295043), SEED(3098150, 6878313),
          SEED(2644328, 7917428), SEED(7029672, 3617159),
          SEED(2979924, 6764155), SEED(3713684, 5989532),
          SEED(6942145, 2589946), SEED(1872090, 2493997),
          SEED(1895444, 3924162), SEED(4431222, 298452), SEED(5452652, 5975395),
          SEED(6137501, 5660355)},
         {SEED(4599057, 448442), SEED(5204966, 1913863), SEED(2556220, 6248215),
          SEED(4662154, 945731), SEED(7249696, 1451407), SEED(6850164, 3176694),
          SEED(4412199, 5295043), SEED(3098150, 6878313),
          SEED(2644328, 7917428), SEED(7029672, 3617159),
          SEED(2979924, 6764155), SEED(3713684, 5989532),
          SEED(6942145, 2589946), SEED(1872090, 2493997),
          SEED(1895444, 3924162), SEED(4431222, 298452),
          SEED(5452652, 5975395)},
         {SEED(4313740, 7426803), SEED(4599057, 448442), SEED(5204966, 1913863),
          SEED(2556220, 6248215), SEED(4662154, 945731), SEED(7249696, 1451407),
          SEED(6850164, 3176694), SEED(4412199, 5295043),
          SEED(3098150, 6878313), SEED(2644328, 7917428),
          SEED(7029672, 3617159), SEED(2979924, 6764155),
          SEED(3713684, 5989532), SEED(6942145, 2589946),
          SEED(1872090, 2493997), SEED(1895444, 3924162),
          SEED(4431222, 298452)},
         {SEED(704768, 1244184), SEED(4313740, 7426803), SEED(4599057, 448442),
          SEED(5204966, 1913863), SEED(2556220, 6248215), SEED(4662154, 945731),
          SEED(7249696, 1451407), SEED(6850164, 3176694),
          SEED(4412199, 5295043), SEED(3098150, 6878313),
          SEED(2644328, 7917428), SEED(7029672, 3617159),
          SEED(2979924, 6764155), SEED(3713684, 5989532),
          SEED(6942145, 2589946), SEED(1872090, 2493997),
          SEED(1895444, 3924162)},
         {SEED(4451664, 1783769), SEED(704768, 1244184), SEED(4313740, 7426803),
          SEED(4599057, 448442), SEED(5204966, 1913863), SEED(2556220, 6248215),
          SEED(4662154, 945731), SEED(7249696, 1451407), SEED(6850164, 3176694),
          SEED(4412199, 5295043), SEED(3098150, 6878313),
          SEED(2644328, 7917428), SEED(7029672, 3617159),
          SEED(2979924, 6764155), SEED(3713684, 5989532),
          SEED(6942145, 2589946), SEED(1872090, 2493997)},
         {SEED(7077056, 4407860), SEED(4451664, 1783769), SEED(704768, 1244184),
          SEED(4313740, 7426803), SEED(4599057, 448442), SEED(5204966, 1913863),
          SEED(2556220, 6248215), SEED(4662154, 945731), SEED(7249696, 1451407),
          SEED(6850164, 3176694), SEED(4412199, 5295043),
          SEED(3098150, 6878313), SEED(2644328, 7917428),
          SEED(7029672, 3617159), SEED(2979924, 6764155),
          SEED(3713684, 5989532), SEED(6942145, 2589946)},
         {SEED(3152594, 3038389), SEED(7077056, 4407860),
          SEED(4451664, 1783769), SEED(704768, 1244184), SEED(4313740, 7426803),
          SEED(4599057, 448442), SEED(5204966, 1913863), SEED(2556220, 6248215),
          SEED(4662154, 945731), SEED(7249696, 1451407), SEED(6850164, 3176694),
          SEED(4412199, 5295043), SEED(3098150, 6878313),
          SEED(2644328, 7917428), SEED(7029672, 3617159),
          SEED(2979924, 6764155), SEED(3713684, 5989532)},
         {SEED(8027424, 5027727), SEED(3152594, 3038389),
          SEED(7077056, 4407860), SEED(4451664, 1783769), SEED(704768, 1244184),
          SEED(4313740, 7426803), SEED(4599057, 448442), SEED(5204966, 1913863),
          SEED(2556220, 6248215), SEED(4662154, 945731), SEED(7249696, 1451407),
          SEED(6850164, 3176694), SEED(4412199, 5295043),
          SEED(3098150, 6878313), SEED(2644328, 7917428),
          SEED(7029672, 3617159), SEED(2979924, 6764155)},
         {SEED(3684692, 8008339), SEED(8027424, 5027727),
          SEED(3152594, 3038389), SEED(7077056, 4407860),
          SEED(4451664, 1783769), SEED(704768, 1244184), SEED(4313740, 7426803),
          SEED(4599057, 448442), SEED(5204966, 1913863), SEED(2556220, 6248215),
          SEED(4662154, 945731), SEED(7249696, 1451407), SEED(6850164, 3176694),
          SEED(4412199, 5295043), SEED(3098150, 6878313),
          SEED(2644328, 7917428), SEED(7029672, 3617159)},
         {SEED(3092728, 5400929), SEED(3684692, 8008339),
          SEED(8027424, 5027727), SEED(3152594, 3038389),
          SEED(7077056, 4407860), SEED(4451664, 1783769), SEED(704768, 1244184),
          SEED(4313740, 7426803), SEED(4599057, 448442), SEED(5204966, 1913863),
          SEED(2556220, 6248215), SEED(4662154, 945731), SEED(7249696, 1451407),
          SEED(6850164, 3176694), SEED(4412199, 5295043),
          SEED(3098150, 6878313), SEED(2644328, 7917428)},
         {SEED(1332776, 3936681), SEED(3092728, 5400929),
          SEED(3684692, 8008339), SEED(8027424, 5027727),
          SEED(3152594, 3038389), SEED(7077056, 4407860),
          SEED(4451664, 1783769), SEED(704768, 1244184), SEED(4313740, 7426803),
          SEED(4599057, 448442), SEED(5204966, 1913863), SEED(2556220, 6248215),
          SEED(4662154, 945731), SEED(7249696, 1451407), SEED(6850164, 3176694),
          SEED(4412199, 5295043), SEED(3098150, 6878313)}},
    },
    {
        {{SEED(1012480, 6982310), SEED(6664000, 2009189),
          SEED(3262529, 1104105), SEED(4960768, 6942360),
          SEED(3069936, 6975315), SEED(5332120, 805239), SEED(6214720, 6726441),
          SEED(3997728, 1620617), SEED(3224892, 8108393),
          SEED(1589404, 2016820), SEED(4410672, 8218482),
          SEED(5928408, 2092599), SEED(5772240, 7393601),
          SEED(2733216, 4631892), SEED(8328820, 7000590),
          SEED(4579104, 1804035), SEED(8050672, 4710392)},
         {SEED(2732000, 3297100), SEED(1012480, 6982310),
          SEED(6664000, 2009189), SEED(3262529, 1104105),
          SEED(4960768, 6942360), SEED(3069936, 6975315), SEED(5332120, 805239),
          SEED(6214720, 6726441), SEED(3997728, 1620617),
          SEED(3224892, 8108393), SEED(1589404, 2016820),
          SEED(4410672, 8218482), SEED(5928408, 2092599),
          SEED(5772240, 7393601), SEED(2733216, 4631892),
          SEED(8328820, 7000590), SEED(4579104, 1804035)},
         {SEED(1151264, 357788), SEED(2732000, 3297100), SEED(1012480, 6982310),
          SEED(6664000, 2009189), SEED(3262529, 1104105),
          SEED(4960768, 6942360), SEED(3069936, 6975315), SEED(5332120, 805239),
          SEED(6214720, 6726441), SEED(3997728, 1620617),
          SEED(3224892, 8108393), SEED(1589404, 2016820),
          SEED(4410672, 8218482), SEED(5928408, 2092599),
          SEED(5772240, 7393601), SEED(2733216, 4631892),
          SEED(8328820, 7000590)},
         {SEED(3202741, 8104696), SEED(1151264, 357788), SEED(2732000, 3297100),
          SEED(1012480, 6982310), SEED(6664000, 2009189),
          SEED(3262529, 1104105), SEED(4960768, 6942360),
          SEED(3069936, 6975315), SEED(5332120, 805239), SEED(6214720, 6726441),
          SEED(3997728, 1620617), SEED(3224892, 8108393),
          SEED(1589404, 2016820), SEED(4410672, 8218482),
          SEED(5928408, 2092599), SEED(5772240, 7393601),
          SEED(2733216, 4631892)},
         {SEED(1008608, 6641082), SEED(3202741, 8104696), SEED(1151264, 357788),
          SEED(2732000, 3297100), SEED(1012480, 6982310),
          SEED(6664000, 2009189), SEED(3262529, 1104105),
          SEED(4960768, 6942360), SEED(3069936, 6975315), SEED(5332120, 805239),
          SEED(6214720, 6726441), SEED(3997728, 1620617),
          SEED(3224892, 8108393), SEED(1589404, 2016820),
          SEED(4410672, 8218482), SEED(5928408, 2092599),
          SEED(5772240, 7393601)},
         {SEED(5772240, 7393601), SEED(2733216, 4631892),
          SEED(8328820, 7000590), SEED(4579104, 1804035),
          SEED(8050672, 4710392), SEED(4068968, 6177070), SEED(449280, 3671356),
          SEED(7653409, 7872095), SEED(1735876, 7222575),
          SEED(1480532, 4958495), SEED(921448, 975365), SEED(286312, 4633842),
          SEED(6614096, 2615623), SEED(491676, 3476501), SEED(1649192, 3404837),
          SEED(8220176, 6414446), SEED(6266344, 5770814)},
         {SEED(5928408, 2092599), SEED(5772240, 7393601),
          SEED(2733216, 4631892), SEED(8328820, 7000590),
          SEED(4579104, 1804035), SEED(8050672, 4710392),
          SEED(4068968, 6177070), SEED(449280, 3671356), SEED(7653409, 7872095),
          SEED(1735876, 7222575), SEED(1480532, 4958495), SEED(921448, 975365),
          SEED(286312, 4633842), SEED(6614096, 2615623), SEED(491676, 3476501),
          SEED(1649192, 3404837), SEED(8220176, 6414446)},
         {SEED(4410672, 8218482), SEED(5928408, 2092599),
          SEED(5772240, 7393601), SEED(2733216, 4631892),
          SEED(8328820, 7000590), SEED(4579104, 1804035),
          SEED(8050672, 4710392), SEED(4068968, 6177070), SEED(449280, 3671356),
          SEED(7653409, 7872095), SEED(1735876, 7222575),
          SEED(1480532, 4958495), SEED(921448, 975365), SEED(286312, 4633842),
          SEED(6614096, 2615623), SEED(491676, 3476501),
          SEED(1649192, 3404837)},
         {SEED(1589404, 2016820), SEED(4410672, 8218482),
          SEED(5928408, 2092599), SEED(5772240, 7393601),
          SEED(2733216, 4631892), SEED(8328820, 7000590),
          SEED(4579104, 1804035), SEED(8050672, 4710392),
          SEED(4068968, 6177070), SEED(449280, 3671356), SEED(7653409, 7872095),
          SEED(1735876, 7222575), SEED(1480532, 4958495), SEED(921448, 975365),
          SEED(286312, 4633842), SEED(6614096, 2615623), SEED(491676, 3476501)},
         {SEED(3224892, 8108393), SEED(1589404, 2016820),
          SEED(4410672, 8218482), SEED(5928408, 2092599),
          SEED(5772240, 7393601), SEED(2733216, 4631892),
          SEED(8328820, 7000590), SEED(4579104, 1804035),
          SEED(8050672, 4710392), SEED(4068968, 6177070), SEED(449280, 3671356),
          SEED(7653409, 7872095), SEED(1735876, 7222575),
          SEED(1480532, 4958495), SEED(921448, 975365), SEED(286312, 4633842),
          SEED(6614096, 2615623)},
         {SEED(3997728, 1620617), SEED(3224892, 8108393),
          SEED(1589404, 2016820), SEED(4410672, 8218482),
          SEED(5928408, 2092599), SEED(5772240, 7393601),
          SEED(2733216, 4631892), SEED(8328820, 7000590),
          SEED(4579104, 1804035), SEED(8050672, 4710392),
          SEED(4068968, 6177070), SEED(449280, 3671356), SEED(7653409, 7872095),
          SEED(1735876, 7222575), SEED(1480532, 4958495), SEED(921448, 975365),
          SEED(286312, 4633842)},
         {SEED(6214720, 6726441), SEED(3997728, 1620617),
          SEED(3224892, 8108393), SEED(1589404, 2016820),
          SEED(4410672, 8218482), SEED(5928408, 2092599),
          SEED(5772240, 7393601), SEED(2733216, 4631892),
          SEED(8328820, 7000590), SEED(4579104, 1804035),
          SEED(8050672, 4710392), SEED(4068968, 6177070), SEED(449280, 3671356),
          SEED(7653409, 7872095), SEED(1735876, 7222575),
          SEED(1480532, 4958495), SEED(921448, 975365)},
         {SEED(5332120, 805239), SEED(6214720, 6726441), SEED(3997728, 1620617),
          SEED(3224892, 8108393), SEED(1589404, 2016820),
          SEED(4410672, 8218482), SEED(5928408, 2092599),
          SEED(5772240, 7393601), SEED(2733216, 4631892),
          SEED(8328820, 7000590), SEED(4579104, 1804035),
          SEED(8050672, 4710392), SEED(4068968, 6177070), SEED(449280, 3671356),
          SEED(7653409, 7872095), SEED(1735876, 7222575),
          SEED(1480532, 4958495)},
         {SEED(3069936, 6975315), SEED(5332120, 805239), SEED(6214720, 6726441),
          SEED(3997728, 1620617), SEED(3224892, 8108393),
          SEED(1589404, 2016820), SEED(4410672, 8218482),
          SEED(5928408, 2092599), SEED(5772240, 7393601),
          SEED(2733216, 4631892), SEED(8328820, 7000590),
          SEED(4579104, 1804035), SEED(8050672, 4710392),
          SEED(4068968, 6177070), SEED(449280, 3671356), SEED(7653409, 7872095),
          SEED(1735876, 7222575)},
         {SEED(4960768, 6942360), SEED(3069936, 6975315), SEED(5332120, 805239),
          SEED(6214720, 6726441), SEED(3997728, 1620617),
          SEED(3224892, 8108393), SEED(1589404, 2016820),
          SEED(4410672, 8218482), SEED(5928408, 2092599),
          SEED(5772240, 7393601), SEED(2733216, 4631892),
          SEED(8328820, 7000590), SEED(4579104, 1804035),
          SEED(8050672, 4710392), SEED(4068968, 6177070), SEED(449280, 3671356),
          SEED(7653409, 7872095)},
         {SEED(3262529, 1104105), SEED(4960768, 6942360),
          SEED(3069936, 6975315), SEED(5332120, 805239), SEED(6214720, 6726441),
          SEED(3997728, 1620617), SEED(3224892, 8108393),
          SEED(1589404, 2016820), SEED(4410672, 8218482),
          SEED(5928408, 2092599), SEED(5772240, 7393601),
          SEED(2733216, 4631892), SEED(8328820, 7000590),
          SEED(4579104, 1804035), SEED(8050672, 4710392),
          SEED(4068968, 6177070), SEED(449280, 3671356)},
         {SEED(6664000, 2009189), SEED(3262529, 1104105),
          SEED(4960768, 6942360), SEED(3069936, 6975315), SEED(5332120, 805239),
          SEED(6214720, 6726441), SEED(3997728, 1620617),
          SEED(3224892, 8108393), SEED(1589404, 2016820),
          SEED(4410672, 8218482), SEED(5928408, 2092599),
          SEED(5772240, 7393601), SEED(2733216, 4631892),
          SEED(8328820, 7000590), SEED(4579104, 1804035),
          SEED(8050672, 4710392), SEED(4068968, 6177070)}},
        {{SEED(7568992, 1618193), SEED(6032080, 3183824),
          SEED(4038528, 5002887), SEED(7433024, 3588575),
          SEED(1346561, 5837659), SEED(5183328, 4923164),
          SEED(7533376, 3146469), SEED(4833584, 8277564),
          SEED(6931072, 3702220), SEED(796800, 2764370), SEED(6632632, 1297283),
          SEED(4426808, 1727550), SEED(5852608, 3460045), SEED(434016, 811691),
          SEED(6159200, 1059350), SEED(2297216, 5361294),
          SEED(4826872, 6447667)},
         {SEED(6173433, 3896718), SEED(7568992, 1618193),
          SEED(6032080, 3183824), SEED(4038528, 5002887),
          SEED(7433024, 3588575), SEED(1346561, 5837659),
          SEED(5183328, 4923164), SEED(7533376, 3146469),
          SEED(4833584, 8277564), SEED(6931072, 3702220), SEED(796800, 2764370),
          SEED(6632632, 1297283), SEED(4426808, 1727550),
          SEED(5852608, 3460045), SEED(434016, 811691), SEED(6159200, 1059350),
          SEED(2297216, 5361294)},
         {SEED(1341632, 561262), SEED(6173433, 3896718), SEED(7568992, 1618193),
          SEED(6032080, 3183824), SEED(4038528, 5002887),
          SEED(7433024, 3588575), SEED(1346561, 5837659),
          SEED(5183328, 4923164), SEED(7533376, 3146469),
          SEED(4833584, 8277564), SEED(6931072, 3702220), SEED(796800, 2764370),
          SEED(6632632, 1297283), SEED(4426808, 1727550),
          SEED(5852608, 3460045), SEED(434016, 811691), SEED(6159200, 1059350)},
         {SEED(1809120, 6062238), SEED(1341632, 561262), SEED(6173433, 3896718),
          SEED(7568992, 1618193), SEED(6032080, 3183824),
          SEED(4038528, 5002887), SEED(7433024, 3588575),
          SEED(1346561, 5837659), SEED(5183328, 4923164),
          SEED(7533376, 3146469), SEED(4833584, 8277564),
          SEED(6931072, 3702220), SEED(796800, 2764370), SEED(6632632, 1297283),
          SEED(4426808, 1727550), SEED(5852608, 3460045), SEED(434016, 811691)},
         {SEED(6466096, 3995515), SEED(1809120, 6062238), SEED(1341632, 561262),
          SEED(6173433, 3896718), SEED(7568992, 1618193),
          SEED(6032080, 3183824), SEED(4038528, 5002887),
          SEED(7433024, 3588575), SEED(1346561, 5837659),
          SEED(5183328, 4923164), SEED(7533376, 3146469),
          SEED(4833584, 8277564), SEED(6931072, 3702220), SEED(796800, 2764370),
          SEED(6632632, 1297283), SEED(4426808, 1727550),
          SEED(5852608, 3460045)},
         {SEED(5852608, 3460045), SEED(434016, 811691), SEED(6159200, 1059350),
          SEED(2297216, 5361294), SEED(4826872, 6447667),
          SEED(2385664, 5083637), SEED(6887312, 37354), SEED(7593552, 5113930),
          SEED(501952, 8274963), SEED(549761, 3073289), SEED(6939304, 3625880),
          SEED(3106568, 1418919), SEED(7369584, 4817518),
          SEED(6497056, 2890529), SEED(3026208, 1705019),
          SEED(4335416, 4324597), SEED(7988544, 3668490)},
         {SEED(4426808, 1727550), SEED(5852608, 3460045), SEED(434016, 811691),
          SEED(6159200, 1059350), SEED(2297216, 5361294),
          SEED(4826872, 6447667), SEED(2385664, 5083637), SEED(6887312, 37354),
          SEED(7593552, 5113930), SEED(501952, 8274963), SEED(549761, 3073289),
          SEED(6939304, 3625880), SEED(3106568, 1418919),
          SEED(7369584, 4817518), SEED(6497056, 2890529),
          SEED(3026208, 1705019), SEED(4335416, 4324597)},
         {SEED(6632632, 1297283), SEED(4426808, 1727550),
          SEED(5852608, 3460045), SEED(434016, 811691), SEED(6159200, 1059350),
          SEED(2297216, 5361294), SEED(4826872, 6447667),
          SEED(2385664, 5083637), SEED(6887312, 37354), SEED(7593552, 5113930),
          SEED(501952, 8274963), SEED(549761, 3073289), SEED(6939304, 3625880),
          SEED(3106568, 1418919), SEED(7369584, 4817518),
          SEED(6497056, 2890529), SEED(3026208, 1705019)},
         {SEED(796800, 2764370), SEED(6632632, 1297283), SEED(4426808, 1727550),
          SEED(5852608, 3460045), SEED(434016, 811691), SEED(6159200, 1059350),
          SEED(2297216, 5361294), SEED(4826872, 6447667),
          SEED(2385664, 5083637), SEED(6887312, 37354), SEED(7593552, 5113930),
          SEED(501952, 8274963), SEED(549761, 3073289), SEED(6939304, 3625880),
          SEED(3106568, 1418919), SEED(7369584, 4817518),
          SEED(6497056, 2890529)},
         {SEED(6931072, 3702220), SEED(796800, 2764370), SEED(6632632, 1297283),
          SEED(4426808, 1727550), SEED(5852608, 3460045), SEED(434016, 811691),
          SEED(6159200, 1059350), SEED(2297216, 5361294),
          SEED(4826872, 6447667), SEED(2385664, 5083637), SEED(6887312, 37354),
          SEED(7593552, 5113930), SEED(501952, 8274963), SEED(549761, 3073289),
          SEED(6939304, 3625880), SEED(3106568, 1418919),
          SEED(7369584, 4817518)},
         {SEED(4833584, 8277564), SEED(6931072, 3702220), SEED(796800, 2764370),
          SEED(6632632, 1297283), SEED(4426808, 1727550),
          SEED(5852608, 3460045), SEED(434016, 811691), SEED(6159200, 1059350),
          SEED(2297216, 5361294), SEED(4826872, 6447667),
          SEED(2385664, 5083637), SEED(6887312, 37354), SEED(7593552, 5113930),
          SEED(501952, 8274963), SEED(549761, 3073289), SEED(6939304, 3625880),
          SEED(3106568, 1418919)},
         {SEED(7533376, 3146469), SEED(4833584, 8277564),
          SEED(6931072, 3702220), SEED(796800, 2764370), SEED(6632632, 1297283),
          SEED(4426808, 1727550), SEED(5852608, 3460045), SEED(434016, 811691),
          SEED(6159200, 1059350), SEED(2297216, 5361294),
          SEED(4826872, 6447667), SEED(2385664, 5083637), SEED(6887312, 37354),
          SEED(7593552, 5113930), SEED(501952, 8274963), SEED(549761, 3073289),
          SEED(6939304, 3625880)},
         {SEED(5183328, 4923164), SEED(7533376, 3146469),
          SEED(4833584, 8277564), SEED(6931072, 3702220), SEED(796800, 2764370),
          SEED(6632632, 1297283), SEED(4426808, 1727550),
          SEED(5852608, 3460045), SEED(434016, 811691), SEED(6159200, 1059350),
          SEED(2297216, 5361294), SEED(4826872, 6447667),
          SEED(2385664, 5083637), SEED(6887312, 37354), SEED(7593552, 5113930),
          SEED(501952, 8274963), SEED(549761, 3073289)},
         {SEED(1346561, 5837659), SEED(5183328, 4923164),
          SEED(7533376, 3146469), SEED(4833584, 8277564),
          SEED(6931072, 3702220), SEED(796800, 2764370), SEED(6632632, 1297283),
          SEED(4426808, 1727550), SEED(5852608, 3460045), SEED(434016, 811691),
          SEED(6159200, 1059350), SEED(2297216, 5361294),
          SEED(4826872, 6447667), SEED(2385664, 5083637), SEED(6887312, 37354),
          SEED(7593552, 5113930), SEED(501952, 8274963)},
         {SEED(7433024, 3588575), SEED(1346561, 5837659),
          SEED(5183328, 4923164), SEED(7533376, 3146469),
          SEED(4833584, 8277564), SEED(6931072, 3702220), SEED(796800, 2764370),
          SEED(6632632, 1297283), SEED(4426808, 1727550),
          SEED(5852608, 3460045), SEED(434016, 811691), SEED(6159200, 1059350),
          SEED(2297216, 5361294), SEED(4826872, 6447667),
          SEED(2385664, 5083637), SEED(6887312, 37354), SEED(7593552, 5113930)},
         {SEED(4038528, 5002887), SEED(7433024, 3588575),
          SEED(1346561, 5837659), SEED(5183328, 4923164),
          SEED(7533376, 3146469), SEED(4833584, 8277564),
          SEED(6931072, 3702220), SEED(796800, 2764370), SEED(6632632, 1297283),
          SEED(4426808, 1727550), SEED(5852608, 3460045), SEED(434016, 811691),
          SEED(6159200, 1059350), SEED(2297216, 5361294),
          SEED(4826872, 6447667), SEED(2385664, 5083637), SEED(6887312, 37354)},
         {SEED(6032080, 3183824), SEED(4038528, 5002887),
          SEED(7433024, 3588575), SEED(1346561, 5837659),
          SEED(5183328, 4923164), SEED(7533376, 3146469),
          SEED(4833584, 8277564), SEED(6931072, 3702220), SEED(796800, 2764370),
          SEED(6632632, 1297283), SEED(4426808, 1727550),
          SEED(5852608, 3460045), SEED(434016, 811691), SEED(6159200, 1059350),
          SEED(2297216, 5361294), SEED(4826872, 6447667),
          SEED(2385664, 5083637)}},
        {{SEED(6022048, 6924872), SEED(5065549, 3714751),
          SEED(3161152, 5586347), SEED(5178320, 5807289),
          SEED(1372352, 2062931), SEED(2992448, 6133212),
          SEED(2647233, 4828178), SEED(8153696, 7869398),
          SEED(4716976, 2109102), SEED(5242504, 4130498),
          SEED(7546432, 7335577), SEED(3726368, 6562661),
          SEED(6159860, 7832162), SEED(3433236, 3280922),
          SEED(3278000, 7854103), SEED(2901336, 3569753),
          SEED(3626096, 4209786)},
         {SEED(4998448, 6272717), SEED(6022048, 6924872),
          SEED(5065549, 3714751), SEED(3161152, 5586347),
          SEED(5178320, 5807289), SEED(1372352, 2062931),
          SEED(2992448, 6133212), SEED(2647233, 4828178),
          SEED(8153696, 7869398), SEED(4716976, 2109102),
          SEED(5242504, 4130498), SEED(7546432, 7335577),
          SEED(3726368, 6562661), SEED(6159860, 7832162),
          SEED(3433236, 3280922), SEED(3278000, 7854103),
          SEED(2901336, 3569753)},
         {SEED(8079656, 988434), SEED(4998448, 6272717), SEED(6022048, 6924872),
          SEED(5065549, 3714751), SEED(3161152, 5586347),
          SEED(5178320, 5807289), SEED(1372352, 2062931),
          SEED(2992448, 6133212), SEED(2647233, 4828178),
          SEED(8153696, 7869398), SEED(4716976, 2109102),
          SEED(5242504, 4130498), SEED(7546432, 7335577),
          SEED(3726368, 6562661), SEED(6159860, 7832162),
          SEED(3433236, 3280922), SEED(3278000, 7854103)},
         {SEED(6439152, 5051842), SEED(8079656, 988434), SEED(4998448, 6272717),
          SEED(6022048, 6924872), SEED(5065549, 3714751),
          SEED(3161152, 5586347), SEED(5178320, 5807289),
          SEED(1372352, 2062931), SEED(2992448, 6133212),
          SEED(2647233, 4828178), SEED(8153696, 7869398),
          SEED(4716976, 2109102), SEED(5242504, 4130498),
          SEED(7546432, 7335577), SEED(3726368, 6562661),
          SEED(6159860, 7832162), SEED(3433236, 3280922)},
         {SEED(110177, 6995674), SEED(6439152, 5051842), SEED(8079656, 988434),
          SEED(4998448, 6272717), SEED(6022048, 6924872),
          SEED(5065549, 3714751), SEED(3161152, 5586347),
          SEED(5178320, 5807289), SEED(1372352, 2062931),
          SEED(2992448, 6133212), SEED(2647233, 4828178),
          SEED(8153696, 7869398), SEED(4716976, 2109102),
          SEED(5242504, 4130498), SEED(7546432, 7335577),
          SEED(3726368, 6562661), SEED(6159860, 7832162)},
         {SEED(6159860, 7832162), SEED(3433236, 3280922),
          SEED(3278000, 7854103), SEED(2901336, 3569753),
          SEED(3626096, 4209786), SEED(3029600, 791660), SEED(2418316, 7275181),
          SEED(3396064, 6105556), SEED(461344, 3698187), SEED(4518456, 6321040),
          SEED(3834624, 7186242), SEED(7309473, 6654124), SEED(1993836, 37236),
          SEED(1283740, 7216788), SEED(1964504, 4665003),
          SEED(4645096, 3765824), SEED(100272, 2352875)},
         {SEED(3726368, 6562661), SEED(6159860, 7832162),
          SEED(3433236, 3280922), SEED(3278000, 7854103),
          SEED(2901336, 3569753), SEED(3626096, 4209786), SEED(3029600, 791660),
          SEED(2418316, 7275181), SEED(3396064, 6105556), SEED(461344, 3698187),
          SEED(4518456, 6321040), SEED(3834624, 7186242),
          SEED(7309473, 6654124), SEED(1993836, 37236), SEED(1283740, 7216788),
          SEED(1964504, 4665003), SEED(4645096, 3765824)},
         {SEED(7546432, 7335577), SEED(3726368, 6562661),
          SEED(6159860, 7832162), SEED(3433236, 3280922),
          SEED(3278000, 7854103), SEED(2901336, 3569753),
          SEED(3626096, 4209786), SEED(3029600, 791660), SEED(2418316, 7275181),
          SEED(3396064, 6105556), SEED(461344, 3698187), SEED(4518456, 6321040),
          SEED(3834624, 7186242), SEED(7309473, 6654124), SEED(1993836, 37236),
          SEED(1283740, 7216788), SEED(1964504, 4665003)},
         {SEED(5242504, 4130498), SEED(7546432, 7335577),
          SEED(3726368, 6562661), SEED(6159860, 7832162),
          SEED(3433236, 3280922), SEED(3278000, 7854103),
          SEED(2901336, 3569753), SEED(3626096, 4209786), SEED(3029600, 791660),
          SEED(2418316, 7275181), SEED(3396064, 6105556), SEED(461344, 3698187),
          SEED(4518456, 6321040), SEED(3834624, 7186242),
          SEED(7309473, 6654124), SEED(1993836, 37236), SEED(1283740, 7216788)},
         {SEED(4716976, 2109102), SEED(5242504, 4130498),
          SEED(7546432, 7335577), SEED(3726368, 6562661),
          SEED(6159860, 7832162), SEED(3433236, 3280922),
          SEED(3278000, 7854103), SEED(2901336, 3569753),
          SEED(3626096, 4209786), SEED(3029600, 791660), SEED(2418316, 7275181),
          SEED(3396064, 6105556), SEED(461344, 3698187), SEED(4518456, 6321040),
          SEED(3834624, 7186242), SEED(7309473, 6654124), SEED(1993836, 37236)},
         {SEED(8153696, 7869398), SEED(4716976, 2109102),
          SEED(5242504, 4130498), SEED(7546432, 7335577),
          SEED(3726368, 6562661), SEED(6159860, 7832162),
          SEED(3433236, 3280922), SEED(3278000, 7854103),
          SEED(2901336, 3569753), SEED(3626096, 4209786), SEED(3029600, 791660),
          SEED(2418316, 7275181), SEED(3396064, 6105556), SEED(461344, 3698187),
          SEED(4518456, 6321040), SEED(3834624, 7186242),
          SEED(7309473, 6654124)},
         {SEED(2647233, 4828178), SEED(8153696, 7869398),
          SEED(4716976, 2109102), SEED(5242504, 4130498),
          SEED(7546432, 7335577), SEED(3726368, 6562661),
          SEED(6159860, 7832162), SEED(3433236, 3280922),
          SEED(3278000, 7854103), SEED(2901336, 3569753),
          SEED(3626096, 4209786), SEED(3029600, 791660), SEED(2418316, 7275181),
          SEED(3396064, 6105556), SEED(461344, 3698187), SEED(4518456, 6321040),
          SEED(3834624, 7186242)},
         {SEED(2992448, 6133212), SEED(2647233, 4828178),
          SEED(8153696, 7869398), SEED(4716976, 2109102),
          SEED(5242504, 4130498), SEED(7546432, 7335577),
          SEED(3726368, 6562661), SEED(6159860, 7832162),
          SEED(3433236, 3280922), SEED(3278000, 7854103),
          SEED(2901336, 3569753), SEED(3626096, 4209786), SEED(3029600, 791660),
          SEED(2418316, 7275181), SEED(3396064, 6105556), SEED(461344, 3698187),
          SEED(4518456, 6321040)},
         {SEED(1372352, 2062931), SEED(2992448, 6133212),
          SEED(2647233, 4828178), SEED(8153696, 7869398),
          SEED(4716976, 2109102), SEED(5242504, 4130498),
          SEED(7546432, 7335577), SEED(3726368, 6562661),
          SEED(6159860, 7832162), SEED(3433236, 3280922),
          SEED(3278000, 7854103), SEED(2901336, 3569753),
          SEED(3626096, 4209786), SEED(3029600, 791660), SEED(2418316, 7275181),
          SEED(3396064, 6105556), SEED(461344, 3698187)},
         {SEED(5178320, 5807289), SEED(1372352, 2062931),
          SEED(2992448, 6133212), SEED(2647233, 4828178),
          SEED(8153696, 7869398), SEED(4716976, 2109102),
          SEED(5242504, 4130498), SEED(7546432, 7335577),
          SEED(3726368, 6562661), SEED(6159860, 7832162),
          SEED(3433236, 3280922), SEED(3278000, 7854103),
          SEED(2901336, 3569753), SEED(3626096, 4209786), SEED(3029600, 791660),
          SEED(2418316, 7275181), SEED(3396064, 6105556)},
         {SEED(3161152, 5586347), SEED(5178320, 5807289),
          SEED(1372352, 2062931), SEED(2992448, 6133212),
          SEED(2647233, 4828178), SEED(8153696, 7869398),
          SEED(4716976, 2109102), SEED(5242504, 4130498),
          SEED(7546432, 7335577), SEED(3726368, 6562661),
          SEED(6159860, 7832162), SEED(3433236, 3280922),
          SEED(3278000, 7854103), SEED(2901336, 3569753),
          SEED(3626096, 4209786), SEED(3029600, 791660),
          SEED(2418316, 7275181)},
         {SEED(5065549, 3714751), SEED(3161152, 5586347),
          SEED(5178320, 5807289), SEED(1372352, 2062931),
          SEED(2992448, 6133212), SEED(2647233, 4828178),
          SEED(8153696, 7869398), SEED(4716976, 2109102),
          SEED(5242504, 4130498), SEED(7546432, 7335577),
          SEED(3726368, 6562661), SEED(6159860, 7832162),
          SEED(3433236, 3280922), SEED(3278000, 7854103),
          SEED(2901336, 3569753), SEED(3626096, 4209786),
          SEED(3029600, 791660)}},
    },
    {
        {{SEED(1395808, 4807518), SEED(6778560, 3257017),
          SEED(3143296, 2668718), SEED(2987569, 6415259),
          SEED(7406400, 5372038), SEED(4826592, 3456058),
          SEED(7493120, 2396775), SEED(2640512, 2552721),
          SEED(1075201, 6640415), SEED(1992256, 6040877),
          SEED(4097792, 3200900), SEED(3070304, 7286614),
          SEED(7808256, 5624490), SEED(2687488, 1488824),
          SEED(1301104, 4137268), SEED(7962224, 2714615),
          SEED(1068288, 7860211)},
         {SEED(86080, 4843642), SEED(1395808, 4807518), SEED(6778560, 3257017),
          SEED(3143296, 2668718), SEED(2987569, 6415259),
          SEED(7406400, 5372038), SEED(4826592, 3456058),
          SEED(7493120, 2396775), SEED(2640512, 2552721),
          SEED(1075201, 6640415), SEED(1992256, 6040877),
          SEED(4097792, 3200900), SEED(3070304, 7286614),
          SEED(7808256, 5624490), SEED(2687488, 1488824),
          SEED(1301104, 4137268), SEED(7962224, 2714615)},
         {SEED(2561185, 741267), SEED(86080, 4843642), SEED(1395808, 4807518),
          SEED(6778560, 3257017), SEED(3143296, 2668718),
          SEED(2987569, 6415259), SEED(7406400, 5372038),
          SEED(4826592, 3456058), SEED(7493120, 2396775),
          SEED(2640512, 2552721), SEED(1075201, 6640415),
          SEED(1992256, 6040877), SEED(4097792, 3200900),
          SEED(3070304, 7286614), SEED(7808256, 5624490),
          SEED(2687488, 1488824), SEED(1301104, 4137268)},
         {SEED(4444400, 6805986), SEED(2561185, 741267), SEED(86080, 4843642),
          SEED(1395808, 4807518), SEED(6778560, 3257017),
          SEED(3143296, 2668718), SEED(2987569, 6415259),
          SEED(7406400, 5372038), SEED(4826592, 3456058),
          SEED(7493120, 2396775), SEED(2640512, 2552721),
          SEED(1075201, 6640415), SEED(1992256, 6040877),
          SEED(4097792, 3200900), SEED(3070304, 7286614),
          SEED(7808256, 5624490), SEED(2687488, 1488824)},
         {SEED(1077440, 4745842), SEED(4444400, 6805986), SEED(2561185, 741267),
          SEED(86080, 4843642), SEED(1395808, 4807518), SEED(6778560, 3257017),
          SEED(3143296, 2668718), SEED(2987569, 6415259),
          SEED(7406400, 5372038), SEED(4826592, 3456058),
          SEED(7493120, 2396775), SEED(2640512, 2552721),
          SEED(1075201, 6640415), SEED(1992256, 6040877),
          SEED(4097792, 3200900), SEED(3070304, 7286614),
          SEED(7808256, 5624490)},
         {SEED(7808256, 5624490), SEED(2687488, 1488824),
          SEED(1301104, 4137268), SEED(7962224, 2714615),
          SEED(1068288, 7860211), SEED(4957824, 1351459), SEED(7674048, 860241),
          SEED(502784, 115997), SEED(1912368, 8163452), SEED(5414144, 7719769),
          SEED(728800, 255158), SEED(4422816, 3498769), SEED(3220864, 5316838),
          SEED(6776321, 5151590), SEED(691152, 1903609), SEED(4524176, 486284),
          SEED(2002016, 7815011)},
         {SEED(3070304, 7286614), SEED(7808256, 5624490),
          SEED(2687488, 1488824), SEED(1301104, 4137268),
          SEED(7962224, 2714615), SEED(1068288, 7860211),
          SEED(4957824, 1351459), SEED(7674048, 860241), SEED(502784, 115997),
          SEED(1912368, 8163452), SEED(5414144, 7719769), SEED(728800, 255158),
          SEED(4422816, 3498769), SEED(3220864, 5316838),
          SEED(6776321, 5151590), SEED(691152, 1903609), SEED(4524176, 486284)},
         {SEED(4097792, 3200900), SEED(3070304, 7286614),
          SEED(7808256, 5624490), SEED(2687488, 1488824),
          SEED(1301104, 4137268), SEED(7962224, 2714615),
          SEED(1068288, 7860211), SEED(4957824, 1351459), SEED(7674048, 860241),
          SEED(502784, 115997), SEED(1912368, 8163452), SEED(5414144, 7719769),
          SEED(728800, 255158), SEED(4422816, 3498769), SEED(3220864, 5316838),
          SEED(6776321, 5151590), SEED(691152, 1903609)},
         {SEED(1992256, 6040877), SEED(4097792, 3200900),
          SEED(3070304, 7286614), SEED(7808256, 5624490),
          SEED(2687488, 1488824), SEED(1301104, 4137268),
          SEED(7962224, 2714615), SEED(1068288, 7860211),
          SEED(4957824, 1351459), SEED(7674048, 860241), SEED(502784, 115997),
          SEED(1912368, 8163452), SEED(5414144, 7719769), SEED(728800, 255158),
          SEED(4422816, 3498769), SEED(3220864, 5316838),
          SEED(6776321, 5151590)},
         {SEED(1075201, 6640415), SEED(1992256, 6040877),
          SEED(4097792, 3200900), SEED(3070304, 7286614),
          SEED(7808256, 5624490), SEED(2687488, 1488824),
          SEED(1301104, 4137268), SEED(7962224, 2714615),
          SEED(1068288, 7860211), SEED(4957824, 1351459), SEED(7674048, 860241),
          SEED(502784, 115997), SEED(1912368, 8163452), SEED(5414144, 7719769),
          SEED(728800, 255158), SEED(4422816, 3498769), SEED(3220864, 5316838)},
         {SEED(2640512, 2552721), SEED(1075201, 6640415),
          SEED(1992256, 6040877), SEED(4097792, 3200900),
          SEED(3070304, 7286614), SEED(7808256, 5624490),
          SEED(2687488, 1488824), SEED(1301104, 4137268),
          SEED(7962224, 2714615), SEED(1068288, 7860211),
          SEED(4957824, 1351459), SEED(7674048, 860241), SEED(502784, 115997),
          SEED(1912368, 8163452), SEED(5414144, 7719769), SEED(728800, 255158),
          SEED(4422816, 3498769)},
         {SEED(7493120, 2396775), SEED(2640512, 2552721),
          SEED(1075201, 6640415), SEED(1992256, 6040877),
          SEED(4097792, 3200900), SEED(3070304, 7286614),
          SEED(7808256, 5624490), SEED(2687488, 1488824),
          SEED(1301104, 4137268), SEED(7962224, 2714615),
          SEED(1068288, 7860211), SEED(4957824, 1351459), SEED(7674048, 860241),
          SEED(502784, 115997), SEED(1912368, 8163452), SEED(5414144, 7719769),
          SEED(728800, 255158)},
         {SEED(4826592, 3456058), SEED(7493120, 2396775),
          SEED(2640512, 2552721), SEED(1075201, 6640415),
          SEED(1992256, 6040877), SEED(4097792, 3200900),
          SEED(3070304, 7286614), SEED(7808256, 5624490),
          SEED(2687488, 1488824), SEED(1301104, 4137268),
          SEED(7962224, 2714615), SEED(1068288, 7860211),
          SEED(4957824, 1351459), SEED(7674048, 860241), SEED(502784, 115997),
          SEED(1912368, 8163452), SEED(5414144, 7719769)},
         {SEED(7406400, 5372038), SEED(4826592, 3456058),
          SEED(7493120, 2396775), SEED(2640512, 2552721),
          SEED(1075201, 6640415), SEED(1992256, 6040877),
          SEED(4097792, 3200900), SEED(3070304, 7286614),
          SEED(7808256, 5624490), SEED(2687488, 1488824),
          SEED(1301104, 4137268), SEED(7962224, 2714615),
          SEED(1068288, 7860211), SEED(4957824, 1351459), SEED(7674048, 860241),
          SEED(502784, 115997), SEED(1912368, 8163452)},
         {SEED(2987569, 6415259), SEED(7406400, 5372038),
          SEED(4826592, 3456058), SEED(7493120, 2396775),
          SEED(2640512, 2552721), SEED(1075201, 6640415),
          SEED(1992256, 6040877), SEED(4097792, 3200900),
          SEED(3070304, 7286614), SEED(7808256, 5624490),
          SEED(2687488, 1488824), SEED(1301104, 4137268),
          SEED(7962224, 2714615), SEED(1068288, 7860211),
          SEED(4957824, 1351459), SEED(7674048, 860241), SEED(502784, 115997)},
         {SEED(3143296, 2668718), SEED(2987569, 6415259),
          SEED(7406400, 5372038), SEED(4826592, 3456058),
          SEED(7493120, 2396775), SEED(2640512, 2552721),
          SEED(1075201, 6640415), SEED(1992256, 6040877),
          SEED(4097792, 3200900), SEED(3070304, 7286614),
          SEED(7808256, 5624490), SEED(2687488, 1488824),
          SEED(1301104, 4137268), SEED(7962224, 2714615),
          SEED(1068288, 7860211), SEED(4957824, 1351459),
          SEED(7674048, 860241)},
         {SEED(6778560, 3257017), SEED(3143296, 2668718),
          SEED(2987569, 6415259), SEED(7406400, 5372038),
          SEED(4826592, 3456058), SEED(7493120, 2396775),
          SEED(2640512, 2552721), SEED(1075201, 6640415),
          SEED(1992256, 6040877), SEED(4097792, 3200900),
          SEED(3070304, 7286614), SEED(7808256, 5624490),
          SEED(2687488, 1488824), SEED(1301104, 4137268),
          SEED(7962224, 2714615), SEED(1068288, 7860211),
          SEED(4957824, 1351459)}},
        {{SEED(7699552, 8367461), SEED(7193153, 4572961),
          SEED(4642624, 1755398), SEED(2747072, 5132968),
          SEED(5179776, 7429719), SEED(7465440, 7388196),
          SEED(2898497, 6109072), SEED(4313216, 2109996),
          SEED(6942912, 3561037), SEED(778624, 8182063), SEED(893184, 2519035),
          SEED(1703265, 4925010), SEED(6631552, 2271174), SEED(7589056, 943175),
          SEED(6844416, 4301656), SEED(7589120, 1558199),
          SEED(6545409, 4169668)},
         {SEED(3336577, 3210780), SEED(7699552, 8367461),
          SEED(7193153, 4572961), SEED(4642624, 1755398),
          SEED(2747072, 5132968), SEED(5179776, 7429719),
          SEED(7465440, 7388196), SEED(2898497, 6109072),
          SEED(4313216, 2109996), SEED(6942912, 3561037), SEED(778624, 8182063),
          SEED(893184, 2519035), SEED(1703265, 4925010), SEED(6631552, 2271174),
          SEED(7589056, 943175), SEED(6844416, 4301656),
          SEED(7589120, 1558199)},
         {SEED(1947584, 6691168), SEED(3336577, 3210780),
          SEED(7699552, 8367461), SEED(7193153, 4572961),
          SEED(4642624, 1755398), SEED(2747072, 5132968),
          SEED(5179776, 7429719), SEED(7465440, 7388196),
          SEED(2898497, 6109072), SEED(4313216, 2109996),
          SEED(6942912, 3561037), SEED(778624, 8182063), SEED(893184, 2519035),
          SEED(1703265, 4925010), SEED(6631552, 2271174), SEED(7589056, 943175),
          SEED(6844416, 4301656)},
         {SEED(3098432, 6057055), SEED(1947584, 6691168),
          SEED(3336577, 3210780), SEED(7699552, 8367461),
          SEED(7193153, 4572961), SEED(4642624, 1755398),
          SEED(2747072, 5132968), SEED(5179776, 7429719),
          SEED(7465440, 7388196), SEED(2898497, 6109072),
          SEED(4313216, 2109996), SEED(6942912, 3561037), SEED(778624, 8182063),
          SEED(893184, 2519035), SEED(1703265, 4925010), SEED(6631552, 2271174),
          SEED(7589056, 943175)},
         {SEED(6393601, 5516137), SEED(3098432, 6057055),
          SEED(1947584, 6691168), SEED(3336577, 3210780),
          SEED(7699552, 8367461), SEED(7193153, 4572961),
          SEED(4642624, 1755398), SEED(2747072, 5132968),
          SEED(5179776, 7429719), SEED(7465440, 7388196),
          SEED(2898497, 6109072), SEED(4313216, 2109996),
          SEED(6942912, 3561037), SEED(778624, 8182063), SEED(893184, 2519035),
          SEED(1703265, 4925010), SEED(6631552, 2271174)},
         {SEED(6631552, 2271174), SEED(7589056, 943175), SEED(6844416, 4301656),
          SEED(7589120, 1558199), SEED(6545409, 4169668), SEED(234112, 979265),
          SEED(4294656, 6852497), SEED(329408, 8034010), SEED(4192768, 1571930),
          SEED(4401152, 7636264), SEED(6572256, 4869161),
          SEED(1195232, 1184062), SEED(6070272, 8227429),
          SEED(7742464, 2617861), SEED(2322816, 3880406), SEED(1692672, 960835),
          SEED(3546464, 755341)},
         {SEED(1703265, 4925010), SEED(6631552, 2271174), SEED(7589056, 943175),
          SEED(6844416, 4301656), SEED(7589120, 1558199),
          SEED(6545409, 4169668), SEED(234112, 979265), SEED(4294656, 6852497),
          SEED(329408, 8034010), SEED(4192768, 1571930), SEED(4401152, 7636264),
          SEED(6572256, 4869161), SEED(1195232, 1184062),
          SEED(6070272, 8227429), SEED(7742464, 2617861),
          SEED(2322816, 3880406), SEED(1692672, 960835)},
         {SEED(893184, 2519035), SEED(1703265, 4925010), SEED(6631552, 2271174),
          SEED(7589056, 943175), SEED(6844416, 4301656), SEED(7589120, 1558199),
          SEED(6545409, 4169668), SEED(234112, 979265), SEED(4294656, 6852497),
          SEED(329408, 8034010), SEED(4192768, 1571930), SEED(4401152, 7636264),
          SEED(6572256, 4869161), SEED(1195232, 1184062),
          SEED(6070272, 8227429), SEED(7742464, 2617861),
          SEED(2322816, 3880406)},
         {SEED(778624, 8182063), SEED(893184, 2519035), SEED(1703265, 4925010),
          SEED(6631552, 2271174), SEED(7589056, 943175), SEED(6844416, 4301656),
          SEED(7589120, 1558199), SEED(6545409, 4169668), SEED(234112, 979265),
          SEED(4294656, 6852497), SEED(329408, 8034010), SEED(4192768, 1571930),
          SEED(4401152, 7636264), SEED(6572256, 4869161),
          SEED(1195232, 1184062), SEED(6070272, 8227429),
          SEED(7742464, 2617861)},
         {SEED(6942912, 3561037), SEED(778624, 8182063), SEED(893184, 2519035),
          SEED(1703265, 4925010), SEED(6631552, 2271174), SEED(7589056, 943175),
          SEED(6844416, 4301656), SEED(7589120, 1558199),
          SEED(6545409, 4169668), SEED(234112, 979265), SEED(4294656, 6852497),
          SEED(329408, 8034010), SEED(4192768, 1571930), SEED(4401152, 7636264),
          SEED(6572256, 4869161), SEED(1195232, 1184062),
          SEED(6070272, 8227429)},
         {SEED(4313216, 2109996), SEED(6942912, 3561037), SEED(778624, 8182063),
          SEED(893184, 2519035), SEED(1703265, 4925010), SEED(6631552, 2271174),
          SEED(7589056, 943175), SEED(6844416, 4301656), SEED(7589120, 1558199),
          SEED(6545409, 4169668), SEED(234112, 979265), SEED(4294656, 6852497),
          SEED(329408, 8034010), SEED(4192768, 1571930), SEED(4401152, 7636264),
          SEED(6572256, 4869161), SEED(1195232, 1184062)},
         {SEED(2898497, 6109072), SEED(4313216, 2109996),
          SEED(6942912, 3561037), SEED(778624, 8182063), SEED(893184, 2519035),
          SEED(1703265, 4925010), SEED(6631552, 2271174), SEED(7589056, 943175),
          SEED(6844416, 4301656), SEED(7589120, 1558199),
          SEED(6545409, 4169668), SEED(234112, 979265), SEED(4294656, 6852497),
          SEED(329408, 8034010), SEED(4192768, 1571930), SEED(4401152, 7636264),
          SEED(6572256, 4869161)},
         {SEED(7465440, 7388196), SEED(2898497, 6109072),
          SEED(4313216, 2109996), SEED(6942912, 3561037), SEED(778624, 8182063),
          SEED(893184, 2519035), SEED(1703265, 4925010), SEED(6631552, 2271174),
          SEED(7589056, 943175), SEED(6844416, 4301656), SEED(7589120, 1558199),
          SEED(6545409, 4169668), SEED(234112, 979265), SEED(4294656, 6852497),
          SEED(329408, 8034010), SEED(4192768, 1571930),
          SEED(4401152, 7636264)},
         {SEED(5179776, 7429719), SEED(7465440, 7388196),
          SEED(2898497, 6109072), SEED(4313216, 2109996),
          SEED(6942912, 3561037), SEED(778624, 8182063), SEED(893184, 2519035),
          SEED(1703265, 4925010), SEED(6631552, 2271174), SEED(7589056, 943175),
          SEED(6844416, 4301656), SEED(7589120, 1558199),
          SEED(6545409, 4169668), SEED(234112, 979265), SEED(4294656, 6852497),
          SEED(329408, 8034010), SEED(4192768, 1571930)},
         {SEED(2747072, 5132968), SEED(5179776, 7429719),
          SEED(7465440, 7388196), SEED(2898497, 6109072),
          SEED(4313216, 2109996), SEED(6942912, 3561037), SEED(778624, 8182063),
          SEED(893184, 2519035), SEED(1703265, 4925010), SEED(6631552, 2271174),
          SEED(7589056, 943175), SEED(6844416, 4301656), SEED(7589120, 1558199),
          SEED(6545409, 4169668), SEED(234112, 979265), SEED(4294656, 6852497),
          SEED(329408, 8034010)},
         {SEED(4642624, 1755398), SEED(2747072, 5132968),
          SEED(5179776, 7429719), SEED(7465440, 7388196),
          SEED(2898497, 6109072), SEED(4313216, 2109996),
          SEED(6942912, 3561037), SEED(778624, 8182063), SEED(893184, 2519035),
          SEED(1703265, 4925010), SEED(6631552, 2271174), SEED(7589056, 943175),
          SEED(6844416, 4301656), SEED(7589120, 1558199),
          SEED(6545409, 4169668), SEED(234112, 979265), SEED(4294656, 6852497)},
         {SEED(7193153, 4572961), SEED(4642624, 1755398),
          SEED(2747072, 5132968), SEED(5179776, 7429719),
          SEED(7465440, 7388196), SEED(2898497, 6109072),
          SEED(4313216, 2109996), SEED(6942912, 3561037), SEED(778624, 8182063),
          SEED(893184, 2519035), SEED(1703265, 4925010), SEED(6631552, 2271174),
          SEED(7589056, 943175), SEED(6844416, 4301656), SEED(7589120, 1558199),
          SEED(6545409, 4169668), SEED(234112, 979265)}},
        {{SEED(5230368, 6973405), SEED(2224928, 7476627),
          SEED(7350226, 5355126), SEED(6334288, 1922703),
          SEED(1763457, 1278528), SEED(472800, 3981363), SEED(4559776, 8036318),
          SEED(1102401, 2395902), SEED(2260880, 8128350),
          SEED(3732449, 7655765), SEED(825568, 2875898), SEED(2866208, 3317773),
          SEED(870976, 4074325), SEED(5278416, 5487916), SEED(2392289, 2860511),
          SEED(4628672, 7405234), SEED(892192, 562468)},
         {SEED(2655649, 1840996), SEED(5230368, 6973405),
          SEED(2224928, 7476627), SEED(7350226, 5355126),
          SEED(6334288, 1922703), SEED(1763457, 1278528), SEED(472800, 3981363),
          SEED(4559776, 8036318), SEED(1102401, 2395902),
          SEED(2260880, 8128350), SEED(3732449, 7655765), SEED(825568, 2875898),
          SEED(2866208, 3317773), SEED(870976, 4074325), SEED(5278416, 5487916),
          SEED(2392289, 2860511), SEED(4628672, 7405234)},
         {SEED(2574352, 939330), SEED(2655649, 1840996), SEED(5230368, 6973405),
          SEED(2224928, 7476627), SEED(7350226, 5355126),
          SEED(6334288, 1922703), SEED(1763457, 1278528), SEED(472800, 3981363),
          SEED(4559776, 8036318), SEED(1102401, 2395902),
          SEED(2260880, 8128350), SEED(3732449, 7655765), SEED(825568, 2875898),
          SEED(2866208, 3317773), SEED(870976, 4074325), SEED(5278416, 5487916),
          SEED(2392289, 2860511)},
         {SEED(1353907, 8215638), SEED(2574352, 939330), SEED(2655649, 1840996),
          SEED(5230368, 6973405), SEED(2224928, 7476627),
          SEED(7350226, 5355126), SEED(6334288, 1922703),
          SEED(1763457, 1278528), SEED(472800, 3981363), SEED(4559776, 8036318),
          SEED(1102401, 2395902), SEED(2260880, 8128350),
          SEED(3732449, 7655765), SEED(825568, 2875898), SEED(2866208, 3317773),
          SEED(870976, 4074325), SEED(5278416, 5487916)},
         {SEED(7503344, 4575935), SEED(1353907, 8215638), SEED(2574352, 939330),
          SEED(2655649, 1840996), SEED(5230368, 6973405),
          SEED(2224928, 7476627), SEED(7350226, 5355126),
          SEED(6334288, 1922703), SEED(1763457, 1278528), SEED(472800, 3981363),
          SEED(4559776, 8036318), SEED(1102401, 2395902),
          SEED(2260880, 8128350), SEED(3732449, 7655765), SEED(825568, 2875898),
          SEED(2866208, 3317773), SEED(870976, 4074325)},
         {SEED(870976, 4074325), SEED(5278416, 5487916), SEED(2392289, 2860511),
          SEED(4628672, 7405234), SEED(892192, 562468), SEED(4757568, 2992042),
          SEED(6053760, 7828916), SEED(6247825, 2959224),
          SEED(4073408, 2182961), SEED(6419616, 2011370),
          SEED(8035840, 1105464), SEED(1693568, 4718545), SEED(231425, 6710185),
          SEED(5371072, 2640433), SEED(1340160, 4795254),
          SEED(4585504, 3859271), SEED(1974016, 2755305)},
         {SEED(2866208, 3317773), SEED(870976, 4074325), SEED(5278416, 5487916),
          SEED(2392289, 2860511), SEED(4628672, 7405234), SEED(892192, 562468),
          SEED(4757568, 2992042), SEED(6053760, 7828916),
          SEED(6247825, 2959224), SEED(4073408, 2182961),
          SEED(6419616, 2011370), SEED(8035840, 1105464),
          SEED(1693568, 4718545), SEED(231425, 6710185), SEED(5371072, 2640433),
          SEED(1340160, 4795254), SEED(4585504, 3859271)},
         {SEED(825568, 2875898), SEED(2866208, 3317773), SEED(870976, 4074325),
          SEED(5278416, 5487916), SEED(2392289, 2860511),
          SEED(4628672, 7405234), SEED(892192, 562468), SEED(4757568, 2992042),
          SEED(6053760, 7828916), SEED(6247825, 2959224),
          SEED(4073408, 2182961), SEED(6419616, 2011370),
          SEED(8035840, 1105464), SEED(1693568, 4718545), SEED(231425, 6710185),
          SEED(5371072, 2640433), SEED(1340160, 4795254)},
         {SEED(3732449, 7655765), SEED(825568, 2875898), SEED(2866208, 3317773),
          SEED(870976, 4074325), SEED(5278416, 5487916), SEED(2392289, 2860511),
          SEED(4628672, 7405234), SEED(892192, 562468), SEED(4757568, 2992042),
          SEED(6053760, 7828916), SEED(6247825, 2959224),
          SEED(4073408, 2182961), SEED(6419616, 2011370),
          SEED(8035840, 1105464), SEED(1693568, 4718545), SEED(231425, 6710185),
          SEED(5371072, 2640433)},
         {SEED(2260880, 8128350), SEED(3732449, 7655765), SEED(825568, 2875898),
          SEED(2866208, 3317773), SEED(870976, 4074325), SEED(5278416, 5487916),
          SEED(2392289, 2860511), SEED(4628672, 7405234), SEED(892192, 562468),
          SEED(4757568, 2992042), SEED(6053760, 7828916),
          SEED(6247825, 2959224), SEED(4073408, 2182961),
          SEED(6419616, 2011370), SEED(8035840, 1105464),
          SEED(1693568, 4718545), SEED(231425, 6710185)},
         {SEED(1102401, 2395902), SEED(2260880, 8128350),
          SEED(3732449, 7655765), SEED(825568, 2875898), SEED(2866208, 3317773),
          SEED(870976, 4074325), SEED(5278416, 5487916), SEED(2392289, 2860511),
          SEED(4628672, 7405234), SEED(892192, 562468), SEED(4757568, 2992042),
          SEED(6053760, 7828916), SEED(6247825, 2959224),
          SEED(4073408, 2182961), SEED(6419616, 2011370),
          SEED(8035840, 1105464), SEED(1693568, 4718545)},
         {SEED(4559776, 8036318), SEED(1102401, 2395902),
          SEED(2260880, 8128350), SEED(3732449, 7655765), SEED(825568, 2875898),
          SEED(2866208, 3317773), SEED(870976, 4074325), SEED(5278416, 5487916),
          SEED(2392289, 2860511), SEED(4628672, 7405234), SEED(892192, 562468),
          SEED(4757568, 2992042), SEED(6053760, 7828916),
          SEED(6247825, 2959224), SEED(4073408, 2182961),
          SEED(6419616, 2011370), SEED(8035840, 1105464)},
         {SEED(472800, 3981363), SEED(4559776, 8036318), SEED(1102401, 2395902),
          SEED(2260880, 8128350), SEED(3732449, 7655765), SEED(825568, 2875898),
          SEED(2866208, 3317773), SEED(870976, 4074325), SEED(5278416, 5487916),
          SEED(2392289, 2860511), SEED(4628672, 7405234), SEED(892192, 562468),
          SEED(4757568, 2992042), SEED(6053760, 7828916),
          SEED(6247825, 2959224), SEED(4073408, 2182961),
          SEED(6419616, 2011370)},
         {SEED(1763457, 1278528), SEED(472800, 3981363), SEED(4559776, 8036318),
          SEED(1102401, 2395902), SEED(2260880, 8128350),
          SEED(3732449, 7655765), SEED(825568, 2875898), SEED(2866208, 3317773),
          SEED(870976, 4074325), SEED(5278416, 5487916), SEED(2392289, 2860511),
          SEED(4628672, 7405234), SEED(892192, 562468), SEED(4757568, 2992042),
          SEED(6053760, 7828916), SEED(6247825, 2959224),
          SEED(4073408, 2182961)},
         {SEED(6334288, 1922703), SEED(1763457, 1278528), SEED(472800, 3981363),
          SEED(4559776, 8036318), SEED(1102401, 2395902),
          SEED(2260880, 8128350), SEED(3732449, 7655765), SEED(825568, 2875898),
          SEED(2866208, 3317773), SEED(870976, 4074325), SEED(5278416, 5487916),
          SEED(2392289, 2860511), SEED(4628672, 7405234), SEED(892192, 562468),
          SEED(4757568, 2992042), SEED(6053760, 7828916),
          SEED(6247825, 2959224)},
         {SEED(7350226, 5355126), SEED(6334288, 1922703),
          SEED(1763457, 1278528), SEED(472800, 3981363), SEED(4559776, 8036318),
          SEED(1102401, 2395902), SEED(2260880, 8128350),
          SEED(3732449, 7655765), SEED(825568, 2875898), SEED(2866208, 3317773),
          SEED(870976, 4074325), SEED(5278416, 5487916), SEED(2392289, 2860511),
          SEED(4628672, 7405234), SEED(892192, 562468), SEED(4757568, 2992042),
          SEED(6053760, 7828916)},
         {SEED(2224928, 7476627), SEED(7350226, 5355126),
          SEED(6334288, 1922703), SEED(1763457, 1278528), SEED(472800, 3981363),
          SEED(4559776, 8036318), SEED(1102401, 2395902),
          SEED(2260880, 8128350), SEED(3732449, 7655765), SEED(825568, 2875898),
          SEED(2866208, 3317773), SEED(870976, 4074325), SEED(5278416, 5487916),
          SEED(2392289, 2860511), SEED(4628672, 7405234), SEED(892192, 562468),
          SEED(4757568, 2992042)}},
    },
    {
        {{SEED(8362692, 3897103), SEED(5658944, 5892359),
          SEED(3114241, 5065176), SEED(7910528, 5930185),
          SEED(3437888, 2472033), SEED(3861059, 463588), SEED(6052544, 5312705),
          SEED(2772865, 7722698), SEED(4718976, 2988879),
          SEED(7831936, 2076526), SEED(4518338, 7388275),
          SEED(3897792, 1705955), SEED(2903553, 5417966),
          SEED(1519232, 3789157), SEED(5959552, 7213901),
          SEED(7156481, 6733734), SEED(1175744, 2954287)},
         {SEED(4613632, 5426320), SEED(8362692, 3897103),
          SEED(5658944, 5892359), SEED(3114241, 5065176),
          SEED(7910528, 5930185), SEED(3437888, 2472033), SEED(3861059, 463588),
          SEED(6052544, 5312705), SEED(2772865, 7722698),
          SEED(4718976, 2988879), SEED(7831936, 2076526),
          SEED(4518338, 7388275), SEED(3897792, 1705955),
          SEED(2903553, 5417966), SEED(1519232, 3789157),
          SEED(5959552, 7213901), SEED(7156481, 6733734)},
         {SEED(6678401, 4275312), SEED(4613632, 5426320),
          SEED(8362692, 3897103), SEED(5658944, 5892359),
          SEED(3114241, 5065176), SEED(7910528, 5930185),
          SEED(3437888, 2472033), SEED(3861059, 463588), SEED(6052544, 5312705),
          SEED(2772865, 7722698), SEED(4718976, 2988879),
          SEED(7831936, 2076526), SEED(4518338, 7388275),
          SEED(3897792, 1705955), SEED(2903553, 5417966),
          SEED(1519232, 3789157), SEED(5959552, 7213901)},
         {SEED(685185, 3890470), SEED(6678401, 4275312), SEED(4613632, 5426320),
          SEED(8362692, 3897103), SEED(5658944, 5892359),
          SEED(3114241, 5065176), SEED(7910528, 5930185),
          SEED(3437888, 2472033), SEED(3861059, 463588), SEED(6052544, 5312705),
          SEED(2772865, 7722698), SEED(4718976, 2988879),
          SEED(7831936, 2076526), SEED(4518338, 7388275),
          SEED(3897792, 1705955), SEED(2903553, 5417966),
          SEED(1519232, 3789157)},
         {SEED(7178176, 1292908), SEED(685185, 3890470), SEED(6678401, 4275312),
          SEED(4613632, 5426320), SEED(8362692, 3897103),
          SEED(5658944, 5892359), SEED(3114241, 5065176),
          SEED(7910528, 5930185), SEED(3437888, 2472033), SEED(3861059, 463588),
          SEED(6052544, 5312705), SEED(2772865, 7722698),
          SEED(4718976, 2988879), SEED(7831936, 2076526),
          SEED(4518338, 7388275), SEED(3897792, 1705955),
          SEED(2903553, 5417966)},
         {SEED(2903553, 5417966), SEED(1519232, 3789157),
          SEED(5959552, 7213901), SEED(7156481, 6733734),
          SEED(1175744, 2954287), SEED(4501633, 3433515), SEED(7995008, 579653),
          SEED(341376, 5731086), SEED(3191552, 2941306), SEED(3994560, 395506),
          SEED(7731329, 1463920), SEED(2154752, 3606750),
          SEED(8257920, 2304731), SEED(3199744, 7588330),
          SEED(1872384, 3251233), SEED(5750465, 654540),
          SEED(2722048, 7140276)},
         {SEED(3897792, 1705955), SEED(2903553, 5417966),
          SEED(1519232, 3789157), SEED(5959552, 7213901),
          SEED(7156481, 6733734), SEED(1175744, 2954287),
          SEED(4501633, 3433515), SEED(7995008, 579653), SEED(341376, 5731086),
          SEED(3191552, 2941306), SEED(3994560, 395506), SEED(7731329, 1463920),
          SEED(2154752, 3606750), SEED(8257920, 2304731),
          SEED(3199744, 7588330), SEED(1872384, 3251233),
          SEED(5750465, 654540)},
         {SEED(4518338, 7388275), SEED(3897792, 1705955),
          SEED(2903553, 5417966), SEED(1519232, 3789157),
          SEED(5959552, 7213901), SEED(7156481, 6733734),
          SEED(1175744, 2954287), SEED(4501633, 3433515), SEED(7995008, 579653),
          SEED(341376, 5731086), SEED(3191552, 2941306), SEED(3994560, 395506),
          SEED(7731329, 1463920), SEED(2154752, 3606750),
          SEED(8257920, 2304731), SEED(3199744, 7588330),
          SEED(1872384, 3251233)},
         {SEED(7831936, 2076526), SEED(4518338, 7388275),
          SEED(3897792, 1705955), SEED(2903553, 5417966),
          SEED(1519232, 3789157), SEED(5959552, 7213901),
          SEED(7156481, 6733734), SEED(1175744, 2954287),
          SEED(4501633, 3433515), SEED(7995008, 579653), SEED(341376, 5731086),
          SEED(3191552, 2941306), SEED(3994560, 395506), SEED(7731329, 1463920),
          SEED(2154752, 3606750), SEED(8257920, 2304731),
          SEED(3199744, 7588330)},
         {SEED(4718976, 2988879), SEED(7831936, 2076526),
          SEED(4518338, 7388275), SEED(3897792, 1705955),
          SEED(2903553, 5417966), SEED(1519232, 3789157),
          SEED(5959552, 7213901), SEED(7156481, 6733734),
          SEED(1175744, 2954287), SEED(4501633, 3433515), SEED(7995008, 579653),
          SEED(341376, 5731086), SEED(3191552, 2941306), SEED(3994560, 395506),
          SEED(7731329, 1463920), SEED(2154752, 3606750),
          SEED(8257920, 2304731)},
         {SEED(2772865, 7722698), SEED(4718976, 2988879),
          SEED(7831936, 2076526), SEED(4518338, 7388275),
          SEED(3897792, 1705955), SEED(2903553, 5417966),
          SEED(1519232, 3789157), SEED(5959552, 7213901),
          SEED(7156481, 6733734), SEED(1175744, 2954287),
          SEED(4501633, 3433515), SEED(7995008, 579653), SEED(341376, 5731086),
          SEED(3191552, 2941306), SEED(3994560, 395506), SEED(7731329, 1463920),
          SEED(2154752, 3606750)},
         {SEED(6052544, 5312705), SEED(2772865, 7722698),
          SEED(4718976, 2988879), SEED(7831936, 2076526),
          SEED(4518338, 7388275), SEED(3897792, 1705955),
          SEED(2903553, 5417966), SEED(1519232, 3789157),
          SEED(5959552, 7213901), SEED(7156481, 6733734),
          SEED(1175744, 2954287), SEED(4501633, 3433515), SEED(7995008, 579653),
          SEED(341376, 5731086), SEED(3191552, 2941306), SEED(3994560, 395506),
          SEED(7731329, 1463920)},
         {SEED(3861059, 463588), SEED(6052544, 5312705), SEED(2772865, 7722698),
          SEED(4718976, 2988879), SEED(7831936, 2076526),
          SEED(4518338, 7388275), SEED(3897792, 1705955),
          SEED(2903553, 5417966), SEED(1519232, 3789157),
          SEED(5959552, 7213901), SEED(7156481, 6733734),
          SEED(1175744, 2954287), SEED(4501633, 3433515), SEED(7995008, 579653),
          SEED(341376, 5731086), SEED(3191552, 2941306), SEED(3994560, 395506)},
         {SEED(3437888, 2472033), SEED(3861059, 463588), SEED(6052544, 5312705),
          SEED(2772865, 7722698), SEED(4718976, 2988879),
          SEED(7831936, 2076526), SEED(4518338, 7388275),
          SEED(3897792, 1705955), SEED(2903553, 5417966),
          SEED(1519232, 3789157), SEED(5959552, 7213901),
          SEED(7156481, 6733734), SEED(1175744, 2954287),
          SEED(4501633, 3433515), SEED(7995008, 579653), SEED(341376, 5731086),
          SEED(3191552, 2941306)},
         {SEED(7910528, 5930185), SEED(3437888, 2472033), SEED(3861059, 463588),
          SEED(6052544, 5312705), SEED(2772865, 7722698),
          SEED(4718976, 2988879), SEED(7831936, 2076526),
          SEED(4518338, 7388275), SEED(3897792, 1705955),
          SEED(2903553, 5417966), SEED(1519232, 3789157),
          SEED(5959552, 7213901), SEED(7156481, 6733734),
          SEED(1175744, 2954287), SEED(4501633, 3433515), SEED(7995008, 579653),
          SEED(341376, 5731086)},
         {SEED(3114241, 5065176), SEED(7910528, 5930185),
          SEED(3437888, 2472033), SEED(3861059, 463588), SEED(6052544, 5312705),
          SEED(2772865, 7722698), SEED(4718976, 2988879),
          SEED(7831936, 2076526), SEED(4518338, 7388275),
          SEED(3897792, 1705955), SEED(2903553, 5417966),
          SEED(1519232, 3789157), SEED(5959552, 7213901),
          SEED(7156481, 6733734), SEED(1175744, 2954287),
          SEED(4501633, 3433515), SEED(7995008, 579653)},
         {SEED(5658944, 5892359), SEED(3114241, 5065176),
          SEED(7910528, 5930185), SEED(3437888, 2472033), SEED(3861059, 463588),
          SEED(6052544, 5312705), SEED(2772865, 7722698),
          SEED(4718976, 2988879), SEED(7831936, 2076526),
          SEED(4518338, 7388275), SEED(3897792, 1705955),
          SEED(2903553, 5417966), SEED(1519232, 3789157),
          SEED(5959552, 7213901), SEED(7156481, 6733734),
          SEED(1175744, 2954287), SEED(4501633, 3433515)}},
        {{SEED(2042140, 3114519), SEED(6071552, 4139333),
          SEED(4306442, 2319135), SEED(6341642, 4490257),
          SEED(5699329, 4179880), SEED(6790805, 2991403),
          SEED(3336576, 3656345), SEED(6102921, 1768050),
          SEED(8211716, 1472048), SEED(1761793, 7713461),
          SEED(4316687, 3137072), SEED(4183808, 394145), SEED(5321608, 822236),
          SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694)},
         {SEED(8289921, 256966), SEED(2042140, 3114519), SEED(6071552, 4139333),
          SEED(4306442, 2319135), SEED(6341642, 4490257),
          SEED(5699329, 4179880), SEED(6790805, 2991403),
          SEED(3336576, 3656345), SEED(6102921, 1768050),
          SEED(8211716, 1472048), SEED(1761793, 7713461),
          SEED(4316687, 3137072), SEED(4183808, 394145), SEED(5321608, 822236),
          SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820)},
         {SEED(5554324, 1202470), SEED(8289921, 256966), SEED(2042140, 3114519),
          SEED(6071552, 4139333), SEED(4306442, 2319135),
          SEED(6341642, 4490257), SEED(5699329, 4179880),
          SEED(6790805, 2991403), SEED(3336576, 3656345),
          SEED(6102921, 1768050), SEED(8211716, 1472048),
          SEED(1761793, 7713461), SEED(4316687, 3137072), SEED(4183808, 394145),
          SEED(5321608, 822236), SEED(2902657, 1969654),
          SEED(1245825, 2863137)},
         {SEED(5552267, 5182272), SEED(5554324, 1202470), SEED(8289921, 256966),
          SEED(2042140, 3114519), SEED(6071552, 4139333),
          SEED(4306442, 2319135), SEED(6341642, 4490257),
          SEED(5699329, 4179880), SEED(6790805, 2991403),
          SEED(3336576, 3656345), SEED(6102921, 1768050),
          SEED(8211716, 1472048), SEED(1761793, 7713461),
          SEED(4316687, 3137072), SEED(4183808, 394145), SEED(5321608, 822236),
          SEED(2902657, 1969654)},
         {SEED(585601, 6108988), SEED(5552267, 5182272), SEED(5554324, 1202470),
          SEED(8289921, 256966), SEED(2042140, 3114519), SEED(6071552, 4139333),
          SEED(4306442, 2319135), SEED(6341642, 4490257),
          SEED(5699329, 4179880), SEED(6790805, 2991403),
          SEED(3336576, 3656345), SEED(6102921, 1768050),
          SEED(8211716, 1472048), SEED(1761793, 7713461),
          SEED(4316687, 3137072), SEED(4183808, 394145), SEED(5321608, 822236)},
         {SEED(5321608, 822236), SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694), SEED(3639943, 123115),
          SEED(2734976, 482988), SEED(6592129, 551084), SEED(6518534, 3018208),
          SEED(3937536, 4855027), SEED(2474118, 8242939),
          SEED(7541376, 3262199), SEED(781313, 945814), SEED(5309059, 7891002),
          SEED(515968, 4850324), SEED(5104005, 6424859),
          SEED(1593216, 4317059)},
         {SEED(4183808, 394145), SEED(5321608, 822236), SEED(2902657, 1969654),
          SEED(1245825, 2863137), SEED(7601290, 5100820),
          SEED(2590592, 4465694), SEED(3639943, 123115), SEED(2734976, 482988),
          SEED(6592129, 551084), SEED(6518534, 3018208), SEED(3937536, 4855027),
          SEED(2474118, 8242939), SEED(7541376, 3262199), SEED(781313, 945814),
          SEED(5309059, 7891002), SEED(515968, 4850324),
          SEED(5104005, 6424859)},
         {SEED(4316687, 3137072), SEED(4183808, 394145), SEED(5321608, 822236),
          SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694), SEED(3639943, 123115),
          SEED(2734976, 482988), SEED(6592129, 551084), SEED(6518534, 3018208),
          SEED(3937536, 4855027), SEED(2474118, 8242939),
          SEED(7541376, 3262199), SEED(781313, 945814), SEED(5309059, 7891002),
          SEED(515968, 4850324)},
         {SEED(1761793, 7713461), SEED(4316687, 3137072), SEED(4183808, 394145),
          SEED(5321608, 822236), SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694), SEED(3639943, 123115),
          SEED(2734976, 482988), SEED(6592129, 551084), SEED(6518534, 3018208),
          SEED(3937536, 4855027), SEED(2474118, 8242939),
          SEED(7541376, 3262199), SEED(781313, 945814), SEED(5309059, 7891002)},
         {SEED(8211716, 1472048), SEED(1761793, 7713461),
          SEED(4316687, 3137072), SEED(4183808, 394145), SEED(5321608, 822236),
          SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694), SEED(3639943, 123115),
          SEED(2734976, 482988), SEED(6592129, 551084), SEED(6518534, 3018208),
          SEED(3937536, 4855027), SEED(2474118, 8242939),
          SEED(7541376, 3262199), SEED(781313, 945814)},
         {SEED(6102921, 1768050), SEED(8211716, 1472048),
          SEED(1761793, 7713461), SEED(4316687, 3137072), SEED(4183808, 394145),
          SEED(5321608, 822236), SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694), SEED(3639943, 123115),
          SEED(2734976, 482988), SEED(6592129, 551084), SEED(6518534, 3018208),
          SEED(3937536, 4855027), SEED(2474118, 8242939),
          SEED(7541376, 3262199)},
         {SEED(3336576, 3656345), SEED(6102921, 1768050),
          SEED(8211716, 1472048), SEED(1761793, 7713461),
          SEED(4316687, 3137072), SEED(4183808, 394145), SEED(5321608, 822236),
          SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694), SEED(3639943, 123115),
          SEED(2734976, 482988), SEED(6592129, 551084), SEED(6518534, 3018208),
          SEED(3937536, 4855027), SEED(2474118, 8242939)},
         {SEED(6790805, 2991403), SEED(3336576, 3656345),
          SEED(6102921, 1768050), SEED(8211716, 1472048),
          SEED(1761793, 7713461), SEED(4316687, 3137072), SEED(4183808, 394145),
          SEED(5321608, 822236), SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694), SEED(3639943, 123115),
          SEED(2734976, 482988), SEED(6592129, 551084), SEED(6518534, 3018208),
          SEED(3937536, 4855027)},
         {SEED(5699329, 4179880), SEED(6790805, 2991403),
          SEED(3336576, 3656345), SEED(6102921, 1768050),
          SEED(8211716, 1472048), SEED(1761793, 7713461),
          SEED(4316687, 3137072), SEED(4183808, 394145), SEED(5321608, 822236),
          SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694), SEED(3639943, 123115),
          SEED(2734976, 482988), SEED(6592129, 551084), SEED(6518534, 3018208)},
         {SEED(6341642, 4490257), SEED(5699329, 4179880),
          SEED(6790805, 2991403), SEED(3336576, 3656345),
          SEED(6102921, 1768050), SEED(8211716, 1472048),
          SEED(1761793, 7713461), SEED(4316687, 3137072), SEED(4183808, 394145),
          SEED(5321608, 822236), SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694), SEED(3639943, 123115),
          SEED(2734976, 482988), SEED(6592129, 551084)},
         {SEED(4306442, 2319135), SEED(6341642, 4490257),
          SEED(5699329, 4179880), SEED(6790805, 2991403),
          SEED(3336576, 3656345), SEED(6102921, 1768050),
          SEED(8211716, 1472048), SEED(1761793, 7713461),
          SEED(4316687, 3137072), SEED(4183808, 394145), SEED(5321608, 822236),
          SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694), SEED(3639943, 123115),
          SEED(2734976, 482988)},
         {SEED(6071552, 4139333), SEED(4306442, 2319135),
          SEED(6341642, 4490257), SEED(5699329, 4179880),
          SEED(6790805, 2991403), SEED(3336576, 3656345),
          SEED(6102921, 1768050), SEED(8211716, 1472048),
          SEED(1761793, 7713461), SEED(4316687, 3137072), SEED(4183808, 394145),
          SEED(5321608, 822236), SEED(2902657, 1969654), SEED(1245825, 2863137),
          SEED(7601290, 5100820), SEED(2590592, 4465694),
          SEED(3639943, 123115)}},
        {{SEED(3891612, 2344137), SEED(4081878, 4225610),
          SEED(3037723, 4337034), SEED(3617598, 4953592),
          SEED(7986896, 1126797), SEED(1721957, 6292666),
          SEED(3842055, 1153011), SEED(5089230, 6028642),
          SEED(5844870, 2102326), SEED(6545487, 2344287),
          SEED(7115832, 7776704), SEED(3603778, 7413672), SEED(727042, 4482452),
          SEED(5736867, 8291737), SEED(241678, 278250), SEED(3131732, 1679012),
          SEED(6702593, 7823403)},
         {SEED(6300881, 561593), SEED(3891612, 2344137), SEED(4081878, 4225610),
          SEED(3037723, 4337034), SEED(3617598, 4953592),
          SEED(7986896, 1126797), SEED(1721957, 6292666),
          SEED(3842055, 1153011), SEED(5089230, 6028642),
          SEED(5844870, 2102326), SEED(6545487, 2344287),
          SEED(7115832, 7776704), SEED(3603778, 7413672), SEED(727042, 4482452),
          SEED(5736867, 8291737), SEED(241678, 278250), SEED(3131732, 1679012)},
         {SEED(6749330, 6632604), SEED(6300881, 561593), SEED(3891612, 2344137),
          SEED(4081878, 4225610), SEED(3037723, 4337034),
          SEED(3617598, 4953592), SEED(7986896, 1126797),
          SEED(1721957, 6292666), SEED(3842055, 1153011),
          SEED(5089230, 6028642), SEED(5844870, 2102326),
          SEED(6545487, 2344287), SEED(7115832, 7776704),
          SEED(3603778, 7413672), SEED(727042, 4482452), SEED(5736867, 8291737),
          SEED(241678, 278250)},
         {SEED(3279401, 4615284), SEED(6749330, 6632604), SEED(6300881, 561593),
          SEED(3891612, 2344137), SEED(4081878, 4225610),
          SEED(3037723, 4337034), SEED(3617598, 4953592),
          SEED(7986896, 1126797), SEED(1721957, 6292666),
          SEED(3842055, 1153011), SEED(5089230, 6028642),
          SEED(5844870, 2102326), SEED(6545487, 2344287),
          SEED(7115832, 7776704), SEED(3603778, 7413672), SEED(727042, 4482452),
          SEED(5736867, 8291737)},
         {SEED(1430137, 4128740), SEED(3279401, 4615284),
          SEED(6749330, 6632604), SEED(6300881, 561593), SEED(3891612, 2344137),
          SEED(4081878, 4225610), SEED(3037723, 4337034),
          SEED(3617598, 4953592), SEED(7986896, 1126797),
          SEED(1721957, 6292666), SEED(3842055, 1153011),
          SEED(5089230, 6028642), SEED(5844870, 2102326),
          SEED(6545487, 2344287), SEED(7115832, 7776704),
          SEED(3603778, 7413672), SEED(727042, 4482452)},
         {SEED(727042, 4482452), SEED(5736867, 8291737), SEED(241678, 278250),
          SEED(3131732, 1679012), SEED(6702593, 7823403),
          SEED(2169655, 4440079), SEED(239823, 3072599), SEED(6337101, 6696999),
          SEED(6161336, 2851265), SEED(1441409, 7171118),
          SEED(2994733, 6904569), SEED(238277, 2127947), SEED(4362188, 1546190),
          SEED(108003, 2199197), SEED(6303809, 2066037), SEED(3984100, 6097692),
          SEED(5289793, 7978876)},
         {SEED(3603778, 7413672), SEED(727042, 4482452), SEED(5736867, 8291737),
          SEED(241678, 278250), SEED(3131732, 1679012), SEED(6702593, 7823403),
          SEED(2169655, 4440079), SEED(239823, 3072599), SEED(6337101, 6696999),
          SEED(6161336, 2851265), SEED(1441409, 7171118),
          SEED(2994733, 6904569), SEED(238277, 2127947), SEED(4362188, 1546190),
          SEED(108003, 2199197), SEED(6303809, 2066037),
          SEED(3984100, 6097692)},
         {SEED(7115832, 7776704), SEED(3603778, 7413672), SEED(727042, 4482452),
          SEED(5736867, 8291737), SEED(241678, 278250), SEED(3131732, 1679012),
          SEED(6702593, 7823403), SEED(2169655, 4440079), SEED(239823, 3072599),
          SEED(6337101, 6696999), SEED(6161336, 2851265),
          SEED(1441409, 7171118), SEED(2994733, 6904569), SEED(238277, 2127947),
          SEED(4362188, 1546190), SEED(108003, 2199197),
          SEED(6303809, 2066037)},
         {SEED(6545487, 2344287), SEED(7115832, 7776704),
          SEED(3603778, 7413672), SEED(727042, 4482452), SEED(5736867, 8291737),
          SEED(241678, 278250), SEED(3131732, 1679012), SEED(6702593, 7823403),
          SEED(2169655, 4440079), SEED(239823, 3072599), SEED(6337101, 6696999),
          SEED(6161336, 2851265), SEED(1441409, 7171118),
          SEED(2994733, 6904569), SEED(238277, 2127947), SEED(4362188, 1546190),
          SEED(108003, 2199197)},
         {SEED(5844870, 2102326), SEED(6545487, 2344287),
          SEED(7115832, 7776704), SEED(3603778, 7413672), SEED(727042, 4482452),
          SEED(5736867, 8291737), SEED(241678, 278250), SEED(3131732, 1679012),
          SEED(6702593, 7823403), SEED(2169655, 4440079), SEED(239823, 3072599),
          SEED(6337101, 6696999), SEED(6161336, 2851265),
          SEED(1441409, 7171118), SEED(2994733, 6904569), SEED(238277, 2127947),
          SEED(4362188, 1546190)},
         {SEED(5089230, 6028642), SEED(5844870, 2102326),
          SEED(6545487, 2344287), SEED(7115832, 7776704),
          SEED(3603778, 7413672), SEED(727042, 4482452), SEED(5736867, 8291737),
          SEED(241678, 278250), SEED(3131732, 1679012), SEED(6702593, 7823403),
          SEED(2169655, 4440079), SEED(239823, 3072599), SEED(6337101, 6696999),
          SEED(6161336, 2851265), SEED(1441409, 7171118),
          SEED(2994733, 6904569), SEED(238277, 2127947)},
         {SEED(3842055, 1153011), SEED(5089230, 6028642),
          SEED(5844870, 2102326), SEED(6545487, 2344287),
          SEED(7115832, 7776704), SEED(3603778, 7413672), SEED(727042, 4482452),
          SEED(5736867, 8291737), SEED(241678, 278250), SEED(3131732, 1679012),
          SEED(6702593, 7823403), SEED(2169655, 4440079), SEED(239823, 3072599),
          SEED(6337101, 6696999), SEED(6161336, 2851265),
          SEED(1441409, 7171118), SEED(2994733, 6904569)},
         {SEED(1721957, 6292666), SEED(3842055, 1153011),
          SEED(5089230, 6028642), SEED(5844870, 2102326),
          SEED(6545487, 2344287), SEED(7115832, 7776704),
          SEED(3603778, 7413672), SEED(727042, 4482452), SEED(5736867, 8291737),
          SEED(241678, 278250), SEED(3131732, 1679012), SEED(6702593, 7823403),
          SEED(2169655, 4440079), SEED(239823, 3072599), SEED(6337101, 6696999),
          SEED(6161336, 2851265), SEED(1441409, 7171118)},
         {SEED(7986896, 1126797), SEED(1721957, 6292666),
          SEED(3842055, 1153011), SEED(5089230, 6028642),
          SEED(5844870, 2102326), SEED(6545487, 2344287),
          SEED(7115832, 7776704), SEED(3603778, 7413672), SEED(727042, 4482452),
          SEED(5736867, 8291737), SEED(241678, 278250), SEED(3131732, 1679012),
          SEED(6702593, 7823403), SEED(2169655, 4440079), SEED(239823, 3072599),
          SEED(6337101, 6696999), SEED(6161336, 2851265)},
         {SEED(3617598, 4953592), SEED(7986896, 1126797),
          SEED(1721957, 6292666), SEED(3842055, 1153011),
          SEED(5089230, 6028642), SEED(5844870, 2102326),
          SEED(6545487, 2344287), SEED(7115832, 7776704),
          SEED(3603778, 7413672), SEED(727042, 4482452), SEED(5736867, 8291737),
          SEED(241678, 278250), SEED(3131732, 1679012), SEED(6702593, 7823403),
          SEED(2169655, 4440079), SEED(239823, 3072599),
          SEED(6337101, 6696999)},
         {SEED(3037723, 4337034), SEED(3617598, 4953592),
          SEED(7986896, 1126797), SEED(1721957, 6292666),
          SEED(3842055, 1153011), SEED(5089230, 6028642),
          SEED(5844870, 2102326), SEED(6545487, 2344287),
          SEED(7115832, 7776704), SEED(3603778, 7413672), SEED(727042, 4482452),
          SEED(5736867, 8291737), SEED(241678, 278250), SEED(3131732, 1679012),
          SEED(6702593, 7823403), SEED(2169655, 4440079),
          SEED(239823, 3072599)},
         {SEED(4081878, 4225610), SEED(3037723, 4337034),
          SEED(3617598, 4953592), SEED(7986896, 1126797),
          SEED(1721957, 6292666), SEED(3842055, 1153011),
          SEED(5089230, 6028642), SEED(5844870, 2102326),
          SEED(6545487, 2344287), SEED(7115832, 7776704),
          SEED(3603778, 7413672), SEED(727042, 4482452), SEED(5736867, 8291737),
          SEED(241678, 278250), SEED(3131732, 1679012), SEED(6702593, 7823403),
          SEED(2169655, 4440079)}},
    },
    {
        {{SEED(5952028, 3476842), SEED(4823781, 5811163),
          SEED(3946032, 6218140), SEED(2078216, 5127233),
          SEED(2392774, 4760400), SEED(6038357, 2101508),
          SEED(3407592, 6714475), SEED(5625000, 7621489),
          SEED(7220505, 4763499), SEED(6073004, 3855051), SEED(319977, 4636927),
          SEED(3169385, 4132990), SEED(6323248, 3868266),
          SEED(7791055, 1213984), SEED(6402713, 4413944),
          SEED(7711435, 5439220), SEED(3176752, 7585771)},
         {SEED(5569526, 3957563), SEED(5952028, 3476842),
          SEED(4823781, 5811163), SEED(3946032, 6218140),
          SEED(2078216, 5127233), SEED(2392774, 4760400),
          SEED(6038357, 2101508), SEED(3407592, 6714475),
          SEED(5625000, 7621489), SEED(7220505, 4763499),
          SEED(6073004, 3855051), SEED(319977, 4636927), SEED(3169385, 4132990),
          SEED(6323248, 3868266), SEED(7791055, 1213984),
          SEED(6402713, 4413944), SEED(7711435, 5439220)},
         {SEED(1401043, 2177846), SEED(5569526, 3957563),
          SEED(5952028, 3476842), SEED(4823781, 5811163),
          SEED(3946032, 6218140), SEED(2078216, 5127233),
          SEED(2392774, 4760400), SEED(6038357, 2101508),
          SEED(3407592, 6714475), SEED(5625000, 7621489),
          SEED(7220505, 4763499), SEED(6073004, 3855051), SEED(319977, 4636927),
          SEED(3169385, 4132990), SEED(6323248, 3868266),
          SEED(7791055, 1213984), SEED(6402713, 4413944)},
         {SEED(1960137, 2243477), SEED(1401043, 2177846),
          SEED(5569526, 3957563), SEED(5952028, 3476842),
          SEED(4823781, 5811163), SEED(3946032, 6218140),
          SEED(2078216, 5127233), SEED(2392774, 4760400),
          SEED(6038357, 2101508), SEED(3407592, 6714475),
          SEED(5625000, 7621489), SEED(7220505, 4763499),
          SEED(6073004, 3855051), SEED(319977, 4636927), SEED(3169385, 4132990),
          SEED(6323248, 3868266), SEED(7791055, 1213984)},
         {SEED(4226228, 7025148), SEED(1960137, 2243477),
          SEED(1401043, 2177846), SEED(5569526, 3957563),
          SEED(5952028, 3476842), SEED(4823781, 5811163),
          SEED(3946032, 6218140), SEED(2078216, 5127233),
          SEED(2392774, 4760400), SEED(6038357, 2101508),
          SEED(3407592, 6714475), SEED(5625000, 7621489),
          SEED(7220505, 4763499), SEED(6073004, 3855051), SEED(319977, 4636927),
          SEED(3169385, 4132990), SEED(6323248, 3868266)},
         {SEED(6323248, 3868266), SEED(7791055, 1213984),
          SEED(6402713, 4413944), SEED(7711435, 5439220),
          SEED(3176752, 7585771), SEED(8302279, 1375333),
          SEED(1416189, 7485296), SEED(6709640, 6985258), SEED(3246319, 363733),
          SEED(4708378, 905348), SEED(5718380, 5853189), SEED(238207, 2581485),
          SEED(7690360, 3753222), SEED(7818058, 3549514),
          SEED(8058899, 7829714), SEED(997150, 7586314),
          SEED(8381241, 4935826)},
         {SEED(3169385, 4132990), SEED(6323248, 3868266),
          SEED(7791055, 1213984), SEED(6402713, 4413944),
          SEED(7711435, 5439220), SEED(3176752, 7585771),
          SEED(8302279, 1375333), SEED(1416189, 7485296),
          SEED(6709640, 6985258), SEED(3246319, 363733), SEED(4708378, 905348),
          SEED(5718380, 5853189), SEED(238207, 2581485), SEED(7690360, 3753222),
          SEED(7818058, 3549514), SEED(8058899, 7829714),
          SEED(997150, 7586314)},
         {SEED(319977, 4636927), SEED(3169385, 4132990), SEED(6323248, 3868266),
          SEED(7791055, 1213984), SEED(6402713, 4413944),
          SEED(7711435, 5439220), SEED(3176752, 7585771),
          SEED(8302279, 1375333), SEED(1416189, 7485296),
          SEED(6709640, 6985258), SEED(3246319, 363733), SEED(4708378, 905348),
          SEED(5718380, 5853189), SEED(238207, 2581485), SEED(7690360, 3753222),
          SEED(7818058, 3549514), SEED(8058899, 7829714)},
         {SEED(6073004, 3855051), SEED(319977, 4636927), SEED(3169385, 4132990),
          SEED(6323248, 3868266), SEED(7791055, 1213984),
          SEED(6402713, 4413944), SEED(7711435, 5439220),
          SEED(3176752, 7585771), SEED(8302279, 1375333),
          SEED(1416189, 7485296), SEED(6709640, 6985258), SEED(3246319, 363733),
          SEED(4708378, 905348), SEED(5718380, 5853189), SEED(238207, 2581485),
          SEED(7690360, 3753222), SEED(7818058, 3549514)},
         {SEED(7220505, 4763499), SEED(6073004, 3855051), SEED(319977, 4636927),
          SEED(3169385, 4132990), SEED(6323248, 3868266),
          SEED(7791055, 1213984), SEED(6402713, 4413944),
          SEED(7711435, 5439220), SEED(3176752, 7585771),
          SEED(8302279, 1375333), SEED(1416189, 7485296),
          SEED(6709640, 6985258), SEED(3246319, 363733), SEED(4708378, 905348),
          SEED(5718380, 5853189), SEED(238207, 2581485),
          SEED(7690360, 3753222)},
         {SEED(5625000, 7621489), SEED(7220505, 4763499),
          SEED(6073004, 3855051), SEED(319977, 4636927), SEED(3169385, 4132990),
          SEED(6323248, 3868266), SEED(7791055, 1213984),
          SEED(6402713, 4413944), SEED(7711435, 5439220),
          SEED(3176752, 7585771), SEED(8302279, 1375333),
          SEED(1416189, 7485296), SEED(6709640, 6985258), SEED(3246319, 363733),
          SEED(4708378, 905348), SEED(5718380, 5853189), SEED(238207, 2581485)},
         {SEED(3407592, 6714475), SEED(5625000, 7621489),
          SEED(7220505, 4763499), SEED(6073004, 3855051), SEED(319977, 4636927),
          SEED(3169385, 4132990), SEED(6323248, 3868266),
          SEED(7791055, 1213984), SEED(6402713, 4413944),
          SEED(7711435, 5439220), SEED(3176752, 7585771),
          SEED(8302279, 1375333), SEED(1416189, 7485296),
          SEED(6709640, 6985258), SEED(3246319, 363733), SEED(4708378, 905348),
          SEED(5718380, 5853189)},
         {SEED(6038357, 2101508), SEED(3407592, 6714475),
          SEED(5625000, 7621489), SEED(7220505, 4763499),
          SEED(6073004, 3855051), SEED(319977, 4636927), SEED(3169385, 4132990),
          SEED(6323248, 3868266), SEED(7791055, 1213984),
          SEED(6402713, 4413944), SEED(7711435, 5439220),
          SEED(3176752, 7585771), SEED(8302279, 1375333),
          SEED(1416189, 7485296), SEED(6709640, 6985258), SEED(3246319, 363733),
          SEED(4708378, 905348)},
         {SEED(2392774, 4760400), SEED(6038357, 2101508),
          SEED(3407592, 6714475), SEED(5625000, 7621489),
          SEED(7220505, 4763499), SEED(6073004, 3855051), SEED(319977, 4636927),
          SEED(3169385, 4132990), SEED(6323248, 3868266),
          SEED(7791055, 1213984), SEED(6402713, 4413944),
          SEED(7711435, 5439220), SEED(3176752, 7585771),
          SEED(8302279, 1375333), SEED(1416189, 7485296),
          SEED(6709640, 6985258), SEED(3246319, 363733)},
         {SEED(2078216, 5127233), SEED(2392774, 4760400),
          SEED(6038357, 2101508), SEED(3407592, 6714475),
          SEED(5625000, 7621489), SEED(7220505, 4763499),
          SEED(6073004, 3855051), SEED(319977, 4636927), SEED(3169385, 4132990),
          SEED(6323248, 3868266), SEED(7791055, 1213984),
          SEED(6402713, 4413944), SEED(7711435, 5439220),
          SEED(3176752, 7585771), SEED(8302279, 1375333),
          SEED(1416189, 7485296), SEED(6709640, 6985258)},
         {SEED(3946032, 6218140), SEED(2078216, 5127233),
          SEED(2392774, 4760400), SEED(6038357, 2101508),
          SEED(3407592, 6714475), SEED(5625000, 7621489),
          SEED(7220505, 4763499), SEED(6073004, 3855051), SEED(319977, 4636927),
          SEED(3169385, 4132990), SEED(6323248, 3868266),
          SEED(7791055, 1213984), SEED(6402713, 4413944),
          SEED(7711435, 5439220), SEED(3176752, 7585771),
          SEED(8302279, 1375333), SEED(1416189, 7485296)},
         {SEED(4823781, 5811163), SEED(3946032, 6218140),
          SEED(2078216, 5127233), SEED(2392774, 4760400),
          SEED(6038357, 2101508), SEED(3407592, 6714475),
          SEED(5625000, 7621489), SEED(7220505, 4763499),
          SEED(6073004, 3855051), SEED(319977, 4636927), SEED(3169385, 4132990),
          SEED(6323248, 3868266), SEED(7791055, 1213984),
          SEED(6402713, 4413944), SEED(7711435, 5439220),
          SEED(3176752, 7585771), SEED(8302279, 1375333)}},
        {{SEED(781224, 4386555), SEED(3094348, 2316722), SEED(247329, 2869000),
          SEED(5187931, 8102235), SEED(6627152, 5731627),
          SEED(7800710, 1732832), SEED(7527342, 6906895), SEED(1181624, 63889),
          SEED(1710474, 5296439), SEED(1375797, 1613392),
          SEED(6768461, 7012639), SEED(5880146, 6026647),
          SEED(2127268, 4462998), SEED(6581571, 1386277),
          SEED(1080654, 1156905), SEED(1139489, 5034197),
          SEED(1202633, 4769136)},
         {SEED(7829785, 2112155), SEED(781224, 4386555), SEED(3094348, 2316722),
          SEED(247329, 2869000), SEED(5187931, 8102235), SEED(6627152, 5731627),
          SEED(7800710, 1732832), SEED(7527342, 6906895), SEED(1181624, 63889),
          SEED(1710474, 5296439), SEED(1375797, 1613392),
          SEED(6768461, 7012639), SEED(5880146, 6026647),
          SEED(2127268, 4462998), SEED(6581571, 1386277),
          SEED(1080654, 1156905), SEED(1139489, 5034197)},
         {SEED(6327420, 4747824), SEED(7829785, 2112155), SEED(781224, 4386555),
          SEED(3094348, 2316722), SEED(247329, 2869000), SEED(5187931, 8102235),
          SEED(6627152, 5731627), SEED(7800710, 1732832),
          SEED(7527342, 6906895), SEED(1181624, 63889), SEED(1710474, 5296439),
          SEED(1375797, 1613392), SEED(6768461, 7012639),
          SEED(5880146, 6026647), SEED(2127268, 4462998),
          SEED(6581571, 1386277), SEED(1080654, 1156905)},
         {SEED(1327983, 4025905), SEED(6327420, 4747824),
          SEED(7829785, 2112155), SEED(781224, 4386555), SEED(3094348, 2316722),
          SEED(247329, 2869000), SEED(5187931, 8102235), SEED(6627152, 5731627),
          SEED(7800710, 1732832), SEED(7527342, 6906895), SEED(1181624, 63889),
          SEED(1710474, 5296439), SEED(1375797, 1613392),
          SEED(6768461, 7012639), SEED(5880146, 6026647),
          SEED(2127268, 4462998), SEED(6581571, 1386277)},
         {SEED(1287311, 3703000), SEED(1327983, 4025905),
          SEED(6327420, 4747824), SEED(7829785, 2112155), SEED(781224, 4386555),
          SEED(3094348, 2316722), SEED(247329, 2869000), SEED(5187931, 8102235),
          SEED(6627152, 5731627), SEED(7800710, 1732832),
          SEED(7527342, 6906895), SEED(1181624, 63889), SEED(1710474, 5296439),
          SEED(1375797, 1613392), SEED(6768461, 7012639),
          SEED(5880146, 6026647), SEED(2127268, 4462998)},
         {SEED(2127268, 4462998), SEED(6581571, 1386277),
          SEED(1080654, 1156905), SEED(1139489, 5034197),
          SEED(1202633, 4769136), SEED(1369122, 2653722),
          SEED(3955614, 3798434), SEED(7454313, 2805110),
          SEED(3477457, 2805796), SEED(5251355, 4118235),
          SEED(1032249, 3108801), SEED(1647196, 880248), SEED(7442964, 3989498),
          SEED(3517511, 3910161), SEED(295143, 456487), SEED(5628972, 1978442),
          SEED(4677513, 1257511)},
         {SEED(5880146, 6026647), SEED(2127268, 4462998),
          SEED(6581571, 1386277), SEED(1080654, 1156905),
          SEED(1139489, 5034197), SEED(1202633, 4769136),
          SEED(1369122, 2653722), SEED(3955614, 3798434),
          SEED(7454313, 2805110), SEED(3477457, 2805796),
          SEED(5251355, 4118235), SEED(1032249, 3108801), SEED(1647196, 880248),
          SEED(7442964, 3989498), SEED(3517511, 3910161), SEED(295143, 456487),
          SEED(5628972, 1978442)},
         {SEED(6768461, 7012639), SEED(5880146, 6026647),
          SEED(2127268, 4462998), SEED(6581571, 1386277),
          SEED(1080654, 1156905), SEED(1139489, 5034197),
          SEED(1202633, 4769136), SEED(1369122, 2653722),
          SEED(3955614, 3798434), SEED(7454313, 2805110),
          SEED(3477457, 2805796), SEED(5251355, 4118235),
          SEED(1032249, 3108801), SEED(1647196, 880248), SEED(7442964, 3989498),
          SEED(3517511, 3910161), SEED(295143, 456487)},
         {SEED(1375797, 1613392), SEED(6768461, 7012639),
          SEED(5880146, 6026647), SEED(2127268, 4462998),
          SEED(6581571, 1386277), SEED(1080654, 1156905),
          SEED(1139489, 5034197), SEED(1202633, 4769136),
          SEED(1369122, 2653722), SEED(3955614, 3798434),
          SEED(7454313, 2805110), SEED(3477457, 2805796),
          SEED(5251355, 4118235), SEED(1032249, 3108801), SEED(1647196, 880248),
          SEED(7442964, 3989498), SEED(3517511, 3910161)},
         {SEED(1710474, 5296439), SEED(1375797, 1613392),
          SEED(6768461, 7012639), SEED(5880146, 6026647),
          SEED(2127268, 4462998), SEED(6581571, 1386277),
          SEED(1080654, 1156905), SEED(1139489, 5034197),
          SEED(1202633, 4769136), SEED(1369122, 2653722),
          SEED(3955614, 3798434), SEED(7454313, 2805110),
          SEED(3477457, 2805796), SEED(5251355, 4118235),
          SEED(1032249, 3108801), SEED(1647196, 880248),
          SEED(7442964, 3989498)},
         {SEED(1181624, 63889), SEED(1710474, 5296439), SEED(1375797, 1613392),
          SEED(6768461, 7012639), SEED(5880146, 6026647),
          SEED(2127268, 4462998), SEED(6581571, 1386277),
          SEED(1080654, 1156905), SEED(1139489, 5034197),
          SEED(1202633, 4769136), SEED(1369122, 2653722),
          SEED(3955614, 3798434), SEED(7454313, 2805110),
          SEED(3477457, 2805796), SEED(5251355, 4118235),
          SEED(1032249, 3108801), SEED(1647196, 880248)},
         {SEED(7527342, 6906895), SEED(1181624, 63889), SEED(1710474, 5296439),
          SEED(1375797, 1613392), SEED(6768461, 7012639),
          SEED(5880146, 6026647), SEED(2127268, 4462998),
          SEED(6581571, 1386277), SEED(1080654, 1156905),
          SEED(1139489, 5034197), SEED(1202633, 4769136),
          SEED(1369122, 2653722), SEED(3955614, 3798434),
          SEED(7454313, 2805110), SEED(3477457, 2805796),
          SEED(5251355, 4118235), SEED(1032249, 3108801)},
         {SEED(7800710, 1732832), SEED(7527342, 6906895), SEED(1181624, 63889),
          SEED(1710474, 5296439), SEED(1375797, 1613392),
          SEED(6768461, 7012639), SEED(5880146, 6026647),
          SEED(2127268, 4462998), SEED(6581571, 1386277),
          SEED(1080654, 1156905), SEED(1139489, 5034197),
          SEED(1202633, 4769136), SEED(1369122, 2653722),
          SEED(3955614, 3798434), SEED(7454313, 2805110),
          SEED(3477457, 2805796), SEED(5251355, 4118235)},
         {SEED(6627152, 5731627), SEED(7800710, 1732832),
          SEED(7527342, 6906895), SEED(1181624, 63889), SEED(1710474, 5296439),
          SEED(1375797, 1613392), SEED(6768461, 7012639),
          SEED(5880146, 6026647), SEED(2127268, 4462998),
          SEED(6581571, 1386277), SEED(1080654, 1156905),
          SEED(1139489, 5034197), SEED(1202633, 4769136),
          SEED(1369122, 2653722), SEED(3955614, 3798434),
          SEED(7454313, 2805110), SEED(3477457, 2805796)},
         {SEED(5187931, 8102235), SEED(6627152, 5731627),
          SEED(7800710, 1732832), SEED(7527342, 6906895), SEED(1181624, 63889),
          SEED(1710474, 5296439), SEED(1375797, 1613392),
          SEED(6768461, 7012639), SEED(5880146, 6026647),
          SEED(2127268, 4462998), SEED(6581571, 1386277),
          SEED(1080654, 1156905), SEED(1139489, 5034197),
          SEED(1202633, 4769136), SEED(1369122, 2653722),
          SEED(3955614, 3798434), SEED(7454313, 2805110)},
         {SEED(247329, 2869000), SEED(5187931, 8102235), SEED(6627152, 5731627),
          SEED(7800710, 1732832), SEED(7527342, 6906895), SEED(1181624, 63889),
          SEED(1710474, 5296439), SEED(1375797, 1613392),
          SEED(6768461, 7012639), SEED(5880146, 6026647),
          SEED(2127268, 4462998), SEED(6581571, 1386277),
          SEED(1080654, 1156905), SEED(1139489, 5034197),
          SEED(1202633, 4769136), SEED(1369122, 2653722),
          SEED(3955614, 3798434)},
         {SEED(3094348, 2316722), SEED(247329, 2869000), SEED(5187931, 8102235),
          SEED(6627152, 5731627), SEED(7800710, 1732832),
          SEED(7527342, 6906895), SEED(1181624, 63889), SEED(1710474, 5296439),
          SEED(1375797, 1613392), SEED(6768461, 7012639),
          SEED(5880146, 6026647), SEED(2127268, 4462998),
          SEED(6581571, 1386277), SEED(1080654, 1156905),
          SEED(1139489, 5034197), SEED(1202633, 4769136),
          SEED(1369122, 2653722)}},
        {{SEED(1198004, 157505), SEED(7703559, 1573923), SEED(7248227, 2232340),
          SEED(7998146, 1495510), SEED(6507526, 5227544),
          SEED(6993671, 7381671), SEED(2725008, 5295086),
          SEED(7693669, 1436288), SEED(1273391, 3088370),
          SEED(3055392, 2125576), SEED(1537819, 5450073),
          SEED(7260648, 4330281), SEED(5893122, 3600825),
          SEED(6845674, 6975300), SEED(2650219, 2997331),
          SEED(7477997, 6571920), SEED(4027132, 5179193)},
         {SEED(2146050, 2018130), SEED(1198004, 157505), SEED(7703559, 1573923),
          SEED(7248227, 2232340), SEED(7998146, 1495510),
          SEED(6507526, 5227544), SEED(6993671, 7381671),
          SEED(2725008, 5295086), SEED(7693669, 1436288),
          SEED(1273391, 3088370), SEED(3055392, 2125576),
          SEED(1537819, 5450073), SEED(7260648, 4330281),
          SEED(5893122, 3600825), SEED(6845674, 6975300),
          SEED(2650219, 2997331), SEED(7477997, 6571920)},
         {SEED(7087535, 8067431), SEED(2146050, 2018130), SEED(1198004, 157505),
          SEED(7703559, 1573923), SEED(7248227, 2232340),
          SEED(7998146, 1495510), SEED(6507526, 5227544),
          SEED(6993671, 7381671), SEED(2725008, 5295086),
          SEED(7693669, 1436288), SEED(1273391, 3088370),
          SEED(3055392, 2125576), SEED(1537819, 5450073),
          SEED(7260648, 4330281), SEED(5893122, 3600825),
          SEED(6845674, 6975300), SEED(2650219, 2997331)},
         {SEED(1509838, 5229672), SEED(7087535, 8067431),
          SEED(2146050, 2018130), SEED(1198004, 157505), SEED(7703559, 1573923),
          SEED(7248227, 2232340), SEED(7998146, 1495510),
          SEED(6507526, 5227544), SEED(6993671, 7381671),
          SEED(2725008, 5295086), SEED(7693669, 1436288),
          SEED(1273391, 3088370), SEED(3055392, 2125576),
          SEED(1537819, 5450073), SEED(7260648, 4330281),
          SEED(5893122, 3600825), SEED(6845674, 6975300)},
         {SEED(6160625, 160616), SEED(1509838, 5229672), SEED(7087535, 8067431),
          SEED(2146050, 2018130), SEED(1198004, 157505), SEED(7703559, 1573923),
          SEED(7248227, 2232340), SEED(7998146, 1495510),
          SEED(6507526, 5227544), SEED(6993671, 7381671),
          SEED(2725008, 5295086), SEED(7693669, 1436288),
          SEED(1273391, 3088370), SEED(3055392, 2125576),
          SEED(1537819, 5450073), SEED(7260648, 4330281),
          SEED(5893122, 3600825)},
         {SEED(5893122, 3600825), SEED(6845674, 6975300),
          SEED(2650219, 2997331), SEED(7477997, 6571920),
          SEED(4027132, 5179193), SEED(2592941, 1164441),
          SEED(4978551, 4667445), SEED(7943166, 796051), SEED(6724755, 6795748),
          SEED(3452134, 3101968), SEED(5455852, 1931598), SEED(3852968, 964804),
          SEED(1800547, 6224071), SEED(2816325, 4501677), SEED(405173, 7516853),
          SEED(2448430, 7266760), SEED(3233516, 7539696)},
         {SEED(7260648, 4330281), SEED(5893122, 3600825),
          SEED(6845674, 6975300), SEED(2650219, 2997331),
          SEED(7477997, 6571920), SEED(4027132, 5179193),
          SEED(2592941, 1164441), SEED(4978551, 4667445), SEED(7943166, 796051),
          SEED(6724755, 6795748), SEED(3452134, 3101968),
          SEED(5455852, 1931598), SEED(3852968, 964804), SEED(1800547, 6224071),
          SEED(2816325, 4501677), SEED(405173, 7516853),
          SEED(2448430, 7266760)},
         {SEED(1537819, 5450073), SEED(7260648, 4330281),
          SEED(5893122, 3600825), SEED(6845674, 6975300),
          SEED(2650219, 2997331), SEED(7477997, 6571920),
          SEED(4027132, 5179193), SEED(2592941, 1164441),
          SEED(4978551, 4667445), SEED(7943166, 796051), SEED(6724755, 6795748),
          SEED(3452134, 3101968), SEED(5455852, 1931598), SEED(3852968, 964804),
          SEED(1800547, 6224071), SEED(2816325, 4501677),
          SEED(405173, 7516853)},
         {SEED(3055392, 2125576), SEED(1537819, 5450073),
          SEED(7260648, 4330281), SEED(5893122, 3600825),
          SEED(6845674, 6975300), SEED(2650219, 2997331),
          SEED(7477997, 6571920), SEED(4027132, 5179193),
          SEED(2592941, 1164441), SEED(4978551, 4667445), SEED(7943166, 796051),
          SEED(6724755, 6795748), SEED(3452134, 3101968),
          SEED(5455852, 1931598), SEED(3852968, 964804), SEED(1800547, 6224071),
          SEED(2816325, 4501677)},
         {SEED(1273391, 3088370), SEED(3055392, 2125576),
          SEED(1537819, 5450073), SEED(7260648, 4330281),
          SEED(5893122, 3600825), SEED(6845674, 6975300),
          SEED(2650219, 2997331), SEED(7477997, 6571920),
          SEED(4027132, 5179193), SEED(2592941, 1164441),
          SEED(4978551, 4667445), SEED(7943166, 796051), SEED(6724755, 6795748),
          SEED(3452134, 3101968), SEED(5455852, 1931598), SEED(3852968, 964804),
          SEED(1800547, 6224071)},
         {SEED(7693669, 1436288), SEED(1273391, 3088370),
          SEED(3055392, 2125576), SEED(1537819, 5450073),
          SEED(7260648, 4330281), SEED(5893122, 3600825),
          SEED(6845674, 6975300), SEED(2650219, 2997331),
          SEED(7477997, 6571920), SEED(4027132, 5179193),
          SEED(2592941, 1164441), SEED(4978551, 4667445), SEED(7943166, 796051),
          SEED(6724755, 6795748), SEED(3452134, 3101968),
          SEED(5455852, 1931598), SEED(3852968, 964804)},
         {SEED(2725008, 5295086), SEED(7693669, 1436288),
          SEED(1273391, 3088370), SEED(3055392, 2125576),
          SEED(1537819, 5450073), SEED(7260648, 4330281),
          SEED(5893122, 3600825), SEED(6845674, 6975300),
          SEED(2650219, 2997331), SEED(7477997, 6571920),
          SEED(4027132, 5179193), SEED(2592941, 1164441),
          SEED(4978551, 4667445), SEED(7943166, 796051), SEED(6724755, 6795748),
          SEED(3452134, 3101968), SEED(5455852, 1931598)},
         {SEED(6993671, 7381671), SEED(2725008, 5295086),
          SEED(7693669, 1436288), SEED(1273391, 3088370),
          SEED(3055392, 2125576), SEED(1537819, 5450073),
          SEED(7260648, 4330281), SEED(5893122, 3600825),
          SEED(6845674, 6975300), SEED(2650219, 2997331),
          SEED(7477997, 6571920), SEED(4027132, 5179193),
          SEED(2592941, 1164441), SEED(4978551, 4667445), SEED(7943166, 796051),
          SEED(6724755, 6795748), SEED(3452134, 3101968)},
         {SEED(6507526, 5227544), SEED(6993671, 7381671),
          SEED(2725008, 5295086), SEED(7693669, 1436288),
          SEED(1273391, 3088370), SEED(3055392, 2125576),
          SEED(1537819, 5450073), SEED(7260648, 4330281),
          SEED(5893122, 3600825), SEED(6845674, 6975300),
          SEED(2650219, 2997331), SEED(7477997, 6571920),
          SEED(4027132, 5179193), SEED(2592941, 1164441),
          SEED(4978551, 4667445), SEED(7943166, 796051),
          SEED(6724755, 6795748)},
         {SEED(7998146, 1495510), SEED(6507526, 5227544),
          SEED(6993671, 7381671), SEED(2725008, 5295086),
          SEED(7693669, 1436288), SEED(1273391, 3088370),
          SEED(3055392, 2125576), SEED(1537819, 5450073),
          SEED(7260648, 4330281), SEED(5893122, 3600825),
          SEED(6845674, 6975300), SEED(2650219, 2997331),
          SEED(7477997, 6571920), SEED(4027132, 5179193),
          SEED(2592941, 1164441), SEED(4978551, 4667445),
          SEED(7943166, 796051)},
         {SEED(7248227, 2232340), SEED(7998146, 1495510),
          SEED(6507526, 5227544), SEED(6993671, 7381671),
          SEED(2725008, 5295086), SEED(7693669, 1436288),
          SEED(1273391, 3088370), SEED(3055392, 2125576),
          SEED(1537819, 5450073), SEED(7260648, 4330281),
          SEED(5893122, 3600825), SEED(6845674, 6975300),
          SEED(2650219, 2997331), SEED(7477997, 6571920),
          SEED(4027132, 5179193), SEED(2592941, 1164441),
          SEED(4978551, 4667445)},
         {SEED(7703559, 1573923), SEED(7248227, 2232340),
          SEED(7998146, 1495510), SEED(6507526, 5227544),
          SEED(6993671, 7381671), SEED(2725008, 5295086),
          SEED(7693669, 1436288), SEED(1273391, 3088370),
          SEED(3055392, 2125576), SEED(1537819, 5450073),
          SEED(7260648, 4330281), SEED(5893122, 3600825),
          SEED(6845674, 6975300), SEED(2650219, 2997331),
          SEED(7477997, 6571920), SEED(4027132, 5179193),
          SEED(2592941, 1164441)}},
    },
    {
        {{SEED(6985568, 3984851), SEED(6494130, 4520488),
          SEED(4144396, 3730990), SEED(960121, 6383268), SEED(215081, 1096084),
          SEED(2609184, 7992267), SEED(2605807, 3045488),
          SEED(5919390, 7990662), SEED(765900, 3068831), SEED(7755415, 158053),
          SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370), SEED(5399006, 4505155)},
         {SEED(5614087, 5601239), SEED(6985568, 3984851),
          SEED(6494130, 4520488), SEED(4144396, 3730990), SEED(960121, 6383268),
          SEED(215081, 1096084), SEED(2609184, 7992267), SEED(2605807, 3045488),
          SEED(5919390, 7990662), SEED(765900, 3068831), SEED(7755415, 158053),
          SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370)},
         {SEED(5074751, 3147030), SEED(5614087, 5601239),
          SEED(6985568, 3984851), SEED(6494130, 4520488),
          SEED(4144396, 3730990), SEED(960121, 6383268), SEED(215081, 1096084),
          SEED(2609184, 7992267), SEED(2605807, 3045488),
          SEED(5919390, 7990662), SEED(765900, 3068831), SEED(7755415, 158053),
          SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456),
          SEED(379708, 7757779)},
         {SEED(4524104, 3100161), SEED(5074751, 3147030),
          SEED(5614087, 5601239), SEED(6985568, 3984851),
          SEED(6494130, 4520488), SEED(4144396, 3730990), SEED(960121, 6383268),
          SEED(215081, 1096084), SEED(2609184, 7992267), SEED(2605807, 3045488),
          SEED(5919390, 7990662), SEED(765900, 3068831), SEED(7755415, 158053),
          SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456)},
         {SEED(5772392, 1157337), SEED(4524104, 3100161),
          SEED(5074751, 3147030), SEED(5614087, 5601239),
          SEED(6985568, 3984851), SEED(6494130, 4520488),
          SEED(4144396, 3730990), SEED(960121, 6383268), SEED(215081, 1096084),
          SEED(2609184, 7992267), SEED(2605807, 3045488),
          SEED(5919390, 7990662), SEED(765900, 3068831), SEED(7755415, 158053),
          SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352)},
         {SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370), SEED(5399006, 4505155),
          SEED(4376384, 4381192), SEED(3888323, 1475000),
          SEED(6613614, 4128935), SEED(194221, 3314437), SEED(848274, 938030),
          SEED(7128952, 3511303), SEED(6783279, 4599849),
          SEED(3692460, 6008310), SEED(1487638, 6431982), SEED(7375707, 788882),
          SEED(8142818, 7717200), SEED(7200738, 2329090)},
         {SEED(4211136, 6834246), SEED(2226930, 1982352),
          SEED(7666870, 5025456), SEED(379708, 7757779), SEED(4114630, 5152370),
          SEED(5399006, 4505155), SEED(4376384, 4381192),
          SEED(3888323, 1475000), SEED(6613614, 4128935), SEED(194221, 3314437),
          SEED(848274, 938030), SEED(7128952, 3511303), SEED(6783279, 4599849),
          SEED(3692460, 6008310), SEED(1487638, 6431982), SEED(7375707, 788882),
          SEED(8142818, 7717200)},
         {SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370), SEED(5399006, 4505155),
          SEED(4376384, 4381192), SEED(3888323, 1475000),
          SEED(6613614, 4128935), SEED(194221, 3314437), SEED(848274, 938030),
          SEED(7128952, 3511303), SEED(6783279, 4599849),
          SEED(3692460, 6008310), SEED(1487638, 6431982),
          SEED(7375707, 788882)},
         {SEED(7755415, 158053), SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370), SEED(5399006, 4505155),
          SEED(4376384, 4381192), SEED(3888323, 1475000),
          SEED(6613614, 4128935), SEED(194221, 3314437), SEED(848274, 938030),
          SEED(7128952, 3511303), SEED(6783279, 4599849),
          SEED(3692460, 6008310), SEED(1487638, 6431982)},
         {SEED(765900, 3068831), SEED(7755415, 158053), SEED(3868840, 4480963),
          SEED(4211136, 6834246), SEED(2226930, 1982352),
          SEED(7666870, 5025456), SEED(379708, 7757779), SEED(4114630, 5152370),
          SEED(5399006, 4505155), SEED(4376384, 4381192),
          SEED(3888323, 1475000), SEED(6613614, 4128935), SEED(194221, 3314437),
          SEED(848274, 938030), SEED(7128952, 3511303), SEED(6783279, 4599849),
          SEED(3692460, 6008310)},
         {SEED(5919390, 7990662), SEED(765900, 3068831), SEED(7755415, 158053),
          SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370), SEED(5399006, 4505155),
          SEED(4376384, 4381192), SEED(3888323, 1475000),
          SEED(6613614, 4128935), SEED(194221, 3314437), SEED(848274, 938030),
          SEED(7128952, 3511303), SEED(6783279, 4599849)},
         {SEED(2605807, 3045488), SEED(5919390, 7990662), SEED(765900, 3068831),
          SEED(7755415, 158053), SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370), SEED(5399006, 4505155),
          SEED(4376384, 4381192), SEED(3888323, 1475000),
          SEED(6613614, 4128935), SEED(194221, 3314437), SEED(848274, 938030),
          SEED(7128952, 3511303)},
         {SEED(2609184, 7992267), SEED(2605807, 3045488),
          SEED(5919390, 7990662), SEED(765900, 3068831), SEED(7755415, 158053),
          SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370), SEED(5399006, 4505155),
          SEED(4376384, 4381192), SEED(3888323, 1475000),
          SEED(6613614, 4128935), SEED(194221, 3314437), SEED(848274, 938030)},
         {SEED(215081, 1096084), SEED(2609184, 7992267), SEED(2605807, 3045488),
          SEED(5919390, 7990662), SEED(765900, 3068831), SEED(7755415, 158053),
          SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370), SEED(5399006, 4505155),
          SEED(4376384, 4381192), SEED(3888323, 1475000),
          SEED(6613614, 4128935), SEED(194221, 3314437)},
         {SEED(960121, 6383268), SEED(215081, 1096084), SEED(2609184, 7992267),
          SEED(2605807, 3045488), SEED(5919390, 7990662), SEED(765900, 3068831),
          SEED(7755415, 158053), SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370), SEED(5399006, 4505155),
          SEED(4376384, 4381192), SEED(3888323, 1475000),
          SEED(6613614, 4128935)},
         {SEED(4144396, 3730990), SEED(960121, 6383268), SEED(215081, 1096084),
          SEED(2609184, 7992267), SEED(2605807, 3045488),
          SEED(5919390, 7990662), SEED(765900, 3068831), SEED(7755415, 158053),
          SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370), SEED(5399006, 4505155),
          SEED(4376384, 4381192), SEED(3888323, 1475000)},
         {SEED(6494130, 4520488), SEED(4144396, 3730990), SEED(960121, 6383268),
          SEED(215081, 1096084), SEED(2609184, 7992267), SEED(2605807, 3045488),
          SEED(5919390, 7990662), SEED(765900, 3068831), SEED(7755415, 158053),
          SEED(3868840, 4480963), SEED(4211136, 6834246),
          SEED(2226930, 1982352), SEED(7666870, 5025456), SEED(379708, 7757779),
          SEED(4114630, 5152370), SEED(5399006, 4505155),
          SEED(4376384, 4381192)}},
        {{SEED(544688, 6485299), SEED(5066986, 4052957), SEED(1184906, 7304212),
          SEED(2303421, 7266530), SEED(6059620, 914511), SEED(1873212, 5753866),
          SEED(7327049, 8320396), SEED(2550757, 1713411),
          SEED(8072649, 4447765), SEED(4610806, 472258), SEED(4022252, 7725527),
          SEED(6687656, 1451695), SEED(3249767, 7721302),
          SEED(7895617, 6410465), SEED(3916272, 3507325), SEED(6311146, 260031),
          SEED(6849756, 540516)},
         {SEED(4520768, 1455028), SEED(544688, 6485299), SEED(5066986, 4052957),
          SEED(1184906, 7304212), SEED(2303421, 7266530), SEED(6059620, 914511),
          SEED(1873212, 5753866), SEED(7327049, 8320396),
          SEED(2550757, 1713411), SEED(8072649, 4447765), SEED(4610806, 472258),
          SEED(4022252, 7725527), SEED(6687656, 1451695),
          SEED(3249767, 7721302), SEED(7895617, 6410465),
          SEED(3916272, 3507325), SEED(6311146, 260031)},
         {SEED(225959, 7526562), SEED(4520768, 1455028), SEED(544688, 6485299),
          SEED(5066986, 4052957), SEED(1184906, 7304212),
          SEED(2303421, 7266530), SEED(6059620, 914511), SEED(1873212, 5753866),
          SEED(7327049, 8320396), SEED(2550757, 1713411),
          SEED(8072649, 4447765), SEED(4610806, 472258), SEED(4022252, 7725527),
          SEED(6687656, 1451695), SEED(3249767, 7721302),
          SEED(7895617, 6410465), SEED(3916272, 3507325)},
         {SEED(5101178, 2422929), SEED(225959, 7526562), SEED(4520768, 1455028),
          SEED(544688, 6485299), SEED(5066986, 4052957), SEED(1184906, 7304212),
          SEED(2303421, 7266530), SEED(6059620, 914511), SEED(1873212, 5753866),
          SEED(7327049, 8320396), SEED(2550757, 1713411),
          SEED(8072649, 4447765), SEED(4610806, 472258), SEED(4022252, 7725527),
          SEED(6687656, 1451695), SEED(3249767, 7721302),
          SEED(7895617, 6410465)},
         {SEED(4573995, 2074815), SEED(5101178, 2422929), SEED(225959, 7526562),
          SEED(4520768, 1455028), SEED(544688, 6485299), SEED(5066986, 4052957),
          SEED(1184906, 7304212), SEED(2303421, 7266530), SEED(6059620, 914511),
          SEED(1873212, 5753866), SEED(7327049, 8320396),
          SEED(2550757, 1713411), SEED(8072649, 4447765), SEED(4610806, 472258),
          SEED(4022252, 7725527), SEED(6687656, 1451695),
          SEED(3249767, 7721302)},
         {SEED(3249767, 7721302), SEED(7895617, 6410465),
          SEED(3916272, 3507325), SEED(6311146, 260031), SEED(6849756, 540516),
          SEED(7060084, 731432), SEED(6128545, 4121168), SEED(7022757, 5590800),
          SEED(2619380, 2818764), SEED(1448814, 442253), SEED(6239568, 6416946),
          SEED(639393, 6868701), SEED(7689598, 2380716), SEED(177032, 6425908),
          SEED(694534, 5353541), SEED(6099714, 7465495), SEED(8226508, 911178)},
         {SEED(6687656, 1451695), SEED(3249767, 7721302),
          SEED(7895617, 6410465), SEED(3916272, 3507325), SEED(6311146, 260031),
          SEED(6849756, 540516), SEED(7060084, 731432), SEED(6128545, 4121168),
          SEED(7022757, 5590800), SEED(2619380, 2818764), SEED(1448814, 442253),
          SEED(6239568, 6416946), SEED(639393, 6868701), SEED(7689598, 2380716),
          SEED(177032, 6425908), SEED(694534, 5353541), SEED(6099714, 7465495)},
         {SEED(4022252, 7725527), SEED(6687656, 1451695),
          SEED(3249767, 7721302), SEED(7895617, 6410465),
          SEED(3916272, 3507325), SEED(6311146, 260031), SEED(6849756, 540516),
          SEED(7060084, 731432), SEED(6128545, 4121168), SEED(7022757, 5590800),
          SEED(2619380, 2818764), SEED(1448814, 442253), SEED(6239568, 6416946),
          SEED(639393, 6868701), SEED(7689598, 2380716), SEED(177032, 6425908),
          SEED(694534, 5353541)},
         {SEED(4610806, 472258), SEED(4022252, 7725527), SEED(6687656, 1451695),
          SEED(3249767, 7721302), SEED(7895617, 6410465),
          SEED(3916272, 3507325), SEED(6311146, 260031), SEED(6849756, 540516),
          SEED(7060084, 731432), SEED(6128545, 4121168), SEED(7022757, 5590800),
          SEED(2619380, 2818764), SEED(1448814, 442253), SEED(6239568, 6416946),
          SEED(639393, 6868701), SEED(7689598, 2380716), SEED(177032, 6425908)},
         {SEED(8072649, 4447765), SEED(4610806, 472258), SEED(4022252, 7725527),
          SEED(6687656, 1451695), SEED(3249767, 7721302),
          SEED(7895617, 6410465), SEED(3916272, 3507325), SEED(6311146, 260031),
          SEED(6849756, 540516), SEED(7060084, 731432), SEED(6128545, 4121168),
          SEED(7022757, 5590800), SEED(2619380, 2818764), SEED(1448814, 442253),
          SEED(6239568, 6416946), SEED(639393, 6868701),
          SEED(7689598, 2380716)},
         {SEED(2550757, 1713411), SEED(8072649, 4447765), SEED(4610806, 472258),
          SEED(4022252, 7725527), SEED(6687656, 1451695),
          SEED(3249767, 7721302), SEED(7895617, 6410465),
          SEED(3916272, 3507325), SEED(6311146, 260031), SEED(6849756, 540516),
          SEED(7060084, 731432), SEED(6128545, 4121168), SEED(7022757, 5590800),
          SEED(2619380, 2818764), SEED(1448814, 442253), SEED(6239568, 6416946),
          SEED(639393, 6868701)},
         {SEED(7327049, 8320396), SEED(2550757, 1713411),
          SEED(8072649, 4447765), SEED(4610806, 472258), SEED(4022252, 7725527),
          SEED(6687656, 1451695), SEED(3249767, 7721302),
          SEED(7895617, 6410465), SEED(3916272, 3507325), SEED(6311146, 260031),
          SEED(6849756, 540516), SEED(7060084, 731432), SEED(6128545, 4121168),
          SEED(7022757, 5590800), SEED(2619380, 2818764), SEED(1448814, 442253),
          SEED(6239568, 6416946)},
         {SEED(1873212, 5753866), SEED(7327049, 8320396),
          SEED(2550757, 1713411), SEED(8072649, 4447765), SEED(4610806, 472258),
          SEED(4022252, 7725527), SEED(6687656, 1451695),
          SEED(3249767, 7721302), SEED(7895617, 6410465),
          SEED(3916272, 3507325), SEED(6311146, 260031), SEED(6849756, 540516),
          SEED(7060084, 731432), SEED(6128545, 4121168), SEED(7022757, 5590800),
          SEED(2619380, 2818764), SEED(1448814, 442253)},
         {SEED(6059620, 914511), SEED(1873212, 5753866), SEED(7327049, 8320396),
          SEED(2550757, 1713411), SEED(8072649, 4447765), SEED(4610806, 472258),
          SEED(4022252, 7725527), SEED(6687656, 1451695),
          SEED(3249767, 7721302), SEED(7895617, 6410465),
          SEED(3916272, 3507325), SEED(6311146, 260031), SEED(6849756, 540516),
          SEED(7060084, 731432), SEED(6128545, 4121168), SEED(7022757, 5590800),
          SEED(2619380, 2818764)},
         {SEED(2303421, 7266530), SEED(6059620, 914511), SEED(1873212, 5753866),
          SEED(7327049, 8320396), SEED(2550757, 1713411),
          SEED(8072649, 4447765), SEED(4610806, 472258), SEED(4022252, 7725527),
          SEED(6687656, 1451695), SEED(3249767, 7721302),
          SEED(7895617, 6410465), SEED(3916272, 3507325), SEED(6311146, 260031),
          SEED(6849756, 540516), SEED(7060084, 731432), SEED(6128545, 4121168),
          SEED(7022757, 5590800)},
         {SEED(1184906, 7304212), SEED(2303421, 7266530), SEED(6059620, 914511),
          SEED(1873212, 5753866), SEED(7327049, 8320396),
          SEED(2550757, 1713411), SEED(8072649, 4447765), SEED(4610806, 472258),
          SEED(4022252, 7725527), SEED(6687656, 1451695),
          SEED(3249767, 7721302), SEED(7895617, 6410465),
          SEED(3916272, 3507325), SEED(6311146, 260031), SEED(6849756, 540516),
          SEED(7060084, 731432), SEED(6128545, 4121168)},
         {SEED(5066986, 4052957), SEED(1184906, 7304212),
          SEED(2303421, 7266530), SEED(6059620, 914511), SEED(1873212, 5753866),
          SEED(7327049, 8320396), SEED(2550757, 1713411),
          SEED(8072649, 4447765), SEED(4610806, 472258), SEED(4022252, 7725527),
          SEED(6687656, 1451695), SEED(3249767, 7721302),
          SEED(7895617, 6410465), SEED(3916272, 3507325), SEED(6311146, 260031),
          SEED(6849756, 540516), SEED(7060084, 731432)}},
        {{SEED(724900, 6167884), SEED(6833752, 6245839), SEED(6675686, 5172319),
          SEED(4651330, 6189344), SEED(7611819, 4985540),
          SEED(2828187, 5991119), SEED(3671207, 4926001),
          SEED(5115336, 6635639), SEED(4535291, 8006873),
          SEED(2586535, 8338866), SEED(3098925, 6282721),
          SEED(1212313, 1136928), SEED(2805736, 5403625),
          SEED(3242152, 7757452), SEED(6621974, 7274716),
          SEED(1030814, 7157480), SEED(7941974, 2479717)},
         {SEED(7165185, 7465258), SEED(724900, 6167884), SEED(6833752, 6245839),
          SEED(6675686, 5172319), SEED(4651330, 6189344),
          SEED(7611819, 4985540), SEED(2828187, 5991119),
          SEED(3671207, 4926001), SEED(5115336, 6635639),
          SEED(4535291, 8006873), SEED(2586535, 8338866),
          SEED(3098925, 6282721), SEED(1212313, 1136928),
          SEED(2805736, 5403625), SEED(3242152, 7757452),
          SEED(6621974, 7274716), SEED(1030814, 7157480)},
         {SEED(5682144, 4958216), SEED(7165185, 7465258), SEED(724900, 6167884),
          SEED(6833752, 6245839), SEED(6675686, 5172319),
          SEED(4651330, 6189344), SEED(7611819, 4985540),
          SEED(2828187, 5991119), SEED(3671207, 4926001),
          SEED(5115336, 6635639), SEED(4535291, 8006873),
          SEED(2586535, 8338866), SEED(3098925, 6282721),
          SEED(1212313, 1136928), SEED(2805736, 5403625),
          SEED(3242152, 7757452), SEED(6621974, 7274716)},
         {SEED(4909052, 4058428), SEED(5682144, 4958216),
          SEED(7165185, 7465258), SEED(724900, 6167884), SEED(6833752, 6245839),
          SEED(6675686, 5172319), SEED(4651330, 6189344),
          SEED(7611819, 4985540), SEED(2828187, 5991119),
          SEED(3671207, 4926001), SEED(5115336, 6635639),
          SEED(4535291, 8006873), SEED(2586535, 8338866),
          SEED(3098925, 6282721), SEED(1212313, 1136928),
          SEED(2805736, 5403625), SEED(3242152, 7757452)},
         {SEED(1687296, 5614684), SEED(4909052, 4058428),
          SEED(5682144, 4958216), SEED(7165185, 7465258), SEED(724900, 6167884),
          SEED(6833752, 6245839), SEED(6675686, 5172319),
          SEED(4651330, 6189344), SEED(7611819, 4985540),
          SEED(2828187, 5991119), SEED(3671207, 4926001),
          SEED(5115336, 6635639), SEED(4535291, 8006873),
          SEED(2586535, 8338866), SEED(3098925, 6282721),
          SEED(1212313, 1136928), SEED(2805736, 5403625)},
         {SEED(2805736, 5403625), SEED(3242152, 7757452),
          SEED(6621974, 7274716), SEED(1030814, 7157480),
          SEED(7941974, 2479717), SEED(6285321, 176764), SEED(3162545, 1319838),
          SEED(1560350, 6925288), SEED(116039, 6571079), SEED(5025284, 5035282),
          SEED(8117870, 8097005), SEED(2458894, 3789073),
          SEED(2309600, 1232014), SEED(1293139, 249421), SEED(4353169, 1064149),
          SEED(2068111, 7513849), SEED(1658947, 7045818)},
         {SEED(1212313, 1136928), SEED(2805736, 5403625),
          SEED(3242152, 7757452), SEED(6621974, 7274716),
          SEED(1030814, 7157480), SEED(7941974, 2479717), SEED(6285321, 176764),
          SEED(3162545, 1319838), SEED(1560350, 6925288), SEED(116039, 6571079),
          SEED(5025284, 5035282), SEED(8117870, 8097005),
          SEED(2458894, 3789073), SEED(2309600, 1232014), SEED(1293139, 249421),
          SEED(4353169, 1064149), SEED(2068111, 7513849)},
         {SEED(3098925, 6282721), SEED(1212313, 1136928),
          SEED(2805736, 5403625), SEED(3242152, 7757452),
          SEED(6621974, 7274716), SEED(1030814, 7157480),
          SEED(7941974, 2479717), SEED(6285321, 176764), SEED(3162545, 1319838),
          SEED(1560350, 6925288), SEED(116039, 6571079), SEED(5025284, 5035282),
          SEED(8117870, 8097005), SEED(2458894, 3789073),
          SEED(2309600, 1232014), SEED(1293139, 249421),
          SEED(4353169, 1064149)},
         {SEED(2586535, 8338866), SEED(3098925, 6282721),
          SEED(1212313, 1136928), SEED(2805736, 5403625),
          SEED(3242152, 7757452), SEED(6621974, 7274716),
          SEED(1030814, 7157480), SEED(7941974, 2479717), SEED(6285321, 176764),
          SEED(3162545, 1319838), SEED(1560350, 6925288), SEED(116039, 6571079),
          SEED(5025284, 5035282), SEED(8117870, 8097005),
          SEED(2458894, 3789073), SEED(2309600, 1232014),
          SEED(1293139, 249421)},
         {SEED(4535291, 8006873), SEED(2586535, 8338866),
          SEED(3098925, 6282721), SEED(1212313, 1136928),
          SEED(2805736, 5403625), SEED(3242152, 7757452),
          SEED(6621974, 7274716), SEED(1030814, 7157480),
          SEED(7941974, 2479717), SEED(6285321, 176764), SEED(3162545, 1319838),
          SEED(1560350, 6925288), SEED(116039, 6571079), SEED(5025284, 5035282),
          SEED(8117870, 8097005), SEED(2458894, 3789073),
          SEED(2309600, 1232014)},
         {SEED(5115336, 6635639), SEED(4535291, 8006873),
          SEED(2586535, 8338866), SEED(3098925, 6282721),
          SEED(1212313, 1136928), SEED(2805736, 5403625),
          SEED(3242152, 7757452), SEED(6621974, 7274716),
          SEED(1030814, 7157480), SEED(7941974, 2479717), SEED(6285321, 176764),
          SEED(3162545, 1319838), SEED(1560350, 6925288), SEED(116039, 6571079),
          SEED(5025284, 5035282), SEED(8117870, 8097005),
          SEED(2458894, 3789073)},
         {SEED(3671207, 4926001), SEED(5115336, 6635639),
          SEED(4535291, 8006873), SEED(2586535, 8338866),
          SEED(3098925, 6282721), SEED(1212313, 1136928),
          SEED(2805736, 5403625), SEED(3242152, 7757452),
          SEED(6621974, 7274716), SEED(1030814, 7157480),
          SEED(7941974, 2479717), SEED(6285321, 176764), SEED(3162545, 1319838),
          SEED(1560350, 6925288), SEED(116039, 6571079), SEED(5025284, 5035282),
          SEED(8117870, 8097005)},
         {SEED(2828187, 5991119), SEED(3671207, 4926001),
          SEED(5115336, 6635639), SEED(4535291, 8006873),
          SEED(2586535, 8338866), SEED(3098925, 6282721),
          SEED(1212313, 1136928), SEED(2805736, 5403625),
          SEED(3242152, 7757452), SEED(6621974, 7274716),
          SEED(1030814, 7157480), SEED(7941974, 2479717), SEED(6285321, 176764),
          SEED(3162545, 1319838), SEED(1560350, 6925288), SEED(116039, 6571079),
          SEED(5025284, 5035282)},
         {SEED(7611819, 4985540), SEED(2828187, 5991119),
          SEED(3671207, 4926001), SEED(5115336, 6635639),
          SEED(4535291, 8006873), SEED(2586535, 8338866),
          SEED(3098925, 6282721), SEED(1212313, 1136928),
          SEED(2805736, 5403625), SEED(3242152, 7757452),
          SEED(6621974, 7274716), SEED(1030814, 7157480),
          SEED(7941974, 2479717), SEED(6285321, 176764), SEED(3162545, 1319838),
          SEED(1560350, 6925288), SEED(116039, 6571079)},
         {SEED(4651330, 6189344), SEED(7611819, 4985540),
          SEED(2828187, 5991119), SEED(3671207, 4926001),
          SEED(5115336, 6635639), SEED(4535291, 8006873),
          SEED(2586535, 8338866), SEED(3098925, 6282721),
          SEED(1212313, 1136928), SEED(2805736, 5403625),
          SEED(3242152, 7757452), SEED(6621974, 7274716),
          SEED(1030814, 7157480), SEED(7941974, 2479717), SEED(6285321, 176764),
          SEED(3162545, 1319838), SEED(1560350, 6925288)},
         {SEED(6675686, 5172319), SEED(4651330, 6189344),
          SEED(7611819, 4985540), SEED(2828187, 5991119),
          SEED(3671207, 4926001), SEED(5115336, 6635639),
          SEED(4535291, 8006873), SEED(2586535, 8338866),
          SEED(3098925, 6282721), SEED(1212313, 1136928),
          SEED(2805736, 5403625), SEED(3242152, 7757452),
          SEED(6621974, 7274716), SEED(1030814, 7157480),
          SEED(7941974, 2479717), SEED(6285321, 176764),
          SEED(3162545, 1319838)},
         {SEED(6833752, 6245839), SEED(6675686, 5172319),
          SEED(4651330, 6189344), SEED(7611819, 4985540),
          SEED(2828187, 5991119), SEED(3671207, 4926001),
          SEED(5115336, 6635639), SEED(4535291, 8006873),
          SEED(2586535, 8338866), SEED(3098925, 6282721),
          SEED(1212313, 1136928), SEED(2805736, 5403625),
          SEED(3242152, 7757452), SEED(6621974, 7274716),
          SEED(1030814, 7157480), SEED(7941974, 2479717),
          SEED(6285321, 176764)}},
    },
    {
        {{SEED(4234192, 5291504), SEED(3093007, 4977074),
          SEED(1948716, 6816489), SEED(45088, 707425), SEED(7245090, 7496495),
          SEED(7799060, 8276075), SEED(8251013, 2983456),
          SEED(6971636, 7103219), SEED(6726684, 1061988),
          SEED(3003096, 6775045), SEED(8230540, 5543718),
          SEED(2464041, 2729620), SEED(424313, 6896652), SEED(1894376, 2830517),
          SEED(1982483, 211362), SEED(4404344, 4988044), SEED(717573, 2511924)},
         {SEED(7962663, 1619811), SEED(4234192, 5291504),
          SEED(3093007, 4977074), SEED(1948716, 6816489), SEED(45088, 707425),
          SEED(7245090, 7496495), SEED(7799060, 8276075),
          SEED(8251013, 2983456), SEED(6971636, 7103219),
          SEED(6726684, 1061988), SEED(3003096, 6775045),
          SEED(8230540, 5543718), SEED(2464041, 2729620), SEED(424313, 6896652),
          SEED(1894376, 2830517), SEED(1982483, 211362),
          SEED(4404344, 4988044)},
         {SEED(4449432, 5695469), SEED(7962663, 1619811),
          SEED(4234192, 5291504), SEED(3093007, 4977074),
          SEED(1948716, 6816489), SEED(45088, 707425), SEED(7245090, 7496495),
          SEED(7799060, 8276075), SEED(8251013, 2983456),
          SEED(6971636, 7103219), SEED(6726684, 1061988),
          SEED(3003096, 6775045), SEED(8230540, 5543718),
          SEED(2464041, 2729620), SEED(424313, 6896652), SEED(1894376, 2830517),
          SEED(1982483, 211362)},
         {SEED(3931199, 7027851), SEED(4449432, 5695469),
          SEED(7962663, 1619811), SEED(4234192, 5291504),
          SEED(3093007, 4977074), SEED(1948716, 6816489), SEED(45088, 707425),
          SEED(7245090, 7496495), SEED(7799060, 8276075),
          SEED(8251013, 2983456), SEED(6971636, 7103219),
          SEED(6726684, 1061988), SEED(3003096, 6775045),
          SEED(8230540, 5543718), SEED(2464041, 2729620), SEED(424313, 6896652),
          SEED(1894376, 2830517)},
         {SEED(4987383, 7807591), SEED(3931199, 7027851),
          SEED(4449432, 5695469), SEED(7962663, 1619811),
          SEED(4234192, 5291504), SEED(3093007, 4977074),
          SEED(1948716, 6816489), SEED(45088, 707425), SEED(7245090, 7496495),
          SEED(7799060, 8276075), SEED(8251013, 2983456),
          SEED(6971636, 7103219), SEED(6726684, 1061988),
          SEED(3003096, 6775045), SEED(8230540, 5543718),
          SEED(2464041, 2729620), SEED(424313, 6896652)},
         {SEED(424313, 6896652), SEED(1894376, 2830517), SEED(1982483, 211362),
          SEED(4404344, 4988044), SEED(717573, 2511924), SEED(4823740, 5404036),
          SEED(3230602, 1993617), SEED(3365688, 8101877),
          SEED(1707012, 8034044), SEED(4241994, 721450), SEED(7957128, 2732356),
          SEED(5786972, 253836), SEED(6547323, 206567), SEED(4832308, 6620079),
          SEED(1020613, 6563683), SEED(3826196, 555674), SEED(1746468, 217696)},
         {SEED(2464041, 2729620), SEED(424313, 6896652), SEED(1894376, 2830517),
          SEED(1982483, 211362), SEED(4404344, 4988044), SEED(717573, 2511924),
          SEED(4823740, 5404036), SEED(3230602, 1993617),
          SEED(3365688, 8101877), SEED(1707012, 8034044), SEED(4241994, 721450),
          SEED(7957128, 2732356), SEED(5786972, 253836), SEED(6547323, 206567),
          SEED(4832308, 6620079), SEED(1020613, 6563683),
          SEED(3826196, 555674)},
         {SEED(8230540, 5543718), SEED(2464041, 2729620), SEED(424313, 6896652),
          SEED(1894376, 2830517), SEED(1982483, 211362), SEED(4404344, 4988044),
          SEED(717573, 2511924), SEED(4823740, 5404036), SEED(3230602, 1993617),
          SEED(3365688, 8101877), SEED(1707012, 8034044), SEED(4241994, 721450),
          SEED(7957128, 2732356), SEED(5786972, 253836), SEED(6547323, 206567),
          SEED(4832308, 6620079), SEED(1020613, 6563683)},
         {SEED(3003096, 6775045), SEED(8230540, 5543718),
          SEED(2464041, 2729620), SEED(424313, 6896652), SEED(1894376, 2830517),
          SEED(1982483, 211362), SEED(4404344, 4988044), SEED(717573, 2511924),
          SEED(4823740, 5404036), SEED(3230602, 1993617),
          SEED(3365688, 8101877), SEED(1707012, 8034044), SEED(4241994, 721450),
          SEED(7957128, 2732356), SEED(5786972, 253836), SEED(6547323, 206567),
          SEED(4832308, 6620079)},
         {SEED(6726684, 1061988), SEED(3003096, 6775045),
          SEED(8230540, 5543718), SEED(2464041, 2729620), SEED(424313, 6896652),
          SEED(1894376, 2830517), SEED(1982483, 211362), SEED(4404344, 4988044),
          SEED(717573, 2511924), SEED(4823740, 5404036), SEED(3230602, 1993617),
          SEED(3365688, 8101877), SEED(1707012, 8034044), SEED(4241994, 721450),
          SEED(7957128, 2732356), SEED(5786972, 253836), SEED(6547323, 206567)},
         {SEED(6971636, 7103219), SEED(6726684, 1061988),
          SEED(3003096, 6775045), SEED(8230540, 5543718),
          SEED(2464041, 2729620), SEED(424313, 6896652), SEED(1894376, 2830517),
          SEED(1982483, 211362), SEED(4404344, 4988044), SEED(717573, 2511924),
          SEED(4823740, 5404036), SEED(3230602, 1993617),
          SEED(3365688, 8101877), SEED(1707012, 8034044), SEED(4241994, 721450),
          SEED(7957128, 2732356), SEED(5786972, 253836)},
         {SEED(8251013, 2983456), SEED(6971636, 7103219),
          SEED(6726684, 1061988), SEED(3003096, 6775045),
          SEED(8230540, 5543718), SEED(2464041, 2729620), SEED(424313, 6896652),
          SEED(1894376, 2830517), SEED(1982483, 211362), SEED(4404344, 4988044),
          SEED(717573, 2511924), SEED(4823740, 5404036), SEED(3230602, 1993617),
          SEED(3365688, 8101877), SEED(1707012, 8034044), SEED(4241994, 721450),
          SEED(7957128, 2732356)},
         {SEED(7799060, 8276075), SEED(8251013, 2983456),
          SEED(6971636, 7103219), SEED(6726684, 1061988),
          SEED(3003096, 6775045), SEED(8230540, 5543718),
          SEED(2464041, 2729620), SEED(424313, 6896652), SEED(1894376, 2830517),
          SEED(1982483, 211362), SEED(4404344, 4988044), SEED(717573, 2511924),
          SEED(4823740, 5404036), SEED(3230602, 1993617),
          SEED(3365688, 8101877), SEED(1707012, 8034044),
          SEED(4241994, 721450)},
         {SEED(7245090, 7496495), SEED(7799060, 8276075),
          SEED(8251013, 2983456), SEED(6971636, 7103219),
          SEED(6726684, 1061988), SEED(3003096, 6775045),
          SEED(8230540, 5543718), SEED(2464041, 2729620), SEED(424313, 6896652),
          SEED(1894376, 2830517), SEED(1982483, 211362), SEED(4404344, 4988044),
          SEED(717573, 2511924), SEED(4823740, 5404036), SEED(3230602, 1993617),
          SEED(3365688, 8101877), SEED(1707012, 8034044)},
         {SEED(45088, 707425), SEED(7245090, 7496495), SEED(7799060, 8276075),
          SEED(8251013, 2983456), SEED(6971636, 7103219),
          SEED(6726684, 1061988), SEED(3003096, 6775045),
          SEED(8230540, 5543718), SEED(2464041, 2729620), SEED(424313, 6896652),
          SEED(1894376, 2830517), SEED(1982483, 211362), SEED(4404344, 4988044),
          SEED(717573, 2511924), SEED(4823740, 5404036), SEED(3230602, 1993617),
          SEED(3365688, 8101877)},
         {SEED(1948716, 6816489), SEED(45088, 707425), SEED(7245090, 7496495),
          SEED(7799060, 8276075), SEED(8251013, 2983456),
          SEED(6971636, 7103219), SEED(6726684, 1061988),
          SEED(3003096, 6775045), SEED(8230540, 5543718),
          SEED(2464041, 2729620), SEED(424313, 6896652), SEED(1894376, 2830517),
          SEED(1982483, 211362), SEED(4404344, 4988044), SEED(717573, 2511924),
          SEED(4823740, 5404036), SEED(3230602, 1993617)},
         {SEED(3093007, 4977074), SEED(1948716, 6816489), SEED(45088, 707425),
          SEED(7245090, 7496495), SEED(7799060, 8276075),
          SEED(8251013, 2983456), SEED(6971636, 7103219),
          SEED(6726684, 1061988), SEED(3003096, 6775045),
          SEED(8230540, 5543718), SEED(2464041, 2729620), SEED(424313, 6896652),
          SEED(1894376, 2830517), SEED(1982483, 211362), SEED(4404344, 4988044),
          SEED(717573, 2511924), SEED(4823740, 5404036)}},
        {{SEED(7766996, 1666410), SEED(6469590, 2730769), SEED(7653865, 782167),
          SEED(1463798, 518266), SEED(3401224, 6004658), SEED(5813359, 4360062),
          SEED(6002618, 7861773), SEED(7207966, 3736237),
          SEED(2171010, 6216965), SEED(947054, 1691224), SEED(4656670, 4491057),
          SEED(5064915, 3056972), SEED(1664885, 1282482),
          SEED(2033034, 1154484), SEED(3960648, 6783965),
          SEED(4266225, 2460753), SEED(8054804, 2773122)},
         {SEED(3067420, 389173), SEED(7766996, 1666410), SEED(6469590, 2730769),
          SEED(7653865, 782167), SEED(1463798, 518266), SEED(3401224, 6004658),
          SEED(5813359, 4360062), SEED(6002618, 7861773),
          SEED(7207966, 3736237), SEED(2171010, 6216965), SEED(947054, 1691224),
          SEED(4656670, 4491057), SEED(5064915, 3056972),
          SEED(1664885, 1282482), SEED(2033034, 1154484),
          SEED(3960648, 6783965), SEED(4266225, 2460753)},
         {SEED(5730023, 2979019), SEED(3067420, 389173), SEED(7766996, 1666410),
          SEED(6469590, 2730769), SEED(7653865, 782167), SEED(1463798, 518266),
          SEED(3401224, 6004658), SEED(5813359, 4360062),
          SEED(6002618, 7861773), SEED(7207966, 3736237),
          SEED(2171010, 6216965), SEED(947054, 1691224), SEED(4656670, 4491057),
          SEED(5064915, 3056972), SEED(1664885, 1282482),
          SEED(2033034, 1154484), SEED(3960648, 6783965)},
         {SEED(3225905, 7566133), SEED(5730023, 2979019), SEED(3067420, 389173),
          SEED(7766996, 1666410), SEED(6469590, 2730769), SEED(7653865, 782167),
          SEED(1463798, 518266), SEED(3401224, 6004658), SEED(5813359, 4360062),
          SEED(6002618, 7861773), SEED(7207966, 3736237),
          SEED(2171010, 6216965), SEED(947054, 1691224), SEED(4656670, 4491057),
          SEED(5064915, 3056972), SEED(1664885, 1282482),
          SEED(2033034, 1154484)},
         {SEED(114016, 3885254), SEED(3225905, 7566133), SEED(5730023, 2979019),
          SEED(3067420, 389173), SEED(7766996, 1666410), SEED(6469590, 2730769),
          SEED(7653865, 782167), SEED(1463798, 518266), SEED(3401224, 6004658),
          SEED(5813359, 4360062), SEED(6002618, 7861773),
          SEED(7207966, 3736237), SEED(2171010, 6216965), SEED(947054, 1691224),
          SEED(4656670, 4491057), SEED(5064915, 3056972),
          SEED(1664885, 1282482)},
         {SEED(1664885, 1282482), SEED(2033034, 1154484),
          SEED(3960648, 6783965), SEED(4266225, 2460753),
          SEED(8054804, 2773122), SEED(1953637, 5694956), SEED(466972, 3257604),
          SEED(445899, 5434538), SEED(7681396, 2689908), SEED(2454170, 4313434),
          SEED(1156689, 8257613), SEED(937703, 4804801), SEED(5543081, 2453755),
          SEED(137976, 5062481), SEED(5375014, 3295866), SEED(390445, 2030304),
          SEED(5398719, 283849)},
         {SEED(5064915, 3056972), SEED(1664885, 1282482),
          SEED(2033034, 1154484), SEED(3960648, 6783965),
          SEED(4266225, 2460753), SEED(8054804, 2773122),
          SEED(1953637, 5694956), SEED(466972, 3257604), SEED(445899, 5434538),
          SEED(7681396, 2689908), SEED(2454170, 4313434),
          SEED(1156689, 8257613), SEED(937703, 4804801), SEED(5543081, 2453755),
          SEED(137976, 5062481), SEED(5375014, 3295866), SEED(390445, 2030304)},
         {SEED(4656670, 4491057), SEED(5064915, 3056972),
          SEED(1664885, 1282482), SEED(2033034, 1154484),
          SEED(3960648, 6783965), SEED(4266225, 2460753),
          SEED(8054804, 2773122), SEED(1953637, 5694956), SEED(466972, 3257604),
          SEED(445899, 5434538), SEED(7681396, 2689908), SEED(2454170, 4313434),
          SEED(1156689, 8257613), SEED(937703, 4804801), SEED(5543081, 2453755),
          SEED(137976, 5062481), SEED(5375014, 3295866)},
         {SEED(947054, 1691224), SEED(4656670, 4491057), SEED(5064915, 3056972),
          SEED(1664885, 1282482), SEED(2033034, 1154484),
          SEED(3960648, 6783965), SEED(4266225, 2460753),
          SEED(8054804, 2773122), SEED(1953637, 5694956), SEED(466972, 3257604),
          SEED(445899, 5434538), SEED(7681396, 2689908), SEED(2454170, 4313434),
          SEED(1156689, 8257613), SEED(937703, 4804801), SEED(5543081, 2453755),
          SEED(137976, 5062481)},
         {SEED(2171010, 6216965), SEED(947054, 1691224), SEED(4656670, 4491057),
          SEED(5064915, 3056972), SEED(1664885, 1282482),
          SEED(2033034, 1154484), SEED(3960648, 6783965),
          SEED(4266225, 2460753), SEED(8054804, 2773122),
          SEED(1953637, 5694956), SEED(466972, 3257604), SEED(445899, 5434538),
          SEED(7681396, 2689908), SEED(2454170, 4313434),
          SEED(1156689, 8257613), SEED(937703, 4804801),
          SEED(5543081, 2453755)},
         {SEED(7207966, 3736237), SEED(2171010, 6216965), SEED(947054, 1691224),
          SEED(4656670, 4491057), SEED(5064915, 3056972),
          SEED(1664885, 1282482), SEED(2033034, 1154484),
          SEED(3960648, 6783965), SEED(4266225, 2460753),
          SEED(8054804, 2773122), SEED(1953637, 5694956), SEED(466972, 3257604),
          SEED(445899, 5434538), SEED(7681396, 2689908), SEED(2454170, 4313434),
          SEED(1156689, 8257613), SEED(937703, 4804801)},
         {SEED(6002618, 7861773), SEED(7207966, 3736237),
          SEED(2171010, 6216965), SEED(947054, 1691224), SEED(4656670, 4491057),
          SEED(5064915, 3056972), SEED(1664885, 1282482),
          SEED(2033034, 1154484), SEED(3960648, 6783965),
          SEED(4266225, 2460753), SEED(8054804, 2773122),
          SEED(1953637, 5694956), SEED(466972, 3257604), SEED(445899, 5434538),
          SEED(7681396, 2689908), SEED(2454170, 4313434),
          SEED(1156689, 8257613)},
         {SEED(5813359, 4360062), SEED(6002618, 7861773),
          SEED(7207966, 3736237), SEED(2171010, 6216965), SEED(947054, 1691224),
          SEED(4656670, 4491057), SEED(5064915, 3056972),
          SEED(1664885, 1282482), SEED(2033034, 1154484),
          SEED(3960648, 6783965), SEED(4266225, 2460753),
          SEED(8054804, 2773122), SEED(1953637, 5694956), SEED(466972, 3257604),
          SEED(445899, 5434538), SEED(7681396, 2689908),
          SEED(2454170, 4313434)},
         {SEED(3401224, 6004658), SEED(5813359, 4360062),
          SEED(6002618, 7861773), SEED(7207966, 3736237),
          SEED(2171010, 6216965), SEED(947054, 1691224), SEED(4656670, 4491057),
          SEED(5064915, 3056972), SEED(1664885, 1282482),
          SEED(2033034, 1154484), SEED(3960648, 6783965),
          SEED(4266225, 2460753), SEED(8054804, 2773122),
          SEED(1953637, 5694956), SEED(466972, 3257604), SEED(445899, 5434538),
          SEED(7681396, 2689908)},
         {SEED(1463798, 518266), SEED(3401224, 6004658), SEED(5813359, 4360062),
          SEED(6002618, 7861773), SEED(7207966, 3736237),
          SEED(2171010, 6216965), SEED(947054, 1691224), SEED(4656670, 4491057),
          SEED(5064915, 3056972), SEED(1664885, 1282482),
          SEED(2033034, 1154484), SEED(3960648, 6783965),
          SEED(4266225, 2460753), SEED(8054804, 2773122),
          SEED(1953637, 5694956), SEED(466972, 3257604), SEED(445899, 5434538)},
         {SEED(7653865, 782167), SEED(1463798, 518266), SEED(3401224, 6004658),
          SEED(5813359, 4360062), SEED(6002618, 7861773),
          SEED(7207966, 3736237), SEED(2171010, 6216965), SEED(947054, 1691224),
          SEED(4656670, 4491057), SEED(5064915, 3056972),
          SEED(1664885, 1282482), SEED(2033034, 1154484),
          SEED(3960648, 6783965), SEED(4266225, 2460753),
          SEED(8054804, 2773122), SEED(1953637, 5694956),
          SEED(466972, 3257604)},
         {SEED(6469590, 2730769), SEED(7653865, 782167), SEED(1463798, 518266),
          SEED(3401224, 6004658), SEED(5813359, 4360062),
          SEED(6002618, 7861773), SEED(7207966, 3736237),
          SEED(2171010, 6216965), SEED(947054, 1691224), SEED(4656670, 4491057),
          SEED(5064915, 3056972), SEED(1664885, 1282482),
          SEED(2033034, 1154484), SEED(3960648, 6783965),
          SEED(4266225, 2460753), SEED(8054804, 2773122),
          SEED(1953637, 5694956)}},
        {{SEED(5657116, 2106534), SEED(4986656, 8040721),
          SEED(8056629, 5026338), SEED(6749381, 6897410),
          SEED(6817192, 4734893), SEED(5621189, 4924579),
          SEED(4192344, 3562257), SEED(7087245, 6650928),
          SEED(7163618, 1984687), SEED(7904258, 934282), SEED(2953254, 1920838),
          SEED(2487009, 4538130), SEED(618862, 589849), SEED(4600147, 5880874),
          SEED(8219969, 1223223), SEED(1209086, 654272),
          SEED(5994023, 2420061)},
         {SEED(4422607, 7154955), SEED(5657116, 2106534),
          SEED(4986656, 8040721), SEED(8056629, 5026338),
          SEED(6749381, 6897410), SEED(6817192, 4734893),
          SEED(5621189, 4924579), SEED(4192344, 3562257),
          SEED(7087245, 6650928), SEED(7163618, 1984687), SEED(7904258, 934282),
          SEED(2953254, 1920838), SEED(2487009, 4538130), SEED(618862, 589849),
          SEED(4600147, 5880874), SEED(8219969, 1223223),
          SEED(1209086, 654272)},
         {SEED(7958467, 7551682), SEED(4422607, 7154955),
          SEED(5657116, 2106534), SEED(4986656, 8040721),
          SEED(8056629, 5026338), SEED(6749381, 6897410),
          SEED(6817192, 4734893), SEED(5621189, 4924579),
          SEED(4192344, 3562257), SEED(7087245, 6650928),
          SEED(7163618, 1984687), SEED(7904258, 934282), SEED(2953254, 1920838),
          SEED(2487009, 4538130), SEED(618862, 589849), SEED(4600147, 5880874),
          SEED(8219969, 1223223)},
         {SEED(7887990, 6249562), SEED(7958467, 7551682),
          SEED(4422607, 7154955), SEED(5657116, 2106534),
          SEED(4986656, 8040721), SEED(8056629, 5026338),
          SEED(6749381, 6897410), SEED(6817192, 4734893),
          SEED(5621189, 4924579), SEED(4192344, 3562257),
          SEED(7087245, 6650928), SEED(7163618, 1984687), SEED(7904258, 934282),
          SEED(2953254, 1920838), SEED(2487009, 4538130), SEED(618862, 589849),
          SEED(4600147, 5880874)},
         {SEED(1198195, 5532988), SEED(7887990, 6249562),
          SEED(7958467, 7551682), SEED(4422607, 7154955),
          SEED(5657116, 2106534), SEED(4986656, 8040721),
          SEED(8056629, 5026338), SEED(6749381, 6897410),
          SEED(6817192, 4734893), SEED(5621189, 4924579),
          SEED(4192344, 3562257), SEED(7087245, 6650928),
          SEED(7163618, 1984687), SEED(7904258, 934282), SEED(2953254, 1920838),
          SEED(2487009, 4538130), SEED(618862, 589849)},
         {SEED(618862, 589849), SEED(4600147, 5880874), SEED(8219969, 1223223),
          SEED(1209086, 654272), SEED(5994023, 2420061), SEED(35927, 5570563),
          SEED(794312, 4478464), SEED(969384, 6764018), SEED(7974371, 4912722),
          SEED(7301542, 3800610), SEED(2667935, 3003741),
          SEED(1705335, 7412735), SEED(6468383, 6061079),
          SEED(2563471, 4492421), SEED(8072897, 8099666),
          SEED(1744168, 1266566), SEED(4881594, 2118068)},
         {SEED(2487009, 4538130), SEED(618862, 589849), SEED(4600147, 5880874),
          SEED(8219969, 1223223), SEED(1209086, 654272), SEED(5994023, 2420061),
          SEED(35927, 5570563), SEED(794312, 4478464), SEED(969384, 6764018),
          SEED(7974371, 4912722), SEED(7301542, 3800610),
          SEED(2667935, 3003741), SEED(1705335, 7412735),
          SEED(6468383, 6061079), SEED(2563471, 4492421),
          SEED(8072897, 8099666), SEED(1744168, 1266566)},
         {SEED(2953254, 1920838), SEED(2487009, 4538130), SEED(618862, 589849),
          SEED(4600147, 5880874), SEED(8219969, 1223223), SEED(1209086, 654272),
          SEED(5994023, 2420061), SEED(35927, 5570563), SEED(794312, 4478464),
          SEED(969384, 6764018), SEED(7974371, 4912722), SEED(7301542, 3800610),
          SEED(2667935, 3003741), SEED(1705335, 7412735),
          SEED(6468383, 6061079), SEED(2563471, 4492421),
          SEED(8072897, 8099666)},
         {SEED(7904258, 934282), SEED(2953254, 1920838), SEED(2487009, 4538130),
          SEED(618862, 589849), SEED(4600147, 5880874), SEED(8219969, 1223223),
          SEED(1209086, 654272), SEED(5994023, 2420061), SEED(35927, 5570563),
          SEED(794312, 4478464), SEED(969384, 6764018), SEED(7974371, 4912722),
          SEED(7301542, 3800610), SEED(2667935, 3003741),
          SEED(1705335, 7412735), SEED(6468383, 6061079),
          SEED(2563471, 4492421)},
         {SEED(7163618, 1984687), SEED(7904258, 934282), SEED(2953254, 1920838),
          SEED(2487009, 4538130), SEED(618862, 589849), SEED(4600147, 5880874),
          SEED(8219969, 1223223), SEED(1209086, 654272), SEED(5994023, 2420061),
          SEED(35927, 5570563), SEED(794312, 4478464), SEED(969384, 6764018),
          SEED(7974371, 4912722), SEED(7301542, 3800610),
          SEED(2667935, 3003741), SEED(1705335, 7412735),
          SEED(6468383, 6061079)},
         {SEED(7087245, 6650928), SEED(7163618, 1984687), SEED(7904258, 934282),
          SEED(2953254, 1920838), SEED(2487009, 4538130), SEED(618862, 589849),
          SEED(4600147, 5880874), SEED(8219969, 1223223), SEED(1209086, 654272),
          SEED(5994023, 2420061), SEED(35927, 5570563), SEED(794312, 4478464),
          SEED(969384, 6764018), SEED(7974371, 4912722), SEED(7301542, 3800610),
          SEED(2667935, 3003741), SEED(1705335, 7412735)},
         {SEED(4192344, 3562257), SEED(7087245, 6650928),
          SEED(7163618, 1984687), SEED(7904258, 934282), SEED(2953254, 1920838),
          SEED(2487009, 4538130), SEED(618862, 589849), SEED(4600147, 5880874),
          SEED(8219969, 1223223), SEED(1209086, 654272), SEED(5994023, 2420061),
          SEED(35927, 5570563), SEED(794312, 4478464), SEED(969384, 6764018),
          SEED(7974371, 4912722), SEED(7301542, 3800610),
          SEED(2667935, 3003741)},
         {SEED(5621189, 4924579), SEED(4192344, 3562257),
          SEED(7087245, 6650928), SEED(7163618, 1984687), SEED(7904258, 934282),
          SEED(2953254, 1920838), SEED(2487009, 4538130), SEED(618862, 589849),
          SEED(4600147, 5880874), SEED(8219969, 1223223), SEED(1209086, 654272),
          SEED(5994023, 2420061), SEED(35927, 5570563), SEED(794312, 4478464),
          SEED(969384, 6764018), SEED(7974371, 4912722),
          SEED(7301542, 3800610)},
         {SEED(6817192, 4734893), SEED(5621189, 4924579),
          SEED(4192344, 3562257), SEED(7087245, 6650928),
          SEED(7163618, 1984687), SEED(7904258, 934282), SEED(2953254, 1920838),
          SEED(2487009, 4538130), SEED(618862, 589849), SEED(4600147, 5880874),
          SEED(8219969, 1223223), SEED(1209086, 654272), SEED(5994023, 2420061),
          SEED(35927, 5570563), SEED(794312, 4478464), SEED(969384, 6764018),
          SEED(7974371, 4912722)},
         {SEED(6749381, 6897410), SEED(6817192, 4734893),
          SEED(5621189, 4924579), SEED(4192344, 3562257),
          SEED(7087245, 6650928), SEED(7163618, 1984687), SEED(7904258, 934282),
          SEED(2953254, 1920838), SEED(2487009, 4538130), SEED(618862, 589849),
          SEED(4600147, 5880874), SEED(8219969, 1223223), SEED(1209086, 654272),
          SEED(5994023, 2420061), SEED(35927, 5570563), SEED(794312, 4478464),
          SEED(969384, 6764018)},
         {SEED(8056629, 5026338), SEED(6749381, 6897410),
          SEED(6817192, 4734893), SEED(5621189, 4924579),
          SEED(4192344, 3562257), SEED(7087245, 6650928),
          SEED(7163618, 1984687), SEED(7904258, 934282), SEED(2953254, 1920838),
          SEED(2487009, 4538130), SEED(618862, 589849), SEED(4600147, 5880874),
          SEED(8219969, 1223223), SEED(1209086, 654272), SEED(5994023, 2420061),
          SEED(35927, 5570563), SEED(794312, 4478464)},
         {SEED(4986656, 8040721), SEED(8056629, 5026338),
          SEED(6749381, 6897410), SEED(6817192, 4734893),
          SEED(5621189, 4924579), SEED(4192344, 3562257),
          SEED(7087245, 6650928), SEED(7163618, 1984687), SEED(7904258, 934282),
          SEED(2953254, 1920838), SEED(2487009, 4538130), SEED(618862, 589849),
          SEED(4600147, 5880874), SEED(8219969, 1223223), SEED(1209086, 654272),
          SEED(5994023, 2420061), SEED(35927, 5570563)}},
    },
    {
        {{SEED(3275440, 562513), SEED(3393624, 6041550), SEED(4772900, 8357991),
          SEED(7967774, 5396489), SEED(285997, 2885701), SEED(4343676, 7955226),
          SEED(253398, 2189612), SEED(4913711, 4251767), SEED(8082380, 3397657),
          SEED(2817254, 6225361), SEED(1568819, 4818124), SEED(5597406, 72820),
          SEED(4556326, 1763960), SEED(5087839, 5182200),
          SEED(4171606, 5042616), SEED(7192842, 6702010),
          SEED(1036142, 5235463)},
         {SEED(1322139, 8121164), SEED(3275440, 562513), SEED(3393624, 6041550),
          SEED(4772900, 8357991), SEED(7967774, 5396489), SEED(285997, 2885701),
          SEED(4343676, 7955226), SEED(253398, 2189612), SEED(4913711, 4251767),
          SEED(8082380, 3397657), SEED(2817254, 6225361),
          SEED(1568819, 4818124), SEED(5597406, 72820), SEED(4556326, 1763960),
          SEED(5087839, 5182200), SEED(4171606, 5042616),
          SEED(7192842, 6702010)},
         {SEED(6772008, 3709892), SEED(1322139, 8121164), SEED(3275440, 562513),
          SEED(3393624, 6041550), SEED(4772900, 8357991),
          SEED(7967774, 5396489), SEED(285997, 2885701), SEED(4343676, 7955226),
          SEED(253398, 2189612), SEED(4913711, 4251767), SEED(8082380, 3397657),
          SEED(2817254, 6225361), SEED(1568819, 4818124), SEED(5597406, 72820),
          SEED(4556326, 1763960), SEED(5087839, 5182200),
          SEED(4171606, 5042616)},
         {SEED(555898, 5012000), SEED(6772008, 3709892), SEED(1322139, 8121164),
          SEED(3275440, 562513), SEED(3393624, 6041550), SEED(4772900, 8357991),
          SEED(7967774, 5396489), SEED(285997, 2885701), SEED(4343676, 7955226),
          SEED(253398, 2189612), SEED(4913711, 4251767), SEED(8082380, 3397657),
          SEED(2817254, 6225361), SEED(1568819, 4818124), SEED(5597406, 72820),
          SEED(4556326, 1763960), SEED(5087839, 5182200)},
         {SEED(92855, 2835143), SEED(555898, 5012000), SEED(6772008, 3709892),
          SEED(1322139, 8121164), SEED(3275440, 562513), SEED(3393624, 6041550),
          SEED(4772900, 8357991), SEED(7967774, 5396489), SEED(285997, 2885701),
          SEED(4343676, 7955226), SEED(253398, 2189612), SEED(4913711, 4251767),
          SEED(8082380, 3397657), SEED(2817254, 6225361),
          SEED(1568819, 4818124), SEED(5597406, 72820), SEED(4556326, 1763960)},
         {SEED(4556326, 1763960), SEED(5087839, 5182200),
          SEED(4171606, 5042616), SEED(7192842, 6702010),
          SEED(1036142, 5235463), SEED(7320372, 995894), SEED(3140226, 3851938),
          SEED(8247797, 4106223), SEED(8274002, 1998831),
          SEED(5857351, 5048947), SEED(2774857, 3137102),
          SEED(3044600, 2116791), SEED(357385, 2487807), SEED(2994541, 6604065),
          SEED(7034256, 1182744), SEED(2764585, 6504721),
          SEED(4561264, 3225965)},
         {SEED(5597406, 72820), SEED(4556326, 1763960), SEED(5087839, 5182200),
          SEED(4171606, 5042616), SEED(7192842, 6702010),
          SEED(1036142, 5235463), SEED(7320372, 995894), SEED(3140226, 3851938),
          SEED(8247797, 4106223), SEED(8274002, 1998831),
          SEED(5857351, 5048947), SEED(2774857, 3137102),
          SEED(3044600, 2116791), SEED(357385, 2487807), SEED(2994541, 6604065),
          SEED(7034256, 1182744), SEED(2764585, 6504721)},
         {SEED(1568819, 4818124), SEED(5597406, 72820), SEED(4556326, 1763960),
          SEED(5087839, 5182200), SEED(4171606, 5042616),
          SEED(7192842, 6702010), SEED(1036142, 5235463), SEED(7320372, 995894),
          SEED(3140226, 3851938), SEED(8247797, 4106223),
          SEED(8274002, 1998831), SEED(5857351, 5048947),
          SEED(2774857, 3137102), SEED(3044600, 2116791), SEED(357385, 2487807),
          SEED(2994541, 6604065), SEED(7034256, 1182744)},
         {SEED(2817254, 6225361), SEED(1568819, 4818124), SEED(5597406, 72820),
          SEED(4556326, 1763960), SEED(5087839, 5182200),
          SEED(4171606, 5042616), SEED(7192842, 6702010),
          SEED(1036142, 5235463), SEED(7320372, 995894), SEED(3140226, 3851938),
          SEED(8247797, 4106223), SEED(8274002, 1998831),
          SEED(5857351, 5048947), SEED(2774857, 3137102),
          SEED(3044600, 2116791), SEED(357385, 2487807),
          SEED(2994541, 6604065)},
         {SEED(8082380, 3397657), SEED(2817254, 6225361),
          SEED(1568819, 4818124), SEED(5597406, 72820), SEED(4556326, 1763960),
          SEED(5087839, 5182200), SEED(4171606, 5042616),
          SEED(7192842, 6702010), SEED(1036142, 5235463), SEED(7320372, 995894),
          SEED(3140226, 3851938), SEED(8247797, 4106223),
          SEED(8274002, 1998831), SEED(5857351, 5048947),
          SEED(2774857, 3137102), SEED(3044600, 2116791),
          SEED(357385, 2487807)},
         {SEED(4913711, 4251767), SEED(8082380, 3397657),
          SEED(2817254, 6225361), SEED(1568819, 4818124), SEED(5597406, 72820),
          SEED(4556326, 1763960), SEED(5087839, 5182200),
          SEED(4171606, 5042616), SEED(7192842, 6702010),
          SEED(1036142, 5235463), SEED(7320372, 995894), SEED(3140226, 3851938),
          SEED(8247797, 4106223), SEED(8274002, 1998831),
          SEED(5857351, 5048947), SEED(2774857, 3137102),
          SEED(3044600, 2116791)},
         {SEED(253398, 2189612), SEED(4913711, 4251767), SEED(8082380, 3397657),
          SEED(2817254, 6225361), SEED(1568819, 4818124), SEED(5597406, 72820),
          SEED(4556326, 1763960), SEED(5087839, 5182200),
          SEED(4171606, 5042616), SEED(7192842, 6702010),
          SEED(1036142, 5235463), SEED(7320372, 995894), SEED(3140226, 3851938),
          SEED(8247797, 4106223), SEED(8274002, 1998831),
          SEED(5857351, 5048947), SEED(2774857, 3137102)},
         {SEED(4343676, 7955226), SEED(253398, 2189612), SEED(4913711, 4251767),
          SEED(8082380, 3397657), SEED(2817254, 6225361),
          SEED(1568819, 4818124), SEED(5597406, 72820), SEED(4556326, 1763960),
          SEED(5087839, 5182200), SEED(4171606, 5042616),
          SEED(7192842, 6702010), SEED(1036142, 5235463), SEED(7320372, 995894),
          SEED(3140226, 3851938), SEED(8247797, 4106223),
          SEED(8274002, 1998831), SEED(5857351, 5048947)},
         {SEED(285997, 2885701), SEED(4343676, 7955226), SEED(253398, 2189612),
          SEED(4913711, 4251767), SEED(8082380, 3397657),
          SEED(2817254, 6225361), SEED(1568819, 4818124), SEED(5597406, 72820),
          SEED(4556326, 1763960), SEED(5087839, 5182200),
          SEED(4171606, 5042616), SEED(7192842, 6702010),
          SEED(1036142, 5235463), SEED(7320372, 995894), SEED(3140226, 3851938),
          SEED(8247797, 4106223), SEED(8274002, 1998831)},
         {SEED(7967774, 5396489), SEED(285997, 2885701), SEED(4343676, 7955226),
          SEED(253398, 2189612), SEED(4913711, 4251767), SEED(8082380, 3397657),
          SEED(2817254, 6225361), SEED(1568819, 4818124), SEED(5597406, 72820),
          SEED(4556326, 1763960), SEED(5087839, 5182200),
          SEED(4171606, 5042616), SEED(7192842, 6702010),
          SEED(1036142, 5235463), SEED(7320372, 995894), SEED(3140226, 3851938),
          SEED(8247797, 4106223)},
         {SEED(4772900, 8357991), SEED(7967774, 5396489), SEED(285997, 2885701),
          SEED(4343676, 7955226), SEED(253398, 2189612), SEED(4913711, 4251767),
          SEED(8082380, 3397657), SEED(2817254, 6225361),
          SEED(1568819, 4818124), SEED(5597406, 72820), SEED(4556326, 1763960),
          SEED(5087839, 5182200), SEED(4171606, 5042616),
          SEED(7192842, 6702010), SEED(1036142, 5235463), SEED(7320372, 995894),
          SEED(3140226, 3851938)},
         {SEED(3393624, 6041550), SEED(4772900, 8357991),
          SEED(7967774, 5396489), SEED(285997, 2885701), SEED(4343676, 7955226),
          SEED(253398, 2189612), SEED(4913711, 4251767), SEED(8082380, 3397657),
          SEED(2817254, 6225361), SEED(1568819, 4818124), SEED(5597406, 72820),
          SEED(4556326, 1763960), SEED(5087839, 5182200),
          SEED(4171606, 5042616), SEED(7192842, 6702010),
          SEED(1036142, 5235463), SEED(7320372, 995894)}},
        {{SEED(1029024, 5515476), SEED(4123568, 1563359),
          SEED(6311204, 5853253), SEED(7372432, 2853663),
          SEED(1966736, 6638717), SEED(6054608, 7763155),
          SEED(8337658, 7423456), SEED(7039668, 545052), SEED(2276037, 5522571),
          SEED(6249708, 5677094), SEED(6086828, 29219), SEED(3507270, 451810),
          SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142),
          SEED(6637640, 3675296)},
         {SEED(215768, 1925406), SEED(1029024, 5515476), SEED(4123568, 1563359),
          SEED(6311204, 5853253), SEED(7372432, 2853663),
          SEED(1966736, 6638717), SEED(6054608, 7763155),
          SEED(8337658, 7423456), SEED(7039668, 545052), SEED(2276037, 5522571),
          SEED(6249708, 5677094), SEED(6086828, 29219), SEED(3507270, 451810),
          SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142)},
         {SEED(4655933, 3825806), SEED(215768, 1925406), SEED(1029024, 5515476),
          SEED(4123568, 1563359), SEED(6311204, 5853253),
          SEED(7372432, 2853663), SEED(1966736, 6638717),
          SEED(6054608, 7763155), SEED(8337658, 7423456), SEED(7039668, 545052),
          SEED(2276037, 5522571), SEED(6249708, 5677094), SEED(6086828, 29219),
          SEED(3507270, 451810), SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861)},
         {SEED(1230055, 7720115), SEED(4655933, 3825806), SEED(215768, 1925406),
          SEED(1029024, 5515476), SEED(4123568, 1563359),
          SEED(6311204, 5853253), SEED(7372432, 2853663),
          SEED(1966736, 6638717), SEED(6054608, 7763155),
          SEED(8337658, 7423456), SEED(7039668, 545052), SEED(2276037, 5522571),
          SEED(6249708, 5677094), SEED(6086828, 29219), SEED(3507270, 451810),
          SEED(7630430, 4426642), SEED(4332416, 7502816)},
         {SEED(67376, 677568), SEED(1230055, 7720115), SEED(4655933, 3825806),
          SEED(215768, 1925406), SEED(1029024, 5515476), SEED(4123568, 1563359),
          SEED(6311204, 5853253), SEED(7372432, 2853663),
          SEED(1966736, 6638717), SEED(6054608, 7763155),
          SEED(8337658, 7423456), SEED(7039668, 545052), SEED(2276037, 5522571),
          SEED(6249708, 5677094), SEED(6086828, 29219), SEED(3507270, 451810),
          SEED(7630430, 4426642)},
         {SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142), SEED(6637640, 3675296),
          SEED(3363024, 6140928), SEED(4174518, 2528510),
          SEED(7660144, 5308200), SEED(5096395, 5719700), SEED(4105636, 961622),
          SEED(8356388, 7733935), SEED(4830388, 6971646),
          SEED(7797846, 4507017), SEED(6332229, 6408362),
          SEED(2942249, 3810233), SEED(414719, 7445685),
          SEED(5258238, 5165121)},
         {SEED(3507270, 451810), SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142), SEED(6637640, 3675296),
          SEED(3363024, 6140928), SEED(4174518, 2528510),
          SEED(7660144, 5308200), SEED(5096395, 5719700), SEED(4105636, 961622),
          SEED(8356388, 7733935), SEED(4830388, 6971646),
          SEED(7797846, 4507017), SEED(6332229, 6408362),
          SEED(2942249, 3810233), SEED(414719, 7445685)},
         {SEED(6086828, 29219), SEED(3507270, 451810), SEED(7630430, 4426642),
          SEED(4332416, 7502816), SEED(3307459, 1866861), SEED(5672109, 972142),
          SEED(6637640, 3675296), SEED(3363024, 6140928),
          SEED(4174518, 2528510), SEED(7660144, 5308200),
          SEED(5096395, 5719700), SEED(4105636, 961622), SEED(8356388, 7733935),
          SEED(4830388, 6971646), SEED(7797846, 4507017),
          SEED(6332229, 6408362), SEED(2942249, 3810233)},
         {SEED(6249708, 5677094), SEED(6086828, 29219), SEED(3507270, 451810),
          SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142), SEED(6637640, 3675296),
          SEED(3363024, 6140928), SEED(4174518, 2528510),
          SEED(7660144, 5308200), SEED(5096395, 5719700), SEED(4105636, 961622),
          SEED(8356388, 7733935), SEED(4830388, 6971646),
          SEED(7797846, 4507017), SEED(6332229, 6408362)},
         {SEED(2276037, 5522571), SEED(6249708, 5677094), SEED(6086828, 29219),
          SEED(3507270, 451810), SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142), SEED(6637640, 3675296),
          SEED(3363024, 6140928), SEED(4174518, 2528510),
          SEED(7660144, 5308200), SEED(5096395, 5719700), SEED(4105636, 961622),
          SEED(8356388, 7733935), SEED(4830388, 6971646),
          SEED(7797846, 4507017)},
         {SEED(7039668, 545052), SEED(2276037, 5522571), SEED(6249708, 5677094),
          SEED(6086828, 29219), SEED(3507270, 451810), SEED(7630430, 4426642),
          SEED(4332416, 7502816), SEED(3307459, 1866861), SEED(5672109, 972142),
          SEED(6637640, 3675296), SEED(3363024, 6140928),
          SEED(4174518, 2528510), SEED(7660144, 5308200),
          SEED(5096395, 5719700), SEED(4105636, 961622), SEED(8356388, 7733935),
          SEED(4830388, 6971646)},
         {SEED(8337658, 7423456), SEED(7039668, 545052), SEED(2276037, 5522571),
          SEED(6249708, 5677094), SEED(6086828, 29219), SEED(3507270, 451810),
          SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142), SEED(6637640, 3675296),
          SEED(3363024, 6140928), SEED(4174518, 2528510),
          SEED(7660144, 5308200), SEED(5096395, 5719700), SEED(4105636, 961622),
          SEED(8356388, 7733935)},
         {SEED(6054608, 7763155), SEED(8337658, 7423456), SEED(7039668, 545052),
          SEED(2276037, 5522571), SEED(6249708, 5677094), SEED(6086828, 29219),
          SEED(3507270, 451810), SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142), SEED(6637640, 3675296),
          SEED(3363024, 6140928), SEED(4174518, 2528510),
          SEED(7660144, 5308200), SEED(5096395, 5719700),
          SEED(4105636, 961622)},
         {SEED(1966736, 6638717), SEED(6054608, 7763155),
          SEED(8337658, 7423456), SEED(7039668, 545052), SEED(2276037, 5522571),
          SEED(6249708, 5677094), SEED(6086828, 29219), SEED(3507270, 451810),
          SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142), SEED(6637640, 3675296),
          SEED(3363024, 6140928), SEED(4174518, 2528510),
          SEED(7660144, 5308200), SEED(5096395, 5719700)},
         {SEED(7372432, 2853663), SEED(1966736, 6638717),
          SEED(6054608, 7763155), SEED(8337658, 7423456), SEED(7039668, 545052),
          SEED(2276037, 5522571), SEED(6249708, 5677094), SEED(6086828, 29219),
          SEED(3507270, 451810), SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142), SEED(6637640, 3675296),
          SEED(3363024, 6140928), SEED(4174518, 2528510),
          SEED(7660144, 5308200)},
         {SEED(6311204, 5853253), SEED(7372432, 2853663),
          SEED(1966736, 6638717), SEED(6054608, 7763155),
          SEED(8337658, 7423456), SEED(7039668, 545052), SEED(2276037, 5522571),
          SEED(6249708, 5677094), SEED(6086828, 29219), SEED(3507270, 451810),
          SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142), SEED(6637640, 3675296),
          SEED(3363024, 6140928), SEED(4174518, 2528510)},
         {SEED(4123568, 1563359), SEED(6311204, 5853253),
          SEED(7372432, 2853663), SEED(1966736, 6638717),
          SEED(6054608, 7763155), SEED(8337658, 7423456), SEED(7039668, 545052),
          SEED(2276037, 5522571), SEED(6249708, 5677094), SEED(6086828, 29219),
          SEED(3507270, 451810), SEED(7630430, 4426642), SEED(4332416, 7502816),
          SEED(3307459, 1866861), SEED(5672109, 972142), SEED(6637640, 3675296),
          SEED(3363024, 6140928)}},
        {{SEED(5797476, 7601581), SEED(1268930, 8116600),
          SEED(7450247, 2183006), SEED(5858020, 8371815),
          SEED(6957544, 3695291), SEED(2324779, 5971224),
          SEED(2086576, 5832088), SEED(6907375, 4702620),
          SEED(6563341, 6878708), SEED(2962374, 3040958),
          SEED(5763418, 4979847), SEED(727133, 991440), SEED(7108207, 6467847),
          SEED(5350065, 4803768), SEED(763066, 3396637), SEED(4368969, 4446203),
          SEED(4735168, 6755307)},
         {SEED(3304104, 2061991), SEED(5797476, 7601581),
          SEED(1268930, 8116600), SEED(7450247, 2183006),
          SEED(5858020, 8371815), SEED(6957544, 3695291),
          SEED(2324779, 5971224), SEED(2086576, 5832088),
          SEED(6907375, 4702620), SEED(6563341, 6878708),
          SEED(2962374, 3040958), SEED(5763418, 4979847), SEED(727133, 991440),
          SEED(7108207, 6467847), SEED(5350065, 4803768), SEED(763066, 3396637),
          SEED(4368969, 4446203)},
         {SEED(1838381, 4429411), SEED(3304104, 2061991),
          SEED(5797476, 7601581), SEED(1268930, 8116600),
          SEED(7450247, 2183006), SEED(5858020, 8371815),
          SEED(6957544, 3695291), SEED(2324779, 5971224),
          SEED(2086576, 5832088), SEED(6907375, 4702620),
          SEED(6563341, 6878708), SEED(2962374, 3040958),
          SEED(5763418, 4979847), SEED(727133, 991440), SEED(7108207, 6467847),
          SEED(5350065, 4803768), SEED(763066, 3396637)},
         {SEED(8213313, 5579643), SEED(1838381, 4429411),
          SEED(3304104, 2061991), SEED(5797476, 7601581),
          SEED(1268930, 8116600), SEED(7450247, 2183006),
          SEED(5858020, 8371815), SEED(6957544, 3695291),
          SEED(2324779, 5971224), SEED(2086576, 5832088),
          SEED(6907375, 4702620), SEED(6563341, 6878708),
          SEED(2962374, 3040958), SEED(5763418, 4979847), SEED(727133, 991440),
          SEED(7108207, 6467847), SEED(5350065, 4803768)},
         {SEED(6618995, 4531760), SEED(8213313, 5579643),
          SEED(1838381, 4429411), SEED(3304104, 2061991),
          SEED(5797476, 7601581), SEED(1268930, 8116600),
          SEED(7450247, 2183006), SEED(5858020, 8371815),
          SEED(6957544, 3695291), SEED(2324779, 5971224),
          SEED(2086576, 5832088), SEED(6907375, 4702620),
          SEED(6563341, 6878708), SEED(2962374, 3040958),
          SEED(5763418, 4979847), SEED(727133, 991440), SEED(7108207, 6467847)},
         {SEED(7108207, 6467847), SEED(5350065, 4803768), SEED(763066, 3396637),
          SEED(4368969, 4446203), SEED(4735168, 6755307),
          SEED(3472697, 1630357), SEED(7570962, 2284511), SEED(542872, 5868994),
          SEED(7683287, 1493106), SEED(3995170, 654333), SEED(4949969, 991376),
          SEED(1359443, 4840648), SEED(8187776, 6623380),
          SEED(1213276, 2074940), SEED(2199308, 8032929), SEED(1394449, 533644),
          SEED(4380573, 2624740)},
         {SEED(727133, 991440), SEED(7108207, 6467847), SEED(5350065, 4803768),
          SEED(763066, 3396637), SEED(4368969, 4446203), SEED(4735168, 6755307),
          SEED(3472697, 1630357), SEED(7570962, 2284511), SEED(542872, 5868994),
          SEED(7683287, 1493106), SEED(3995170, 654333), SEED(4949969, 991376),
          SEED(1359443, 4840648), SEED(8187776, 6623380),
          SEED(1213276, 2074940), SEED(2199308, 8032929),
          SEED(1394449, 533644)},
         {SEED(5763418, 4979847), SEED(727133, 991440), SEED(7108207, 6467847),
          SEED(5350065, 4803768), SEED(763066, 3396637), SEED(4368969, 4446203),
          SEED(4735168, 6755307), SEED(3472697, 1630357),
          SEED(7570962, 2284511), SEED(542872, 5868994), SEED(7683287, 1493106),
          SEED(3995170, 654333), SEED(4949969, 991376), SEED(1359443, 4840648),
          SEED(8187776, 6623380), SEED(1213276, 2074940),
          SEED(2199308, 8032929)},
         {SEED(2962374, 3040958), SEED(5763418, 4979847), SEED(727133, 991440),
          SEED(7108207, 6467847), SEED(5350065, 4803768), SEED(763066, 3396637),
          SEED(4368969, 4446203), SEED(4735168, 6755307),
          SEED(3472697, 1630357), SEED(7570962, 2284511), SEED(542872, 5868994),
          SEED(7683287, 1493106), SEED(3995170, 654333), SEED(4949969, 991376),
          SEED(1359443, 4840648), SEED(8187776, 6623380),
          SEED(1213276, 2074940)},
         {SEED(6563341, 6878708), SEED(2962374, 3040958),
          SEED(5763418, 4979847), SEED(727133, 991440), SEED(7108207, 6467847),
          SEED(5350065, 4803768), SEED(763066, 3396637), SEED(4368969, 4446203),
          SEED(4735168, 6755307), SEED(3472697, 1630357),
          SEED(7570962, 2284511), SEED(542872, 5868994), SEED(7683287, 1493106),
          SEED(3995170, 654333), SEED(4949969, 991376), SEED(1359443, 4840648),
          SEED(8187776, 6623380)},
         {SEED(6907375, 4702620), SEED(6563341, 6878708),
          SEED(2962374, 3040958), SEED(5763418, 4979847), SEED(727133, 991440),
          SEED(7108207, 6467847), SEED(5350065, 4803768), SEED(763066, 3396637),
          SEED(4368969, 4446203), SEED(4735168, 6755307),
          SEED(3472697, 1630357), SEED(7570962, 2284511), SEED(542872, 5868994),
          SEED(7683287, 1493106), SEED(3995170, 654333), SEED(4949969, 991376),
          SEED(1359443, 4840648)},
         {SEED(2086576, 5832088), SEED(6907375, 4702620),
          SEED(6563341, 6878708), SEED(2962374, 3040958),
          SEED(5763418, 4979847), SEED(727133, 991440), SEED(7108207, 6467847),
          SEED(5350065, 4803768), SEED(763066, 3396637), SEED(4368969, 4446203),
          SEED(4735168, 6755307), SEED(3472697, 1630357),
          SEED(7570962, 2284511), SEED(542872, 5868994), SEED(7683287, 1493106),
          SEED(3995170, 654333), SEED(4949969, 991376)},
         {SEED(2324779, 5971224), SEED(2086576, 5832088),
          SEED(6907375, 4702620), SEED(6563341, 6878708),
          SEED(2962374, 3040958), SEED(5763418, 4979847), SEED(727133, 991440),
          SEED(7108207, 6467847), SEED(5350065, 4803768), SEED(763066, 3396637),
          SEED(4368969, 4446203), SEED(4735168, 6755307),
          SEED(3472697, 1630357), SEED(7570962, 2284511), SEED(542872, 5868994),
          SEED(7683287, 1493106), SEED(3995170, 654333)},
         {SEED(6957544, 3695291), SEED(2324779, 5971224),
          SEED(2086576, 5832088), SEED(6907375, 4702620),
          SEED(6563341, 6878708), SEED(2962374, 3040958),
          SEED(5763418, 4979847), SEED(727133, 991440), SEED(7108207, 6467847),
          SEED(5350065, 4803768), SEED(763066, 3396637), SEED(4368969, 4446203),
          SEED(4735168, 6755307), SEED(3472697, 1630357),
          SEED(7570962, 2284511), SEED(542872, 5868994),
          SEED(7683287, 1493106)},
         {SEED(5858020, 8371815), SEED(6957544, 3695291),
          SEED(2324779, 5971224), SEED(2086576, 5832088),
          SEED(6907375, 4702620), SEED(6563341, 6878708),
          SEED(2962374, 3040958), SEED(5763418, 4979847), SEED(727133, 991440),
          SEED(7108207, 6467847), SEED(5350065, 4803768), SEED(763066, 3396637),
          SEED(4368969, 4446203), SEED(4735168, 6755307),
          SEED(3472697, 1630357), SEED(7570962, 2284511),
          SEED(542872, 5868994)},
         {SEED(7450247, 2183006), SEED(5858020, 8371815),
          SEED(6957544, 3695291), SEED(2324779, 5971224),
          SEED(2086576, 5832088), SEED(6907375, 4702620),
          SEED(6563341, 6878708), SEED(2962374, 3040958),
          SEED(5763418, 4979847), SEED(727133, 991440), SEED(7108207, 6467847),
          SEED(5350065, 4803768), SEED(763066, 3396637), SEED(4368969, 4446203),
          SEED(4735168, 6755307), SEED(3472697, 1630357),
          SEED(7570962, 2284511)},
         {SEED(1268930, 8116600), SEED(7450247, 2183006),
          SEED(5858020, 8371815), SEED(6957544, 3695291),
          SEED(2324779, 5971224), SEED(2086576, 5832088),
          SEED(6907375, 4702620), SEED(6563341, 6878708),
          SEED(2962374, 3040958), SEED(5763418, 4979847), SEED(727133, 991440),
          SEED(7108207, 6467847), SEED(5350065, 4803768), SEED(763066, 3396637),
          SEED(4368969, 4446203), SEED(4735168, 6755307),
          SEED(3472697, 1630357)}},
    },
};

/*
 * The seed vector is advanced by n elements and the resulting pseudo-random
 * value is returned. Skip lengths greater than or equal to CUTOFF are
 * performed by matrix-vector multiplication, multiplying a matrix from the
 * table corresponding to each bit set in n beyond CUTMASK.
 */

static __BIGREAL_T
advance_seed_lf(__INT_T n)
{
  __INT_T i, j, m, old_offset;
  const Seed *t0;
  __BIGREAL_T *t1;
  __BIGREAL_T yhi, ylo;

#ifdef DEBUG
  /*
   * Check for zero advance.
   */
  if (n < 1)
    rnum_abort(__FILE__, __LINE__,
               "random_number:  internal error in advance_seed_lf:  n < 1");
#endif
  /*
   * Update seed_lf values.
   */
  if (n & CUTMASK)
    for (i = n & CUTMASK; i > 0; --i) {
      offset = (offset + 1) & MASK;
      seed_lf[offset] = seed_lf[(offset - SHORT_LAG) & MASK] +
                        seed_lf[(offset - LONG_LAG) & MASK];
      if (seed_lf[offset] > 1.0)
        seed_lf[offset] -= 1.0;
    }
  if (n > CUTMASK) {
    n -= n & CUTMASK;
    /*
     * Adjust to fit.  This way no offsets span the ends of the seed_lf
     * array, and the MASK is not needed below.
     */
    if (LONG_LAG > (offset & (MASK >> 1))) {
      old_offset = offset;
      offset += LONG_LAG - (offset & (MASK >> 1));
      offset &= MASK;
      for (i = 0; i < LONG_LAG; ++i)
        seed_lf[offset - i] = seed_lf[(old_offset - i) & MASK];
    }
    offset &= MASK;
    /*
     * Do big jumps by matrix multiplication.
     */
    m = 0;
    while (n > 0) {
      /*
       * Advance vector of size LONG_LAG of seed_lf values by 2**m.
       * Matrix-vector multiplication is performed from seed_lf[old_offset]
       * into seed_lf[offset].
       */
      i = n & (DIGIT);
      if (i) {
        old_offset = offset;
        offset ^= TOGGLE;
        t0 = table_lf[m][i - 1][0];
        t1 = seed_lf + old_offset;
        i = T23 * *t1;
        yhi = R23 * i;
        ylo = *t1 - yhi;
        for (i = 0; i < LONG_LAG; ++i)
          seed_lf[offset - i] = mul46(t0++, ylo, yhi);
        for (j = 1; j < LONG_LAG; ++j) {
          --t1;
          i = T23 * *t1;
          yhi = R23 * i;
          ylo = *t1 - yhi;
          for (i = 0; i < LONG_LAG; ++i)
            seed_lf[offset - i] += mul46(t0++, ylo, yhi);
        }
        for (i = 0; i < LONG_LAG; ++i) {
          j = seed_lf[offset - i];
          seed_lf[offset - i] -= j;
        }
      }
      /*
       * Update power of two index (m) and skip counter (n).
       */
      ++m;
      n >>= NBITS;
    }
  }
  /*
   * Return new value.
   */
  return seed_lf[offset];
}

/*
 * Routine that loops through a dimension of the double precision output.
 * Recursive down to last dimension, where work is done.
 */

static void I8(prng_loop_d_lf)(__REAL8_T *hb, F90_Desc *harvest, __INT_T li,
                               int dim, __INT_T section_offset, __INT_T limit)
{
  DECL_DIM_PTRS(hdd);
  DECL_DIM_PTRS(tdd);
  __INT_T cl, cn, current, i, il, iu, lo, clof, n;
  __INT_T hi, tcl, tcn, tclof;

  SET_DIM_PTRS(hdd, harvest, dim - 1);
  cl = DIST_DPTR_CL_G(hdd);
  cn = DIST_DPTR_CN_G(hdd);
  clof = DIST_DPTR_CLOF_G(hdd);

  if (dim > 1)
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      for (i = 0; i < n; ++i) {
        I8(prng_loop_d_lf)(hb, harvest, lo, dim - 1, current + i, limit);
        lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      }
    }
  /*
   * Optimization collapsing non-distributed leading dimensions.
   */
  else if (limit > 0) {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      /*
       * Find first current and low value of fill range.
       */
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      hi = lo + (n - 1) * F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      for (i = dim - 1; i > 0; --i) {
        SET_DIM_PTRS(tdd, harvest, i - 1);
        tcl = DIST_DPTR_CL_G(tdd);
        tcn = DIST_DPTR_CN_G(tdd);
        tclof = DIST_DPTR_CLOF_G(tdd);
        (void)I8(__fort_block_bounds)(harvest, i, tcl, &il, &iu);
        lo = lo +
             (F90_DPTR_SSTRIDE_G(tdd) * il + F90_DPTR_SOFFSET_G(tdd) - tclof) *
                 F90_DPTR_LSTRIDE_G(hdd);
        current =
            F90_DPTR_EXTENT_G(tdd) * current + (il - F90_DPTR_LBOUND_G(tdd));
        n = I8(__fort_block_bounds)(
            harvest, i, tcl + (tcn - 1) * DIST_DPTR_CS_G(tdd), &il, &iu);
        hi = hi +
             (F90_DPTR_SSTRIDE_G(tdd) * (il + n - 1) + F90_DPTR_SOFFSET_G(tdd) -
              tclof) *
                 F90_DPTR_LSTRIDE_G(tdd);
      }
      /*
       * Fill the array with random numbers.
       */
      hb[lo] = advance_seed_lf(current - last_i);
      last_i = current + hi - lo;
      for (i = lo + 1; i <= hi; ++i) {
        offset = (offset + 1) & MASK;
        seed_lf[offset] = seed_lf[(offset - SHORT_LAG) & MASK] +
                          seed_lf[(offset - LONG_LAG) & MASK];
        if (seed_lf[offset] > 1.0)
          seed_lf[offset] -= 1.0;
        hb[i] = seed_lf[offset];
      }
    }
  } else {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      if (n > 0) {
        lo = li +
             (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
                 F90_DPTR_LSTRIDE_G(hdd);
        current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                  (il - F90_DPTR_LBOUND_G(hdd));
        hb[lo] = advance_seed_lf(current - last_i);
        for (i = 1; i < n; ++i) {
          lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
          offset = (offset + 1) & MASK;
          seed_lf[offset] = seed_lf[(offset - SHORT_LAG) & MASK] +
                            seed_lf[(offset - LONG_LAG) & MASK];
          if (seed_lf[offset] > 1.0)
            seed_lf[offset] -= 1.0;
          hb[lo] = seed_lf[offset];
        }
        last_i = current + n - 1;
      }
    }
  }
}

#ifdef TARGET_SUPPORTS_QUADFP
static void I8(prng_loop_q_lf)(__REAL16_T *hb, F90_Desc *harvest, __INT_T li,
                               int dim, __INT_T section_offset, __INT_T limit)
{
  DECL_DIM_PTRS(hdd);
  DECL_DIM_PTRS(tdd);
  __INT_T cl, cn, current, i, il, iu, lo, clof, n;
  __INT_T hi, tcl, tcn, tclof;

  SET_DIM_PTRS(hdd, harvest, dim - 1);
  cl = DIST_DPTR_CL_G(hdd);
  cn = DIST_DPTR_CN_G(hdd);
  clof = DIST_DPTR_CLOF_G(hdd);

  if (dim > 1)
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      for (i = 0; i < n; ++i) {
        I8(prng_loop_q_lf)(hb, harvest, lo, dim - 1, current + i, limit);
        lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      }
    }
  /*
   * Optimization collapsing non-distributed leading dimensions.
   */
  else if (limit > 0) {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      /*
       * Find first current and low value of fill range.
       */
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      hi = lo + (n - 1) * F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      for (i = dim - 1; i > 0; --i) {
        SET_DIM_PTRS(tdd, harvest, i - 1);
        tcl = DIST_DPTR_CL_G(tdd);
        tcn = DIST_DPTR_CN_G(tdd);
        tclof = DIST_DPTR_CLOF_G(tdd);
        (void)I8(__fort_block_bounds)(harvest, i, tcl, &il, &iu);
        lo = lo +
             (F90_DPTR_SSTRIDE_G(tdd) * il + F90_DPTR_SOFFSET_G(tdd) - tclof) *
                 F90_DPTR_LSTRIDE_G(hdd);
        current =
            F90_DPTR_EXTENT_G(tdd) * current + (il - F90_DPTR_LBOUND_G(tdd));
        n = I8(__fort_block_bounds)(
            harvest, i, tcl + (tcn - 1) * DIST_DPTR_CS_G(tdd), &il, &iu);
        hi = hi +
             (F90_DPTR_SSTRIDE_G(tdd) * (il + n - 1) + F90_DPTR_SOFFSET_G(tdd) -
              tclof) *
                 F90_DPTR_LSTRIDE_G(tdd);
      }
      /*
       * Fill the array with random numbers.
       */
      hb[lo] = advance_seed_lf(current - last_i);
      last_i = current + hi - lo;
      for (i = lo + 1; i <= hi; ++i) {
        offset = (offset + 1) & MASK;
        seed_lf[offset] = seed_lf[(offset - SHORT_LAG) & MASK] +
                          seed_lf[(offset - LONG_LAG) & MASK];
        if (seed_lf[offset] > 1.0)
          seed_lf[offset] -= 1.0;
        hb[i] = seed_lf[offset];
      }
    }
  } else {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      if (n > 0) {
        lo = li +
             (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
                 F90_DPTR_LSTRIDE_G(hdd);
        current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                  (il - F90_DPTR_LBOUND_G(hdd));
        hb[lo] = advance_seed_lf(current - last_i);
        for (i = 1; i < n; ++i) {
          lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
          offset = (offset + 1) & MASK;
          seed_lf[offset] = seed_lf[(offset - SHORT_LAG) & MASK] +
                            seed_lf[(offset - LONG_LAG) & MASK];
          if (seed_lf[offset] > 1.0)
            seed_lf[offset] -= 1.0;
          hb[lo] = seed_lf[offset];
        }
        last_i = current + n - 1;
      }
    }
  }
}
#endif

/*
 * Routine that loops through a dimension of the single precision output.
 * Recursive down to last dimension, where work is done.
 */

static void I8(prng_loop_r_lf)(__REAL4_T *hb, F90_Desc *harvest, __INT_T li,
                               int dim, __INT_T section_offset, __INT_T limit)
{
  DECL_DIM_PTRS(hdd);
  DECL_DIM_PTRS(tdd);
  __INT_T cl, cn, current, i, il, iu, lo, clof, n;
  __INT_T hi, tcl, tcn, tclof;

  SET_DIM_PTRS(hdd, harvest, dim - 1);
  cl = DIST_DPTR_CL_G(hdd);
  cn = DIST_DPTR_CN_G(hdd);
  clof = DIST_DPTR_CLOF_G(hdd);

  if (dim > 1)
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      for (i = 0; i < n; ++i) {
        I8(prng_loop_r_lf)(hb, harvest, lo, dim - 1, current + i, limit);
        lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      }
    }
  /*
   * Optimization collapsing non-distributed leading dimensions.
   */
  else if (limit > 0) {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      /*
       * Find first current and low value of fill range.
       */
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      lo = li +
           (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
               F90_DPTR_LSTRIDE_G(hdd);
      current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                (il - F90_DPTR_LBOUND_G(hdd));
      hi = lo + (n - 1) * F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
      for (i = dim - 1; i > 0; --i) {
        SET_DIM_PTRS(tdd, harvest, i - 1);
        tcl = DIST_DPTR_CL_G(tdd);
        tcn = DIST_DPTR_CN_G(tdd);
        tclof = DIST_DPTR_CLOF_G(tdd);
        (void)I8(__fort_block_bounds)(harvest, i, tcl, &il, &iu);
        lo = lo +
             (F90_DPTR_SSTRIDE_G(tdd) * il + F90_DPTR_SOFFSET_G(tdd) - tclof) *
                 F90_DPTR_LSTRIDE_G(hdd);
        current =
            F90_DPTR_EXTENT_G(tdd) * current + (il - F90_DPTR_LBOUND_G(tdd));
        n = I8(__fort_block_bounds)(
            harvest, i, tcl + (tcn - 1) * DIST_DPTR_CS_G(tdd), &il, &iu);
        hi = hi +
             (F90_DPTR_SSTRIDE_G(tdd) * (il + n - 1) + F90_DPTR_SOFFSET_G(tdd) -
              tclof) *
                 F90_DPTR_LSTRIDE_G(hdd);
      }
      /*
       * Fill the array with random numbers.
       */
      hb[lo] = advance_seed_lf(current - last_i);
      last_i = current + hi - lo;
      for (i = lo + 1; i <= hi; ++i) {
        offset = (offset + 1) & MASK;
        seed_lf[offset] = seed_lf[(offset - SHORT_LAG) & MASK] +
                          seed_lf[(offset - LONG_LAG) & MASK];
        if (seed_lf[offset] > 1.0)
          seed_lf[offset] -= 1.0;
        hb[i] = seed_lf[offset];
      }
    }
  } else {
    for (; cn > 0;
         --cn, cl += DIST_DPTR_CS_G(hdd), clof += DIST_DPTR_CLOS_G(hdd)) {
      n = I8(__fort_block_bounds)(harvest, dim, cl, &il, &iu);
      if (n > 0) {
        lo = li +
             (F90_DPTR_SSTRIDE_G(hdd) * il + F90_DPTR_SOFFSET_G(hdd) - clof) *
                 F90_DPTR_LSTRIDE_G(hdd);
        current = F90_DPTR_EXTENT_G(hdd) * section_offset +
                  (il - F90_DPTR_LBOUND_G(hdd));
        hb[lo] = advance_seed_lf(current - last_i);
        for (i = 1; i < n; ++i) {
          lo += F90_DPTR_SSTRIDE_G(hdd) * F90_DPTR_LSTRIDE_G(hdd);
          offset = (offset + 1) & MASK;
          seed_lf[offset] = seed_lf[(offset - SHORT_LAG) & MASK] +
                            seed_lf[(offset - LONG_LAG) & MASK];
          if (seed_lf[offset] > 1.0)
            seed_lf[offset] -= 1.0;
          hb[lo] = seed_lf[offset];
        }
        last_i = current + n - 1;
      }
    }
  }
}

/*
 * ========================================================================
 * Common code.
 * ========================================================================
 */

static int fibonacci = 1;

static __BIGREAL_T (*advance_seed)(__INT_T) = advance_seed_lf;
#ifdef TARGET_SUPPORTS_QUADFP
static void (*prng_loop_q)(__REAL16_T *, F90_Desc *, __INT_T, int, __INT_T,
                           __INT_T) = I8(prng_loop_q_lf);
#endif
static void (*prng_loop_d)(__REAL8_T *, F90_Desc *, __INT_T, int, __INT_T,
                           __INT_T) = I8(prng_loop_d_lf);
static void (*prng_loop_r)(__REAL4_T *, F90_Desc *, __INT_T, int, __INT_T,
                           __INT_T) = I8(prng_loop_r_lf);

static void
set_fibonacci(void)
{
  fibonacci = 1;
  advance_seed = advance_seed_lf;
#ifdef TARGET_SUPPORTS_QUADFP
  prng_loop_q = I8(prng_loop_q_lf);
#endif
  prng_loop_d = I8(prng_loop_d_lf);
  prng_loop_r = I8(prng_loop_r_lf);
}

static void
set_npb(void)
{
  fibonacci = 0;
  advance_seed = advance_seed_npb;
#ifdef TARGET_SUPPORTS_QUADFP
  prng_loop_q = I8(prng_loop_q_npb);
#endif
  prng_loop_d = I8(prng_loop_d_npb);
  prng_loop_r = I8(prng_loop_r_npb);
}

/*
 * Determine how many dimensions are exactly contained on this processor of
 * this array section.  The stride must be one, the dimension must not be
 * partitioned, and the extent of the section must span the array.  This
 * optimizes random number generation in arrays with dimensions such as
 * (2,N) distributed (*,block).
 */

static int I8(level)(F90_Desc *harvest)
{
  DECL_DIM_PTRS(hdd);
  int i;

  i = 0;
  while (i < (F90_RANK_G(harvest) - 1)) {
    if ((DIST_MAPPED_G(harvest) >> i) & 1)
      break;
    SET_DIM_PTRS(hdd, harvest, i);
    if (DPTR_UBOUND_G(hdd) - F90_DPTR_LBOUND_G(hdd) !=
        DIST_DPTR_UAB_G(hdd) - DIST_DPTR_LAB_G(hdd))
      break;
    ++i;
  }
  return i;
}

/*
 * Single precision, pseudo-random number generator, RANDOM_NUMBER.
 */

void ENTFTN(RNUM, rnum)(__REAL4_T *hb, F90_Desc *harvest)
{
  __INT_T final, i;
  int itmp;
  __BIGREAL_T tmp1, tmp2;

  MP_P(sem);
  if (F90_TAG_G(harvest) == __DESC) {
    if (F90_GSIZE_G(harvest) <= 0) {
      MP_V(sem);
      return;
    }
    last_i = -1;
    if (~F90_FLAGS_G(harvest) & __OFF_TEMPLATE) {
      I8(__fort_cycle_bounds)(harvest);
      i = I8(level)(harvest);
      prng_loop_r(hb, harvest, F90_LBASE_G(harvest) - 1, F90_RANK_G(harvest), 0,
                  i);
    }
    final = F90_GSIZE_G(harvest) - 1;
    if (last_i < final)
      (void)advance_seed(final - last_i);
#ifdef DEBUG
    else if (last_i != final)
      rnum_abort(__FILE__, __LINE__,
                 "random_number:  internal error:  last_i != final");
#endif
  } else {
    if (fibonacci) {
      offset = (offset + 1) & MASK;
      seed_lf[offset] = seed_lf[(offset - SHORT_LAG) & MASK] +
                        seed_lf[(offset - LONG_LAG) & MASK];
      if (seed_lf[offset] > 1.0)
        seed_lf[offset] -= 1.0;
      *hb = seed_lf[offset];
      if (*hb == (float)1.0) {
        itmp = 0x3F7FFFFF;
        *hb = *(float *)&itmp;
      }
    } else {
      tmp1 = seed_lo * table[0][0];
      itmp = T23 * tmp1;
      tmp2 = R23 * itmp;
      seed_hi = tmp2 + seed_lo * table[0][1] + seed_hi * table[0][0];
      seed_lo = tmp1 - tmp2;
      itmp = seed_hi;
      seed_hi -= itmp;
      *hb = seed_lo + seed_hi;
    }
  }
  MP_V(sem);
}

/*
 * Double precision, pseudo-random number generator, RANDOM_NUMBER.
 */

void ENTFTN(RNUMD, rnumd)(__REAL8_T *hb, F90_Desc *harvest)
{
  __INT_T final, i;
  int itmp, tmp3[2];
  __BIGREAL_T tmp1, tmp2;

  MP_P(sem);
  if (F90_TAG_G(harvest) == __DESC) {
    if (F90_GSIZE_G(harvest) <= 0) {
      MP_V(sem);
      return;
    }
    last_i = -1;
    if ((~(unsigned int)F90_FLAGS_G(harvest)) & __OFF_TEMPLATE) {
      I8(__fort_cycle_bounds)(harvest);
      i = I8(level)(harvest);
      prng_loop_d(hb, harvest, F90_LBASE_G(harvest) - 1, F90_RANK_G(harvest), 0,
                  i);
    }
    final = F90_GSIZE_G(harvest) - 1;
    if (last_i < final)
      (void)advance_seed(final - last_i);
#ifdef DEBUG
    else if (last_i != final)
      rnum_abort(__FILE__, __LINE__,
                 "random_number:  internal error:  last_i != final");
#endif
  } else {
    if (fibonacci) {
      offset = (offset + 1) & MASK;
      seed_lf[offset] = seed_lf[(offset - SHORT_LAG) & MASK] +
                        seed_lf[(offset - LONG_LAG) & MASK];
      if (seed_lf[offset] > 1.0)
        seed_lf[offset] -= 1.0;
      *hb = seed_lf[offset];
      /* According to standard, *hb should >= 0 and < 1,
       * when *hb == 1.0, assign number that
       * is nearest to 1 and less than 1 to *hb */
      if (*hb == (double)1.0) {
        tmp3[1] = 0x3FEFFFFF;
        tmp3[0] = 0xFFFFFFFF;
        *hb = *(double *)&tmp3;
      }
    } else {
      tmp1 = seed_lo * table[0][0];
      itmp = T23 * tmp1;
      tmp2 = R23 * itmp;
      seed_hi = tmp2 + seed_lo * table[0][1] + seed_hi * table[0][0];
      seed_lo = tmp1 - tmp2;
      itmp = seed_hi;
      seed_hi -= itmp;
      *hb = seed_lo + seed_hi;
    }
  }
  MP_V(sem);
}

#ifdef TARGET_SUPPORTS_QUADFP
void ENTFTN(RNUMQ, rnumq)(__REAL16_T *hb, F90_Desc *harvest)
{
  __INT_T final, i;
  int itmp, tmp3[4];
  __BIGREAL_T tmp1, tmp2;

  MP_P(sem);
  if (F90_TAG_G(harvest) == __DESC) {
    if (F90_GSIZE_G(harvest) <= 0) {
      MP_V(sem);
      return;
    }
    last_i = -1;
    if (~(unsigned int)F90_FLAGS_G(harvest) & __OFF_TEMPLATE) {
      I8(__fort_cycle_bounds)(harvest);
      i = I8(level)(harvest);
      prng_loop_q(hb, harvest, F90_LBASE_G(harvest) - 1, F90_RANK_G(harvest), 0,
                  i);
    }
    final = F90_GSIZE_G(harvest) - 1;
    if (last_i < final)
      (void)advance_seed(final - last_i);
#ifdef DEBUG
    else if (last_i != final)
      rnum_abort(__FILE__, __LINE__,
                 "random_number:  internal error:  last_i != final");
#endif
  } else {
    if (fibonacci) {
      offset = (offset + 1) & MASK;
      seed_lf[offset] = seed_lf[(offset - SHORT_LAG) & MASK] +
                        seed_lf[(offset - LONG_LAG) & MASK];
      if (seed_lf[offset] > 1.0)
        seed_lf[offset] -= 1.0;
      *hb = seed_lf[offset];
      /* According to standard, *hb should >= 0 and < 1,
       * when *hb == 1.0, assign number that
       * is nearest to 1 and less than 1 to *hb */
      if (*hb == (__BIGREAL_T)1.0) {
        tmp3[3] = 0x3FFEFFFF;
        tmp3[2] = 0xFFFFFFFF;
        tmp3[1] = 0xFFFFFFFF;
        tmp3[0] = 0xFFFFFFFF;
        *hb = *(__BIGREAL_T *)&tmp3;
      }
    } else {
      tmp1 = seed_lo * table[0][0];
      itmp = T23 * tmp1;
      tmp2 = R23 * itmp;
      seed_hi = tmp2 + seed_lo * table[0][1] + seed_hi * table[0][0];
      seed_lo = tmp1 - tmp2;
      itmp = seed_hi;
      seed_hi -= itmp;
      *hb = seed_lo + seed_hi;
    }
  }
  MP_V(sem);
}
#endif

/*
 * put_int writes a single integer.
 */

static void
put_int(void *b, F90_Desc *d, __INT_T val)
{
  dtype kind;

  if (F90_TAG_G(d) == __DESC) {
    if (F90_RANK_G(d) != 0)
      __fort_abort("put_int: non-scalar destination");
    if (F90_FLAGS_G(d) & __OFF_TEMPLATE)
      return;
    b += DIST_SCOFF_G(d) * F90_LEN_G(d);
    kind = F90_KIND_G(d);
  } else
    kind = Abs(F90_TAG_G(d));

  switch (kind) {
  case __INT1:
    *(__INT1_T *)b = (__INT1_T)val;
    break;
  case __INT2:
    *(__INT2_T *)b = (__INT2_T)val;
    break;
  case __INT4:
    *(__INT4_T *)b = (__INT4_T)val;
    break;
  case __INT8:
    *(__INT8_T *)b = (__INT8_T)val;
    break;
  default:
    __fort_abort("put_int: non-integer type");
  }
}

/*
 * Fortran 90 pseudo-random number generator seed routine, RANDOM_SEED.
 */

void ENTFTN(RSEED, rseed)(void *size, __INT_T *putb, __INT_T *getb,
                          F90_Desc *sized, F90_Desc *putd, F90_Desc *getd)
{
  int i, j, no_args_present, vhi, vlo;
  __INT_T list[LONG_LAG][2];
  __INT_T extent, index;
  char *static_seed;
  unsigned int shift_val=0;
  

  MP_P(sem);
  no_args_present = 1;
  vhi = vlo = 0;
  /*
   * Handle valid GET section.
   */
  if (ISPRESENT(getb)) {
    if (F90_TAG_G(getd) != __DESC)
      __fort_abort("random_seed:  argument GET is not array section");
    if (F90_RANK_G(getd) != 1)
      __fort_abort("random_seed:  argument GET is not rank 1");

    extent = F90_DIM_EXTENT_G(getd, 0);
    if (extent < 2)
      __fort_abort("random_seed:  argument GET is wrong size");

    if (extent < (2 * LONG_LAG)) {
      set_npb();
/*
 * SEED_LO:
 */
      vlo = T46 * seed_lo;
      I8(__fort_store_int_element)(getb, getd, 1, vlo);
/*
 * SEED_HI:
 */
      vhi = T23 * seed_hi;
      I8(__fort_store_int_element)(getb, getd, 2, vhi);

    } else {

      set_fibonacci();
      for (i = 0; i < LONG_LAG; ++i) {
        vhi = T23 * seed_lf[(offset + (CYCLE - LONG_LAG + 1) + i) & MASK];
        vlo =
            T23 *
            (T23 * seed_lf[(offset + (CYCLE - LONG_LAG + 1) + i) & MASK] - vhi);
        I8(__fort_store_int_element)(getb, getd, 2 * i + 1, vlo);
        I8(__fort_store_int_element)(getb, getd, 2 * i + 2, vhi);
      }
    }
    no_args_present = 0;
  }
  /*
   * Handle valid PUT section.
   */
  if (ISPRESENT(putb)) {
    /*
     * Tidy up offset so MASK is not needed in indexing below.
     */
    if (F90_TAG_G(putd) == __DESC) {
      if (F90_RANK_G(putd) != 1)
        __fort_abort("random_seed:  argument PUT is not rank 1 array section");

      extent = F90_DIM_EXTENT_G(putd, 0);
      if (extent < 2) {
        __fort_abort("random_seed:  argument PUT is wrong size array section");
      }

      if (extent < (2 * LONG_LAG)) {
        shift_val = 0;
        do {
          set_npb();
          /*
           * SEED_LO:
           */
          index = F90_DIM_LBOUND_G(putd, 0);
          I8(__fort_get_scalar)(list[0] + 0, putb, putd, &index);
          list[0][0] >>= shift_val;
          list[0][0] &= MASK23;
          vlo = list[0][0];
          seed_lo = R46 * vlo;
          /*
           * SEED_HI:
           */
          index = F90_DIM_LBOUND_G(putd, 0) + 1;
          I8(__fort_get_scalar)(list[0] + 1, putb, putd, &index);
          list[0][1] >>= shift_val;
          list[0][1] &= MASK23;
          vhi = list[0][1];
          seed_hi = R23 * vhi;
          shift_val += 23;
        } while (!(vlo | vhi) && shift_val < 64);
      } else {
        shift_val = 0;
        do {
          if (shift_val != 0)
            vlo = vhi = 0;
          set_fibonacci();
          offset = LONG_LAG - 1;
          for (i = 0; i < LONG_LAG; ++i)
            for (j = 0; j < 2; ++j) {
              index = F90_DIM_LBOUND_G(putd, 0) + (2 * i + j);
              I8(__fort_get_scalar)(list[i] + j, putb, putd, &index);
              list[i][j] >>= shift_val;
              list[i][j] &= 0x7fffff;
            }
          for (i = 0; i < LONG_LAG; ++i) {
            seed_lf[i] = R23 * (R23 * list[i][0] + list[i][1]);
            vlo |= list[i][0];
            vhi |= list[i][1];
          }
          shift_val += 23;
        } while (!(vlo | vhi) && shift_val < 64);
      }
    } else {
      shift_val = 0;
      do {
        /*
         * Mask seed value that was input.
         */
        vlo = (*putb >> shift_val) & MASK23;
        vhi = (*putb >> shift_val) & MASK23;
        shift_val += 23;
      } while (!(vlo | vhi) && shift_val < 64);

      if (fibonacci)
        for (i = 0; i < LONG_LAG; ++i)
          seed_lf[i] = R23 * (R23 * vlo + vhi);
      else {
        seed_lo = R46 * vlo;
        seed_hi = R23 * vhi;
      }
    }
    /*
     * Check suitability of input seed.  Seed values should have at least one
     * nonzero value.
     */
    if (!(vlo | vhi))
      __fort_abort(
          "random_seed:  input seed must have at least one nonzero value");
    no_args_present = 0;
  }
  /*
   * Return size of seed table.
   */
  if (ISPRESENT(size)) {
    if (fibonacci)
      put_int(size, sized, 2 * LONG_LAG);
    else
      put_int(size, sized, 2);
    no_args_present = 0;
  }
  /*
   * Seed with default seed values.
   */
  if (no_args_present) {
    if (fibonacci) {
      offset = LONG_LAG - 1;
      for (i = 0; i < LONG_LAG; ++i)
        seed_lf[i] = R46 * default_seed_lf[i];

      static_seed = __fort_getenv("STATIC_RANDOM_SEED");
      if (static_seed == NULL || strstr(static_seed, "yes") == 0) {

        /* advance the random seed by a variable amount
           each time the program is called.  If the
           user calls RANDOM_NUMBER() then the new
           "variable" default is used - Mat Colgrove 1/27/05
        */
        if (!start_time_is_set) {
          int start_time_int;
          start_time_is_set = 1;
          time(&start_time);

          /* first cast our value into a 32bit signed type */
          start_time_int = start_time;

          /* If the time value has the sign bit lit in this
             representation, convert the number to its positive
             equivalent, then assign it back start_time for use
             after it has been sanitized.
          */
          if (start_time_int < 0)
            start_time = start_time_int & 0x7fffffff;
        }
        advance_seed_lf(start_time);
      }
    } else {
      seed_lo = DEFAULT_SEED_LO;
      seed_hi = DEFAULT_SEED_HI;
    }
  }
  MP_V(sem);
}
