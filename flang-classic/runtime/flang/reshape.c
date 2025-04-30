/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

#include "stdioInterf.h"
#include "fioMacros.h"

extern void (*__fort_scalar_copy[__NTYPES])(void *rp, const void *sp, int len);

/* advance index n elements and return the remaining extent in the
   first dimension from that point.  When the end of the array is
   reached, index is reset to the first element and 0 is returned. */

static int I8(advance)(F90_Desc *d, __INT_T *index, __INT_T n)
{
  __INT_T i, r;

  index[0] += n;
  if (index[0] <= DIM_UBOUND_G(d, 0))
    return DIM_UBOUND_G(d, 0) - index[0] + 1;
  else if (index[0] > DIM_UBOUND_G(d, 0) + 1)
    __fort_abort("RESHAPE: internal error, advance past ubound");
  else
    index[0] = F90_DIM_LBOUND_G(d, 0);
  r = F90_RANK_G(d);
  for (i = 1; i < r; ++i) {
    index[i]++;
    if (index[i] <= DIM_UBOUND_G(d, i))
      return DIM_UBOUND_G(d, 0) - index[0] + 1;
    else
      index[i] = F90_DIM_LBOUND_G(d, i);
  }
  return 0;
}

/* note: dimensions in order vector are zero-based. */

static int I8(advance_permuted)(F90_Desc *d, __INT_T *index, int *order,
                                __INT_T n)
{
  int j, k;
  __INT_T i, r;

  k = order[0];
  index[k] += n;
  if (index[k] <= DIM_UBOUND_G(d, k))
    return DIM_UBOUND_G(d, k) - index[k] + 1;
  else if (index[k] > DIM_UBOUND_G(d, k) + 1)
    __fort_abort("RESHAPE: internal error, advance past ubound");
  else
    index[k] = F90_DIM_LBOUND_G(d, k);
  r = F90_RANK_G(d);
  for (i = 1; i < r; i++) {
    j = order[i];
    index[j]++;
    if (index[j] <= DIM_UBOUND_G(d, j))
      return DIM_UBOUND_G(d, k) - index[k] + 1;
    else
      index[j] = F90_DIM_LBOUND_G(d, j);
  }
  return 0;
}

/* reshape intrinsic */

void ENTFTN(RESHAPE, reshape)(char *resb,     /* result base */
                              char *srcb,     /* source base */
                              char *shpb,     /* shape base */
                              char *padb,     /* pad base */
                              char *ordb,     /* order base */
                              F90_Desc *resd, /* result descriptor */
                              F90_Desc *srcd, /* source descriptor */
                              F90_Desc *shpd, /* shape descriptor */
                              F90_Desc *padd, /* pad descriptor */
                              F90_Desc *ordd) /* order descriptor */
{
  __INT_T resx[MAXDIMS];
  __INT_T srcx[MAXDIMS];
  __INT_T padx[MAXDIMS];
  int shape[MAXDIMS];
  int order[MAXDIMS];
  DECL_HDR_VARS(fromd);
  DECL_HDR_VARS(tod);
  char *fromb, *tob;
  chdr *ch;
  __INT_T more_res, more_src, more_pad, n;
  int i, j, k, m, r;

#if defined(DEBUG)
  if (resd == NULL || F90_TAG_G(resd) != __DESC)
    __fort_abort("RESHAPE: invalid result descriptor");
  if (srcd == NULL || F90_TAG_G(srcd) != __DESC)
    __fort_abort("RESHAPE: invalid SOURCE descriptor");
  if (shpd == NULL || F90_TAG_G(shpd) != __DESC)
    __fort_abort("RESHAPE: invalid SHAPE descriptor");
  if (padd == NULL || F90_TAG_G(padd) != __DESC && F90_TAG_G(padd) != __NONE)
    __fort_abort("RESHAPE: invalid PAD descriptor");
  if (ordd == NULL || F90_TAG_G(ordd) != __DESC && F90_TAG_G(ordd) != __NONE)
    __fort_abort("RESHAPE: invalid ORDER descriptor");
#endif

  if (F90_KIND_G(resd) != F90_KIND_G(srcd) ||
      F90_LEN_G(resd) != F90_LEN_G(srcd))
    __fort_abort("RESHAPE: result type != SOURCE type");
  if (F90_TAG_G(padd) == __DESC && (F90_KIND_G(padd) != F90_KIND_G(srcd) ||
                                    F90_LEN_G(padd) != F90_LEN_G(srcd)))
    __fort_abort("RESHAPE: PAD type != SOURCE type");

  /* don't really need the shape vector because the shape is already
     set in the result descriptor, but check its validity anyway */

  if (F90_RANK_G(shpd) <= 0)
    __fort_abort("RESHAPE: invalid SHAPE argument");

  r = DIM_UBOUND_G(shpd, 0) - F90_DIM_LBOUND_G(shpd, 0) + 1;
  if (r < 0 || r > MAXDIMS || r != F90_RANK_G(resd))
    __fort_abort("RESHAPE: invalid SHAPE argument");

  I8(__fort_fetch_int_vector)(shpb, shpd, shape, r);
  for (i = r; --i >= 0;) {
    if (shape[i] < 0)
      __fort_abort("RESHAPE: invalid SHAPE argument");
  }

  /* get the order vector */

  if (F90_TAG_G(ordd) == __DESC) {
    I8(__fort_fetch_int_vector)(ordb, ordd, order, r);
    m = 0;
    for (i = r; --i >= 0;) {
      if (order[i] < 1 || order[i] > r)
        __fort_abort("RESHAPE: invalid ORDER argument");
      --order[i]; /* zero-based dimension number */
      m |= 1 << order[i];
    }
    if (m != ~(-1 << r))
      __fort_abort("RESHAPE: invalid ORDER argument");
  } else { /* absent */
    for (i = r; --i >= 0;)
      order[i] = i;
  }

  /* initialize indices and first column extents */

  if (F90_GSIZE_G(resd) <= 0)
    return;
  for (i = r; --i >= 0;)
    resx[i] = F90_DIM_LBOUND_G(resd, i);
  k = order[0];
  more_res = DIM_UBOUND_G(resd, k) - F90_DIM_LBOUND_G(resd, k) + 1;

  if (F90_GSIZE_G(srcd) > 0) {
    for (i = F90_RANK_G(srcd); --i >= 0;)
      srcx[i] = F90_DIM_LBOUND_G(srcd, i);
    more_src = DIM_UBOUND_G(srcd, 0) - F90_DIM_LBOUND_G(srcd, 0) + 1;
  } else
    more_src = 0;

  if (F90_TAG_G(padd) == __DESC && F90_GSIZE_G(padd) > 0) {
    for (i = F90_RANK_G(padd); --i >= 0;)
      padx[i] = F90_DIM_LBOUND_G(padd, i);
    more_pad = DIM_UBOUND_G(padd, 0) - F90_DIM_LBOUND_G(padd, 0) + 1;
  } else
    more_pad = 0;

  /* loop -- transfer matching column vector sections and advance
     indices until result array is filled */

  while (more_res) {

    if (more_src) {
      n = Min(more_src, more_res);

      __DIST_INIT_SECTION(fromd, 1, srcd);
      I8(__fort_set_section)(fromd, 1, srcd, 1, srcx[0], srcx[0] + n - 1, 1);
      for (i = 1; i < F90_RANK_G(srcd); ++i)
        I8(__fort_set_single)(fromd, srcd, i + 1, srcx[i], __SCALAR);
      I8(__fort_finish_section)(fromd);
      fromb = srcb;

      more_src = I8(advance)(srcd, srcx, n);
    } else if (more_pad) {
      n = Min(more_pad, more_res);

      fromb = padb;
      __DIST_INIT_SECTION(fromd, 1, padd);
      I8(__fort_set_section)(fromd, 1, padd, 1, padx[0], padx[0] + n - 1, 1);
      for (i = 1; i < F90_RANK_G(padd); ++i)
        I8(__fort_set_single)(fromd, padd, i + 1, padx[i], __SCALAR);
      I8(__fort_finish_section)(fromd);

      more_pad = I8(advance)(padd, padx, n);
      if (!more_pad) /* start over if end reached */
        more_pad = DIM_UBOUND_G(padd, 0) - F90_DIM_LBOUND_G(padd, 0) + 1;
    } else
      __fort_abort("RESHAPE: not enough elements in SOURCE array");

    __DIST_INIT_SECTION(tod, 1, resd);
    I8(__fort_set_section)(tod, 1, resd, k + 1, resx[k], resx[k] + n - 1, 1);
    for (i = 1; i < F90_RANK_G(resd); ++i) {
      j = order[i];
      I8(__fort_set_single)(tod, resd, j + 1, resx[j], __SCALAR);
    }
    I8(__fort_finish_section)(tod);

    fromb += DIST_SCOFF_G(fromd) * F90_LEN_G(fromd);
    tob = resb + DIST_SCOFF_G(tod) * F90_LEN_G(tod);

    ch = I8(__fort_copy)(tob, fromb, tod, fromd, NULL);
    __fort_doit(ch);
    __fort_frechn(ch);

    more_res = I8(advance_permuted)(resd, resx, order, n);
  }
}

void ENTFTN(RESHAPECA, reshapeca)(DCHAR(resb),    /* result char base */
                                DCHAR(srcb),    /* source char base */
                                char *shpb,     /* shape base */
                                DCHAR(padb),    /* pad char base */
                                char *ordb,     /* order base */
                                F90_Desc *resd, /* result descriptor */
                                F90_Desc *srcd, /* source descriptor */
                                F90_Desc *shpd, /* shape descriptor */
                                F90_Desc *padd, /* pad descriptor */
                                F90_Desc *ordd  /* order descriptor */
                                DCLEN64(resb)     /* result char len */
                                DCLEN64(srcb)     /* source char len */
                                DCLEN64(padb))    /* pad char len */
{
  ENTFTN(RESHAPE, reshape)
  (CADR(resb), CADR(srcb), shpb, CADR(padb), ordb, resd, srcd, shpd, padd,
   ordd);
}
/* 32 bit CLEN version */
void ENTFTN(RESHAPEC, reshapec)(DCHAR(resb),    /* result char base */
                                DCHAR(srcb),    /* source char base */
                                char *shpb,     /* shape base */
                                DCHAR(padb),    /* pad char base */
                                char *ordb,     /* order base */
                                F90_Desc *resd, /* result descriptor */
                                F90_Desc *srcd, /* source descriptor */
                                F90_Desc *shpd, /* shape descriptor */
                                F90_Desc *padd, /* pad descriptor */
                                F90_Desc *ordd  /* order descriptor */
                                DCLEN(resb)     /* result char len */
                                DCLEN(srcb)     /* source char len */
                                DCLEN(padb))    /* pad char len */
{
  ENTFTN(RESHAPECA, reshapeca)(CADR(resb), CADR(srcb), shpb, CADR(padb), ordb,
         resd, srcd, shpd, padd, ordd, (__CLEN_T)CLEN(resb),
         (__CLEN_T)CLEN(srcb), (__CLEN_T)CLEN(padb));
}
