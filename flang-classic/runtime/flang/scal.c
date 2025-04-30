/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* scal.c -- scalar communication routines */

#include "stdioInterf.h"
#include "fioMacros.h"

#include "fort_vars.h"
extern void (*__fort_scalar_copy[__NTYPES])(void *rp, const void *sp, int size);

/* all processors get the value of the selected array element */

void I8(__fort_get_scalar)(void *dst, void *ab, F90_Desc *ad, __INT_T *idx)
{
  char *af, *src;
  int from;

  af = (char *)ab + DIST_SCOFF_G(ad) * F90_LEN_G(ad);

/* shortcut for local or fully replicated arrays */

  if (F90_FLAGS_G(ad) & __LOCAL || (DIST_MAPPED_G(DIST_ALIGN_TARGET_G(ad)) == 0))
  {
    src = I8(__fort_local_address)(af, ad, idx);

#if defined(DEBUG)
    if (src == NULL)
      __fort_abort("get_scalar: index out of bounds");
#endif

    __fort_scalar_copy[F90_KIND_G(ad)](dst, src, F90_LEN_G(ad));
    return;
  }

  from = I8(__fort_owner)(ad, idx);
  if (from == GET_DIST_LCPU) {
    src = I8(__fort_local_address)(af, ad, idx);
#if defined(DEBUG)
    if (src == NULL) {
      int i, j;
      printf("%d get_scalar: localization error\n", GET_DIST_LCPU);
      for (i = 0; i < F90_RANK_G(ad); ++i) {
        printf("%d dim %d: idx=%d lb=%d ub=%d ol=%d ou=%d tl=%d tu=%d ts=%d "
               "to=%d\n",
               GET_DIST_LCPU, i, idx[i], F90_DIM_LBOUND_G(ad, i),
               DIM_UBOUND_G(ad, i), DIST_DIM_OLB_G(ad, i), DIST_DIM_OUB_G(ad, i),
               DIST_DIM_TLB_G(ad, i), DIST_DIM_TUB_G(ad, i),
               DIST_DIM_TSTRIDE_G(ad, i), DIST_DIM_TOFFSET_G(ad, i));
        if (DFMT(ad, i + 1) == DFMT_GEN_BLOCK) {
          printf("%d dim %d: gen_block format: ", GET_DIST_LCPU, i);
          for (j = 0; j < DIST_DIM_PSHAPE_G(ad, i); ++j)
            printf("%d ", DIST_DIM_GEN_BLOCK_G(ad, i)[j]);
          __io_putchar('\n');
        }
      }
      __fort_abort((char *)0);
    }
    if (__fort_test & DEBUG_SCAL) {
      int i;
      printf("%d get_scalar bcst ", GET_DIST_LCPU);
      for (i = 0; i < F90_RANK_G(ad); ++i)
        printf("idx[%d]=%d ", i, idx[i]);
      __fort_show_scalar(src, F90_KIND_G(ad));
      printf("\n");
    }
#endif
    __fort_scalar_copy[F90_KIND_G(ad)](dst, src, F90_LEN_G(ad));
  }

  __fort_rbcstl(from, dst, 1, 1, F90_KIND_G(ad), F90_LEN_G(ad));

#if defined(DEBUG)
  if (__fort_test & DEBUG_SCAL) {
    int i;
    printf("%d get_scalar from=%d ", GET_DIST_LCPU, from);
    for (i = 0; i < F90_RANK_G(ad); ++i)
      printf("idx[%d]=%d ", i, idx[i]);
    __fort_show_scalar(dst, F90_KIND_G(ad));
    printf("\n");
  }
#endif
}

/* varargs version for fortran */

void ENTFTN(GET_SCALAR, get_scalar)(void *dst, void *ab, F90_Desc *ad, ...)
{
  va_list va;
  __INT_T i, idx[MAXDIMS];

#if defined(DEBUG)
  if (dst == NULL)
    __fort_abort("get_scalar: invalid dest address");
  if (ab == NULL)
    __fort_abort("get_scalar: invalid base address");
  if (ad == NULL || F90_TAG_G(ad) != __DESC)
    __fort_abort("get_scalar: invalid descriptor");
#endif

  va_start(va, ad);
  for (i = 0; i < F90_RANK_G(ad); i++)
    idx[i] = *va_arg(va, __INT_T *);
  va_end(va);

  I8(__fort_get_scalar)(dst, ab, ad, idx);
}

#ifndef DESC_I8
void ENTFTN(BCST_SCALAR, bcst_scalar)(void *dst, __INT_T *cpu, void *src,
                                      __INT_T *kind, __INT_T *len)
{
  if (GET_DIST_LCPU == *cpu && dst != src)
    __fort_scalar_copy[*kind](dst, src, *len);
  if (GET_DIST_TCPUS > 1)
    __fort_rbcstl(*cpu, dst, 1, 1, *kind, *len);
}
#endif

