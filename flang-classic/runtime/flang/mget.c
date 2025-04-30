/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* mget.c -- scalar broadcast routine */

#include "stdioInterf.h"
#include "fioMacros.h"

#define QTREE

#include "fort_vars.h"
extern void (*__fort_scalar_copy[__NTYPES])(void *rp, const void *sp, int size);

static int _1 = 1;

static void
mget_send(chdr **ch, int to, int me, char *rb, dtype kind, int size)
{
  int *tcpus_addr;

  tcpus_addr = GET_DIST_TCPUS_ADDR;
  if (to == me)
    return;
  if (*ch == NULL)
    *ch = __fort_chn_1to1(NULL, 1, 0, tcpus_addr, &_1, 1, 0, tcpus_addr, &_1);
  __fort_sendl(*ch, to, rb, 1, 1, kind, size);
#if defined(DEBUG)
  if (__fort_test & DEBUG_SCAL)
    printf("%d mget_scalar send to=%d\n", me, to);
#endif
}

static void
mget_recv(chdr **ch, int me, int from, char *rb, dtype kind, int size)
{
  int *tcpus_addr;

  tcpus_addr = GET_DIST_TCPUS_ADDR;
  if (from == me)
    return;
  if (*ch == NULL)
    *ch = __fort_chn_1to1(NULL, 1, 0, tcpus_addr, &_1, 1, 0, tcpus_addr, &_1);
  __fort_recvl(*ch, from, rb, 1, 1, kind, size);
#if defined(DEBUG)
  if (__fort_test & DEBUG_SCAL)
    printf("%d mget_scalar recv from=%d\n", me, from);
#endif
}

void ENTFTN(MGET_SCALAR, mget_scalar)(__INT_T *nb, ...)
/* ... = {void *rb, void *ab,F90_Desc *as, __INT_T *i1, ..., __INT_T *iR}* */
{
  va_list va;
  char *rb, *ab, *ap;
  DECL_HDR_PTRS(as);
  chdr *ch;
  int me, np, from, partner;
  __INT_T i, n, idx[MAXDIMS];

  me = GET_DIST_LCPU;
  np = GET_DIST_TCPUS;

  ch = NULL;

  va_start(va, nb);
  for (n = *nb; n > 0; --n) {
    rb = va_arg(va, char *);
    ab = va_arg(va, char *);
    as = va_arg(va, F90_Desc *);
#if defined(DEBUG)
    if (rb == NULL)
      __fort_abort("mget_scalar: invalid result address");
    if (ab == NULL)
      __fort_abort("mget_scalar: invalid array address");
    if (F90_TAG_G(as) != __DESC)
      __fort_abort("mget_scalar: invalid section descriptor");
#endif
    for (i = 0; i < F90_RANK_G(as); ++i)
      idx[i] = *va_arg(va, __INT_T *);

/* shortcut for replicated arrays */

    if (DIST_MAPPED_G(DIST_ALIGN_TARGET_G(as)) == 0)
    {
      ap = I8(__fort_local_address)(ab, as, idx);
      if (ap == NULL)
        __fort_abort("mget_scalar: localization error");
      __fort_scalar_copy[F90_KIND_G(as)](rb, ap, F90_LEN_G(as));
      continue;
    }

    from = I8(__fort_owner)(as, idx);

    if (from == me) {
      ap = I8(__fort_local_address)(ab, as, idx);
#if defined(DEBUG)
      if (ap == NULL)
        __fort_abort("mget_scalar: localization error");

      if (__fort_test & DEBUG_SCAL) {
        printf("%d mget_scalar bcst a", me);
        I8(__fort_show_index)(F90_RANK_G(as), idx);
        printf("@%x =", ap);
        __fort_show_scalar(ap, F90_KIND_G(as));
        printf("\n");
      }
#endif
      __fort_scalar_copy[F90_KIND_G(as)](rb, ap, F90_LEN_G(as));
    }

    if (from == me) {
      for (partner = 0; partner < np; ++partner)
        mget_send(&ch, partner, me, rb, F90_KIND_G(as), F90_LEN_G(as));
    } else
      mget_recv(&ch, me, from, rb, F90_KIND_G(as), F90_LEN_G(as));
  }
  va_end(va);

  if (ch) {
    __fort_chn_prune(ch);
    __fort_doit(ch);
    __fort_frechn(ch);
  }
}
