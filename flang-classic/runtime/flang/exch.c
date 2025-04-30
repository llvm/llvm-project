/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* exch.c -- total exchange routine */

#include "stdioInterf.h"
#include "fioMacros.h"

#include "fort_vars.h"

static int _1 = 1;
static chdr *exch[16] = {NULL};
static int *rbuf, *sbuf, *vcounts, *vrbuf, *vsbuf;
static int rmax, smax, vrmax; /* * * * NOT THREAD SAFE * * * */

/* counts is an integer array of length 2**ceil(log2 np).  For each
   pair of processors {i,j}, counts[j] on processor i is exchanged
   with counts[i] on processor j, and the maximum count value is
   returned.  Ceil(log2 np) communication steps are required.  If the
   number of processors is not a power of two, some processors do
   double duty as "virtual processors". */

int
__fort_exchange_counts(int *counts)
{
  chdr *c;
  int cpu, lcpu, i, j, k, l, m, n, tcpus, vcpu, vme;
  int *tcpus_addr;

  lcpu = GET_DIST_LCPU;
  tcpus = GET_DIST_TCPUS;
  tcpus_addr = GET_DIST_TCPUS_ADDR;
  smax = counts[0]; /* initial maximum */
  if (tcpus == 1)
    return smax;

  for (i = tcpus; --i > 0;) {
    if (counts[i] > smax)
      smax = counts[i];
  }

#if defined(DEBUG)
  if (__fort_test & DEBUG_EXCH) {
    for (i = tcpus; i < __fort_np2; ++i)
      counts[i] = -lcpu;
    printf("%d exch counts", lcpu);
    for (i = 0; i < tcpus; ++i)
      printf(" %d", counts[i]);
    printf("\n");
  }
#endif

  m = __fort_np2 >> 1; /* message length, also most
                          significant bit of processor number */
  vme = lcpu ^ m;     /* virtual processor number */

  /* one-time setup... */

  if (exch[0] == NULL) {

    /* allocate buffers */

    n = __fort_np2;
    if (__fort_np2 != tcpus)
      n *= 3; /* more buffer needed for virtual processor */
    rbuf = (int *)__fort_gmalloc(n * sizeof(int));
    sbuf = rbuf + m;
    if (__fort_np2 != tcpus) {
      vrbuf = sbuf + m;
      vsbuf = vrbuf + m;
      vcounts = vsbuf + m;
    } else
      vsbuf = vrbuf = vcounts = NULL;

    /* initialize channels */

    for (n = 0, l = 1; l < tcpus; ++n, l <<= 1) {
      c = __fort_chn_1to1(NULL, 1, 0, tcpus_addr, &_1, 1, 0, tcpus_addr, &_1);
      cpu = lcpu ^ l;
      if (cpu >= tcpus)
        cpu ^= m;
      vcpu = vme ^ l;
      if (vcpu >= tcpus)
        vcpu ^= m;
#if defined(DEBUG)
      if (__fort_test & DEBUG_EXCH) {
        printf("%d exch l=%d cpu=%d(%d)\n", lcpu, l, lcpu ^ l, cpu);
        if (vme >= tcpus) {
          printf("%d exch l=%d cpu=%d(%d)\n", vme, l, vme ^ l, vcpu);
        }
      }
#endif
      __fort_sendl(c, cpu, &smax, 1, 1, __CINT, sizeof(int));
      __fort_sendl(c, cpu, sbuf, m, 1, __CINT, sizeof(int));
      if (vme >= tcpus) {
        __fort_sendl(c, vcpu, &smax, 1, 1, __CINT, sizeof(int));
        __fort_sendl(c, vcpu, vsbuf, m, 1, __CINT, sizeof(int));
        if (cpu == lcpu) {
          __fort_recvl(c, vcpu, &vrmax, 1, 1, __CINT, sizeof(int));
          __fort_recvl(c, vcpu, vrbuf, m, 1, __CINT, sizeof(int));
          __fort_recvl(c, cpu, &rmax, 1, 1, __CINT, sizeof(int));
          __fort_recvl(c, cpu, rbuf, m, 1, __CINT, sizeof(int));
        } else {
          __fort_recvl(c, cpu, &rmax, 1, 1, __CINT, sizeof(int));
          __fort_recvl(c, cpu, rbuf, m, 1, __CINT, sizeof(int));
          __fort_recvl(c, vcpu, &vrmax, 1, 1, __CINT, sizeof(int));
          __fort_recvl(c, vcpu, vrbuf, m, 1, __CINT, sizeof(int));
        }
      } else {
        __fort_recvl(c, cpu, &rmax, 1, 1, __CINT, sizeof(int));
        __fort_recvl(c, cpu, rbuf, m, 1, __CINT, sizeof(int));
      }
      __fort_chn_prune(c);
      exch[n] = c;
    }
  }

#if defined(DEBUG)
  if (vme >= tcpus) {
    for (i = 0; i < __fort_np2; ++i)
      vcounts[i] = -vme;
  }
#endif

  /* do the exchanges */

  for (n = 0, l = 1; l < tcpus; ++n, l <<= 1) {
    cpu = lcpu ^ l;
    vcpu = vme ^ l;
    for (i = cpu & l, j = 0; j < m; i += l) {
      for (k = j + l; j < k; ++i, ++j) {
        sbuf[j] = counts[i];
      }
    }
    if (vme >= tcpus) {
      for (i = vcpu & l, j = 0; j < m; i += l) {
        for (k = j + l; j < k; ++i, ++j) {
          vsbuf[j] = vcounts[i];
        }
      }
    }
    __fort_doit(exch[n]);
    if (rmax > smax)
      smax = rmax;
    for (i = cpu & l, j = 0; j < m; i += l) {
      for (k = j + l; j < k; ++i, ++j) {
        counts[i] = rbuf[j];
      }
    }
    if (vme >= tcpus) {
      if (vrmax > smax)
        smax = vrmax;
      for (i = vcpu & l, j = 0; j < m; i += l) {
        for (k = j + l; j < k; ++i, ++j) {
          vcounts[i] = vrbuf[j];
        }
      }
    }
#if defined(DEBUG)
    if (__fort_test & DEBUG_EXCH) {
      printf("%d exch l=%d cpu=%d rmax=%d", lcpu, l, cpu, rmax);
      for (i = 0; i < __fort_np2; ++i)
        printf(" %d", counts[i]);
      printf("\t>");
      for (i = 0; i < m; ++i)
        printf(" %d", sbuf[i]);
      printf("\t<");
      for (i = 0; i < m; ++i)
        printf(" %d", rbuf[i]);
      printf("\n");
      if (vme >= tcpus) {
        printf("%d exch l=%d cpu=%d rmax=%d", vme, l, vcpu, vrmax);
        for (i = 0; i < __fort_np2; ++i)
          printf(" %d", vcounts[i]);
        printf("\t>");
        for (i = 0; i < m; ++i)
          printf(" %d", vsbuf[i]);
        printf("\t<");
        for (i = 0; i < m; ++i)
          printf(" %d", vrbuf[i]);
        printf("\n");
      }
    }
#endif
  }
  return smax;
}
