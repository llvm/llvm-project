/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "genlist.h"
#include "xfer.h"

/* 1toN */

struct chdr *__fort_chn_1toN(struct chdr *cp, /* previous structure in list */
                             int dnd,         /* destination cpu dimensions */
                             int dlow,   /* destination lowest numbered cpu */
                             int *dcnts, /* destination number of cpus */
                             int *dstrs, /* destination cpu stride */
                             int snd,    /* source cpu dimensions */
                             int slow,   /* source lowest numbered cpu */
                             int *scnts, /* source number of cpus */
                             int *sstrs) /* source cpu stride */
{
  int dnstrs[MAXDIMS];
  int dncnts[MAXDIMS];
  int snstrs[MAXDIMS];
  int sncnts[MAXDIMS];
  struct chdr *c;
  struct cgrp *sg; /* source group */
  struct cgrp *dg; /* destination group */
  struct cgrp *tg; /* transfer group */
  int si;
  int di;
  int ti;
  int n;
  int lcpu;
  int scpu; /* sending cpu */
  int tcnt;
  int copy;
  int i;

  /* sort destination strides and counts by ascending stride */

  for (i = 0; i < dnd; i++) {
    dnstrs[i] = dstrs[i];
    dncnts[i] = dcnts[i];
  }
  i = 0;
  while (i < (dnd - 1)) {
    if (dnstrs[i] > dnstrs[i + 1]) {
      n = dnstrs[i];
      dnstrs[i] = dnstrs[i + 1];
      dnstrs[i + 1] = n;
      n = dncnts[i];
      dncnts[i] = dncnts[i + 1];
      dncnts[i + 1] = n;
      if (i > 0) {
        i--;
        continue;
      }
    }
    i++;
  }

  /* sort source strides and counts by ascending stride */

  for (i = 0; i < snd; i++) {
    snstrs[i] = sstrs[i];
    sncnts[i] = scnts[i];
  }
  i = 0;
  while (i < (snd - 1)) {
    if (snstrs[i] > snstrs[i + 1]) {
      n = snstrs[i];
      snstrs[i] = snstrs[i + 1];
      snstrs[i + 1] = n;
      n = sncnts[i];
      sncnts[i] = sncnts[i + 1];
      sncnts[i + 1] = n;
      if (i > 0) {
        i--;
        continue;
      }
    }
    i++;
  }

  /* generate destination and source cpu lists */

  dg = __fort_genlist(dnd, dlow, dncnts, dnstrs);
  sg = __fort_genlist(snd, slow, sncnts, snstrs);

  /* allocate temporary area to hold cpus */

  tg = (struct cgrp *)__fort_malloc(sizeof(struct cgrp) +
                                   (dg->ncpus) * sizeof(int));
  lcpu = __fort_myprocnum();

  /* match up cpus for transfers */

  si = 0;
  di = 0;
  ti = 0;
  while ((di < dg->ncpus) && (si < sg->ncpus)) {
    if (dg->cpus[di] < sg->cpus[si]) {
      tg->cpus[di++] = sg->cpus[ti++];
      ti = (ti >= sg->ncpus ? 0 : ti);
    } else if (dg->cpus[di] > sg->cpus[si]) {
      si++;
    } else {
      tg->cpus[di++] = sg->cpus[si++];
    }
  }
  while (di < dg->ncpus) {
    tg->cpus[di++] = sg->cpus[ti++];
    ti = (ti >= sg->ncpus ? 0 : ti);
  }

  /* cpus are matched, tg contains senders corresponding to
  receivers in dg */

  __fort_free(sg);

  /* collect cpus in our group into tg, set sending cpu in scpu */

  copy = 0;
  tcnt = 0;
  for (di = 0; di < dg->ncpus; di++) {
    if ((tg->cpus[di] != lcpu) && (dg->cpus[di] != lcpu)) {
      continue; /* ignore these cpus */
    }
    if (dg->cpus[di] == tg->cpus[di]) {
      copy = 1; /* bcopy */
      continue;
    }
    scpu = tg->cpus[di]; /* our group */
    tg->cpus[tcnt++] = dg->cpus[di];
  }

  __fort_free(dg);

  /* allocate channel */

  c = __fort_allchn(cp, 1, 1, tcnt + 1);
  c->cn = 0;

  /* determine actions */

  if (tcnt == 1) {
    if (scpu == lcpu) {
      c->cp[c->cn].op = CPU_SEND;
      c->cp[c->cn].cpu = tg->cpus[0];
      c->cp[c->cn].sp = &(c->sp[0]);
    } else {
      c->cp[c->cn].op = CPU_RECV;
      c->cp[c->cn].cpu = scpu;
      c->cp[c->cn].rp = &(c->rp[0]);
    }
    c->cn++;
  } else if (tcnt > 1) {
    __fort_bcstchn(c, scpu, tcnt, tg->cpus);
  }
  if (copy) {
    c->cp[c->cn].op = CPU_COPY;
    c->cp[c->cn].sp = &(c->sp[0]);
    c->cp[c->cn].rp = &(c->rp[0]);
    c->cn++;
  }

  __fort_free(tg);
  return (c);
}
