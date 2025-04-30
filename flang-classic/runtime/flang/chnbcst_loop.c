/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "stdioInterf.h"
#include "fioMacros.h"

extern struct cgrp *__fort_genlist();
extern struct chdr *__fort_allchn();

/* compute the recv/send order for a binary broadcast */

void __fort_bcstchn(struct chdr *c,
                   int scpu,  /* sending cpu */
                   int ncpus,/* number of receiving cpus */
                   int *cpus)
{
  int lcpu;
  int n;

  lcpu = __fort_myprocnum();

  if (lcpu != scpu) {
    c->cp[c->cn].op = CPU_RECV;
    c->cp[c->cn].cpu = scpu;
    c->cp[c->cn].rp = &(c->rp[0]);
    c->cn++;
    return;
  }

  for (n = 0; n < ncpus; n++) {
    c->cp[c->cn].op = CPU_SEND;
    c->cp[c->cn].cpu = cpus[n];
    c->cp[c->cn].sp = &(c->sp[0]);
    c->cn++;
  }
}
