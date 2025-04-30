/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "xfer.h"

/* merge ent structures */

static void
merge_ent(struct ents *a, struct ents *b, struct ents *c)
{
  a->beg = (struct ent *)__fort_realloc((char *)b->beg,
                                       ((b->avl - b->beg) + (c->avl - c->beg)) *
                                           sizeof(struct ent));
  a->avl = a->beg + (b->avl - b->beg);
  a->end = a->avl + (c->avl - c->beg);
  __fort_bcopy((char *)a->avl, (char *)c->beg,
              (c->avl - c->beg) * sizeof(struct ent));
  a->avl = a->end;
  __fort_free((char *)c->beg);
}

/* merge channel structures */

chdr *
__fort_comm_merge(chdr *b, chdr *c)
{
  chdr *a;              /* new channel */
  int acpi, bcpi, ccpi; /* channel cpu entry index */
  int aspi, arpi;       /* next avail ents index */
  int nmax;             /* next higher power of 2 cpus */
  int cpu;              /* next cpu to check */
  int lcpu;
  int r; /* cpu index */
  int tcpus;

  /* check for channels that shouldn't be merged */

  if ((b->next != (chdr *)0) || (c->next != (chdr *)0)) {
    __fort_abort("__fort_comm_merge: cannot merge linked channels");
  }
  if ((b->typ != c->typ) || (b->typ != c->typ)) {
    __fort_abort("__fort_comm_merge: differing base types");
  }
  for (cpu = 0; cpu < b->cn; cpu++) {
    if (b->cp[cpu].opt != (char *)0) { /* free opt area */
      __fort_free(b->cp[cpu].opt);
    }
  }
  for (cpu = 0; cpu < c->cn; cpu++) {
    if (c->cp[cpu].opt != (char *)0) { /* free opt area */
      __fort_free(c->cp[cpu].opt);
    }
  }

  /* allocate new channel, ents, and cpu structures */

  a = __fort_allchn((chdr *)0, b->rn + c->rn, b->sn + c->sn, b->cn + c->cn);
  if ((b->bases != c->bases) || (b->baser != c->baser) || (b->typ != c->typ)) {
    a->bases = (char *)0;
    a->baser = (char *)0;
    a->typ = 0;
    a->ilen = 0;
    a->flags = 0;
  } else {
    a->bases = b->bases;
    a->baser = b->baser;
    a->typ = b->typ;
    a->ilen = b->ilen;
    a->flags = b->flags & CHDR_BASE;
  }

  /* find next higher power than number of cpus */

  nmax = 1;
  tcpus = GET_DIST_TCPUS;
  while (tcpus > nmax) {
    nmax = nmax << 1;
  }

  /* merge structures */

  acpi = 0;
  aspi = 0;
  arpi = 0;
  lcpu = GET_DIST_LCPU;
  for (r = 0; r < nmax; r++) {

    cpu = r ^ lcpu; /* next cpu to check */
    bcpi = 0;       /* channel b cpu index */
    ccpi = 0;       /* channel c cpu index */

    while ((bcpi < b->cn) || (ccpi < c->cn)) {
      while (bcpi < b->cn) { /* check channel b */
        if (b->cp[bcpi].cpu == cpu) {
          break;
        }
        bcpi++;
      }
      while (ccpi < c->cn) { /* check channel c */
        if (c->cp[ccpi].cpu == cpu) {
          break;
        }
        ccpi++;
      }

      /* both channels access this cpu? */

      if (((bcpi < b->cn) && (b->cp[bcpi].cpu == cpu)) &&
          ((ccpi < c->cn) && (c->cp[ccpi].cpu == cpu))) {

        a->cp[acpi].cpu = b->cp[bcpi].cpu;

        /* same op? */

        if (b->cp[bcpi].op == c->cp[ccpi].op) {
          a->cp[acpi].op = b->cp[bcpi].op;
          if (b->cp[bcpi].op == CPU_COPY) {
            a->cp[acpi].rp = &(a->rp[arpi++]);
            a->cp[acpi].sp = &(a->sp[aspi++]);
            merge_ent(a->sp, b->sp, c->sp);
            merge_ent(a->rp, b->rp, c->rp);
          } else if (b->cp[bcpi].op == CPU_RECV) {
            a->cp[acpi].rp = &(a->rp[arpi++]);
            merge_ent(a->rp, b->rp, c->rp);
          } else if (b->cp[bcpi].op == CPU_SEND) {
            a->cp[acpi].sp = &(a->sp[aspi++]);
            merge_ent(a->sp, b->sp, c->sp);
          }
          acpi++;
          bcpi++;
          ccpi++;

          /* channel b copy? */

        } else if (b->cp[bcpi].op == CPU_COPY) {
          a->cp[acpi].op = b->cp[bcpi].op;
          a->cp[acpi].rp = &(a->rp[arpi++]);
          a->cp[acpi].sp = &(a->sp[aspi++]);
          *(a->cp[acpi].sp) = *(b->cp[bcpi].sp);
          *(a->cp[acpi].rp) = *(b->cp[bcpi].rp);
          acpi++;
          bcpi++;

          /* channel c copy? */

        } else if (c->cp[ccpi].op == CPU_COPY) {
          a->cp[acpi].op = c->cp[ccpi].op;
          a->cp[acpi].rp = &(a->rp[arpi++]);
          a->cp[acpi].sp = &(a->sp[aspi++]);
          *(a->cp[acpi].sp) = *(c->cp[ccpi].sp);
          *(a->cp[acpi].rp) = *(c->cp[ccpi].rp);
          acpi++;
          ccpi++;

          /* do send first */

        } else if (cpu > lcpu) {
          if (b->cp[bcpi].op == CPU_SEND) {
            a->cp[acpi].op = b->cp[bcpi].op;
            a->cp[acpi].sp = &(a->sp[aspi++]);
            *(a->cp[acpi].sp) = *(b->cp[bcpi].sp);
            acpi++;
            bcpi++;
          } else if (c->cp[ccpi].op == CPU_SEND) {
            a->cp[acpi].op = c->cp[ccpi].op;
            a->cp[acpi].sp = &(a->sp[aspi++]);
            *(a->cp[acpi].sp) = *(c->cp[ccpi].sp);
            acpi++;
            ccpi++;
          }

          /* do recv first */

        } else {
          a->cp[acpi].rp = &(a->rp[arpi++]);
          if (b->cp[bcpi].op == CPU_RECV) {
            a->cp[acpi].op = b->cp[bcpi].op;
            a->cp[acpi].rp = &(a->rp[arpi++]);
            *(a->cp[acpi].rp) = *(b->cp[bcpi].rp);
            acpi++;
            bcpi++;
          } else if (c->cp[ccpi].op == CPU_RECV) {
            a->cp[acpi].op = c->cp[ccpi].op;
            a->cp[acpi].rp = &(a->rp[arpi++]);
            *(a->cp[acpi].rp) = *(c->cp[ccpi].rp);
            acpi++;
            ccpi++;
          }
        }

        /* only channel b op */

      } else if (((bcpi < b->cn) && (b->cp[bcpi].cpu == cpu))) {
        a->cp[acpi].cpu = b->cp[bcpi].cpu;
        a->cp[acpi].op = b->cp[bcpi].op;
        if (a->cp[acpi].op == CPU_RECV) {
          a->cp[acpi].rp = &(a->rp[arpi++]);
          *(a->cp[acpi].rp) = *(b->cp[bcpi].rp);
        } else if (a->cp[acpi].op == CPU_SEND) {
          a->cp[acpi].sp = &(a->sp[aspi++]);
          *(a->cp[acpi].sp) = *(b->cp[bcpi].sp);
        } else if (a->cp[acpi].op == CPU_COPY) {
          a->cp[acpi].rp = &(a->rp[arpi++]);
          *(a->cp[acpi].rp) = *(b->cp[bcpi].rp);
          a->cp[acpi].sp = &(a->sp[aspi++]);
          *(a->cp[acpi].sp) = *(b->cp[bcpi].sp);
        }
        acpi++;
        bcpi++;

        /* only channel c op */

      } else if (((ccpi < c->cn) && (c->cp[ccpi].cpu == cpu))) {
        a->cp[acpi].cpu = c->cp[ccpi].cpu;
        a->cp[acpi].op = c->cp[ccpi].op;
        if (a->cp[acpi].op == CPU_RECV) {
          a->cp[acpi].rp = &(a->rp[arpi++]);
          *(a->cp[acpi].rp) = *(c->cp[ccpi].rp);
        } else if (a->cp[acpi].op == CPU_SEND) {
          a->cp[acpi].sp = &(a->sp[aspi++]);
          *(a->cp[acpi].sp) = *(c->cp[ccpi].sp);
        } else if (a->cp[acpi].op == CPU_COPY) {
          a->cp[acpi].rp = &(a->rp[arpi++]);
          *(a->cp[acpi].rp) = *(c->cp[ccpi].rp);
          a->cp[acpi].sp = &(a->sp[aspi++]);
          *(a->cp[acpi].sp) = *(c->cp[ccpi].sp);
        }
        acpi++;
        ccpi++;
      }
    }
  }

  a->cn = acpi;

  /* free original channels */

  __fort_free((char *)b);
  __fort_free((char *)c);

  return (a);
}
