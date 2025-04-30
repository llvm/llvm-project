/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "stdioInterf.h"
#include "fioMacros.h"

#define DEFENTS (512)

/* allocate channel structure */

struct chdr *
__fort_allchn(struct chdr *cp, int dents, int sents, int cpus)
{
  struct chdr *c;
  struct chdr *ct;

  /* allocate new channel, ents, and cpu structures */

  c = (chdr *)__fort_calloc(sizeof(chdr) +
                               (sizeof(struct ents) * (dents + sents)) +
                               (sizeof(struct ccpu) * cpus),
                           1);
  c->sp = (struct ents *)((char *)c + sizeof(chdr));
  c->sn = sents;
  c->rp =
      (struct ents *)((char *)c + sizeof(chdr) + (sizeof(struct ents) * sents));
  c->rn = dents;
  c->cp = (struct ccpu *)((char *)c + sizeof(chdr) +
                          (sizeof(struct ents) * (sents + dents)));
  c->cn = cpus;
  c->flags = CHDR_1INT | CHDR_1DBL;

  /* chain structure into list */

  if (cp != (struct chdr *)0) {
    ct = cp;
    while (ct->next != (struct chdr *)0) {
      ct = ct->next;
    }
    ct->next = c;
  }

  return (c);
}

/* free a list of channels */

void
__fort_frechn(struct chdr *c)
{
  struct chdr *d;
  int n;

  while (c != (struct chdr *)0) {
    for (n = 0; n < c->sn; n++) {
      if (c->sp[n].beg != (struct ent *)0) {
        __fort_free((char *)c->sp[n].beg);
      }
    }
    for (n = 0; n < c->rn; n++) {
      if (c->rp[n].beg != (struct ent *)0) {
        __fort_free((char *)c->rp[n].beg);
      }
    }
    for (n = 0; n < c->cn; n++) {
      if (c->cp[n].opt != (void *)0) {
        __fort_free((char *)c->cp[n].opt);
      }
    }
    d = c->next;
    __fort_free(c);
    c = d;
  }
}

/* reset channels */

void
__fort_rstchn(struct chdr *c)
{
  int n;

  while (c != (struct chdr *)0) {
    for (n = 0; n < c->sn; n++) {
      c->sp[n].avl = c->sp[n].beg;
    }
    for (n = 0; n < c->rn; n++) {
      c->rp[n].avl = c->rp[n].beg;
    }
    c = c->next;
  }
}

/* send data */

void __fort_sendl(struct chdr *c, /* channels */
                 int indx,       /* indx to send to */
                 void *adr,      /* adr of first data item */
                 long cnt,       /* number of data items */
                 long str,       /* stride between data items */
                 int typ,        /* data type (see pghpft.h) */
                 long ilen)      /* data item length */
{
  register struct ents *s;
  register int n;

#ifdef DEBUG
  if (indx >= c->sn) {
    __fort_abort("__fort_send: index >= cpu count\n");
  }
#endif
  s = &(c->sp[indx]);
  if (s->avl == s->end) {
    n = s->end - s->beg;
    if (s->beg == (struct ent *)0) {
      s->beg = (struct ent *)__fort_malloc((n + DEFENTS) * sizeof(struct ent));
    } else {
      s->beg = (struct ent *)__fort_realloc(s->beg,
                                           (n + DEFENTS) * sizeof(struct ent));
    }
    s->end = s->beg + n + DEFENTS;
    s->avl = s->beg + n;
  }
  s->avl->adr = adr;
  s->avl->cnt = cnt;
  s->avl->str = str;
  s->avl->typ = typ;
  s->avl->ilen = ilen;
  s->avl->len = cnt * ilen;
  if (c->flags & (CHDR_1INT | CHDR_1DBL)) {
    if (cnt != 1) {
      c->flags &= ~(CHDR_1INT | CHDR_1DBL);
    } else {
      if (ilen != sizeof(int)) {
        c->flags &= ~CHDR_1INT;
      }
      if (ilen != sizeof(double)) {
        c->flags &= ~CHDR_1DBL;
      }
    }
  }
  s->avl++;
}

/* recv data */

void __fort_recvl(struct chdr *c, /* channels */
                 int indx,       /* indx to receive from */
                 void *adr,      /* adr of first data item */
                 long cnt,       /* number of data items */
                 long str,       /* stride between data items */
                 int typ,        /* data type (see pghpft.h) */
                 long ilen)      /* data item length */
{
  register struct ents *r;
  register int n;

#ifdef DEBUG
  if (indx >= c->rn) {
    __fort_abort("__fort_recv: index >= cpu count\n");
  }
#endif
  r = &(c->rp[indx]);
  if (r->avl == r->end) {
    n = r->end - r->beg;
    if (r->beg == (struct ent *)0) {
      r->beg = (struct ent *)__fort_malloc((n + DEFENTS) * sizeof(struct ent));
    } else {
      r->beg = (struct ent *)__fort_realloc(r->beg,
                                           (n + DEFENTS) * sizeof(struct ent));
    }
    r->end = r->beg + n + DEFENTS;
    r->avl = r->beg + n;
  }
  r->avl->adr = adr;
  r->avl->cnt = cnt;
  r->avl->str = str;
  r->avl->typ = typ;
  r->avl->ilen = ilen;
  r->avl->len = cnt * ilen;
  if (c->flags & (CHDR_1INT | CHDR_1DBL)) {
    if (cnt != 1) {
      c->flags &= ~(CHDR_1INT | CHDR_1DBL);
    } else {
      if (ilen != sizeof(int)) {
        c->flags &= ~CHDR_1INT;
      }
      if (ilen != sizeof(double)) {
        c->flags &= ~CHDR_1DBL;
      }
    }
  }
  r->avl++;
}

/* chain channels together */

struct chdr *__fort_chain_em_up(struct chdr *list, /* current and new head */
                               struct chdr *c)    /* new tail */
{
  struct chdr *ct;

  if (list == (struct chdr *)0) {
    return (c);
  }
  ct = list;
  while (ct->next != (struct chdr *)0) {
    ct = ct->next;
  }
  ct->next = c;
  return (list);
}

/* change data addresses and type in list of channels */

void __fort_adjbase(struct chdr *c, /* list of channels */
                   char *bases,    /* send base address */
                   char *baser,    /* recv base address */
                   int typ,        /* data type */
                   long ilen)      /* data item length */
{
  struct ent *p;
  int n;
  long l;

  while (c != (struct chdr *)0) {
    if (~c->flags & CHDR_BASE) {
      __fort_abort("__fort_adjbase: setbase not called");
    }
    c->flags &= ~(CHDR_1INT | CHDR_1DBL);
    if ((c->bases != bases) || (c->typ != typ) || (c->ilen != ilen)) {
      if (c->ilen == ilen) {
        for (n = 0; n < c->sn; n++) {
          p = c->sp[n].beg;
          while (p < c->sp[n].avl) {
            l = p->adr - c->bases;
            p->adr = l + bases;
#ifdef DEBUG
            if (c->typ != p->typ)
              __fort_abort("__fort_adjbase: inconsistent send data types");
#endif
            p->typ = typ;
            p++;
          }
        }
      } else {
        for (n = 0; n < c->sn; n++) {
          p = c->sp[n].beg;
          while (p < c->sp[n].avl) {
            l = p->adr - c->bases;
            l /= c->ilen;
            l *= ilen;
            p->adr = l + bases;
#ifdef DEBUG
            if (c->typ != p->typ)
              __fort_abort("__fort_adjbase: inconsistent send data types");
#endif
            p->typ = typ;
            p->ilen = ilen;
            p->len = p->cnt * ilen;
            p++;
          }
        }
      }
      c->bases = bases;
    }
    if ((c->baser != baser) || (c->typ != typ) || (c->ilen != ilen)) {
      if (c->ilen == ilen) {
        for (n = 0; n < c->rn; n++) {
          p = c->rp[n].beg;
          while (p < c->rp[n].avl) {
            l = p->adr - c->baser;
            p->adr = l + baser;
#ifdef DEBUG
            if (c->typ != p->typ)
              __fort_abort("__fort_adjbase: inconsistent recv data types");
#endif
            p->typ = typ;
            p++;
          }
        }
      } else {
        for (n = 0; n < c->rn; n++) {
          p = c->rp[n].beg;
          while (p < c->rp[n].avl) {
            l = p->adr - c->baser;
            l /= c->ilen;
            l *= ilen;
            p->adr = l + baser;
#ifdef DEBUG
            if (c->typ != p->typ)
              __fort_abort("__fort_adjbase: inconsistent recv data types");
#endif
            p->typ = typ;
            p->ilen = ilen;
            p->len = p->cnt * ilen;
            p++;
          }
        }
      }
      c->baser = baser;
    }
    c->typ = typ;
    c->ilen = ilen;
    c = c->next;
  }
}

/* set bases addresses and data type (called once after __fort_chn_xxx) */

void __fort_setbase(chdr *c,     /* list of channels */
                   char *bases, /* send base address */
                   char *baser, /* recv base address */
                   int typ,     /* data type */
                   long ilen)   /* data item length */
{

  while (c != (struct chdr *)0) {
    c->flags |= CHDR_BASE;
    c->bases = bases;
    c->baser = baser;
    c->typ = typ;
    c->ilen = ilen;
    c = c->next;
  }
}

/*
 * keep track of the global buffer
 *
 * this routine should ONLY be used by routines called by __fort_doit and
 * then only very carefully.
 */

#define GBUFA 4096 /* global buffer size alignment */

static int gbufz;
static char *gbuf;

char *
__fort_getgbuf(long len)
{
  if (len <= gbufz) {
    return (gbuf);
  }
  if (gbuf != (char *)0) {
    __fort_gfree(gbuf);
  }
  len = (len + GBUFA - 1) & ~(GBUFA - 1);
  gbuf = __fort_gmalloc(len);
  gbufz = len;
  return (gbuf);
}

/* prune unsed entries and do special optimizations */

#define MINSHIFT 1024 /* smaller send values may be send */

void
__fort_chn_prune(struct chdr *c)
{
  int n;
  int m;

  while (c != (struct chdr *)0) {

    /* prune unused entries */

    m = 0;
    for (n = 0; n < c->cn; n++) {
      if (n != m) {
        c->cp[m] = c->cp[n];
      }
      if (((c->cp[n].sp != (struct ents *)0) &&
           (c->cp[n].sp->avl > c->cp[n].sp->beg)) ||
          ((c->cp[n].rp != (struct ents *)0) &&
           (c->cp[n].rp->avl > c->cp[n].rp->beg))) {
        m++;
      }
    }
    c->cn = m;

    c = c->next;
  }
}
