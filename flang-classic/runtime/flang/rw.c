/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "stdioInterf.h"
#include "fioMacros.h"

/* receive single data item */

void
__fort_rrecvl(int cpu, void *adr, long cnt, long str, int typ, long ilen)
{
  struct ents s;
  struct ent e;

  if (cnt <= 0) {
    return;
  }
  s.beg = &e;
  s.avl = (&e) + 1;
  s.end = s.avl;
  e.adr = adr;
  e.str = (str == 0 ? 1 : str);
  e.cnt = cnt;
  e.typ = typ;
  e.ilen = ilen;
  e.len = cnt * e.ilen;
  __fort_erecv(cpu, &s);
}

/* send a single data item */

void
__fort_rsendl(int cpu, void *adr, long cnt, long str, int typ, long ilen)
{
  struct ents s;
  struct ent e;

  if (cnt <= 0) {
    return;
  }
  s.beg = &e;
  s.avl = (&e) + 1;
  s.end = s.avl;
  e.adr = adr;
  e.str = (str == 0 ? 1 : str);
  e.cnt = cnt;
  e.typ = typ;
  e.ilen = ilen;
  e.len = cnt * e.ilen;
  __fort_esend(cpu, &s);
}

/* receive single data item */

void
__fort_rrecv(int cpu, void *adr, long cnt, long str, int typ)
{
  __fort_rrecvl(cpu, adr, cnt, str, typ, GET_DIST_SIZE_OF(typ));
}

/* send a single data item */

void
__fort_rsend(int cpu, void *adr, long cnt, long str, int typ)
{
  __fort_rsendl(cpu, adr, cnt, str, typ, GET_DIST_SIZE_OF(typ));
}

/* simple broadcast from "src" processor to other processors */

void
__fort_rbcstl(int src, void *adr, long cnt, long str, int typ, long ilen)
{
  register int in; /* increment or decrement for other cpu */
  register int me; /* this cpu's adjusted number */
  register int cp; /* other cpu's real number */
  struct ents s;
  struct ent e;
  int lcpu, tcpus;

  if (cnt <= 0) {
    return;
  }
  s.beg = &e;
  s.avl = (&e) + 1;
  s.end = s.avl;
  lcpu = GET_DIST_LCPU;
  e.adr = adr;
  e.str = (str == 0 ? 1 : str);
  e.cnt = cnt;
  e.typ = typ;
  e.ilen = ilen;
  e.len = cnt * e.ilen;

  in = 1;
  tcpus = GET_DIST_TCPUS;
  while (in < tcpus) {
    in <<= 1;
  }
  in >>= 1;

  me = lcpu - src;
  me = (me < 0 ? me + tcpus : me);
  if (me != 0) {
    while (((in - 1) & me) != 0) {
      in >>= 1;
    }
    cp = (me - in) + src;
    cp = (cp >= tcpus ? cp - tcpus : cp);
    __fort_erecv(cp, &s);
    in >>= 1;
  }
  while (in > 0) {
    if ((me + in) < tcpus) {
      cp = (me + in) + src;
      cp = (cp >= tcpus ? cp - tcpus : cp);
      __fort_esend(cp, &s);
    }
    in >>= 1;
  }
}

/* simple broadcast from "src" processor to other processors */

void
__fort_rbcst(int src, void *adr, long cnt, long str, int typ)
{
  __fort_rbcstl(src, adr, cnt, str, typ, GET_DIST_SIZE_OF(typ));
}
