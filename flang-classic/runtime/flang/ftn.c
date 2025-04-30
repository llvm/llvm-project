/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

#include "stdioInterf.h"
#include "fioMacros.h"

/*
 * Translation from user local communication types to internal types.
 * These types must match the types in pglocal.h.
 */

static int ltypes[] = {
    __INT1,   /* integer*1 */
    __INT2,   /* integer*2 */
    __INT4,   /* integer*4 */
    __INT8,   /* integer*8 */
    __LOG1,   /* logical*1 */
    __LOG2,   /* logical*2 */
    __LOG4,   /* logical*4 */
    __LOG8,   /* logical*8 */
    __REAL4,  /* real*4 */
    __REAL8,  /* real*8 */
    __REAL16, /* real*16 */
    __CPLX8,  /* complex*8 (2x real*4) */
    __CPLX16, /* complex*16 (2x real*8) */
    __CPLX32, /* complex*32 (2x real*16) */
    __WORD4,  /* typeless */
    __WORD8,  /* double typeless */
    __WORD16  /* quad typeless */
};

/* local send routine (C interface) */

void
__fort_csend(int cpu, void *adr, int cnt, int str, int typ)
{
  if (cpu == GET_DIST_LCPU) {
    __fort_abort("__fort_csend: cannot send to self");
  }
  __fort_rsend(cpu, adr, cnt, str, ltypes[typ]);
}

/* local receive routine (C interface) */

void
__fort_crecv(int cpu, void *adr, int cnt, int str, int typ)
{
  if (cpu == GET_DIST_LCPU) {
    __fort_abort("__fort_crecv: cannot receive from self");
  }
  __fort_rrecv(cpu, adr, cnt, str, ltypes[typ]);
}

/* local send routines (Fortran interface) */

void ENTFTN(CSEND, csend)(__INT_T *cpu, void *adr, __INT_T *cnt, __INT_T *str,
                          __INT_T *typ)
{
  if (*cpu == GET_DIST_LCPU) {
    __fort_abort("__fort_csend: cannot send to self");
  }
  __fort_rsend(*cpu, adr, *cnt, *str, ltypes[*typ]);
}

void ENTFTN(CSENDCHARA, csendchara)(__INT_T *cpu, DCHAR(buf), __INT_T *cnt,
                                  __INT_T *str DCLEN64(buf))
{
  char *adr;
  __CLEN_T n, len, skip;

  if (*cpu == GET_DIST_LCPU) {
    __fort_abort("__fort_csendchar: cannot send to self");
  }
  adr = CADR(buf);
  len = CLEN(buf);
  skip = len * (*str);
  for (n = *cnt; n > 0; --n) {
    __fort_rsend(*cpu, adr, len, 1, __STR);
    adr += skip;
  }
}

/* 32 bit CLEN version */
void ENTFTN(CSENDCHAR, csendchar)(__INT_T *cpu, DCHAR(buf), __INT_T *cnt,
                                  __INT_T *str DCLEN(buf))
{
  ENTFTN(CSENDCHARA, csendchara)(cpu, CADR(buf), cnt, str, (__CLEN_T)CLEN(buf));
}

/* local receive routines (Fortran interface) */

void ENTFTN(CRECV, crecv)(__INT_T *cpu, void *adr, __INT_T *cnt, __INT_T *str,
                          __INT_T *typ)
{
  if (*cpu == GET_DIST_LCPU) {
    __fort_abort("__fort_crecv: cannot receive from self");
  }
  __fort_rrecv(*cpu, adr, *cnt, *str, ltypes[*typ]);
}

void ENTFTN(CRECVCHARA, crecvchara)(__INT_T *cpu, DCHAR(buf), __INT_T *cnt,
                                  __INT_T *str DCLEN64(buf))
{
  char *adr;
  __CLEN_T n, len, skip;

  if (*cpu == GET_DIST_LCPU) {
    __fort_abort("__fort_crecvchar: cannot receive from self");
  }
  adr = CADR(buf);
  len = CLEN(buf);
  skip = len * (*str);
  for (n = *cnt; n > 0; --n) {
    __fort_rrecv(*cpu, adr, len, 1, __STR);
    adr += skip;
  }
}

/* 32 bit CLEN version */
void ENTFTN(CRECVCHAR, crecvchar)(__INT_T *cpu, DCHAR(buf), __INT_T *cnt,
                                  __INT_T *str DCLEN(buf))
{
  ENTFTN(CRECVCHARA, crecvchara)(cpu, CADR(buf), cnt, str, (__CLEN_T)CLEN(buf));
}
