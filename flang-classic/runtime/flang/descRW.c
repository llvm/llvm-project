/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* hpf i/o array handling routines */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "descRW.h"

char *__fort_getgbuf(long len);

#include "fort_vars.h"
#ifdef DESC_I8
#define __fort_owner_i8(a, b) 0
#define __fort_next_owner_i8(a, b, c, d) -1
#else
#define __fort_owner(a, b) 0
#define __fort_next_owner(a, b, c, d) -1
#endif
#define __fortio_stat_bcst(a) (*(a))

static void I8(__io_read)(fio_parm *z)
{
  DECL_HDR_PTRS(ac);
  char *adr, *buf;
  int i, ioproc, owner, pc[MAXDIMS], str;

  ac = z->ac;
  adr = I8(__fort_local_address)(z->ab, ac, z->index);
  str = z->str;
#if defined(DEBUG)
  if (__fort_test & DEBUG_HFIO) {
    printf("%d __io_read index=", GET_DIST_LCPU);
    I8(__fort_show_index)(F90_RANK_G(ac), z->index);
    printf(" cnt=%d str=%d adr=%x\n", z->cnt, str, adr);
  }
#endif
  if (LOCAL_MODE) {
    if (!z->stat)
      z->stat = z->f90io_rw(F90_KIND_G(ac), z->cnt, z->str * F90_LEN_G(ac), adr,
                            F90_LEN_G(ac));
    return;
  }
  ioproc = GET_DIST_IOPROC;
  buf = __fort_getgbuf(z->cnt * F90_LEN_G(ac));
  if (GET_DIST_LCPU == ioproc) {
    if (adr == NULL) {
      adr = buf;
      str = 1;
    }
    if (!z->stat)
      z->stat = z->f90io_rw(F90_KIND_G(ac), z->cnt, str * F90_LEN_G(ac), adr,
                            F90_LEN_G(ac));
    owner = I8(__fort_owner)(ac, z->index);
    for (i = z->repl.ndim; --i >= 0;)
      pc[i] = 0;
    while (owner >= 0) {
      if (owner != GET_DIST_LCPU)
        __fort_rsendl(owner, adr, z->cnt, str, F90_KIND_G(ac), F90_LEN_G(ac));
      owner = I8(__fort_next_owner)(ac, &z->repl, pc, owner);
    }
  } else if (adr != NULL)
    __fort_rrecvl(ioproc, adr, z->cnt, str, F90_KIND_G(ac), F90_LEN_G(ac));
}

static void I8(__io_write)(fio_parm *z)
{
  DECL_HDR_PTRS(ac);
  char *adr;
  int str;

  ac = z->ac;
  adr = I8(__fort_local_address)(z->ab, ac, z->index);
  str = z->str;
#if defined(DEBUG)
  if (__fort_test & DEBUG_HFIO) {
    printf("%d __io_write index=", GET_DIST_LCPU);
    I8(__fort_show_index)(F90_RANK_G(ac), z->index);
    printf(" cnt=%d str=%d adr=%x\n", z->cnt, str, adr);
  }
#endif
  if (!z->stat)
    z->stat = z->f90io_rw(F90_KIND_G(ac), z->cnt, str * F90_LEN_G(ac), adr,
                          F90_LEN_G(ac));
}

int I8(__fortio_main)(char *ab,             /* base address */
                      F90_Desc *ac,         /* array descriptor */
                      int rw,               /* 0 => read, 1 => write */
                      f90io_rw_fn f90io_rw) /* f90io function */
{
  int ioproc, size_of_kind;
  fio_parm z;

  z.stat = 0;
  if (F90_TAG_G(ac) != __DESC) { /* scalar */
    dtype kind = TYPEKIND(ac);
#if defined(DEBUG)
    if (kind == __STR || kind == __DERIVED)
      __fort_abort("__fortio_main: character or derived type not handled");
#endif
    ioproc = GET_DIST_IOPROC;
    size_of_kind = GET_DIST_SIZE_OF(kind);
    if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
      z.stat = f90io_rw(kind, 1, 1, ab, size_of_kind);
    if (!rw && !LOCAL_MODE) /* if global read... */
      __fort_rbcstl(ioproc, ab, 1, 1, kind, size_of_kind);
    return __fortio_stat_bcst(&z.stat);
  }
  if (F90_GSIZE_G(ac) <= 0)
    return 0; /* zero-size array */
  z.ab = ab + DIST_SCOFF_G(ac) * F90_LEN_G(ac);
  z.ac = ac;
  z.f90io_rw = f90io_rw;
  z.fio_rw = rw ? I8(__io_write) : I8(__io_read);
  if (!rw && !LOCAL_MODE) /* if global read... */
    I8(__fort_describe_replication)(ac, &z.repl);
  if (F90_RANK_G(ac) > 0)
    I8(__fortio_loop)(&z, F90_RANK_G(ac));
  else {
    z.cnt = 1;
    z.str = 1;
    z.fio_rw(&z);
  }
  return __fortio_stat_bcst(&z.stat);
}
