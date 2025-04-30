/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* common fio routines */

#include "fioMacros.h"
#include "descRW.h"
#include "global.h"

/* loop over local blocks in each array dimension, reducing to
   one-dimensional strided transfers */

void I8(__fortio_loop)(fio_parm *z, /* parameter struct */
                      int dim)       /* dimension */
{
  DECL_HDR_PTRS(ac);
  DECL_DIM_PTRS(acd);

  __INT_T n;

  ac = z->ac;
  SET_DIM_PTRS(acd, ac, dim - 1);

  z->index[dim - 1] = F90_DPTR_LBOUND_G(acd);
  n = F90_DPTR_EXTENT_G(acd); /* extent */
  if (n <= 0)
    return;

/* handle unmapped dimension */

  if (dim > 1) {
    while (--n >= 0) {
      I8(__fortio_loop)(z, dim - 1);
      z->index[dim - 1]++;
    }
  } else {
    z->cnt = n;
    z->str = F90_DPTR_SSTRIDE_G(acd) * F90_DPTR_LSTRIDE_G(acd);
    z->fio_rw(z);
  }
  return;
}

/*
 * routines to broadcast status and io status
 */

#if !defined(DESC_I8)

static __INT_T fio_bitv;
static __INT_T *fio_iostat;

/* init bitv and iostat */

void __fort_status_init(__INT_T *bitv, __INT_T *iostat)
{
  fio_bitv = *bitv;
  fio_iostat = iostat;
}

/* status and iostat broadcast from i/o processor to others */

#undef DIST_STATUS_BCST
__INT_T
__fort_status_bcst(__INT_T s)
{
  __INT_T msg[2];
  int ioproc, lcpu;

  if (((fio_bitv &
        (FIO_BITV_IOSTAT | FIO_BITV_ERR | FIO_BITV_EOF | FIO_BITV_EOR)) == 0) ||
      LOCAL_MODE) {
    return (s);
  }
  ioproc = GET_DIST_IOPROC;
  lcpu = GET_DIST_LCPU;
  if (lcpu == ioproc) {/* i/o proc sets up data */
    msg[0] = s;
    if (fio_bitv & FIO_BITV_IOSTAT) {
      msg[1] = *fio_iostat;
    } else {
      msg[1] = 0;
    }
  }
  __fort_rbcst(ioproc, msg, 2, 1, __INT);
  if (lcpu != ioproc) {/* others get data */
    if (fio_bitv & FIO_BITV_IOSTAT) {
      *fio_iostat = msg[1];
    }
  }
  return (msg[0]);
}

/* initialize i/o condition handling bit vector and iostat address */

void
__fortio_stat_init(__INT_T *bitv, __INT_T *iostat)
{
  fio_bitv = *bitv;
  fio_iostat = iostat;
}

/* broadcast status and iostat from i/o processor to others */

int
__fortio_stat_bcst(int *stat)
{
  __INT_T msg[2];
  int ioproc, lcpu;

  if (!LOCAL_MODE &&
      fio_bitv &
          (FIO_BITV_IOSTAT | FIO_BITV_ERR | FIO_BITV_EOF | FIO_BITV_EOR)) {
    ioproc = GET_DIST_IOPROC;
    lcpu = GET_DIST_LCPU;
    if (lcpu == ioproc) {
      msg[0] = *stat;
      msg[1] = (fio_bitv & FIO_BITV_IOSTAT) ? *fio_iostat : 0;
    }
    __fort_rbcstl(ioproc, msg, 2, 1, __INT, sizeof(__INT_T));
    if (lcpu != ioproc) {
      *stat = msg[0];
      if (fio_bitv & FIO_BITV_IOSTAT)
        *fio_iostat = msg[1];
    }
  }
  return *stat;
}

#endif
