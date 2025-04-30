/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

typedef int (*f90io_rw_fn)(int, long, int, char *, __CLEN_T);

typedef struct fio_parm fio_parm;
struct fio_parm {
  char *ab;          /* array base address */
  DECL_HDR_PTRS(ac); /* array descriptor */

  f90io_rw_fn f90io_rw; /* f90io read/write function ptr */

  int (*pario_rw)(int fd, char *adr, int cnt, int str, int typ, int ilen,
                  int own); /* pario read/write function ptr */

  void (*fio_rw)(fio_parm *z); /* fio read/write function ptr */

  __INT_T index[MAXDIMS]; /* first element index */
  int cnt;                /* element count */
  int str;                /* element stride */
  int stat;               /* f90io function return status */
  int tcnt;               /* pario total transfer count */
  int fd;                 /* pario file descriptor */

  repl_t repl; /* replication descriptor */
};

int I8(__fortio_main)(char *ab,              /* base address */
                      F90_Desc *ac,          /* array descriptor */
                      int rw,                /* 0 => read, 1 => write */
                      f90io_rw_fn f90io_rw); /* f90io function */

void I8(__fortio_loop)(fio_parm *z, /* parameter struct */
                       int dim);    /* loop dimension */
