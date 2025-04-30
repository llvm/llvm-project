/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Implements Fortran CLOSE statement.
 */

#include <errno.h>
#include "global.h"
#if !defined(_WIN64)
#include <unistd.h>
#endif
#include "stdioInterf.h"

#if defined(_WIN64)
#include <io.h> // for _access, _unlink
#define unlink _unlink
#define access _access
#endif

/* ------------------------------------------------------------------ */

/** brief Fortran file close
 *
 * \param f 0 (use default stream)
 * \param flag FIO_KEEP, FIO_DELETE
 */
int
__fortio_close(FIO_FCB *f, int flag) 
{

  /* append carriage return (maybe) */

  if (f->nonadvance) {
    f->nonadvance = FALSE;
#if defined(_WIN64)
    if (__io_binary_mode(f->fp))
      __io_fputc('\r', f->fp);
#endif
    __io_fputc('\n', f->fp);
    if (__io_ferror(f->fp))
      return __io_errno();
  }

  if (!f->stdunit) {
    if (__io_fclose(f->fp) != 0) {
      return __fortio_error(__io_errno());
    }
    if (flag == 0 && f->dispose == FIO_DELETE)
      flag = FIO_DELETE;

    /* (note SCRATCH files are unlinked when they are opened... ) */
    if (flag == FIO_DELETE && f->status != FIO_SCRATCH) {
      if (__fort_access(f->name, 2)) /* check write permission */
        __fortio_error(FIO_EREADONLY);
      else
        __fort_unlink(f->name);
    }
#if defined(_WIN64)
    else if (f->status == FIO_SCRATCH)
      unlink(f->name);
#endif
  } else { /* stdin, stdout, stderr - just flush */
#if defined(TARGET_OSX)
    if (f->unit != 5 && f->unit != -5)
#endif
      if (__io_fflush(f->fp) != 0)
        return __fortio_error(__io_errno());
  }

  __fortio_free_fcb(f); /*  free up FCB for later use  */

  return 0;
}

static int
_f90io_close(__INT_T *unit, __INT_T *bitv, __INT_T *iostat, char *status,
            __CLEN_T status_siz)
{
  int status_flag;
  FIO_FCB *f;

  __fortio_errinit03(*unit, *bitv, iostat, "CLOSE");
  if (ILLEGAL_UNIT(*unit))
    return __fortio_error(FIO_EUNIT); /* illegal unit number */

  /* find the FIO_FCB struct, else return */

  f = __fortio_find_unit(*unit);
  if (f == NULL)
    return 0; /* no error occurred */

  /*  check for legal status parameter and assign status flag:  */

  status_flag = 0;
  if (status != NULL) {
    if (__fortio_eq_str(status, status_siz, "DELETE")) {
      if (f->acc == FIO_READ) /* cannot delete readonly file */
        return __fortio_error(FIO_EREADONLY);
      status_flag = FIO_DELETE;
    } else if (__fortio_eq_str(status, status_siz, "KEEP") ||
               __fortio_eq_str(status, status_siz, "SAVE")) {
      if (f->status == FIO_SCRATCH) /* cannot keep scratch file */
        return __fortio_error(FIO_ECOMPAT);
      status_flag = FIO_KEEP;
    } else
      return __fortio_error(FIO_ESPEC);
  }

  return __fortio_close(f, status_flag);
}

__INT_T
ENTF90IO(CLOSEA, closea)
(__INT_T *unit, __INT_T *bitv, __INT_T *iostat, DCHAR(status) DCLEN64(status))
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC) {
    char *p;
    __CLEN_T n;
    if (ISPRESENTC(status)) {
      p = CADR(status);
      n = CLEN(status);
    } else {
      p = NULL;
      n = 0;
    }
    s = _f90io_close(unit, bitv, iostat, p, n);
  }
  __fortio_errend03();
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(CLOSE, close)
(__INT_T *unit, __INT_T *bitv, __INT_T *iostat, DCHAR(status) DCLEN(status))
{
  return ENTF90IO(CLOSEA, closea) (unit, bitv, iostat, CADR(status),
                      (__CLEN_T)CLEN(status));
}

__INT_T
ENTCRF90IO(CLOSEA, closea)
(__INT_T *unit, __INT_T *bitv, __INT_T *iostat, DCHAR(status) DCLEN64(status))
{
  char *p;
  __CLEN_T n;
  int s = 0;

  if (ISPRESENTC(status)) {
    p = CADR(status);
    n = CLEN(status);
  } else {
    p = NULL;
    n = 0;
  }
  s = _f90io_close(unit, bitv, iostat, p, n);
  __fortio_errend03();
  return s;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(CLOSE, close)
(__INT_T *unit, __INT_T *bitv, __INT_T *iostat, DCHAR(status) DCLEN(status))
{
  return ENTCRF90IO(CLOSEA, closea) (unit, bitv, iostat, CADR(status),
                         (__CLEN_T)CLEN(status));
}

/** \brief IO cleanup routine */
void
__fortio_cleanup(void)
{
  FIO_FCB *f, *f_next;

  if (!LOCAL_MODE) {
    __fort_barrier();
  }
  if ((LOCAL_MODE || (GET_DIST_LCPU == GET_DIST_IOPROC))) {
    for (f = fioFcbTbls.fcbs; f != (FIO_FCB *)0; f = f_next) {
      /*
       * WARNING: __fortio_close() calls __fortio_free_fcb()
       * which removes 'f' from the fioFcbTbls.fcbs list;
       * consequently, need to extract the 'next' field now.
       */
      f_next = f->next;
      if (f->fp == NULL) { /* open? */
        continue;
      }
      __io_fflush(f->fp);
      if (f->stdunit) { /* standard unit? */
        continue;
      }
      __fortio_close(f, 0);
      if ((f->dispose == FIO_DELETE) && (f->status != FIO_SCRATCH) &&
          (__fort_access(f->name, 2) == 0)) {
        __fort_unlink(f->name);
      }
    }
    __fortio_cleanup_fcb();
  }
}
