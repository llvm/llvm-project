/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Utility functions for fortran i.o.
 */

#if defined(_WIN64)
#include <io.h>
#include <fcntl.h>
#include <math.h>
#include <Windows.h>
#include <time.h>
#include "wintimes.h"
#endif
#include <errno.h>
#include "global.h"
#include "open_close.h"
#include "stdioInterf.h"
#include "fioMacros.h"
#include "async.h"

/* --------------------------------------------------------------- */

/* number of FCBs to malloc at a time: */
#define CHUNKSZ 100

/* pointer to list of available File Control Blocks: */
static FIO_FCB *fcb_avail = NULL;

/* pointer to the allocatated chunks of File Control Blocks: */
static FIO_FCB *fcb_chunks;

static int __fortio_trunc(FIO_FCB *, seekoffx_t);

extern void *
__fortio_fiofcb_asyptr(FIO_FCB *f)
{
  return f->asyptr;
}

extern bool
__fortio_fiofcb_asy_rw(FIO_FCB *f)
{
  return f->asy_rw;
}

extern void
__fortio_set_asy_rw(FIO_FCB *f, bool value)
{
  f->asy_rw = (sbool)value;
}

extern bool
__fortio_fiofcb_stdunit(FIO_FCB *f)
{
  return f->stdunit;
}

extern FILE *
__fortio_fiofcb_fp(FIO_FCB *f)
{
  return f->fp;
}

extern short
__fortio_fiofcb_form(FIO_FCB *f)
{
  return f->form;
}

extern const char *
__fortio_fiofcb_name(FIO_FCB *f)
{
  return f->name;
}

extern void *
__fortio_fiofcb_next(FIO_FCB *f)
{
  return f->next;
}

extern FIO_FCB *
__fortio_alloc_fcb(void)
{
  FIO_FCB *p;

  if (fcb_avail) { /* return item from avail list */
    p = fcb_avail;
    fcb_avail = p->next;
  } else { /* call malloc for some new space */
    int i;
    p = (FIO_FCB *)malloc(CHUNKSZ * sizeof(FIO_FCB));
    assert(p);
    /*
     * Waste the first element of the chunk; the first element is
     * used to link all of the chunks together so that they can
     * be freed upon program termination.
     */
    for (i = 2; i < CHUNKSZ - 1; i++) /* create avail list */
      p[i].next = &p[i + 1];
    p[CHUNKSZ - 1].next = NULL; /* end of avail list */
    fcb_avail = &p[2];

    p[0].next = fcb_chunks;
    fcb_chunks = p;

    p++;
  }

  memset(p, 0, sizeof(FIO_FCB));
  p[0].next = fioFcbTbls.fcbs; /* add new FCB to front of list */
  fioFcbTbls.fcbs = p;
  return p;
}

extern void
__fortio_free_fcb(FIO_FCB *p)
{
  if (p->name) {
    free(p->name);
    p->name = NULL;
  }

  if (fioFcbTbls.fcbs == p) /* delete p from list */
    fioFcbTbls.fcbs = p->next;
  else {
    FIO_FCB *q;
    for (q = fioFcbTbls.fcbs; q; q = q->next) /* find predecessor of p */
      if (q->next == p)
        break;
    assert(q != NULL); /* trying to free unallocated block */
    q->next = p->next;
  }

  p->next = fcb_avail; /* add to front of avail list */
  fcb_avail = p;
}

extern void
__fortio_cleanup_fcb()
{
  FIO_FCB *p, *p_next;
  for (p = fcb_chunks; p; p = p_next) {
    p_next = p->next;
    free(p);
  }
  fcb_avail = NULL;
  fcb_chunks = NULL;
}

/* --------------------------------------------------------------- */

extern FIO_FCB *__fortio_rwinit(
    int unit, int form, /* FIO_FORMATTED, FIO_UNFORMATTED */
    __INT_T *recptr,    /* ptr to record number, may be present or NULL */
    int optype)         /* 0,1,2 - read, write, endfile */
{
  FIO_FCB *f;
  int errflag;
  seekoffx_t pos; /* typedef in pgstdio.h */
  seekoffx_t rec;
  bool rec_specified;

#define ERR(code)               \
  {                             \
    (void)__fortio_error(code); \
    return NULL;                \
  }

  if (recptr == 0 || !ISPRESENT(recptr)) {
    rec = 0;
    rec_specified = FALSE;
  } else {
    rec = *recptr;
    rec_specified = TRUE;
  }

  f = __fortio_find_unit(unit);
  if (f == NULL) { /* unit not connected */
    int status = FIO_UNKNOWN;

    if (optype == 0) /* if READ, error if file does not exist */
      status = FIO_OLD;
    if (!fioFcbTbls.pos_present) {
      errflag = __fortio_open(unit, FIO_READWRITE, status, FIO_KEEP,
                              FIO_SEQUENTIAL, FIO_NULL, form, FIO_NONE,
                              FIO_ASIS, FIO_YES, 0, NULL /*name*/, 0);
      if (errflag != 0)
        return NULL;
      f = __fortio_find_unit(unit);
      assert(f && f->acc == FIO_SEQUENTIAL);
    } else {
      errflag = __fortio_open(unit, FIO_READWRITE, status, FIO_KEEP, FIO_STREAM,
                              FIO_NULL, form, FIO_NONE, FIO_ASIS, FIO_YES, 0,
                              NULL /*name*/, 0);
      if (errflag != 0)
        return NULL;
      f = __fortio_find_unit(unit);
      assert(f && f->acc == FIO_STREAM);
      if (f->form == FIO_UNFORMATTED)
        f->binary = TRUE;
      fioFcbTbls.pos_present = FALSE;
      pos = fioFcbTbls.pos;
      /*
       * spec says  a pos of 1 => beginning of file
       * INQUIRE always adds one.
       */
      if (pos > 0)
        --pos;
      else
        ERR(FIO_EPOSV);
      if (__io_fseek(f->fp, (seekoffx_t)pos, SEEK_SET) != 0)
        ERR(__io_errno());
      f->coherent = 0;
    }
  } else { /* unit is already connected: */

    /* check for outstanding async i/o */

    if (f->asy_rw) { /* stop any async i/o */
      f->asy_rw = 0;
      if (Fio_asy_disable(f->asyptr) == -1) {
        return (NULL);
      }
    }

    if (fioFcbTbls.pos_present) {
      fioFcbTbls.pos_present = FALSE;
      if (f->acc != FIO_STREAM) {
        ERR(FIO_EPOS);
      } else {
        pos = fioFcbTbls.pos;
        /*
         * spec says  a pos of 1 => beginning of file
         * INQUIRE always adds one.
         */
        if (pos > 0)
          --pos;
        else
          ERR(FIO_EPOSV);
        if (__io_fseek(f->fp, (seekoffx_t)pos, SEEK_SET) != 0)
          ERR(__io_errno());
        f->coherent = 0;
        f->eof_flag = FALSE; /* clear it for the ensuing test */
      }
    }

    if (optype != 0 && f->action == FIO_READ)
      ERR(FIO_EREADONLY);
    if (optype == 0 && f->action == FIO_WRITE)
      ERR(FIO_EWRITEONLY);

    /*  error if attempt operation while positioned past end of file.
        For write or endfile statements, ignore error for compatibility
        with other Fortran compilers.  */
    if (f->eof_flag && optype == 0 /*read*/)
      ERR(FIO_EEOFERR);
    f->eof_flag = FALSE;

    if (form != f->form && optype != 2)
      ERR(FIO_EFORM);

    if (f->acc == FIO_DIRECT) {
      assert(f->reclen > 0);

      if (!rec_specified || rec == 0)
        /* since rec not specified, assume the next record */
        rec = f->nextrec;
      else if (rec < 1)
        ERR(FIO_EDIRECT);
      if (optype == 0 && rec > f->maxrec) {
        seekoffx_t len;
        seekoffx_t sav_pos;

        sav_pos = __io_ftell(f->fp);
        if (__io_fseek(f->fp, (seekoffx_t)0, SEEK_END) != 0)
          ERR(__io_errno());
        len = __io_ftell(f->fp);
        f->partial = len % f->reclen;
        if (form == FIO_UNFORMATTED && f->partial) {
          /* allow read of partial record */
          if (__io_fseek(f->fp, sav_pos, SEEK_SET) != 0)
            ERR(__io_errno());
        } else {

          /* Add simple check to see if maxrec has been
             changed by another process before bailing out */
          f->maxrec = len / f->reclen;

          /* Now check with recomputed maxrec */
          if (rec > f->maxrec) {
            f->nextrec = rec + 1; /* make error info come out correct */
            ERR(FIO_EDREAD);      /* read of non-existing record */
          }

          /* We recovered, so seek to the right point */
          pos = f->reclen * (rec - 1);
          if (__io_fseek(f->fp, (seekoffx_t)pos, SEEK_SET) != 0)
            ERR(__io_errno());
          f->coherent = 0;
        }
      }

      if (f->nextrec != rec) {
        /* FS 3662 Add simple check to see if maxrec has been
           changed by another process before bailing out.
           Certainly need to recompute it before bb is calculated
           below, as multiple writers can cause a file to grow
           enormously.  If it has changed, and the record is not
           past the end, we can just use the normal seek and reset
           coherent section of code below.
        */
        if (rec > f->maxrec + 1) {
          seekoffx_t len;
          if (__io_fseek(f->fp, 0L, SEEK_END) != 0)
            ERR(__io_errno());
          len = __io_ftell(f->fp);
          f->maxrec = len / f->reclen;
        } /* Now go to next if-check with recomputed maxrec */

        if (rec <= f->maxrec + 1) {
          pos = f->reclen * (rec - 1);
          if (__io_fseek(f->fp, (seekoffx_t)pos, SEEK_SET) != 0)
            ERR(__io_errno());
          f->coherent = 0;
        } else {
          /* pad with (rec-maxrec-1)*reclen bytes: */

          seekoffx_t bb = (rec - f->maxrec - 1) * f->reclen;
          /*
           * It has been reported that extending the file by writing
           * to the file is very slow when the number of bytes is
           * large. Rather than use writes to pad the file, use fseek
           * to extend the file 1 less the number of bytes and
           * complete the padding by writing a single byte. A write
           * is necessary after the fseek to ensure that the file's
           * physical size is increased.
          if (__io_fseek(f->fp, 0L, SEEK_END) != 0)
              ERR(__io_errno());
          errflag =
              __fortio_zeropad(f->fp, (rec-f->maxrec-1) * f->reclen);
           */
          /* With multiple writers, there is a chance that
             this could clobber a byte of data, but a very
             small chance now that we've added the
             recomputation of f->maxrec above
          */
          if (__io_fseek(f->fp, (seekoffx_t)(bb - 1), SEEK_END) != 0)
            ERR(__io_errno());
          errflag = __fortio_zeropad(f->fp, 1);
          if (errflag != 0)
            ERR(errflag);
          f->coherent = 1;
        }
      }
      if (optype == 0 /*read*/ && form == FIO_FORMATTED)
        f->nextrec = rec;
      else
        f->nextrec = rec + 1;

      if (rec > f->maxrec)
        f->maxrec = rec;
    }
  }

  /* for write into sequential file, may need to truncate file: */

  assert(f->form == FIO_FORMATTED || f->form == FIO_UNFORMATTED);
  assert(f->acc == FIO_DIRECT || f->acc == FIO_SEQUENTIAL ||
         f->acc == FIO_STREAM);
  if (f->acc == FIO_SEQUENTIAL) {
    if (f->form == FIO_UNFORMATTED)
      f->skip = 0;
    if (rec_specified)
      ERR(FIO_ECOMPAT);
    if (optype != 0 && f->truncflag) {
      pos = __io_ftell(f->fp);
      if (__io_fseek(f->fp, 0L, SEEK_END) != 0)
        ERR(__io_errno());
      f->coherent = 0;
      /* if not currently positioned at end of file, need to trunc: */
      if (pos != __io_ftell(f->fp)) {
        if (__io_fseek(f->fp, (seekoffx_t)pos, SEEK_SET) != 0)
          ERR(__io_errno());
        errflag = __fortio_trunc(f, pos);
        if (errflag != 0)
          return NULL;
      }
      f->truncflag = FALSE;
    }
    if (optype == 0 /*read*/) {
      if (f->ispipe) {
        f->truncflag = FALSE;
        f->nextrec = 1;
        if (f->coherent == 1)
          /* last operation was a write */
          fflush(f->fp);
        f->coherent = 0;
        f->skip = 0;
        return f;
      }
      f->truncflag = TRUE;
    } else
      f->nextrec++; /* endfile or write */
  } else            /* FIO_DIRECT */
    f->skip = 0;

  /*	coherent flag of fcb indicates how the file was last accessed
      (read/write).  If an operation occurs which is not identical with
      the previous operation, the file's buffer must be
      flushed.  NOTE: coherent is set to 0 by open and rewind. */

  if (optype != 2) {
    if (f->coherent && (f->coherent != 2 - optype)) {
      (void)__io_fseek(f->fp, 0L, SEEK_CUR);
      f->skip = 0;
    }
    f->coherent = 2 - optype; /* write ==> 1, read ==> 2*/
  } else
    f->skip = 0;

  return f;
}

/* --------------------------------------------------------------- */

extern FIO_FCB *
__fortio_find_unit(
    /* search FCB table for entry with matching unit number: */
    int unit)
{
  FIO_FCB *p;

  for (p = fioFcbTbls.fcbs; p; p = p->next)
    if (p->unit == unit)
      return p;

  return NULL; /* not found */
}

/* ---------------------------------------------------------------- */

extern int
__fortio_zeropad(FILE *fp, long len)
{
#define BFSZ 512L
  static struct { /* (double aligned buff may be faster) */
    char b[BFSZ];
    double dalign;
  } b = {{0}, 0};

  while (len >= BFSZ) {
    if (FWRITE(b.b, BFSZ, 1, fp) != 1)
      return __io_errno();
    len -= BFSZ;
  }

  if (len > 0)
    if (FWRITE(b.b, len, 1, fp) != 1)
      return __io_errno();

  return 0;
}

/* --------------------------------------------------------------- */

/* return TRUE if string 'str' of length 'len' is equal to 'pattern'. */
extern bool __fortio_eq_str(
    char *str,           /* user-specified string, not null terminated */
    __CLEN_T len,        /* maximum number of characters to comprae */
    const char *pattern) /* upper case, null terminated string */
{
  char c1, c2;

  if (str == NULL || len <= 0)
    return FALSE;

  while (1) {
    c1 = *str++;
    c2 = *pattern++;

    if (len == 0)
      break;
    len--;

    if (c1 >= 'a' && c1 <= 'z') /* convert to upper case */
      c1 = c1 + ('A' - 'a');

    if (c2 == '\0' || c1 != c2)
      break;
  }

  if (c2 != 0)
    return FALSE;

  if (len == 0)
    return TRUE;

  /*  verify that remaining characters of str are blank:  */

  while (len--)
    if (*str++ != ' ')
      return FALSE;

  return TRUE;
}

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

#define SWAPB(p, b1, b2, tmp) \
  {                           \
    tmp = p[b1];              \
    p[b1] = p[b2];            \
    p[b2] = tmp;              \
  }

void __fortio_swap_bytes(
    /*
     * swap bytes where the value located by p is in the wrong endian order.
     * swapping is endian-independent.
     */
    char *p,  /* locates first byte of items */
    int type, /* data type of item */
    long cnt) /* number of 'unit_sz' items to be swapped */
{
  char btmp;
  int unit_sz; /* basic size of item to be swapped */

  switch (type) {
  case __STR:
    return;
  case __CPLX8:
    unit_sz = FIO_TYPE_SIZE(__REAL4);
    cnt <<= 1;
    break;
  case __CPLX16:
    unit_sz = FIO_TYPE_SIZE(__REAL8);
    cnt <<= 1;
    break;
  case __CPLX32:
    unit_sz = FIO_TYPE_SIZE(__REAL16);
    cnt <<= 1;
    break;
  default:
    unit_sz = FIO_TYPE_SIZE(type);
    break;
  }
  while (cnt--) {
    switch (unit_sz) {
    case 1: /* byte */
      return;
    case 2: /* half-word */
      SWAPB(p, 0, 1, btmp);
      break;
    case 4: /* word */
      SWAPB(p, 0, 3, btmp);
      SWAPB(p, 1, 2, btmp);
      break;
    case 8: /* double-word */
      SWAPB(p, 0, 7, btmp);
      SWAPB(p, 1, 6, btmp);
      SWAPB(p, 2, 5, btmp);
      SWAPB(p, 3, 4, btmp);
      break;
    default: /* error */
      assert(0);
      return;
    }
    p += unit_sz;
  }
}

/* ---------------------------------------------------------------------- */

static int
__fortio_trunc(FIO_FCB *p, seekoffx_t length)
{
  __io_fflush(p->fp);
  if (ftruncate(__fort_getfd(p->fp), length))
    return __fortio_error(__io_errno());
  if (length == 0) {
    /*
     * For a file which is now empty, ensure that certain FCB attributes
     * are reset.
     */
    p->nextrec = 1;
    p->truncflag = FALSE;
    p->coherent = 0;
    p->eof_flag = FALSE;
  }
  return 0;
}

#if defined(_WIN64)
void
sincos(double x, double *sine, double *cosine) {
    *sine = sin(x);
    *cosine = cos(x);
}

void
sincosf(float x, float *sine, float *cosine) {
    *sine = sinf(x);
    *cosine = cosf(x);
}

int ftruncate(int fd, __int64 length) {
  _chsize_s(fd, length);
}

struct timezone 
{
    int tz_minuteswest; /* minutes W of Greenwich */
    int tz_dsttime;     /* type of dst correction */
};

#define EPOCHFILETIME (116444736000000000LL)

int
gettimeofday(struct timeval *tv, struct timezone *tz)
{
    FILETIME        ft;
    LARGE_INTEGER   li;
    __int64         t;
    static int      tzflag;

    if(tv)
    {
        GetSystemTimeAsFileTime(&ft);
        li.LowPart  = ft.dwLowDateTime;
        li.HighPart = ft.dwHighDateTime;
        t  = li.QuadPart; 
        t -= EPOCHFILETIME;
        t /= 10;
        tv->tv_sec  = (long)(t / 1000000);
        tv->tv_usec = (long)(t % 1000000);
    }

    if (tz)
    {
        if (!tzflag)
        {
            _tzset();
            tzflag++;
        }
        tz->tz_minuteswest = _timezone / 60;
        tz->tz_dsttime = _daylight;
    }

    return 0;
}

clock_t convert_filetime( const FILETIME *ac_FileTime )  {
    ULARGE_INTEGER    lv_Large ;

    lv_Large.LowPart  = ac_FileTime->dwLowDateTime   ;
    lv_Large.HighPart = ac_FileTime->dwHighDateTime  ;

    return (clock_t)lv_Large.QuadPart ;
}

/*
    Thin emulation of the unix times function
*/
void times(tms *time_struct) {
    FILETIME time_create, time_exit, accum_sys, accum_user;

    GetProcessTimes( GetCurrentProcess(),
            &time_create, &time_exit, &accum_sys, &accum_user );

    time_struct->tms_utime = convert_filetime(&accum_user);
    time_struct->tms_stime = convert_filetime(&accum_sys);
}
#endif
