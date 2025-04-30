/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief
 * Fortran formatted read support.
 */

#include <string.h>
#include "global.h"
#include "feddesc.h"
#include "format.h"
#include "fioMacros.h"

#define RPSTACK_SIZE 20 /* determines max paren nesting level */

typedef struct {
  int count;
  int code;
  int fmtpos;
} rpstack_struct;

static rpstack_struct rpstack[RPSTACK_SIZE];

union ieee {
  double d;
  struct {
    unsigned int lm : 32;
    unsigned int hm : 20;
    unsigned int e : 11;
    unsigned int s : 1;
  } v;
  int i[2];
};

#define INIT_BUFF_LEN 2008

struct struct_G {
  bool internal_file;
  long obuff_len;
  char *obuff;
  char *rec_buff;
  FIO_FCB *fcb;
  INT *fmt_base;
  __INT8_T *size_ptr; /* # of chars read, nonadvancing i/o */
  long rec_len;       /* for direct access or internal files, rec_len
                         is both buffer len and actual record len */
  long max_pos;       /* for variable length records, the actual
                         record len is max_pos.  max_pos is not
                         defined for internal files.  */
  long curr_pos;      /* because of X, TL, curr_pos may be greater
                         than max_pos in some cases. */
  int eor_seen;
  int eor_len;
  int fmt_pos;
  int scale_factor;
  int num_internal_recs;
  int rpstack_top;
  INT last_curr_pos; /* last current offset, nonadvancing i/o */
  short blank_zero;  /* FIO_ ZERO or NULL */
  short pad;         /* FIO_ YES or NULL */
  short decimal;     /* COMMA, POINT, NONE */
  short round;       /* FIO_ UP, DOWN, ZERO, NEAREST, COMPATIBLE,
                      *      PROCESSOR_DEFINED, NONE
                      */
  short sign;        /* FIO_ PLUS, SUPPRESS, PROCESSOR_DEFINED,
                      *      NONE
                      */
  bool repeat_flag;
  bool nonadvance; /* set if advance="no" was specified */

  bool fmt_alloc; /* if this fmt is allocated */
  int move_fwd_eor;
  struct struct_G *same_fcb;
  int same_fcb_idx;
  rpstack_struct rpstack[RPSTACK_SIZE];
};

typedef struct struct_G G;

#define GBL_SIZE 5

static G static_gbl[GBL_SIZE];
static G *gbl = &static_gbl[0];
static G *gbl_head = &static_gbl[0];
static int gbl_avl = 0;
static int gbl_size = GBL_SIZE;

static int move_fwd_eor;

static int fr_read(char *, int, int);

static INT fr_get_fmtcode(void);
static INT fr_get_val(G *);
static int fr_readnum(int, char *, int);
static int fr_init(__INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                   __INT8_T *, char *, __CLEN_T);

static int fr_assign(char *, int, __BIGINT_T, DBLINT64, __BIGREAL_T);
static int fr_OZreadnum(int, char *, int, int);
static int fr_Breadnum(char *, int, int);
static __BIGREAL_T fr_getreal(char *, int, int, int *);
static int fr_move_fwd(int);
static int fr_read_record(void);
static int malloc_obuff(G *, size_t);
static int realloc_obuff(G *, size_t);

/* ----------------------------------------------------------------------- */
static void
save_gbl()
{
  int i;
  if (gbl_avl && gbl) {
    gbl->move_fwd_eor = move_fwd_eor;
    for (i = 0; i < RPSTACK_SIZE; ++i) {
      gbl->rpstack[i].count = rpstack[i].count;
      gbl->rpstack[i].code = rpstack[i].code;
      gbl->rpstack[i].fmtpos = rpstack[i].fmtpos;
    }
  }
}

static void
restore_gbl()
{
  int i;
  if (gbl_avl && gbl) {
    move_fwd_eor = gbl->move_fwd_eor;
    for (i = 0; i < RPSTACK_SIZE; ++i) {
      rpstack[i].count = gbl->rpstack[i].count;
      rpstack[i].code = gbl->rpstack[i].code;
      rpstack[i].fmtpos = gbl->rpstack[i].fmtpos;
    }
  }
}

static void
save_samefcb()
{
  G *tmp_gbl;
  tmp_gbl = gbl->same_fcb;
  if (tmp_gbl) {
    tmp_gbl = &gbl_head[gbl->same_fcb_idx];
    tmp_gbl->nonadvance = gbl->nonadvance;
    tmp_gbl->last_curr_pos = gbl->last_curr_pos;
    tmp_gbl->curr_pos = gbl->curr_pos;
    tmp_gbl->size_ptr = gbl->size_ptr;
    if (tmp_gbl->obuff_len < gbl->obuff_len) {
      tmp_gbl->obuff =
          realloc((void *)tmp_gbl->obuff, (size_t)(gbl->obuff_len));
      /*	    if (!tmp_gbl->obuff)
                      return __fortio_error(FIO_ENOMEM);
      */
    }
    memcpy((void *)tmp_gbl->obuff, (void *)gbl->obuff,
           (size_t)(gbl->obuff_len));
    tmp_gbl->obuff_len = gbl->obuff_len;
    tmp_gbl->max_pos = gbl->max_pos;
    tmp_gbl->move_fwd_eor = gbl->move_fwd_eor;
    tmp_gbl->rec_buff = tmp_gbl->rec_buff + (gbl->rec_buff - gbl->obuff);
    tmp_gbl->rec_len = gbl->rec_len;
  }
}

static void
allocate_new_gbl()
{
  G *tmp_gbl;
  char *obuff = 0;
  char *rec_buff = 0;
  long obuff_len = 0;
  int eor_seen;
  int gsize = sizeof(G);
  if (gbl_avl >= gbl_size) {
    if (gbl_size == GBL_SIZE) {
      gbl_size = gbl_size + GBL_SIZE;
      tmp_gbl = (G *)malloc(gsize * gbl_size);
      memcpy(tmp_gbl, gbl_head, (gsize * gbl_avl));
      memset(tmp_gbl + gbl_avl, 0, gsize * GBL_SIZE);
      gbl_head = tmp_gbl;
    } else {
      gbl_size = gbl_size + GBL_SIZE;
      gbl_head = (G *)realloc(gbl_head, gsize * (gbl_size));
      memset(gbl_head + gbl_avl, 0, gsize * GBL_SIZE);
    }
  }
  gbl = &gbl_head[gbl_avl];
  if (gbl_avl == 0) { /* keep buffer instead of allocate */
    obuff = gbl->obuff;
    obuff_len = gbl->obuff_len;
    rec_buff = gbl->rec_buff;
    eor_seen = gbl->eor_seen;
  } else if (gbl->obuff && !gbl->same_fcb) {
    free(gbl->obuff);
    gbl->obuff = NULL;
  }
  memset(gbl, 0, gsize);
  if (gbl_avl == 0) {
    gbl->obuff = obuff;
    gbl->obuff_len = obuff_len;
    gbl->rec_buff = rec_buff;
    gbl->eor_seen = eor_seen;
  }
  ++gbl_avl;
}

static void
free_gbl()
{
  G *tmp_gbl;
  tmp_gbl = gbl;
  --gbl_avl;
  if (gbl_avl <= 0)
    gbl_avl = 0;
  if (gbl_avl == 0) {
    gbl = &gbl_head[gbl_avl];
  } else {
    gbl = &gbl_head[gbl_avl - 1];
  }
  if (tmp_gbl && tmp_gbl->fmt_alloc) {
    free(tmp_gbl->fmt_base);
    tmp_gbl->fmt_base = NULL;
    tmp_gbl->fmt_alloc = 0;
  }
}
/* ----------------------------------------------------------------------- */

static int
fr_init(__INT_T *unit,   /* unit number */
        __INT_T *rec,    /* record number for direct access I/O */
        __INT_T *bitv,   /* same as for ENTF90IO(open_) */
        __INT_T *iostat, /* same as for ENTF90IO(open_) */
        __INT_T *fmt,    /* encoded format array.  A value of
                          * null indicates that format was
                          * previously encoded by a call to
                          * ENTF90IO(encode_fmt) */
        __INT8_T *size,  /* number of chars read before EOR
                          * (non-advancing only) */
        char *advance,   /* YES, NO, or NULL */
        __CLEN_T advancelen)
{
  FIO_FCB *f;
  long len;
  G *g, *tmp_gbl;
  int errcode, i;
  errcode = 0;

  /* ----- perform initializations.  Get pointer to file control block: */

  save_gbl();
  __fortio_errinit03(*unit, *bitv, iostat, "formatted read");
  allocate_new_gbl();
  f = __fortio_rwinit(*unit, FIO_FORMATTED, rec, 0 /*read*/);
  if (f == NULL) {
    if (fioFcbTbls.eof)
      return EOF_FLAG;
    /* TBD - does there need to be fioFcbTbls.eor */
    return ERR_FLAG;
  }
  g = gbl;
  g->fcb = f;

  /* check if same file */
  tmp_gbl = NULL;
  i = 0;
  if (gbl_avl > 1) {
    for (i = gbl_avl - 2; i >= 0; --i) {
      if (gbl_head[i].fcb == f) {
        tmp_gbl = &gbl_head[i];
        break;
      }
    }
  }
  if (tmp_gbl) {
    gbl->same_fcb = tmp_gbl;
    gbl->same_fcb_idx = i;
  } else {
    gbl->same_fcb = NULL;
  }

  /* ---- set up base pointer to encoded format:  */

  if (ISPRESENT(fmt)) {
    g->fmt_base = fmt;
    g->fmt_alloc = FALSE;
  } else {
    g->fmt_alloc = TRUE;
    g->fmt_base = fioFcbTbls.enctab;
    /*  check for error flag set by encode_format:  */
    if (g->fmt_base[0] == FED_ERROR) {
      return __fortio_error(g->fmt_base[1]);
    }
  }
  g->fmt_pos = 0;

  /* ---- set up char buffer to hold formatted record: */

  if (f->acc == FIO_DIRECT) {
    if (advancelen) {
      return __fortio_error(FIO_ECOMPAT);
    }
    len = f->reclen;
  } else
    len = INIT_BUFF_LEN;

  if (g->obuff_len < len) {
    int err;
    err = malloc_obuff(g, (size_t)len);
    if (err)
      return err;
  } else
    g->rec_buff = g->obuff;
  if (f->acc == FIO_DIRECT)
    g->rec_len = len;
  /* else, rec_len is undefined until fr_read_record is called */

  /* ----- initialize other variables:  */

  g->blank_zero = f->blank;
  g->internal_file = FALSE;
  g->pad = f->pad;
  g->scale_factor = 0;
  g->repeat_flag = FALSE;
  g->rpstack_top = -1;
  g->decimal = f->decimal;
  g->round = f->round;
  if (g->same_fcb) {
    g->nonadvance = tmp_gbl->nonadvance;
    g->last_curr_pos = tmp_gbl->last_curr_pos;
    g->curr_pos = tmp_gbl->curr_pos;
    g->size_ptr = tmp_gbl->size_ptr;
    if (g->obuff_len < tmp_gbl->obuff_len) {
      g->obuff = realloc(g->obuff, tmp_gbl->obuff_len);
      /*	    if (!g->obuff)
                      return __fortio_error(FIO_ENOMEM);
      */
    }

    memcpy(g->obuff, tmp_gbl->obuff, tmp_gbl->obuff_len);
    g->obuff_len = tmp_gbl->obuff_len;
    g->max_pos = tmp_gbl->max_pos;
    g->move_fwd_eor = tmp_gbl->move_fwd_eor;
    g->rec_buff = g->rec_buff + (tmp_gbl->rec_buff - tmp_gbl->obuff);
    g->rec_len = tmp_gbl->rec_len;
  }
  if (advancelen && __fortio_eq_str(advance, advancelen, "NO")) {
    g->nonadvance = TRUE;
    if (ISPRESENT(size)) {
      g->size_ptr = size;
      *g->size_ptr = 0;
      g->last_curr_pos = 0;
    } else {
      g->size_ptr = (__INT8_T *)0;
    }
    if (g->fcb->eor_flag) {
      /*
       * the previous nonadvancing read of stdin left the file
       * at EOR.
       */
      g->fcb->eor_flag = FALSE;
      return __fortio_error(FIO_EEOR);
    }
  } else {
    g->nonadvance = FALSE;
    if (g->fcb->eor_flag) {
      /*
       * the previous nonadvancing read of stdin left the file
       * at EOR. Since this is now 'advancing', treat the current
       * position as 'end of line'.
       */
      g->fcb->eor_flag = FALSE;
      g->obuff[0] = ' ';
      g->rec_len = 1;
      g->curr_pos = 0;
      g->max_pos = 0;
      return 0;
    }
  }

/* ---- read first record: */
  if (!g->same_fcb)
    errcode = fr_read_record();
  if (errcode != 0) {
    return __fortio_error(errcode);
  }
  return 0;
}

__INT_T
ENTF90IO(FMTR_INITA, fmtr_inita)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * null indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 __INT_T *size,   /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN64(advance))
{
  int s = 0;
  char *p;
  __CLEN_T n;
  __INT8_T newsize;

  __fort_status_init(bitv, iostat);
  if ((GET_DIST_LCPU == GET_DIST_IOPROC) || LOCAL_MODE) {
    if (ISPRESENTC(advance)) {
      p = CADR(advance);
      n = CLEN(advance);
    } else {
      p = NULL;
      n = 0;
    }
    if (ISPRESENT(size)) {
      newsize = (__INT8_T)*size;
      s = fr_init(unit, rec, bitv, iostat, fmt, &newsize, p, n);
      *size = (__INT8_T)newsize;
    } else {
      s = fr_init(unit, rec, bitv, iostat, fmt, (__INT8_T *)size, p, n);
    }
  }
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTR_INIT, fmtr_init)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * null indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 __INT_T *size,   /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN(advance))
{
  return ENTF90IO(FMTR_INITA, fmtr_inita) (unit, rec, bitv, iostat, fmt, size,
                             CADR(advance), (__CLEN_T)CLEN(advance));
}

__INT_T
ENTF90IO(FMTR_INIT2003A, fmtr_init2003a)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * null indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 __INT8_T *size,  /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN64(advance))
{
  int s = 0;
  char *p;
  __CLEN_T n;

  __fort_status_init(bitv, iostat);
  if ((GET_DIST_LCPU == GET_DIST_IOPROC) || LOCAL_MODE) {
    if (ISPRESENTC(advance)) {
      p = CADR(advance);
      n = CLEN(advance);
    } else {
      p = NULL;
      n = 0;
    }
    s = fr_init(unit, rec, bitv, iostat, fmt, size, p, n);
  }
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTR_INIT2003, fmtr_init2003)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * null indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 __INT8_T *size,  /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN(advance))
{
  return ENTF90IO(FMTR_INIT2003A, fmtr_init2003a) (unit, rec, bitv, iostat, fmt,
                                  size, CADR(advance), (__CLEN_T)CLEN(advance));
}

__INT_T
ENTF90IO(FMTR_INIT03A, fmtr_init03a)
(__INT_T *istat, DCHAR(blank), DCHAR(decimal), DCHAR(pad),
 DCHAR(round) DCLEN64(blank) DCLEN64(decimal) DCLEN64(pad) DCLEN64(round))
{
  int s = *istat;

  if (s)
    return DIST_STATUS_BCST(s);

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC) {
    if (ISPRESENTC(blank)) {
      if (__fortio_eq_str(CADR(blank), CLEN(blank), "ZERO")) {
        gbl->blank_zero = FIO_ZERO;
      } else if (__fortio_eq_str(CADR(blank), CLEN(blank), "NULL")) {
        gbl->blank_zero = FIO_NULL;
      } else
        s = __fortio_error(FIO_ESPEC);
    }
    if (ISPRESENTC(decimal)) {
      if (__fortio_eq_str(CADR(decimal), CLEN(decimal), "COMMA")) {
        gbl->decimal = FIO_COMMA;
      } else if (__fortio_eq_str(CADR(decimal), CLEN(decimal), "POINT")) {
        gbl->decimal = FIO_POINT;
      } else
        s = __fortio_error(FIO_ESPEC);
    }
    if (ISPRESENTC(pad)) {
      if (__fortio_eq_str(CADR(pad), CLEN(pad), "YES"))
        gbl->pad = FIO_YES;
      else if (__fortio_eq_str(CADR(pad), CLEN(pad), "NO"))
        gbl->pad = FIO_NO;
      else
        return __fortio_error(FIO_ESPEC);
    }
    if (ISPRESENTC(round)) {
      if (__fortio_eq_str(CADR(round), CLEN(round), "UP")) {
        gbl->round = FIO_UP;
      } else if (__fortio_eq_str(CADR(round), CLEN(round), "DOWN")) {
        gbl->round = FIO_DOWN;
      } else if (__fortio_eq_str(CADR(round), CLEN(round), "ZERO")) {
        gbl->round = FIO_ZERO;
      } else if (__fortio_eq_str(CADR(round), CLEN(round), "NEAREST")) {
        gbl->round = FIO_NEAREST;
      } else if (__fortio_eq_str(CADR(round), CLEN(round), "COMPATIBLE")) {
        gbl->round = FIO_COMPATIBLE;
      } else if (__fortio_eq_str(CADR(round), CLEN(round),
                                "PROCESSOR_DEFINED")) {
        gbl->round = FIO_PROCESSOR_DEFINED;
      } else
        s = __fortio_error(FIO_ESPEC);
    }
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTR_INIT03, fmtr_init03)
(__INT_T *istat, DCHAR(blank), DCHAR(decimal), DCHAR(pad),
 DCHAR(round) DCLEN(blank) DCLEN(decimal) DCLEN(pad) DCLEN(round))
{
  return ENTF90IO(FMTR_INIT03A, fmtr_init03a) (istat, CADR(blank),
                  CADR(decimal), CADR(pad), CADR(round), (__CLEN_T)CLEN(blank),
		  (__CLEN_T)CLEN(decimal), (__CLEN_T)CLEN(pad),
		  (__CLEN_T)CLEN(round));
}

__INT_T
ENTCRF90IO(FMTR_INITA, fmtr_inita)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * null indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 __INT_T *size,   /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN64(advance))
{
  int s = 0;
  __CLEN_T n;
  char *p;
  __INT8_T newsize;

  if (ISPRESENTC(advance)) {
    p = CADR(advance);
    n = CLEN(advance);
  } else {
    p = NULL;
    n = 0;
  }
  if (ISPRESENT(size)) {
    newsize = (__INT8_T)*size;
    s = fr_init(unit, rec, bitv, iostat, fmt, &newsize, p, n);
    *size = (__INT_T)newsize;
  } else {
    s = fr_init(unit, rec, bitv, iostat, fmt, (__INT8_T *)size, p, n);
  }
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMTR_INIT, fmtr_init)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * null indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 __INT_T *size,   /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN(advance))
{
  return ENTCRF90IO(FMTR_INITA, fmtr_inita) (unit, rec, bitv, iostat, fmt, size,
                               CADR(advance), (__CLEN_T)CLEN(advance));
}

__INT_T
ENTCRF90IO(FMTR_INIT2003A, fmtr_init2003a)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * null indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 __INT8_T *size,  /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN64(advance))
{
  int s = 0;
  __CLEN_T n;
  char *p;

  if (ISPRESENTC(advance)) {
    p = CADR(advance);
    n = CLEN(advance);
  } else {
    p = NULL;
    n = 0;
  }

  s = fr_init(unit, rec, bitv, iostat, fmt, size, p, n);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMTR_INIT2003, fmtr_init2003)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * null indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 __INT8_T *size,  /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN(advance))
{
  return ENTCRF90IO(FMTR_INIT2003A, fmtr_init2003a) (unit, rec, bitv, iostat,
                             fmt, size, CADR(advance), (__CLEN_T)CLEN(advance));
}

__INT_T
ENTF90IO(FMTR_INITVA, fmtr_initva)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   * containing address of encoded
                   * format array */
 __INT_T *size,   /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN64(advance))
{
  int s = 0;
  char *p;
  __CLEN_T n;
  __INT8_T newsize;

  __fort_status_init(bitv, iostat);
  if ((GET_DIST_LCPU == GET_DIST_IOPROC) || LOCAL_MODE) {
    if (ISPRESENTC(advance)) {
      p = CADR(advance);
      n = CLEN(advance);
    } else {
      p = NULL;
      n = 0;
    }
    if (ISPRESENT(size)) {
      newsize = (__INT8_T)*size;
      s = fr_init(unit, rec, bitv, iostat, *fmt, &newsize, p, n);
      *size = (__INT_T)newsize;
    } else {
      s = fr_init(unit, rec, bitv, iostat, *fmt, (__INT8_T *)size, p, n);
    }
  }
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTR_INITV, fmtr_initv)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   * containing address of encoded
                   * format array */
 __INT_T *size,   /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN(advance))
{
  return ENTF90IO(FMTR_INITVA, fmtr_initva) (unit, rec, bitv, iostat, fmt, size,
                              CADR(advance), (__CLEN_T)CLEN(advance));
}

__INT_T
ENTF90IO(FMTR_INITV2003A, fmtr_initv2003a)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   * containing address of encoded
                   * format array */
 __INT8_T *size,  /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN64(advance))
{
  int s = 0;
  char *p;
  __CLEN_T n;

  __fort_status_init(bitv, iostat);
  if ((GET_DIST_LCPU == GET_DIST_IOPROC) || LOCAL_MODE) {
    if (ISPRESENTC(advance)) {
      p = CADR(advance);
      n = CLEN(advance);
    } else {
      p = NULL;
      n = 0;
    }
    s = fr_init(unit, rec, bitv, iostat, *fmt, size, p, n);
  }
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTR_INITV2003, fmtr_initv2003)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   * containing address of encoded
                   * format array */
 __INT8_T *size,  /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN(advance))
{
  return ENTF90IO(FMTR_INITV2003A, fmtr_initv2003a) (unit, rec, bitv, iostat,
                           fmt, size, CADR(advance), (__CLEN_T)CLEN(advance));
}

__INT_T
ENTCRF90IO(FMTR_INITVA, fmtr_initva)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   * containing address of encoded
                   * format array */
 __INT_T *size,   /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN64(advance))
{
  int s;
  __CLEN_T n;
  char *p;
  __INT8_T newsize;

  if (ISPRESENTC(advance)) {
    p = CADR(advance);
    n = CLEN(advance);
  } else {
    p = NULL;
    n = 0;
  }

  if (ISPRESENT(size)) {
    newsize = (__INT8_T)*size;
    s = fr_init(unit, rec, bitv, iostat, *fmt, &newsize, p, n);
    *size = (__INT_T)newsize;
  } else {
    s = fr_init(unit, rec, bitv, iostat, *fmt, (__INT8_T *)size, p, n);
  }
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return (s);
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMTR_INITV, fmtr_initv)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   * containing address of encoded
                   * format array */
 __INT_T *size,   /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN(advance))
{
  return ENTCRF90IO(FMTR_INITVA, fmtr_initva) (unit, rec, bitv, iostat, fmt,
                                size, CADR(advance), (__CLEN_T)CLEN(advance));
}

__INT_T
ENTCRF90IO(FMTR_INITV2003A, fmtr_initv2003a)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   * containing address of encoded
                   * format array */
 __INT8_T *size,  /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN64(advance))
{
  int s;
  __CLEN_T n;
  char *p;

  if (ISPRESENTC(advance)) {
    p = CADR(advance);
    n = CLEN(advance);
  } else {
    p = NULL;
    n = 0;
  }

  s = fr_init(unit, rec, bitv, iostat, *fmt, size, p, n);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return (s);
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMTR_INITV2003, fmtr_initv2003)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   * containing address of encoded
                   * format array */
 __INT8_T *size,  /* number of chars read before EOR
                   * (non-advancing only) */
 DCHAR(advance)   /* YES, NO, or NULL */
 DCLEN(advance))
{
  return ENTCRF90IO(FMTR_INITV2003A, fmtr_initv2003a) (unit, rec, bitv, iostat,
                            fmt, size, CADR(advance), (__CLEN_T)CLEN(advance));
}

/* ------------------------------------------------------------------ */

static int
fr_intern_init(char *cunit,      /* pointer to variable or array to read from */
               __INT_T *rec_num, /* number of records in internal file-- 0 if
                                  * the file is an assumed size character
                                  * array */
               __INT_T *bitv,    /* same as for ENTF90IO(open_) */
               __INT_T *iostat,  /* same as for ENTF90IO(open_) */
               __INT_T *fmt,     /* same as for ENTF90IO(fmtr)/w_init */
               __CLEN_T cunitlen)
{
  G *g;
  long w;
  __CLEN_T i;
  char *p;

  save_gbl();
  __fortio_errinit03(-99, *bitv, iostat, "formatted read");
  assert(*rec_num > 0);
  allocate_new_gbl();
  g = gbl;

  if (ISPRESENT(fmt)) {
    g->fmt_base = fmt;
    g->fmt_alloc = FALSE;
  } else {
    g->fmt_alloc = TRUE;
    g->fmt_base = fioFcbTbls.enctab;
    /*  check for error flag set by encode_format:  */
    if (g->fmt_base[0] == FED_ERROR)
      return __fortio_error(g->fmt_base[1]);
  }

  g->fmt_pos = 0;
  g->rec_len = cunitlen;

  /* malloc buffer for a local version of cunit and copy contents of
     cunit to buffer */

  w = g->rec_len * *rec_num;
  i = INIT_BUFF_LEN;
  if (w > INIT_BUFF_LEN)
    i = w;
  if (g->obuff_len < i) {
    int err;
    err = malloc_obuff(g, (size_t)i);
    if (err)
      return err;
  } else
    g->rec_buff = g->obuff;
  p = cunit;
  i = 0;
  while (w-- > 0)
    g->rec_buff[i++] = *p++;

  g->curr_pos = 0;
  g->blank_zero = FIO_NULL;
  g->internal_file = TRUE;
  g->pad = FIO_YES;
  g->num_internal_recs = *rec_num - 1;
  g->scale_factor = 0;
  g->repeat_flag = FALSE;
  g->rpstack_top = -1;
  g->nonadvance = FALSE;
  g->decimal = FIO_POINT;
  g->round = FIO_COMPATIBLE;

  return 0;
}

__INT_T
ENTF90IO(FMTR_INTERN_INITA, fmtr_intern_inita)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt      /* same as for ENTF90IO(fmtr)/w_init */
 DCLEN64(cunit))
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if ((GET_DIST_LCPU == GET_DIST_IOPROC) || LOCAL_MODE) {
    s = fr_intern_init(CADR(cunit), rec_num, bitv, iostat, fmt, CLEN(cunit));
  }
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTR_INTERN_INIT, fmtr_intern_init)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt      /* same as for ENTF90IO(fmtr)/w_init */
 DCLEN(cunit))
{
  return ENTF90IO(FMTR_INTERN_INITA, fmtr_intern_inita) (CADR(cunit), rec_num,
                                   bitv, iostat, fmt, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTCRF90IO(FMTR_INTERN_INITA, fmtr_intern_inita)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt      /* same as for ENTF90IO(fmtr)/w_init */
 DCLEN64(cunit))
{
  int s = 0;

  s = fr_intern_init(CADR(cunit), rec_num, bitv, iostat, fmt, CLEN(cunit));
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return (s);
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMTR_INTERN_INIT, fmtr_intern_init)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt      /* same as for ENTF90IO(fmtr)/w_init */
 DCLEN(cunit))
{
  return ENTCRF90IO(FMTR_INTERN_INITA, fmtr_intern_inita) (CADR(cunit), rec_num,
                                      bitv, iostat, fmt, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTF90IO(FMTR_INTERN_INITVA, fmtr_intern_initva)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt     /* same as for ENTF90IO(fmtr)/w_initv */
 DCLEN64(cunit))
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if ((GET_DIST_LCPU == GET_DIST_IOPROC) || LOCAL_MODE) {
    s = fr_intern_init(CADR(cunit), rec_num, bitv, iostat, *fmt, CLEN(cunit));
  }
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTR_INTERN_INITV, fmtr_intern_initv)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt     /* same as for ENTF90IO(fmtr)/w_initv */
 DCLEN(cunit))
{
  return ENTF90IO(FMTR_INTERN_INITVA, fmtr_intern_initva) (CADR(cunit), rec_num,
                                      bitv, iostat, fmt, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTCRF90IO(FMTR_INTERN_INITVA, fmtr_intern_initva)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt     /* same as for ENTF90IO(fmtr)/w_initv */
 DCLEN64(cunit))
{
  int s = 0;
  s = fr_intern_init(CADR(cunit), rec_num, bitv, iostat, *fmt, CLEN(cunit));
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMTR_INTERN_INITV, fmtr_intern_initv)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt     /* same as for ENTF90IO(fmtr)/w_initv */
 DCLEN(cunit))
{
  return ENTCRF90IO(FMTR_INTERN_INITVA, fmtr_intern_initva) (CADR(cunit),
                          rec_num, bitv, iostat, fmt, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTF90IO(FMTR_INTERN_INITE, fmtr_intern_inite)
(char **cunit,     /* variable containing address to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt,     /* same as for ENTF90IO(fmtr)/w_init */
 __INT_T *len)     /* len of 'cunit' */
{
  /* DECODE initialization */
  int s = 0;

  __fort_status_init(bitv, iostat);
  if ((GET_DIST_LCPU == GET_DIST_IOPROC) || LOCAL_MODE) {
    s = fr_intern_init(*cunit, rec_num, bitv, iostat, fmt, *len);
  }
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(FMTR_INTERN_INITE, fmtr_intern_inite)
(char **cunit,     /* variable containing address to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt,     /* same as for ENTF90IO(fmtr)/w_init */
 __INT_T *len)     /* len of 'cunit' */
{
  int s = 0;

  s = fr_intern_init(*cunit, rec_num, bitv, iostat, fmt, *len);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return (s);
}

__INT_T
ENTF90IO(FMTR_INTERN_INITEV, fmtr_intern_initev)
(char **cunit,     /* variable containing address to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt,    /* same as for ENTF90IO(fmtr)/w_initv */
 __INT_T *len)     /* len of 'cunit' */
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if ((GET_DIST_LCPU == GET_DIST_IOPROC) || LOCAL_MODE) {
    s = fr_intern_init(*cunit, rec_num, bitv, iostat, *fmt, *len);
  }
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(FMTR_INTERN_INITEV, fmtr_intern_initev)
(char **cunit,     /* variable containing address to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt,    /* same as for ENTF90IO(fmtr)/w_initv */
 __INT_T *len)     /* len of 'cunit' */
{
  int s = 0;
  s = fr_intern_init(*cunit, rec_num, bitv, iostat, *fmt, *len);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}

/* --------------------------------------------------------------------- */

int
__f90io_fmt_read(int type,    /* data type (as defined in pghpft.h) */
                 long length, /* # items of type to read. May be <= 0 */
                 int stride,  /* distance in bytes between items*/
                 char *item,  /* where to transfer data to */
                 __CLEN_T item_length)
{
  long i;
  int sz;
  int tmptype;   /* scratch copy of type */
  char *tmpitem; /* scratch copy of item */
  G *g = gbl;
  FIO_FCB *f;
  int ist;
  int ret_err = 0;

  if (fioFcbTbls.error) {
    ret_err = ERR_FLAG;
    goto fmtr_err;
  }
  if (fioFcbTbls.eof) {
    ret_err = EOF_FLAG;
    goto fmtr_err;
  }
  assert(item != NULL);

  f = g->fcb;

  sz = 0;
  tmptype = type;
  if (tmptype == __CPLX8)
    tmptype = __REAL4, sz = FIO_TYPE_SIZE(tmptype);
  else if (tmptype == __CPLX16)
    tmptype = __REAL8, sz = FIO_TYPE_SIZE(tmptype);
  else if (tmptype == __CPLX32)
    tmptype = __REAL16, sz = FIO_TYPE_SIZE(tmptype);

  tmpitem = item;
  for (i = 0; i < length; i++, tmpitem += stride) {
    ist = fr_read(tmpitem, tmptype, item_length);
    if (ist != 0) {
      if (fioFcbTbls.eof) {
        ret_err = EOF_FLAG;
        goto fmtr_err;
      }
      if (ist == EOR_FLAG) {
        ret_err = EOR_FLAG;
        goto fmtr_err;
      }
      ret_err = ERR_FLAG;
      goto fmtr_err;
    }
    /*  read second half of complex if necessary:  */
    if (sz != 0 && fr_read(tmpitem + sz, tmptype, item_length) != 0) {
      if (fioFcbTbls.eof) {
        ret_err = EOF_FLAG;
        goto fmtr_err;
      }
      ret_err = ERR_FLAG;
      goto fmtr_err;
    }
  }

  /* nonadvancing i/o: update size variable */

  if (g->nonadvance && (g->size_ptr != (__INT8_T *)0)) {
    if (g->curr_pos < g->last_curr_pos) {
      g->last_curr_pos = 0;
    }
    sz = g->curr_pos - g->last_curr_pos;
    *g->size_ptr += (__INT8_T)sz;
    g->last_curr_pos = g->curr_pos;
  }

  return 0;

fmtr_err:
  free_gbl();
  restore_gbl();
  __fortio_errend03();
  return ret_err;
}

__INT_T
ENTF90IO(FMT_READA, fmt_reada)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items*/
 DCHAR(item)      /* where to transfer data to */
 DCLEN64(item))
{
  int typ;
  int cnt, ioproc;
  int len, str;
  char *adr;
  int s = 0;

  typ = *type;
  cnt = *count;
  str = *stride;
  adr = CADR(item);
  len = (typ == __STR) ? (int)CLEN(item) : GET_DIST_SIZE_OF(typ);

#if defined(DEBUG)
  if ((str / len) * len != str)
    __fort_abort("FMT_READ: stride not a multiple of item length");
#endif

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_fmt_read(typ, cnt, str, adr, len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, adr, cnt, str / len, typ, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMT_READ, fmt_read)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items*/
 DCHAR(item)      /* where to transfer data to */
 DCLEN(item))
{
  return ENTF90IO(FMT_READA, fmt_reada) (type, count, stride, CADR(item), (__CLEN_T)CLEN(item));
}

/* same as fmt_read, but item may be array - for fmt_read, the compiler
 * scalarizes.
 */
__INT_T
ENTF90IO(FMT_READ_AA, fmt_read_aa)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items*/
 DCHAR(item)      /* where to transfer data to */
 DCLEN64(item))
{
  int typ;
  int cnt, ioproc;
  int len, str;
  char *adr;
  int s = 0;

  typ = *type;
  cnt = *count;
  str = *stride;
  adr = CADR(item);
  len = (typ == __STR) ? (int)CLEN(item) : GET_DIST_SIZE_OF(typ);

#if defined(DEBUG)
  if ((str / len) * len != str)
    __fort_abort("FMT_READ_A: stride not a multiple of item length");
#endif

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_fmt_read(typ, cnt, str, adr, len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, adr, cnt, str / len, typ, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMT_READ_A, fmt_read_a)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items*/
 DCHAR(item)      /* where to transfer data to */
 DCLEN(item))
{
  return ENTF90IO(FMT_READ_AA, fmt_read_aa) (type, count, stride, CADR(item),
                                             (__CLEN_T)CLEN(item));
}

__INT_T
ENTF90IO(FMT_READ64_AA, fmt_read64_aa)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT8_T *count, /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items*/
 DCHAR(item)      /* where to transfer data to */
 DCLEN64(item))
{
  int typ;
  long cnt;
  int ioproc;
  int len, str;
  char *adr;
  int s = 0;

  typ = *type;
  cnt = *count;
  str = *stride;
  adr = CADR(item);
  len = (typ == __STR) ? (int)CLEN(item) : GET_DIST_SIZE_OF(typ);

#if defined(DEBUG)
  if ((str / len) * len != str)
    __fort_abort("FMT_READ_A: stride not a multiple of item length");
#endif

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_fmt_read(typ, cnt, str, adr, len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, adr, cnt, str / len, typ, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMT_READ64_A, fmt_read64_a)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT8_T *count, /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items*/
 DCHAR(item)      /* where to transfer data to */
 DCLEN(item))
{
  return ENTF90IO(FMT_READ64_AA, fmt_read64_aa) (type, count, stride,
                                CADR(item), (__CLEN_T)CLEN(item));
}

__INT_T
ENTCRF90IO(FMT_READA, fmt_reada)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items*/
 DCHAR(item)      /* where to transfer data to */
 DCLEN64(item))
{
  int typ;
  int cnt;
  int len, str;
  char *adr;

  typ = *type;
  cnt = *count;
  str = *stride;
  adr = CADR(item);
  len = (typ == __STR) ? (int)CLEN(item) : GET_DIST_SIZE_OF(typ);

#if defined(DEBUG)
  if ((str / len) * len != str)
    __fort_abort("FMT_READ: stride not a multiple of item length");
#endif
  return __f90io_fmt_read(typ, cnt, str, adr, len);
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMT_READ, fmt_read)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items*/
 DCHAR(item)      /* where to transfer data to */
 DCLEN(item))
{
  return ENTCRF90IO(FMT_READA, fmt_reada) (type, count, stride, CADR(item),
                                           (__CLEN_T)CLEN(item));
}

/* --------------------------------------------------------------------- */

static int
fr_read(char *item,      /* where to transfer data to.  The value of item may
                          * be NULL to indicate finishing format processing */
        int type,        /* data type (as defined in pghpft.h) */
        int item_length) /* optional-- passed if type is character */
{
  int code;
  G *g = gbl;
  bool endreached = FALSE;
  int i, w;

  move_fwd_eor = 0;
  while (TRUE) {
    code = fr_get_fmtcode();

    switch (code) {
    case FED_END:
      if (item == NULL) {
        goto exit_loop;
      }
      i = fr_read_record();
      if (g->same_fcb) {
        goto exit_loop;
      }
      if (i != 0)
        return __fortio_error(i);
      g->fmt_pos = g->fmt_base[g->fmt_pos];
      if (endreached)
        return __fortio_error(FIO_EINFINITE_REVERSION);
      endreached = TRUE;
      break;

    case FED_LPAREN:
      /*  get_fmtcode created stack entry for this paren only if it had
          a repeat count != 1.  */
      assert(g->repeat_flag == FALSE);
      break;

    case FED_RPAREN:
      assert(g->rpstack_top >= -1);
      assert(g->repeat_flag == FALSE);
      i = g->fmt_base[g->fmt_pos++]; /* get back-pointer */
      if (g->rpstack_top != -1 && rpstack[g->rpstack_top].fmtpos == i) {
        /*  this paren has an active repeat count, go back ...  */
        assert(rpstack[g->rpstack_top].code == FED_LPAREN);
        g->fmt_pos = i;
        i = rpstack[g->rpstack_top].count;
        rpstack[g->rpstack_top].count = i - 1; /* decrement rpcount */
        if (i <= 1) {                          /* repeat count used up */
          assert(i == 1);
          g->rpstack_top--;
        }
      }
      break;

    case FED_P:
      /* pick up scale factor put of format and save for later use: */
      i = fr_get_val(g);
      if (i < -128 || i > 127)
        return __fortio_error(FIO_ESCALEF);
      g->scale_factor = i;
      break;

    case FED_STR:
    case FED_KANJI_STRING:
      /*  advance curr_pos without giving error for record length
          exceeded: */
      i = g->fmt_base[g->fmt_pos++]; /* string length */
      g->curr_pos += i;
      g->fmt_pos += (i + 3) >> 2;
      break;

    case FED_T:
      i = fr_get_val(g);
      if (i < 1)
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      g->curr_pos = i - 1;
      break;

    case FED_TL:
      i = fr_get_val(g);
      if (i < 1)
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      g->curr_pos -= i;
      if (g->curr_pos < 0)
        g->curr_pos = 0;
      break;

    case FED_TR:
    case FED_X:
      i = fr_get_val(g);
      if (i < 1)
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      g->curr_pos += i;
      if (g->curr_pos > g->rec_len) {
        if (g->internal_file || g->fcb->acc == FIO_DIRECT)
          g->curr_pos = g->rec_len;
        else {
          /* sequential, external, formatted input - extend record
           * w blanks:
           */
          if (g->curr_pos > g->obuff_len)
            g->curr_pos = g->obuff_len;
          else {
            while (g->rec_len < g->curr_pos)
              g->rec_buff[g->rec_len++] = ' ';
          }
        }
      }
      break;

    case FED_SP:
    case FED_S:
    case FED_SS: /* ignore these for read */
      break;

    case FED_BN:
      g->blank_zero = FIO_NULL;
      break;
    case FED_BZ:
      g->blank_zero = FIO_ZERO;
      break;

    case FED_DC:
      g->decimal = FIO_COMMA;
      break;
    case FED_DP:
      g->decimal = FIO_POINT;
      break;

    case FED_RU:
      g->round = FIO_UP;
      break;
    case FED_RD:
      g->round = FIO_DOWN;
      break;
    case FED_RZ:
      g->round = FIO_ZERO;
      break;
    case FED_RN:
      g->round = FIO_NEAREST;
      break;
    case FED_RC:
      g->round = FIO_COMPATIBLE;
      break;
    case FED_RP:
      g->round = FIO_PROCESSOR_DEFINED;
      break;

    case FED_SLASH:
      i = fr_read_record();
      if (i != 0)
        return __fortio_error(i);
      break;

    case FED_COLON:
      if (item == NULL)
        goto exit_loop;
      break;

    case FED_Q:
      if (item != NULL) {
        i = fr_assign(item, type, g->rec_len - g->curr_pos, 0, 0.0);
        if (i != 0)
          return __fortio_error(i);
      }
      goto exit_loop;

    case FED_DOLLAR: /* ignore $ in input format */
      break;

    case FED_Aw:
    case FED_A:
      if (item != NULL) {
        int idx;
        int pad = 0;

        if (type != __STR)
          item_length = FIO_TYPE_SIZE(type);

        w = item_length;
        if (code == FED_Aw) {
          w = fr_get_val(g); /*  field width  */
          if (w > item_length) {
            g->curr_pos += (w - item_length);
            w = item_length;
          } else if (w < item_length)
            pad = item_length - w;
        }
        idx = g->curr_pos;
        i = fr_move_fwd(w);
        if (i != 0)
          return i;
        while (w-- > 0)
          *item++ = g->rec_buff[idx++];
        if (g->pad == FIO_YES) {
          while (pad > 0)
            *item++ = ' ', pad--;
        }
      }
      goto exit_loop;

    case FED_Gw_d:
    case FED_G:
      if (item == NULL)
        goto exit_loop;

      if (type == __STR) {
        int idx;
        int pad = 0;

        w = item_length;
        if (code == FED_Gw_d) {
          w = fr_get_val(g); /*  field width  */
          if (w > item_length) {
            g->curr_pos += (w - item_length);
            w = item_length;
          } else if (w < item_length)
            pad = item_length - w;
          (void) fr_get_val(g);                    /*ignore w */
          if (g->fmt_base[g->fmt_pos] == FED_Ee) { /*ignore e */
            g->fmt_pos++;
            (void) fr_get_val(g);
          }
        }
        idx = g->curr_pos;
        i = fr_move_fwd(w);
        if (i != 0)
          return i;
        while (w-- > 0)
          *item++ = g->rec_buff[idx++];
        if (g->pad == FIO_YES) {
          while (pad > 0)
            *item++ = ' ', pad--;
        }
        goto exit_loop;
      }

      i = fr_readnum(code, item, type);
      if (i != 0)
        return i;
      goto exit_loop;

    case FED_Lw:
    case FED_Iw_m:
    case FED_Fw_d:
    case FED_Ew_d:
    case FED_ENw_d:
    case FED_ESw_d:
    case FED_Dw_d:
    case FED_I:
    case FED_L:
    case FED_F:
    case FED_E:
    case FED_D:
      if (item == NULL)
        goto exit_loop;

      i = fr_readnum(code, item, type);
      if (i != 0)
        return i;
      goto exit_loop;

    case FED_Ow_m:
    case FED_Zw_m:
    case FED_O:
    case FED_Z:
      if (item == NULL)
        goto exit_loop;

      i = fr_OZreadnum(code, item, type, item_length);
      if (i != 0)
        return i;
      goto exit_loop;

    case FED_Bw_m:
      if (item == NULL)
        goto exit_loop;

      i = fr_Breadnum(item, type, item_length);
      if (i != 0)
        return i;
      goto exit_loop;

    case FED_ERROR:
      return ERR_FLAG;
    case FED_DT:
      goto exit_loop;
    case FED_Ee:
    default:
      return __fortio_error(FIO_EEDITDSCR);
    } /* end switch */
  }   /* end while(TRUE) */

exit_loop:
  if (move_fwd_eor) {
    move_fwd_eor = 0;
    return __fortio_error(FIO_EEOR);
  }
  return 0;
} /*  end fr_read()  */

/* --------------------------------------------------------------------- */

static INT
fr_get_fmtcode(void)
{
  G *g = gbl;
  INT k;
  int repeatcount;

  if (g->repeat_flag) { /* return previous edit descriptor ... */
    repeatcount = rpstack[g->rpstack_top].count;
    k = rpstack[g->rpstack_top].code;
    g->fmt_pos = rpstack[g->rpstack_top].fmtpos;
    rpstack[g->rpstack_top].count = repeatcount - 1;

    if (repeatcount <= 1) { /* pop stack if this repeat count used up: */
      assert(repeatcount == 1);
      g->rpstack_top--;
      g->repeat_flag = FALSE;
    }
    return k;
  }

  repeatcount = 1;             /* default repeat count */
  k = g->fmt_base[g->fmt_pos]; /* FED code or repeat count */
  if (k >= 0) {                /* repeat count specified */
    repeatcount = fr_get_val(g);
    k = g->fmt_base[g->fmt_pos]; /* FED code */
  }
  g->fmt_pos++;

  if (repeatcount != 1) {
    if (repeatcount <= 0) { /* check for valid repeat cnt value */
      (void) __fortio_error(FIO_EREPCNT);
      return FED_ERROR;
    }

    /*  push new entry on to repeat stack for this item:  */

    g->rpstack_top++;
    if (g->rpstack_top >= RPSTACK_SIZE) {
      (void) __fortio_error(FIO_EPNEST); /* parens nested too deep */
      return FED_ERROR;
    }
    rpstack[g->rpstack_top].count = repeatcount - 1;
    rpstack[g->rpstack_top].code = k;
    rpstack[g->rpstack_top].fmtpos = g->fmt_pos;
    if (k != FED_LPAREN)
      g->repeat_flag = TRUE;
  }

  return k;
}

/* --------------------------------------------------------------------- */

static INT
fr_get_val(G *g)
{
  INT flag = g->fmt_base[g->fmt_pos];
  INT val = g->fmt_base[g->fmt_pos + 1];

  g->fmt_pos += 2;

  if (flag != 0) { /* must call function to value */
    INT (*fp)();
    fp = (INT((*)()))(long)val;
    val = (*fp)();
  }

  return val;
}

/* ------------------------------------------------------------------- */

static int
fr_readnum(int code, char *item, int type)
{
  __BIGINT_T ival;
  __BIGREAL_T dval;
#undef IS_INT
  DBLINT64 i8val; /* always declare because of fr_assign() */
#define IS_INT(t) (t == __INT || t == __INT8)
  int ty;
  int w, d, e, c;
  int idx;
  int tmp_idx;
  G *g = gbl;
  int errflag;
  int comma_seen, width;
  bool is_logical;

  dval = 0; /* prevent purify UMR when type isn't floating point */
  is_logical = FALSE;

  switch (type) {
  case __INT1:
    ty = __INT;
    w = 7;
    break;

  case __LOG1:
    ty = __INT;
    w = 7;
    is_logical = TRUE;
    break;

  case __INT2:
    ty = __INT;
    w = 7;
    break;

  case __LOG2:
    ty = __INT;
    w = 7;
    is_logical = TRUE;
    break;

  case __INT4:
  case __WORD4:
    ty = __INT;
    w = 12;
    break;

  case __LOG4:
    ty = __INT;
    w = 12;
    is_logical = TRUE;
    break;

  case __REAL4:
    ty = __REAL4;
    w = REAL4_W;
    d = REAL4_D;
    ival = TRUE;
    break;

  case __REAL8:
    ty = __REAL8;
    w = REAL8_W;
    d = REAL8_D;
    ival = FALSE;
    break;

  case __REAL16:
    ty = __REAL16;
    w = G_REAL16_W;
    d = G_REAL16_D;
    ival = FALSE; /* I don't think this is valid for this code. */
    break;

  case __INT8:
    ty = __INT8;
    w = 24;
    break;

  case __LOG8:
    ty = __INT8;
    w = 24;
    is_logical = TRUE;
    break;

  default:
    goto fmt_mismatch;
  }

  /* may need to deal with comma terminating the input field. w is
     overridden with the actual input field length, including the
     comma. curr_pos is adjusted accordingly and then the value
     (minus the comma) is processed. */

  comma_seen = FALSE;
  idx = tmp_idx = g->curr_pos;

  switch (code) {
  case FED_Gw_d:
    if (IS_INT(ty)) {
      w = fr_get_val(g);
      (void) fr_get_val(g);                    /*ignore w */
      if (g->fmt_base[g->fmt_pos] == FED_Ee) { /*ignore Ee for read*/
        g->fmt_pos++;
        (void) fr_get_val(g);
      }
      if (is_logical)
        goto L_shared;
      goto I_shared;
    }
    goto Ew_d_shared;

  case FED_G:
    if (IS_INT(ty)) {
      if (is_logical)
        goto L_shared;
      goto I_shared;
    }
    goto E_shared;

  case FED_Lw:
    w = fr_get_val(g);
    goto L_shared;
  case FED_L:
    w = 2;
  L_shared:
    if (!IS_INT(ty)) {
      if (__fortio_check_format())
        goto fmt_mismatch;
      switch (ty) {
      case __REAL4:
        type = __INT4;
        ty = __INT;
        break;
      case __REAL8:
      case __REAL16:
        type = __INT8;
        ty = __INT8;
        break;
      }
    }
    width = g->rec_len - g->curr_pos;
    if (width > w)
      width = w;
    if (gbl->decimal == FIO_COMMA) {
      while (width > 0 && g->rec_buff[tmp_idx] != ';')
        tmp_idx++, width--;
    } else {
      while (width > 0 && g->rec_buff[tmp_idx] != ',')
        tmp_idx++, width--;
    }
    if (width) {
      w = tmp_idx - idx + 1;
      comma_seen++;
    }
    ival = fr_move_fwd(w);
    if (ival != 0)
      return ival;
    while (w > 0 && g->rec_buff[idx] == ' ')
      idx++, w--;
    if (comma_seen)
      w -= 1;
    if (w <= 0)
      ival = FTN_FALSE;
    else {
      if (g->rec_buff[idx] == '.') {
        if (w < 2)
          return __fortio_error(FIO_EERR_DATA_CONVERSION);
        w--;
        idx++;
      }
      if (g->rec_buff[idx] == 'T' || g->rec_buff[idx] == 't')
        ival = FTN_TRUE;
      else if (g->rec_buff[idx] == 'F' || g->rec_buff[idx] == 'f')
        ival = FTN_FALSE;
      else
        return __fortio_error(FIO_EERR_DATA_CONVERSION);
    }
    if (ty == __INT8) {
      i8val[1] = 0;
      i8val[0] = ival;
    }
    break;

  case FED_Iw_m:
    w = fr_get_val(g);
    (void) fr_get_val(g);
    if (!IS_INT(ty)) {
      if (__fortio_check_format())
        goto fmt_mismatch;
      switch (ty) {
      case __REAL4:
        type = __INT4;
        ty = __INT;
        break;
      case __REAL8:
      case __REAL16:
        type = __INT8;
        ty = __INT8;
        break;
      }
    }
    goto I_shared;
  case FED_I:
    if (!IS_INT(ty)) {
      if (__fortio_check_format())
        goto fmt_mismatch;
      switch (ty) {
      case __REAL4:
        type = __INT4;
        ty = __INT;
        w = 12;
        break;
      case __REAL8:
      case __REAL16:
        type = __INT8;
        ty = __INT8;
        w = 24;
        break;
      }
    }
  I_shared:
    width = g->rec_len - g->curr_pos;
    if (width > w)
      width = w;
    if (gbl->decimal == FIO_COMMA) {
      while (width > 0 && g->rec_buff[tmp_idx] != ';')
        tmp_idx++, width--;
    } else {
      while (width > 0 && g->rec_buff[tmp_idx] != ',')
        tmp_idx++, width--;
    }
    if (width) {
      w = tmp_idx - idx + 1;
      comma_seen++;
    }
    ival = fr_move_fwd(w);
    if (ival != 0)
      return ival;
    while (g->rec_buff[idx] == ' ' && w > 0)
      idx++, w--;
    if (comma_seen)
      w -= 1;
    if (w == 0)
      ival = i8val[0] = i8val[1] = 0;
    else {
      c = g->rec_buff[idx];
      e = FALSE; /* sign flag */
      if (ty == __INT8) {
        /* Before the conversion, replace embedded blanks with '0'
         * if the blank_zero specifier is FIO_ZERO. Otherwise, need
         * to remove the blanks by shifting over the remaining
         * characters (this reduces the number of the characters
         * processed by the conversion routine).
         * Note that it has already been established that the character
         * at position 'idx' is not a blank.
         */
        int tmp_w = w;
        int cpos = idx; /* 'last' character copied */
        i8val[0] = i8val[1] = 0;
        tmp_idx = idx;
        while (--tmp_w > 0) {
          ++tmp_idx;
          c = g->rec_buff[tmp_idx];
          if (c != ' ')
            g->rec_buff[++cpos] = c;
          else {
            if (g->blank_zero == FIO_ZERO)
              g->rec_buff[++cpos] = '0';
            else
              w--; /* ignore blank and continue */
          }
        }
        if (__fort_atoxi64(&g->rec_buff[idx], i8val, w, 10) != 0)
          return __fortio_error(FIO_EERR_DATA_CONVERSION);
        break;
      }
      if (c == '-') {
        ++idx;
        w--;
        e = TRUE;
      } else if (c == '+') {
        ++idx;
        w--;
      }

      ival = 0;
      while (w-- > 0) {
        c = g->rec_buff[idx++];
        if (c > '9' || c < '0') {
          if (c != ' ')
            return __fortio_error(FIO_EERR_DATA_CONVERSION);
          if (g->blank_zero == FIO_ZERO)
            c = '0';
          else
            continue;
        }
        ival = (ival * 10) + (c - '0');
      }
      if (e)
        ival = -ival;
    }
    break;

  case FED_Ew_d:
  case FED_Fw_d:
  case FED_Dw_d:
  case FED_ENw_d:
  case FED_ESw_d:
  Ew_d_shared:
    w = fr_get_val(g);
    d = fr_get_val(g);
    if (code == FED_Ew_d || code == FED_Gw_d || code == FED_ENw_d ||
        code == FED_ESw_d)
      if (g->fmt_base[g->fmt_pos] == FED_Ee) { /*ignore Ee for read*/
        g->fmt_pos++;
        (void) fr_get_val(g);
      }
    switch (ty) {
    case __INT:
      if (__fortio_check_format())
        goto fmt_mismatch;
      switch (type) {
      case __INT1:
      case __LOG1:
      case __INT2:
      case __LOG2:
        goto fmt_mismatch;
      case __INT4:
      case __WORD4:
      case __LOG4:
        type = ty = __REAL4;
        ival = TRUE;
        break;
      default:
        goto fmt_mismatch;
      }
      break;
    case __INT8:
      if (__fortio_check_format())
        goto fmt_mismatch;
      type = ty = __REAL8;
      ival = FALSE;
      break;
    }
    goto f_shared;

  case FED_E:
  case FED_D:
  case FED_F:
  E_shared:
    switch (ty) {
    case __INT:
      if (__fortio_check_format())
        goto fmt_mismatch;
      switch (type) {
      case __INT1:
      case __LOG1:
      case __INT2:
      case __LOG2:
        goto fmt_mismatch;
      case __INT4:
      case __WORD4:
      case __LOG4:
        type = ty = __REAL4;
        w = REAL4_W;
        d = REAL4_D;
        ival = TRUE;
        break;
      default:
        goto fmt_mismatch;
      }
      break;
    case __INT8:
      if (__fortio_check_format())
        goto fmt_mismatch;
      type = ty = __REAL8;
      w = REAL8_W;
      d = REAL8_D;
      ival = FALSE;
      break;
    }
  f_shared:
    width = g->rec_len - g->curr_pos;
    if (width > w)
      width = w;
    if (gbl->decimal == FIO_COMMA) {
      while (width > 0 && g->rec_buff[tmp_idx] != ';') {
        tmp_idx++, width--;
      }
    } else {
      while (width > 0 && g->rec_buff[tmp_idx] != ',') {
        tmp_idx++, width--;
      }
    }
    if (width) {
      w = tmp_idx - idx + 1;
      comma_seen++;
    }
    ival = fr_move_fwd(w);
    if (ival != 0)
      return ival;
    if (comma_seen)
      w -= 1;
    dval = fr_getreal(&g->rec_buff[idx], w, d, &errflag);
    if (errflag)
      return __fortio_error(errflag);
    break;

  default:
  fmt_mismatch:
    return __fortio_error(FIO_EMISMATCH);
  }

  ival = fr_assign(item, type, ival, i8val, dval);
  if (ival != 0)
    return __fortio_error(ival);

  return 0;
}

/* ------------------------------------------------------------------ */

static int
fr_assign(char *item, int type, __BIGINT_T ival, DBLINT64 i8val, __BIGREAL_T dval)
{
  switch (type) {
  case __INT1:
    if (ival > 127 || ival < -128)
      return FIO_EERR_DATA_CONVERSION;
    *((__INT1_T *)item) = ival;
    break;
  case __LOG1:
    if (ival > 127 || ival < -128)
      return FIO_EERR_DATA_CONVERSION;
    *((__LOG1_T *)item) = ival;
    break;
  case __INT2:
    if (ival > 32767 || ival < -32768)
      return FIO_EERR_DATA_CONVERSION;
    *((__INT2_T *)item) = ival;
    break;
  case __LOG2:
    if (ival > 32767 || ival < -32768)
      return FIO_EERR_DATA_CONVERSION;
    *((__LOG2_T *)item) = ival;
    break;
  case __INT4:
    *((__INT4_T *)item) = ival;
    break;
  case __LOG4:
    *((__LOG4_T *)item) = ival;
    break;
  case __WORD4:
    *((__WORD4_T *)item) = ival;
    break;
  case __REAL4:
    *((__REAL4_T *)item) = (__REAL4_T)dval;
    break;
  case __REAL8:
    *((__REAL8_T *)item) = (__REAL8_T)dval;
    break;

  case __REAL16:
    *((__REAL16_T *)item) = (__REAL16_T)dval;
    break;

  case __LOG8:
    if (__ftn_32in64_)
      I64_MSH(i8val) = 0;
    ((__INT4_T *)item)[0] = i8val[0];
    ((__INT4_T *)item)[1] = i8val[1];
    break;
  case __INT8:
    if (__ftn_32in64_)
      I64_MSH(i8val) = 0;
    ((__INT4_T *)item)[0] = i8val[0];
    ((__INT4_T *)item)[1] = i8val[1];
    break;

  default:
    return FIO_EMISMATCH;
  }
  return 0;
}

/* ------------------------------------------------------------------- */

/*  local static variables for octal/hex conversion:  */

static int OZbase;
static unsigned char *OZbuff;
static int numbits;
static unsigned char *buff_pos, *buff_end;

static void fr_OZconv_init(int, int);
static void fr_OZbyte(int);
static void fr_Bbyte(int);

static int
fr_OZreadnum(int code, char *item, int type, int item_length)
{
  int idx, tmp_idx;
  G *g = gbl;
  int w, k, sz;
  int comma_seen, width;

  switch (type) {
  case __INT1:
    w = 7;
    sz = FIO_TYPE_SIZE(__INT1);
    break;

  case __LOG1:
    w = 7;
    sz = FIO_TYPE_SIZE(__LOG1);
    break;

  case __INT2:
    w = 7;
    sz = FIO_TYPE_SIZE(__INT2);
    break;

  case __LOG2:
    w = 7;
    sz = FIO_TYPE_SIZE(__LOG2);
    break;

  case __INT4:
    w = 12;
    sz = FIO_TYPE_SIZE(__INT4);
    break;

  case __LOG4:
    w = 12;
    sz = FIO_TYPE_SIZE(__LOG4);
    break;

  case __REAL4:
    w = 12;
    sz = FIO_TYPE_SIZE(__REAL4);
    break;

  case __WORD4:
    w = 12;
    sz = FIO_TYPE_SIZE(__WORD4);
    break;

  case __REAL8:
    w = 23;
    sz = FIO_TYPE_SIZE(__REAL8);
    break;

  case __REAL16:
    w = 44;
    sz = FIO_TYPE_SIZE(__REAL16);
    break;

  case __LOG8:
    w = 23;
    sz = FIO_TYPE_SIZE(__INT8);
    break;

  case __INT8:
    w = 23;
    sz = FIO_TYPE_SIZE(__INT8);
    break;

  case __STR:
    /*  note, just disallow FED_O and FED_Z with character item */
    if (code == FED_O || code == FED_Z)
      goto fmt_mismatch;
    sz = item_length;
    break;

  default: /*  actually, internal error ..... */
  fmt_mismatch:
    return __fortio_error(FIO_EMISMATCH);
  }

  /*  compute base and field width based on format code:  */

  OZbase = 16;
  if (code == FED_O || code == FED_Ow_m)
    OZbase = 8;
  if (code == FED_Zw_m || code == FED_Ow_m) {
    w = fr_get_val(g);
    (void) fr_get_val(g);
  }

  /* may need to deal with comma terminating the input field. w is
     overridden with the actual input field length, including the
     comma. curr_pos is adjusted accordingly and then the value
     (minus the comma) is processed. */

  comma_seen = FALSE;
  idx = tmp_idx = g->curr_pos;

  width = g->rec_len - g->curr_pos;
  if (width > w)
    width = w;
  while (width > 0 && g->rec_buff[tmp_idx] != ',')
    tmp_idx++, width--;
  if (width) {
    w = tmp_idx - idx + 1;
    comma_seen++;
  }
  k = fr_move_fwd(w);
  if (k != 0)
    return k;
  while (w > 0 && g->rec_buff[idx] == ' ')
    idx++, w--;
  if (comma_seen)
    w -= 1;
  fr_OZconv_init(w, sz);

  if (w != 0) {
    assert(w > 0);
    idx = idx + w - 1;
    while (w--) {
      int c;
      c = g->rec_buff[idx--];
      if (c > '7' || c < '0') {
        if (c == ' ') {
          if (g->blank_zero == FIO_ZERO)
            c = '0';
          else
            continue;
        } else if (OZbase == 8)
          return __fortio_error(FIO_EERR_DATA_CONVERSION);
        else if (c >= 'a' && c <= 'z') /* convert to upper */
          c = c + ('A' - 'a');
        if (!(c >= '0' && c <= '9') && !(c >= 'A' && c <= 'F'))
          return __fortio_error(FIO_EERR_DATA_CONVERSION);
      }
      fr_OZbyte(c);
    }
  }

  /*  remove leading zeros from byte buffer just created:  */

  while (buff_pos < buff_end) {
    if (*buff_pos != 0)
      break;
    buff_pos++;
  }

  if (buff_end - buff_pos > sz) /*  overflow */
    return __fortio_error(FIO_EERR_DATA_CONVERSION);
  buff_pos = buff_end - sz;

  switch (sz) {
  case 1:
    *((char *)item) = *buff_pos;
    break;

  case 2:
    *(item + 1) = *buff_pos;
    *(item) = *(buff_pos + 1);
    break;

  case 4:
    *(item + 3) = *buff_pos;
    *(item + 2) = *(buff_pos + 1);
    *(item + 1) = *(buff_pos + 2);
    *(item) = *(buff_pos + 3);
    break;

  case 8:
    *(item + 7) = *buff_pos;
    *(item + 6) = *(buff_pos + 1);
    *(item + 5) = *(buff_pos + 2);
    *(item + 4) = *(buff_pos + 3);
    *(item + 3) = *(buff_pos + 4);
    *(item + 2) = *(buff_pos + 5);
    *(item + 1) = *(buff_pos + 6);
    *(item) = *(buff_pos + 7);
    break;

  case 16:
    *(item + 15) = *(buff_pos);
    *(item + 14) = *(buff_pos + 1);
    *(item + 13) = *(buff_pos + 2);
    *(item + 12) = *(buff_pos + 3);
    *(item + 11) = *(buff_pos + 4);
    *(item + 10) = *(buff_pos + 5);
    *(item + 9) = *(buff_pos + 6);
    *(item + 8) = *(buff_pos + 7);
    *(item + 7) = *(buff_pos + 8);
    *(item + 6) = *(buff_pos + 9);
    *(item + 5) = *(buff_pos + 10);
    *(item + 4) = *(buff_pos + 11);
    *(item + 3) = *(buff_pos + 12);
    *(item + 2) = *(buff_pos + 13);
    *(item + 1) = *(buff_pos + 14);
    *(item) = *(buff_pos + 15);
    break;

  default:
    assert(type == __STR);
/* NOTE: str type might go thru other cases if the size is right,
   but only str type should get here */
    while (sz--)
      *(item++) = buff_pos[sz];
    break;
  }
  return 0;
}

/* ------------------------------------------------------------------- */

static void
fr_OZconv_init(int w, int sz)
{
  static int buff_len = 0;
  int len;

  if (OZbase == 16)
    len = (w + 1) >> 1;
  else if (OZbase == 2)
    len = (w + 7) >> 3;
  else
    len = ((w * 3) + 7) >> 3;
  if (len < sz)
    len = sz;

  if (buff_len < len) {
    if (buff_len != 0)
      free(OZbuff);
    buff_len = len + 8;
    OZbuff = (unsigned char *)malloc(buff_len);
  }

  buff_end = OZbuff + len;
  buff_pos = OZbuff + len; /* start at right end of buffer */
  numbits = 0;
  while (--len >= 0)
    *(OZbuff + len) = 0;
}

static void
fr_OZbyte(int c)
{
  if (OZbase == 16) {
    if (c <= '9')
      c -= '0';
    else
      c -= ('A' - 10);
    assert(c >= 0 && c <= 15);
    numbits += 4;
    if ((numbits & 0x7) != 0) {
      buff_pos--;
      *buff_pos = c;
    } else
      *buff_pos |= (c << 4);
  } else {
    int k;
    assert(c >= '0' && c <= '7');
    c = c - '0';

    k = (numbits & 0x7); /* number of bits in last-filled byte */
    if (k == 0) {
      buff_pos--;
      *buff_pos = c; /* first value put into this byte */
    } else
      *buff_pos |= (c << k);

    numbits += 3;
    if ((k = (numbits & 0x7)) < 3) {
      if (k > 0) {
        buff_pos--;
        assert(buff_pos >= OZbuff);
        *buff_pos = (c >> (3 - k));
      }
    }
  }

}

/* ------------------------------------------------------------------- */

static int
fr_Breadnum(char *item, int type, int item_length)
{
  char *p, *tmp_ptr;
  G *g = gbl;
  int w, k, sz;
  int comma_seen, width;

  switch (type) {
  case __INT1:
    sz = FIO_TYPE_SIZE(__INT1);
    break;

  case __LOG1:
    sz = FIO_TYPE_SIZE(__LOG1);
    break;

  case __INT2:
    sz = FIO_TYPE_SIZE(__INT2);
    break;

  case __LOG2:
    sz = FIO_TYPE_SIZE(__LOG2);
    break;

  case __INT4:
    sz = FIO_TYPE_SIZE(__INT4);
    break;

  case __LOG4:
    sz = FIO_TYPE_SIZE(__LOG4);
    break;

  case __REAL4:
    sz = FIO_TYPE_SIZE(__REAL4);
    break;

  case __WORD4:
    sz = FIO_TYPE_SIZE(__WORD4);
    break;

  case __REAL8:
    sz = FIO_TYPE_SIZE(__REAL8);
    break;

  case __REAL16:
    sz = FIO_TYPE_SIZE(__REAL16);
    break;

  case __LOG8:
    sz = FIO_TYPE_SIZE(__LOG8);
    break;

  case __INT8:
    sz = FIO_TYPE_SIZE(__INT8);
    break;

  case __STR:
    sz = item_length;
    break;

  default: /*  actually, internal error ..... */
    return __fortio_error(FIO_EMISMATCH);
  }

  /*  compute base and field width based on format code:  */

  OZbase = 2;
  w = fr_get_val(g);
  (void) fr_get_val(g);

  /* may need to deal with comma terminating the input field. w is
     overridden with the actual input field length, including the
     comma. curr_pos is adjusted accordingly and then the value
     (minus the comma) is processed. */

  comma_seen = FALSE;
  p = tmp_ptr = g->rec_buff + g->curr_pos;

  width = g->rec_len - g->curr_pos;
  if (width > w)
    width = w;
  while (*tmp_ptr != ',' && width > 0)
    tmp_ptr++, width--;
  if (width) {
    w = (tmp_ptr - p) + 1;
    comma_seen++;
  }
  k = fr_move_fwd(w);
  if (k != 0)
    return k;
  while (*p == ' ' && w > 0)
    p++, w--;
  if (comma_seen)
    w -= 1;
  fr_OZconv_init(w, sz);

  if (w != 0) {
    assert(w > 0);
    p = p + w - 1;
    while (w--) {
      int c;
      c = *p--;
      if (c != '0' && c != '1') {
        if (c == ' ') {
          if (g->blank_zero == FIO_ZERO)
            c = '0';
          else
            continue;
        } else
          return __fortio_error(FIO_EERR_DATA_CONVERSION);
      }
      fr_Bbyte(c);
    }
  }

  /*  remove leading zeros from byte buffer just created:  */

  while (buff_pos < buff_end) {
    if (*buff_pos != 0)
      break;
    buff_pos++;
  }

  if (buff_end - buff_pos > sz) /*  overflow */
    return __fortio_error(FIO_EERR_DATA_CONVERSION);
  buff_pos = buff_end - sz;

  switch (sz) {
  case 1:
    *((char *)item) = *buff_pos;
    break;

  case 2:
    *(item + 1) = *buff_pos;
    *(item) = *(buff_pos + 1);
    break;

  case 4:
    *(item + 3) = *buff_pos;
    *(item + 2) = *(buff_pos + 1);
    *(item + 1) = *(buff_pos + 2);
    *(item) = *(buff_pos + 3);
    break;

  case 8:
    *(item + 7) = *buff_pos;
    *(item + 6) = *(buff_pos + 1);
    *(item + 5) = *(buff_pos + 2);
    *(item + 4) = *(buff_pos + 3);
    *(item + 3) = *(buff_pos + 4);
    *(item + 2) = *(buff_pos + 5);
    *(item + 1) = *(buff_pos + 6);
    *(item) = *(buff_pos + 7);
    break;

  case 16:
    *(item + 15) = *(buff_pos);
    *(item + 14) = *(buff_pos + 1);
    *(item + 13) = *(buff_pos + 2);
    *(item + 12) = *(buff_pos + 3);
    *(item + 11) = *(buff_pos + 4);
    *(item + 10) = *(buff_pos + 5);
    *(item + 9) = *(buff_pos + 6);
    *(item + 8) = *(buff_pos + 7);
    *(item + 7) = *(buff_pos + 8);
    *(item + 6) = *(buff_pos + 9);
    *(item + 5) = *(buff_pos + 10);
    *(item + 4) = *(buff_pos + 11);
    *(item + 3) = *(buff_pos + 12);
    *(item + 2) = *(buff_pos + 13);
    *(item + 1) = *(buff_pos + 14);
    *(item) = *(buff_pos + 15);
    break;

  default:
    assert(type == __STR);
    while (sz--)
      *(item++) = buff_pos[sz];
    break;
  }
  return 0;
}

static void
fr_Bbyte(int c)
{
  c -= '0';
  if (numbits == 0) {
    buff_pos--;
    *buff_pos = c;
  } else
    *buff_pos |= (c << numbits);

  numbits = (numbits + 1) & 7;

}

/* ------------------------------------------------------------------- */

static __BIGREAL_T
fr_getreal(char *p, int w, int d, int *errflag)
{
#define MAXFLEN 400
  int expval = 0;
  int dotflag = -1;
  char *errp;
  __BIGREAL_T dval;
  char buff[MAXFLEN], buff2[MAXFLEN];
  int ipos = 0, jpos = 0;
  int c;
  unsigned int nval, ntval, nshplaces;
  union ieee ieee_v;
  bool negflag = FALSE;
  bool expflag = FALSE;

  *errflag = 0;

  while (w > 0 && *p == ' ')
    p++, w--; /* scan past blanks */
  if (w == 0)
    return 0.0;

  if (w + 5 > MAXFLEN) /* fixed buffer overflow */
    goto conv_error;

  if (*p == '-' || *p == '+') {
    if (*p == '-')
      buff[ipos++] = '-';
    p++;
    if (--w == 0)
      return 0.0;
  }

  while (w--) {
    c = *p++;
    if (c > '9' || c < '0') {
      if (c == ' ') {
        if (gbl->blank_zero == FIO_ZERO)
          c = '0';
        else
          continue;
      } else if (c == '.') {
        if (dotflag != -1) {
          goto conv_error;
        }
        dotflag = ipos;
        continue;
      } else if (c == ',' && gbl->decimal == FIO_COMMA) {
        dotflag = ipos;
        continue;
      } else if (c == 'i' || c == 'I' || c == 'n' || c == 'N') {
        goto either_inf_nan_or_error;
      } else
        break;
    }
    buff[ipos++] = c;
  }

  if (ipos == 0) /* need at least one digit for strtod */
    buff[ipos++] = '0';

  if (w == -1)
    goto after_exponent;

  if (c != 'E' && c != 'D' && c != 'Q' && c != 'e' && c != 'd' && c != 'q'  &&
      c != '+' && c != '-')
    goto conv_error;

  if (c == '+' || c == '-')
    p--, w++; /* backup so that sign is processed below... */

  expflag = TRUE;

  while (w > 0 && *p == ' ')
    p++, w--; /* scan past blanks */
  if (w == 0)
    goto after_exponent;

  if (*p == '-' || *p == '+') {
    if (--w == 0)
      goto after_exponent;
    if (*p == '-')
      negflag = TRUE;
    p++;
  }

  while (w--) {
    c = *p++;
    if (c > '9' || c < '0') {
      if (c == ' ') {
        if (gbl->blank_zero == FIO_ZERO)
          c = '0';
        else
          continue;
      } else
        break;
    }
    buff2[jpos++] = c;
  }

  if (w != -1)
    goto conv_error;

  /* convert string in buff2 to integer: */

  for (p = buff2; jpos-- > 0; p++)
    expval = (10 * expval) + (*p - '0');
  if (negflag > 0)
    expval = -expval;

after_exponent:
  if (!expflag)
    expval -= gbl->scale_factor;

  if (dotflag != -1)
    expval -= (ipos - dotflag);
  else /*if (!expflag)*/
    expval -= d;

  if (expval != 0) {
    buff[ipos] = 'E';
    sprintf(buff + ipos + 1, "%d", expval);
  } else
    buff[ipos] = '\0';

  dval = __io_strtold(buff, &errp);

  if (errp != buff)
    return dval; /* successful conversion */

either_inf_nan_or_error:
  if (ipos <= 1) {
    if ((ipos == 1) && (buff[0] == '-'))
      ieee_v.v.s = 1;
    else
      ieee_v.v.s = 0;
    if (c == 'i' || c == 'I') {
      if (w == 0)
        goto conv_error;
      w--;
      c = *p++;
      if (c == 'n' || c == 'N') {
        if (w == 0)
          goto conv_error;
        w--;
        c = *p++;
        if (c == 'f' || c == 'F') {
          if (w == 0) {
            ieee_v.i[0] = 0x0;
            ieee_v.v.hm = 0x0;
            ieee_v.v.e = 2047;
            dval = ieee_v.d;
            return dval; /* successful conversion */
          }
          w--; /* scan past blanks */
          while (w > 0 && *p == ' ')
            p++, w--;
          if (w == 0) {
            ieee_v.i[0] = 0x0;
            ieee_v.v.hm = 0x0;
            ieee_v.v.e = 2047;
            dval = ieee_v.d;
            return dval; /* successful conversion */
          }
          c = *p++;
          if (c == 'i' || c == 'I') {
            if (w == 0)
              goto conv_error;
            w--;
            c = *p++;
            if (c == 'n' || c == 'N') {
              if (w == 0)
                goto conv_error;
              w--;
              c = *p++;
              if (c == 'i' || c == 'I') {
                if (w == 0)
                  goto conv_error;
                w--;
                c = *p++;
                if (c == 't' || c == 'T') {
                  if (w == 0)
                    goto conv_error;
                  w--;
                  c = *p++;
                  if (c == 'y' || c == 'Y') {
                    if (w == 0) {
                      ieee_v.i[0] = 0x0;
                      ieee_v.v.hm = 0x0;
                      ieee_v.v.e = 2047;
                      dval = ieee_v.d;
                      return dval; /* success */
                    }
                    w--; /* scan past blanks */
                    while (w > 0 && *p == ' ')
                      p++, w--;
                    if (w == 0) {
                      ieee_v.i[0] = 0x0;
                      ieee_v.v.hm = 0x0;
                      ieee_v.v.e = 2047;
                      dval = ieee_v.d;
                      return dval; /* success */
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else if (c == 'n' || c == 'N') {
      if (w == 0)
        goto conv_error;
      w--;
      c = *p++;
      if (c == 'a' || c == 'A') {
        if (w == 0)
          goto conv_error;
        w--;
        c = *p++;
        if (c == 'n' || c == 'N') {
          if (w == 0) {
            ieee_v.i[0] = 0x0;
            ieee_v.v.hm = 0x80000;
            ieee_v.v.e = 2047;
            dval = ieee_v.d;
            return dval; /* Found a NaN */
          }
          while (w > 0 && *p == ' ')
            p++, w--;
          if (w == 0) {
            ieee_v.i[0] = 0x0;
            ieee_v.v.hm = 0x80000;
            ieee_v.v.e = 2047;
            dval = ieee_v.d;
            return dval; /* successful conversion */
          }
          c = *p++;
          if (c == '(') {
            if (w == 0)
              goto conv_error;
            ieee_v.i[0] = 0x0;
            ieee_v.v.hm = 0x0;
            ieee_v.v.e = 2047;
            ntval = 0;
            w--;
            c = *p++;
            nshplaces = 48;
            while (w--) {
              if ((c >= '0') && (c <= '9'))
                nval = c - '0';
              else if ((c >= 'a') && (c <= 'f'))
                nval = c - 'a' + 10;
              else if ((c >= 'A') && (c <= 'F'))
                nval = c - 'A' + 10;
              else if (c == ')')
                break;
              else
                goto conv_error;
              if (nshplaces > 28)
                ieee_v.v.hm |= nval << (nshplaces - 32);
              else
                ieee_v.v.lm |= nval << nshplaces;
              ntval += nval;
              c = *p++;
              if (nshplaces)
                nshplaces -= 4;
            }
            if (c == ')') {
              while (w > 0 && *p == ' ')
                p++, w--;
              if (w <= 0) {
                /* Set this quiet */
                if (ntval == 0)
                  ieee_v.v.hm |= 0x80000;
                dval = ieee_v.d;
                return dval; /* NaN success */
              }
            }
          }
        }
      }
    }
  }

conv_error:
  *errflag = FIO_EERR_DATA_CONVERSION;
  return 0.0;
}

/* ------------------------------------------------------------------- */

static int
fr_move_fwd(int len)
{
  G *g = gbl;

  move_fwd_eor = 0;
  g->curr_pos += len;
  if (g->curr_pos > g->rec_len) {
    if (!g->internal_file && g->fcb->acc == FIO_DIRECT)
      return __fortio_error(FIO_ETOOBIG);

    /*  sequential, external, formatted input - extend record w blanks: */

    if (g->curr_pos > g->obuff_len) {
      int err;
      err = realloc_obuff(g, (size_t)(g->curr_pos + INIT_BUFF_LEN));
      if (err)
        return err;
    }

    /* set size variable before rec_len is adjusted */
    if (g->nonadvance) {
      if (g->size_ptr != (__INT8_T *)0)
        *g->size_ptr = (__INT8_T)g->rec_len;
      move_fwd_eor = 1;
    }

    while (g->rec_len < g->curr_pos)
      g->rec_buff[g->rec_len++] = ' ';
  }
  g->max_pos = g->curr_pos;
  return 0;
}

/* ------------------------------------------------------------------- */

static int
fr_read_record(void)
{
  G *g = gbl;
  int idx = 0;

  if (g->internal_file) {
    if (g->num_internal_recs <= 0)
      return FIO_EEOF;
    g->num_internal_recs--;
    g->rec_buff += g->rec_len; /* point to next record */
  } else {                     /* external file */
    FIO_FCB *f = g->fcb;
    FILE *fp = f->fp;
    if (f->pread) {
      int idx = 0;
      char *p = f->pread;

      f->pread = 0;
      while (TRUE) { /* read one char per iteration until '\n' */
        int c = *(p++);
        if (c == EOF) {
          if (__io_feof(fp)) {
            if (idx)
              break;
            return FIO_EEOF;
          }
          return __io_errno();
        }
        if (c == '\r' && EOR_CRLF) {
          c = *(p++);
          if (c == '\n')
            break;
          --p;
          c = '\r';
        }
        if (c == '\n')
          break;

        /* else, put character into record:  */
        g->obuff[idx++] = c;
        if (idx >= g->obuff_len) {
          int err;
          err = realloc_obuff(g, (size_t)(g->obuff_len + INIT_BUFF_LEN));
          if (err)
            return err;
        }
      }
      g->rec_len = idx;
    } else {

      ++(f->nextrec); /* increment here to get errmsg recnum right */

      if (f->acc == FIO_DIRECT) {
        if (f->nextrec > f->maxrec + 1)
          return FIO_EDREAD; /* attempt to read non-existent rec */
        if (__io_fread(g->rec_buff, 1, g->rec_len, fp) != g->rec_len)
          return __io_errno();
      } else { /* sequential read */
        idx = 0;

        while (TRUE) { /* read one char per iteration until '\n' */
          int c = __io_fgetc(fp);
          if (c == EOF) {
            if (__io_feof(fp)) {
              if (idx)
                break;
              if (g->nonadvance && !g->eor_seen && g->rec_len != 0)
                break;

              g->eor_seen = 0;
              return FIO_EEOF;
            }
            return __io_errno();
          }
          if (c == '\r' && EOR_CRLF) {
            c = __io_fgetc(fp);
            if (c == '\n') {
              g->eor_len = 2;
              break;
            }
            __io_ungetc(c, fp);
            c = '\r';
          }
          if (c == '\n') {
            g->eor_len = 1;
            break;
          }

          /* else, put character into record:  */
          g->obuff[idx++] = c;
          if (idx >= g->obuff_len) {
            int err;
            err = realloc_obuff(g, (size_t)(g->obuff_len + INIT_BUFF_LEN));
            if (err)
              return err;
          }
        }
        g->rec_len = idx;
      }
    }
  }
  if (g->nonadvance && g->rec_len == 0) {
    g->eor_seen = 1;
    return FIO_EEOR;
  }

  g->curr_pos = 0;
  g->max_pos = 0;
  g->eor_seen = 0;

  return 0;
}

/* ------------------------------------------------------------------ */

static int
malloc_obuff(G *g, size_t len)
{
  if (g->obuff) {
    free(g->obuff);
    g->obuff = NULL;
  }
  g->obuff = malloc(len);
  if (g->obuff == NULL) {
    return __fortio_error(FIO_ENOMEM);
  }
  g->rec_buff = g->obuff;
  g->obuff_len = len;
  return 0;
}

/* ------------------------------------------------------------------ */

static int
realloc_obuff(G *g, size_t len)
{
  assert(g->obuff == g->rec_buff);
  g->obuff = realloc(g->obuff, len);
  if (g->obuff == NULL) {
    return __fortio_error(FIO_ENOMEM);
  }
  g->obuff_len = len;
  g->rec_buff = g->obuff;
  return 0;
}

/* ------------------------------------------------------------------ */

static int
_f90io_fmtr_end(void)
{
  G *g = gbl;
  int err;
  if (fioFcbTbls.error)
    return ERR_FLAG;
  if (fioFcbTbls.eof)
    return EOF_FLAG;

  if (!g->internal_file && g->same_fcb)
    return 0;
  err = fr_read((char *)NULL, -1, -1);
  if (err)
    return err;

  if (g->nonadvance) {
    if (!g->internal_file) {
      if (g->curr_pos <= g->rec_len) {
        /* just reposition the file pointer to the point after
         * the last character in the record which was transferred.
         */
        int i;
        i = g->rec_len - g->curr_pos + g->eor_len;
        --(g->fcb->nextrec); /* decr errmsg recnum */
        if (__io_fseek(g->fcb->fp, (seekoffx_t)-i, SEEK_CUR) != 0) {
          if (g->fcb->stdunit) {
            /*
             * Can't seek stdin, but need to leave the postion
             * after the last character.  Since it's stdin,
             * just mark it as 'eor'.
             */
            g->fcb->eor_flag = TRUE;
            return 0;
          }
          return __fortio_error(__io_errno());
        }
      }
    }
  }

  return 0;
}

__INT_T
ENTF90IO(FMTR_END, fmtr_end)()
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_fmtr_end();

  save_samefcb();
  free_gbl();
  restore_gbl();
  __fortio_fmtend();
  __fortio_errend03();
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(FMTR_END, fmtr_end)()
{
  int s = 0;

  s = _f90io_fmtr_end();

  save_samefcb();
  free_gbl();
  restore_gbl();
  __fortio_fmtend();
  __fortio_errend03();
  return s;
}

#define CHAR_ONLY 1
#define CHAR_AND_VLIST 2

/* --------------------------------------------------------------------- */
__INT_T
ENTF90IO(DTS_FMTR,dts_fmtr)(char** cptr, void** iptr, __INT_T* len, F90_Desc* sd, int* flag)
{
  int i;
  INT code, k, first;
  __INT_T ubnd = 0;
  __INT8_T **tptr8 = (__INT8_T **)iptr;
  INT **tptr4 = (INT **)iptr;
  G *g = gbl;
  while (TRUE) {
    code = fr_get_fmtcode();
    switch (code) {
    case FED_END:
      if (!g->repeat_flag && !fioFcbTbls.error) {
        i = fr_read_record();
        if (i != 0)
          return __fortio_error(i);
      }
      g->fmt_pos = g->fmt_base[g->fmt_pos];
      break;
    case FED_T:
      i = fr_get_val(g);
      if (i < 1) {
        i = 0;
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      }

      g->curr_pos = i - 1;
      break;
    case FED_TL:
      i = fr_get_val(g);

      if (i < 1) {
        i = 0;
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      }

      g->curr_pos -= i;
      if (g->curr_pos < 0)
        g->curr_pos = 0;
      break;
    case FED_TR:
    case FED_X:
      i = fr_get_val(g);
      if (i < 1) {
        i = 0;
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      }
      g->curr_pos += i;
      break;
    case FED_DT:
      goto exit_loop;
    default:
      break;
    }
  }
exit_loop:

  /* get DT_ value */
  k = fr_get_val(g);
  switch (k) {
  case CHAR_ONLY:
    k = fr_get_val(g); /* length of this DT....*/
    *len = k;
    *cptr = (char *)&(g->fmt_base[g->fmt_pos]);
    *iptr = NULL; /* need to make its descriptor size to 0 */
    /* update int array descriptor */
    if (sd) {
      if (*flag == 3 || *flag == 2) {
        get_vlist_desc_i8(sd, ubnd);
      } else {
        get_vlist_desc(sd, ubnd);
      }
    }
    g->fmt_pos += (k + 3) >> 2;
    break;
  case CHAR_AND_VLIST:
    k = fr_get_val(g);
    *len = k;
    *cptr = (char *)&(g->fmt_base[g->fmt_pos]);
    g->fmt_pos += (k + 3) >> 2;
    k = fr_get_val(g);     /* how many item is the vlist */
    first = fr_get_val(g); /* changed by for loop below */

    if (*flag == 3 || *flag == 1) {
      *tptr8 = (__INT8_T *)&(g->fmt_base[g->fmt_pos]);
    } else {
      *tptr4 = (INT *)&(g->fmt_base[g->fmt_pos]);
      if (first == 0) {
        (g->fmt_base[(g->fmt_pos) - 1]) = 1;
        for (i = 0; i < k; ++i) {
          (*tptr4)[i] = (INT)((*tptr8)[i]);
        }
      }
    }
    ubnd = k;
    if (sd) {
      if (*flag == 3 || *flag == 2) {
        get_vlist_desc_i8(sd, ubnd);
      } else {
        get_vlist_desc(sd, ubnd);
      }
    }
    g->fmt_pos += 2*k;
    break;
  default:
    /* error */
    break;
  }
  return 0;
}
