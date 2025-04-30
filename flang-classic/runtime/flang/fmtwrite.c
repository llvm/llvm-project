/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief
 * Fortran formatted write support.
 */

#include "global.h"
#include "feddesc.h"
#include "format.h"
#include "format-double.h"
#include <string.h>

#define RPSTACK_SIZE 20 /* determines max paren nesting level */

typedef struct {
  int count;
  int code;
  int fmtpos;
} rpstack_struct;

static rpstack_struct rpstack[RPSTACK_SIZE];

#define INIT_BUFF_LEN 200

struct struct_G {
  bool internal_file;
  char *internal_unit;
  long obuff_len;
  char *obuff;
  char *rec_buff;
  FIO_FCB *fcb;
  INT *fmt_base;
  long rec_len;  /* for direct access or internal files, rec_len
                    is both buffer len and actual record len. can be negative */
  long max_pos;  /* for variable length records, the actual
                    record len is max_pos.  max_pos is not
                    defined for internal files.  */
  long curr_pos; /* because of X, TL, curr_pos may be greater
                    than max_pos in some cases. */
  bool record_written;
  int fmt_pos;
  int scale_factor;
  int num_internal_recs;
  int rpstack_top;
  short decimal; /* FIO_ COMMA, POINT, NONE */
  short round;   /* FIO_ UP, DOWN, ZERO, NEAREST, COMPATIBLE,
                  *      PROCESSOR_DEFINED, NONE
                  */
  short sign;    /* FIO_ PLUS, SUPPRESS, PROCESSOR_DEFINED,
                  *      NONE
                  */
  bool plus_flag;
  bool suppress_crlf;
  bool repeat_flag;
  bool nonadvance;
  bool fmt_alloc; /* if this fmt is allocated */

  rpstack_struct rpstack[RPSTACK_SIZE];
  struct struct_G *same_fcb;
  int same_fcb_idx;
};

#define GBL_SIZE 5
typedef struct struct_G G;

static G static_gbl[GBL_SIZE];
static G *gbl = &static_gbl[0];
static G *gbl_head = &static_gbl[0];
static int gbl_avl = 0;
static int gbl_size = GBL_SIZE;

static int fw_write(char *, int, int);
static int fw_slashes(G *, int);
static int fw_end_nonadvance(void);
static INT fw_get_fmtcode(void);
static INT fw_get_val(G *);
static int fw_writenum(int, char *, int);
static int fw_OZwritenum(int, char *, int, int);
static int fw_Bwritenum(char *, int, __CLEN_T);
static int fw_write_item(const char *, int);
static int fw_check_size(long);
static int fw_write_record(void);
/* ----------------------------------------------------------------------- */
static void
save_gbl()
{
  int i;
  if (gbl_avl) {
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
  if (gbl_avl) {
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
    tmp_gbl->curr_pos = gbl->curr_pos;
    tmp_gbl->obuff_len = gbl->obuff_len;
    tmp_gbl->rec_buff = gbl->rec_buff;
    tmp_gbl->obuff = gbl->obuff;
    tmp_gbl->max_pos = gbl->max_pos;
    tmp_gbl->rec_len = gbl->rec_len;
    tmp_gbl->record_written = gbl->record_written;
  }
}

static void
allocate_new_gbl()
{
  G *tmp_gbl;
  char *obuff = 0;
  char *rec_buff = 0;
  long obuff_len = 0;
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
      gbl_head = (G *)realloc(gbl_head, (size_t)(gsize * gbl_size));
      memset(gbl_head + gbl_avl, 0, gsize * GBL_SIZE);
    }
  }
  gbl = &gbl_head[gbl_avl];
  if (gbl_avl == 0) { /* keep buffer instead of allocate every time for
                         non-recursive i/o */
    obuff = gbl->obuff;
    obuff_len = gbl->obuff_len;
    rec_buff = gbl->rec_buff;
  } else if (gbl->obuff && !gbl->same_fcb) {
    free(gbl->obuff);
    gbl->obuff = NULL;
  }
  memset(gbl, 0, gsize);
  if (gbl_avl == 0) {
    gbl->obuff = obuff;
    gbl->obuff_len = obuff_len;
    gbl->rec_buff = rec_buff;
  }
  ++gbl_avl;
}

static void
free_gbl()
{
  G *tmp_gbl = gbl;
  if (tmp_gbl && tmp_gbl->fmt_alloc) {
    free(tmp_gbl->fmt_base);
    tmp_gbl->fmt_base = NULL;
    tmp_gbl->fmt_alloc = 0;
  }
  --gbl_avl;
  if (gbl_avl <= 0)
    gbl_avl = 0;
  if (gbl_avl == 0)
    gbl = &gbl_head[0];
  else
    gbl = &gbl_head[gbl_avl - 1];
}

/* ----------------------------------------------------------------------- */

static int
fw_init(__INT_T *unit,   /* unit number */
        __INT_T *rec,    /* record number for direct access I/O */
        __INT_T *bitv,   /* same as for ENTF90IO(open_) */
        __INT_T *iostat, /* same as for ENTF90IO(open_) */
        __INT_T *fmt,    /* encoded format array.  A value of
                          * NULL indicates that format was
                          * previously encoded by a call to
                          * ENTF90IO(encode_fmt) */
        char *advance,   /* 'YES', 'NO', or 'NULL' */
        __CLEN_T advancelen)
{
  FIO_FCB *f;
  long len;
  G *g = gbl;
  G *tmp_gbl;
  int i;

  /* ----- perform initializations.  Get pointer to file control block: */

  __fortio_errinit03(*unit, *bitv, iostat, "formatted write");
  f = __fortio_rwinit(*unit, FIO_FORMATTED, rec, 1 /*write*/);

  if (f == NULL)
    return ERR_FLAG;

  g->fcb = f;

  /* ---- set up base pointer to encoded format:  */

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

  /* ---- set up char buffer to hold formatted record: */

  if (f->acc == FIO_DIRECT)
    len = (long)f->reclen;
  else
    len = INIT_BUFF_LEN;

  /* check all recursive fcb, starting from latest recursive */
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
    gbl->curr_pos = tmp_gbl->curr_pos;
    gbl->max_pos = tmp_gbl->max_pos;
    gbl->rec_len = tmp_gbl->rec_len;
    gbl->obuff_len = tmp_gbl->obuff_len;
    gbl->rec_buff = tmp_gbl->rec_buff;
    gbl->obuff = tmp_gbl->obuff;
    gbl->same_fcb = tmp_gbl;
    gbl->same_fcb_idx = i;
  } else {
    gbl->same_fcb = NULL;
    gbl->same_fcb_idx = 0;
    g->curr_pos = 0;
  }

  if (g->obuff_len < len) {
    if (g->obuff_len != 0)
      free(g->obuff);
    g->obuff = NULL;
    if (tmp_gbl) {
      len = len + gbl->obuff_len;
      g->obuff = realloc(g->obuff, len);
    } else {
      g->obuff = malloc(len);
    }
    if (g->obuff == NULL)
      return __fortio_error(FIO_ENOMEM);
    if (tmp_gbl)
      memset((void *)(g->obuff + gbl->obuff_len), ' ',
             (size_t)(len - g->obuff_len));
    else
      memset(g->obuff, ' ', len);
    g->obuff_len = len;
  }
  g->rec_buff = g->obuff;
  if (f->acc == FIO_DIRECT) {
    if (advancelen)
      return __fortio_error(FIO_ECOMPAT);
    g->rec_len = len;
  } else
    g->rec_len = g->obuff_len;

  g->max_pos = 0;

  if (f->skip) {
    memcpy(g->rec_buff + gbl->curr_pos, f->skip_buff, f->skip);
    g->max_pos = f->skip;
    f->skip = 0;
    free(f->skip_buff);
  }

  /* ----- initialize other variables:  */

  g->scale_factor = 0;
  g->sign = f->sign;
  g->suppress_crlf = FALSE;
  g->record_written = FALSE;
  g->decimal = f->decimal;
  g->round = f->round;
  g->repeat_flag = FALSE;
  g->rpstack_top = -1;

  if (g->sign == FIO_PLUS)
    g->plus_flag = TRUE;
  else
    g->plus_flag = FALSE;

  if (advancelen && __fortio_eq_str(advance, advancelen, "NO"))
    g->nonadvance = TRUE;
  else
    g->nonadvance = FALSE;
  return 0;
}

__INT_T
ENTF90IO(FMTW_INITA, fmtw_inita)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * NULL indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 DCHAR(advance)   /* 'YES', 'NO', or 'NULL' */
 DCLEN64(advance))
{
  G *g;
  char *advadr;
  __CLEN_T advlen;
  int s = 0;

  save_gbl();
  allocate_new_gbl();
  g = gbl;
  g->internal_file = FALSE;

  if (ISPRESENTC(advance)) {
    advadr = CADR(advance);
    advlen = CLEN(advance);
  } else {
    advadr = NULL;
    advlen = 0;
  }

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = fw_init(unit, rec, bitv, iostat, fmt, advadr, advlen);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTW_INIT, fmtw_init)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * NULL indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 DCHAR(advance)   /* 'YES', 'NO', or 'NULL' */
 DCLEN(advance))
{
  return ENTF90IO(FMTW_INITA, fmtw_inita) (unit, rec, bitv, iostat, fmt,
                             CADR(advance), (__CLEN_T)CLEN(advance));
}

__INT_T
ENTF90IO(FMTW_INIT03A, fmtw_init03a)
(__INT_T *istat, DCHAR(decimal), DCHAR(sign),
 DCHAR(round) DCLEN64(decimal) DCLEN64(sign) DCLEN64(round))
{
  int s = *istat;

  if (s)
    return DIST_STATUS_BCST(s);

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC) {
    if (ISPRESENTC(decimal)) {
      if (__fortio_eq_str(CADR(decimal), CLEN(decimal), "COMMA")) {
        gbl->decimal = FIO_COMMA;
      } else if (__fortio_eq_str(CADR(decimal), CLEN(decimal), "POINT")) {
        gbl->decimal = FIO_POINT;
      } else
        s = __fortio_error(FIO_ESPEC);
    }
    if (ISPRESENTC(sign)) {
      if (__fortio_eq_str(CADR(sign), CLEN(sign), "PLUS")) {
        gbl->sign = FIO_PLUS;
      } else if (__fortio_eq_str(CADR(sign), CLEN(sign), "SUPPRESS")) {
        gbl->sign = FIO_SUPPRESS;
      } else if (__fortio_eq_str(CADR(sign), CLEN(sign), "PROCESSOR_DEFINED")) {
        gbl->sign = FIO_PROCESSOR_DEFINED;
      } else
        s = __fortio_error(FIO_ESPEC);
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
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTW_INIT03, fmtw_init03)
(__INT_T *istat, DCHAR(decimal), DCHAR(sign),
 DCHAR(round) DCLEN(decimal) DCLEN(sign) DCLEN(round))
{
  return ENTF90IO(FMTW_INIT03A, fmtw_init03a) (istat, CADR(decimal), CADR(sign),
                  CADR(round), (__CLEN_T)CLEN(decimal), (__CLEN_T)CLEN(sign), (__CLEN_T)CLEN(round));
}

__INT_T
ENTCRF90IO(FMTW_INITA, fmtw_inita)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * NULL indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 DCHAR(advance)   /* 'YES', 'NO', or 'NULL' */
 DCLEN64(advance))
{
  G *g;
  __CLEN_T advlen;
  char *advadr;
  int s = 0;

  save_gbl();
  allocate_new_gbl();
  g = gbl;
  g->internal_file = FALSE;

  if (ISPRESENTC(advance)) {
    advadr = CADR(advance);
    advlen = CLEN(advance);
  } else {
    advadr = NULL;
    advlen = 0;
  }
  s = fw_init(unit, rec, bitv, iostat, fmt, advadr, advlen);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMTW_INIT, fmtw_init)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T *fmt,    /* encoded format array.  A value of
                   * NULL indicates that format was
                   * previously encoded by a call to
                   * ENTF90IO(encode_fmt) */
 DCHAR(advance)   /* 'YES', 'NO', or 'NULL' */
 DCLEN(advance))
{
  return ENTCRF90IO(FMTW_INITA, fmtw_inita) (unit, rec, bitv, iostat, fmt,
                              CADR(advance), (__CLEN_T)CLEN(advance));
}

__INT_T
ENTF90IO(FMTW_INITVA, fmtw_initva)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   * containing address of encoded
                   * format array */
 DCHAR(advance)   /* 'YES', 'NO', or 'NULL' */
 DCLEN64(advance))
{
  G *g;
  char *advadr;
  __CLEN_T advlen;
  int s = 0;

  save_gbl();
  allocate_new_gbl();
  g = gbl;
  g->internal_file = FALSE;

  if (ISPRESENTC(advance)) {
    advadr = CADR(advance);
    advlen = CLEN(advance);
  } else {
    advadr = NULL;
    advlen = 0;
  }

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = fw_init(unit, rec, bitv, iostat, *fmt, advadr, advlen);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTW_INITV, fmtw_initv)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   * containing address of encoded
                   * format array */
 DCHAR(advance)   /* 'YES', 'NO', or 'NULL' */
 DCLEN(advance))
{
  return ENTF90IO(FMTW_INITVA, fmtw_initva) (unit, rec, bitv, iostat, fmt,
                            CADR(advance), (__CLEN_T)CLEN(advance));
}

__INT_T
ENTCRF90IO(FMTW_INITVA, fmtw_initva)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   *ontaining address of encoded
                   * format array */
 DCHAR(advance)   /* 'YES', 'NO', or 'NULL' */
 DCLEN64(advance))
{
  G *g;
  int s;
  __CLEN_T advlen;
  char *advadr;

  save_gbl();
  allocate_new_gbl();
  g = gbl;
  g->internal_file = FALSE;

  if (ISPRESENTC(advance)) {
    advadr = CADR(advance);
    advlen = CLEN(advance);
  } else {
    advadr = NULL;
    advlen = 0;
  }
  s = fw_init(unit, rec, bitv, iostat, *fmt, advadr, advlen);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMTW_INITV, fmtw_initv)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat, /* same as for ENTF90IO(open_) */
 __INT_T **fmt,   /* address of format variable
                   *ontaining address of encoded
                   * format array */
 DCHAR(advance)   /* 'YES', 'NO', or 'NULL' */
 DCLEN(advance))
{
  return ENTCRF90IO(FMTW_INITVA, fmtw_initva) (unit, rec, bitv, iostat, fmt,
                               CADR(advance), (__CLEN_T)CLEN(advance));
}

/* ------------------------------------------------------------------ */

static int
fw_intern_init(char *cunit,      /* pointer to variable or array to read from */
               __INT_T *rec_num, /* number of records in internal file-- 0 if
                                  * the file is an assumed size character
                                  * array */
               __INT_T *bitv,    /* same as for ENTF90IO(open_) */
               __INT_T *iostat,  /* same as for ENTF90IO(open_) */
               __INT_T *fmt,     /* same as for ENTF90IO(fmtr)/w_init */
               __CLEN_T cunitlen)
{
  G *g = gbl;

  __fortio_errinit03(-99, *bitv, iostat, "formatted write");

  assert(ISPRESENT(fmt));
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

  g->rec_len = -cunitlen;
  g->rec_buff = cunit;
  g->curr_pos = 0;
  g->num_internal_recs = *rec_num;
  g->scale_factor = 0;
  g->plus_flag = FALSE;
  g->suppress_crlf = FALSE;
  g->repeat_flag = FALSE;
  g->rpstack_top = -1;
  g->nonadvance = FALSE;
  g->decimal = FIO_POINT;
  g->round = FIO_COMPATIBLE;
  g->sign = FIO_PROCESSOR_DEFINED;

  return 0;
}

__INT_T
ENTF90IO(FMTW_INTERN_INITA, fmtw_intern_inita)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt      /* same as for ENTF90IO(fmtr)/w_init */
 DCLEN64(cunit))
{
  G *g;
  int s = 0;

  save_gbl();
  allocate_new_gbl();
  g = gbl;

  g->internal_file = TRUE;
  g->internal_unit = CADR(cunit);

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = fw_intern_init(CADR(cunit), rec_num, bitv, iostat, fmt, CLEN(cunit));
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTW_INTERN_INIT, fmtw_intern_init)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt      /* same as for ENTF90IO(fmtr)/w_init */
 DCLEN64(cunit))
{
  return ENTF90IO(FMTW_INTERN_INITA, fmtw_intern_inita) (CADR(cunit), rec_num,
                              bitv, iostat, fmt, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTCRF90IO(FMTW_INTERN_INITA, fmtw_intern_inita)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt      /* same as for ENTF90IO(fmtr)/w_init */
 DCLEN64(cunit))
{
  G *g;
  int s = 0;

  save_gbl();
  allocate_new_gbl();
  g = gbl;

  g->internal_file = TRUE;
  g->internal_unit = CADR(cunit);

  s = fw_intern_init(CADR(cunit), rec_num, bitv, iostat, fmt, CLEN(cunit));
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMTW_INTERN_INIT, fmtw_intern_init)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt      /* same as for ENTF90IO(fmtr)/w_init */
 DCLEN(cunit))
{
  return ENTCRF90IO(FMTW_INTERN_INITA, fmtw_intern_inita) (CADR(cunit), rec_num,
                             bitv, iostat, fmt, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTF90IO(FMTW_INTERN_INITVA, fmtw_intern_initva)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt     /* same as for ENTF90IO(fmtr)/w_initv */
 DCLEN64(cunit))
{
  G *g;
  int s = 0;

  save_gbl();
  allocate_new_gbl();
  g = gbl;
  g->internal_file = TRUE;
  g->internal_unit = CADR(cunit);

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = fw_intern_init(CADR(cunit), rec_num, bitv, iostat, *fmt, CLEN(cunit));
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMTW_INTERN_INITV, fmtw_intern_initv)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt     /* same as for ENTF90IO(fmtr)/w_initv */
 DCLEN(cunit))
{
  return ENTF90IO(FMTW_INTERN_INITVA, fmtw_intern_initva) (CADR(cunit), rec_num,
                bitv, iostat, fmt, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTCRF90IO(FMTW_INTERN_INITVA, fmtw_intern_initva)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt     /* same as for ENTF90IO(fmtr)/w_initv */
 DCLEN64(cunit))
{
  G *g;
  int s = 0;

  save_gbl();
  allocate_new_gbl();
  g = gbl;
  g->internal_file = TRUE;
  g->internal_unit = CADR(cunit);

  s = fw_intern_init(CADR(cunit), rec_num, bitv, iostat, *fmt, CLEN(cunit));
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMTW_INTERN_INITV, fmtw_intern_initv)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt     /* same as for ENTF90IO(fmtr)/w_initv */
 DCLEN(cunit))
{
  return ENTCRF90IO(FMTW_INTERN_INITVA, fmtw_intern_initva) (CADR(cunit),
                        rec_num, bitv, iostat, fmt, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTF90IO(FMTW_INTERN_INITE, fmtw_intern_inite)
(char **cunit,     /* variable containing address to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt,     /* same as for ENTF90IO(fmtr)/w_init */
 __INT_T *len)     /* size of 'cunit' */
{
  /* ENCODE initialization */
  G *g;
  int s = 0;

  save_gbl();
  allocate_new_gbl();
  g = gbl;
  g->internal_file = TRUE;
  g->internal_unit = *cunit;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = fw_intern_init(*cunit, rec_num, bitv, iostat, fmt, *len);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(FMTW_INTERN_INITE, fmtw_intern_inite)
(char **cunit,     /* variable containing address to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *fmt,     /* same as for ENTF90IO(fmtr)/w_init */
 __INT_T *len)     /* size of 'cunit' */
{
  /* ENCODE initialization */
  G *g;
  int s = 0;

  save_gbl();
  allocate_new_gbl();
  g = gbl;
  g->internal_file = TRUE;
  g->internal_unit = *cunit;

  s = fw_intern_init(*cunit, rec_num, bitv, iostat, fmt, *len);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}

__INT_T
ENTF90IO(FMTW_INTERN_INITEV, fmtw_intern_initev)
(char **cunit,     /* variable containing address to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt,    /* same as for ENTF90IO(fmtr)/w_initv */
 __INT_T *len)     /* size of 'cunit' */
{
  /* ENCODE initialization */
  G *g;
  int s = 0;

  save_gbl();
  allocate_new_gbl();
  g = gbl;
  g->internal_file = TRUE;
  g->internal_unit = *cunit;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = fw_intern_init(*cunit, rec_num, bitv, iostat, *fmt, *len);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(FMTW_INTERN_INITEV, fmtw_intern_initev)
(char **cunit,     /* variable containing address to read from */
 __INT_T *rec_num, /* number of records in internal file-- 0 if
                    * the file is an assumed size character
                    * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T **fmt,    /* same as for ENTF90IO(fmtr)/w_initv */
 __INT_T *len)     /* size of 'cunit' */
{
  /* ENCODE initialization */
  G *g;
  int s = 0;

  save_gbl();
  allocate_new_gbl();
  g = gbl;
  g->internal_file = TRUE;
  g->internal_unit = *cunit;

  s = fw_intern_init(*cunit, rec_num, bitv, iostat, *fmt, *len);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}

/* --------------------------------------------------------------------- */

int
__f90io_fmt_write(int type,    /* data type (as defined in pghpft.h) */
                  long length, /* # items of type to read. May be <= 0 */
                  int stride,  /* distance in bytes between items */
                  char *item,  /* where to transfer data from */
                  __CLEN_T item_length)
{
  long i;
  int sz;
  int tmptype;   /* scratch copy of type */
  char *tmpitem; /* scratch copy of item */
  int ret_err = 0;

  if (fioFcbTbls.error) {
    ret_err = ERR_FLAG;
    goto fmtr_err;
  }
  assert(fioFcbTbls.eof == 0);
  assert(item != NULL);

  sz = 0;
  tmptype = type;
  if (tmptype == __CPLX8) {
    tmptype = __REAL4;
    sz = FIO_TYPE_SIZE(tmptype);
  } else if (tmptype == __CPLX16) {
    tmptype = __REAL8;
    sz = FIO_TYPE_SIZE(tmptype);
  } else if (tmptype == __CPLX32) {
    tmptype = __REAL16;
    sz = FIO_TYPE_SIZE(tmptype);
  }

  tmpitem = item;
  for (i = 0; i < length; i++, tmpitem += stride) {
    if (fw_write(tmpitem, tmptype, item_length) != 0) {
      ret_err = ERR_FLAG;
      goto fmtr_err;
    }
    /*  write second half of complex if necessary:  */
    if (sz != 0 && fw_write(tmpitem + sz, tmptype, item_length) != 0) {
      ret_err = ERR_FLAG;
      goto fmtr_err;
    }
  }
  return 0;
fmtr_err:
  return ret_err;
}

__INT_T
ENTF90IO(FMT_WRITEA, fmt_writea)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN64(item))
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_fmt_write(*type, *length, *stride, CADR(item),
                          (*type == __STR) ? CLEN(item) : 0);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMT_WRITE, fmt_write)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN(item))
{
  return ENTF90IO(FMT_WRITEA, fmt_writea) (type, length, stride, CADR(item),
                         (__CLEN_T)CLEN(item));
}

/* same as fmt_write, but item may be array - for fmt_write, the compiler
 * scalarizes.
 */
__INT_T
ENTF90IO(FMT_WRITE_AA, fmt_write_aa)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN64(item))
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_fmt_write(*type, *length, *stride, CADR(item),
                          (*type == __STR) ? CLEN(item) : 0);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMT_WRITE_A, fmt_write_a)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN(item))
{
  return ENTF90IO(FMT_WRITE_AA, fmt_write_aa) (type, length, stride, CADR(item), (__CLEN_T)CLEN(item));
}

__INT_T
ENTF90IO(FMT_WRITE64_AA, fmt_write64_aa)
(__INT_T *type,    /* data type (as defined in pghpft.h) */
 __INT8_T *length, /* # items of type to read. May be <= 0 */
 __INT_T *stride,  /* distance in bytes between items */
 DCHAR(item)       /* where to transfer data from */
 DCLEN64(item))
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_fmt_write(*type, *length, *stride, CADR(item),
                          (*type == __STR) ? CLEN(item) : 0);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(FMT_WRITE64_A, fmt_write64_a)
(__INT_T *type,    /* data type (as defined in pghpft.h) */
 __INT8_T *length, /* # items of type to read. May be <= 0 */
 __INT_T *stride,  /* distance in bytes between items */
 DCHAR(item)       /* where to transfer data from */
 DCLEN(item))
{
  return ENTF90IO(FMT_WRITE64_AA, fmt_write64_aa) (type, length, stride, CADR(item), (__CLEN_T)CLEN(item));
}

__INT_T
ENTCRF90IO(FMT_WRITEA, fmt_writea)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN64(item))
{
  return __f90io_fmt_write(*type, *length, *stride, CADR(item),
                           (*type == __STR) ? CLEN(item) : 0);
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(FMT_WRITE, fmt_write)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to read. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN(item))
{
  return ENTCRF90IO(FMT_WRITEA, fmt_writea) (type, length, stride, CADR(item),
                                             (__CLEN_T)CLEN(item));
}

/* --------------------------------------------------------------------- */

static int
fw_write(char *item,      /* where to transfer data from.  The value of item
                           * may be NULL to indicate the end of format
                           * processing.
                           */
         int type,        /* data type (as defined in pghpft.h) */
         int item_length) /*  optional-- passed if type is character */
{
  INT code;
  int i, errflag;
  G *g = gbl;
  bool endreached = FALSE;

  while (TRUE) {
    /*
     * W A R N I N G:  New FED cases need to be added to fw_write()
     * and fw_end_nonadvance().
     */
    code = fw_get_fmtcode();

    switch (code) {
    case FED_END:
      i = fw_write_record();
      if (i != 0)
        return __fortio_error(i);
      g->fmt_pos = g->fmt_base[g->fmt_pos];
      if (item == NULL)
        goto exit_loop;
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
      i = fw_get_val(g);
      if (i < -128 || i > 127)
        return __fortio_error(FIO_ESCALEF);
      g->scale_factor = i;
      break;

    case FED_STR:
    case FED_KANJI_STRING:
      i = g->fmt_base[g->fmt_pos++]; /* string length */
      errflag = fw_write_item((char *)&(g->fmt_base[g->fmt_pos]), i);
      if (errflag)
        return ERR_FLAG;
      g->fmt_pos += (i + 3) >> 2;
      break;

    case FED_T:
      i = fw_get_val(g);
      if (i < 1)
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      g->curr_pos = i - 1;
      break;

    case FED_TL:
      i = fw_get_val(g);
      if (i < 1)
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      g->curr_pos -= i;
      if (g->curr_pos < 0)
        g->curr_pos = 0;
      break;

    case FED_TR:
    case FED_X:
      i = fw_get_val(g);
      if (i < 1)
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      g->curr_pos += i;
      break;

    case FED_SP:
      g->plus_flag = TRUE;
      g->sign = FIO_PLUS;
      break;

    case FED_S: /* set to default ... */
    case FED_SS:
      g->plus_flag = FALSE;
      g->sign = FIO_PROCESSOR_DEFINED;
      break;

    case FED_BN:
    case FED_BZ: /* ignore these edit descriptors during write */
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
      i = fw_slashes(g, 1);
      if (i != 0)
        return __fortio_error(i);
      break;

    case FED_COLON:
      if (item == NULL) {
        g->record_written = FALSE;
        goto exit_loop;
      }
      break;

    case FED_Q: /* act as if this causes item to be printed */
      goto exit_loop;

    case FED_DOLLAR:
      /*  set line-feed suppress flag if 1st char of current record
          is blank or '+':  */
      if (envar_fortranopt != NULL &&
          strstr(envar_fortranopt, "vaxio") != NULL) {
        FIO_FCB *f = g->fcb;
        i = g->rec_buff[0];
        if ((i == ' ' || i == '+') && f->stdunit)
          g->suppress_crlf = TRUE;
      } else
        g->suppress_crlf = TRUE;
      break;

    case FED_Aw:
    case FED_A:
      if (item != NULL) {
        if (type != __STR)
          item_length = FIO_TYPE_SIZE(type);
        if (code == FED_Aw) {
          i = fw_get_val(g); /*  field width  */
          if (i > item_length) {
            g->curr_pos += (i - item_length); /* blank pad */
            i = item_length;
          }
          i = fw_write_item(item, i);
        } else
          i = fw_write_item(item, item_length);
        if (i != 0)
          return i;
      }
      goto exit_loop;

    case FED_Gw_d:
    case FED_G:
    case FED_G0_d:
    case FED_G0:
      if (item == NULL)
        goto exit_loop;

      if (type == __STR) {
        if (code == FED_Gw_d) {
          i = fw_get_val(g); /*  field width  */
          if (i > item_length) {
            g->curr_pos += (i - item_length); /* blank pad */
            i = item_length;
          }
          (void) fw_get_val(g); /* skip d */
          if (g->fmt_base[g->fmt_pos] == FED_Ee) {
            g->fmt_pos++;
            (void) fw_get_val(g); /* skip e */
          }
          i = fw_write_item(item, i);
        } else
          i = fw_write_item(item, item_length);
        if (i != 0)
          return i;
        goto exit_loop;
      }

      i = fw_writenum(code, item, type);
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

      i = fw_writenum(code, item, type);
      if (i != 0)
        return i;
      goto exit_loop;

    case FED_Ow_m:
    case FED_Zw_m:
    case FED_O:
    case FED_Z:
      if (item == NULL)
        goto exit_loop;

      i = fw_OZwritenum(code, item, type, item_length);
      if (i != 0)
        return i;
      goto exit_loop;

    case FED_Bw_m:
      if (item == NULL)
        goto exit_loop;

      i = fw_Bwritenum(item, type, item_length);
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

  return 0;
} /*  end fw_write()  */

static int
fw_slashes(G *g, int cnt)
{
  bool save_nonadvance;
  int reterr;

  /* ensure that the 'nonadvance' test in fw_write_record() is
   * ignored for the slash edit descriptor
   */
  save_nonadvance = g->nonadvance;
  g->nonadvance = FALSE;
  while (cnt--) {
    reterr = fw_write_record();
    if (reterr != 0)
      break;
  }
  g->nonadvance = save_nonadvance;
  g->record_written = FALSE;
  return reterr;
}

/*
 * Nonadvancing terminates when a data edit descriptor, colon edit
 * descriptor, or 'end' is seen.
 * Encountered string edit descriptors are written. Position descriptors
 * must be processed as well; forward skipping cannot be processed
 * in the subsequent write, since the position after applying the
 * skip defines the 'left tabbing limit' for the subsequent write.
 * All other edit descriptors are ignored.
 */
static int
fw_end_nonadvance(void)
{
  INT code;
  int i, errflag;
  G *g = gbl;

  while (TRUE) {
    /*
     * W A R N I N G:  New FED cases need to be added to fw_write()
     * and fw_end_nonadvance().
     */
    code = fw_get_fmtcode();

    switch (code) {
    case FED_END:
      goto exit_loop;

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

    /***** Control edit descriptors *****/
    case FED_P:
      i = fw_get_val(g);
      if (i < -128 || i > 127)
        return __fortio_error(FIO_ESCALEF);
      break;

    case FED_T:
      i = fw_get_val(g);
      if (i < 1)
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      g->curr_pos = i - 1;
      break;

    case FED_TL:
      i = fw_get_val(g);
      if (i < 1)
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      g->curr_pos -= i;
      if (g->curr_pos < 0)
        g->curr_pos = 0;
      break;

    case FED_TR:
    case FED_X:
      i = fw_get_val(g);
      if (i < 1)
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      g->curr_pos += i;
      break;

    case FED_SP:
    case FED_S:
    case FED_SS:
    case FED_BN:
    case FED_BZ:
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
      i = fw_slashes(g, 1);
      if (i != 0)
        return __fortio_error(i);
      break;

    case FED_COLON:
      g->record_written = FALSE;
      goto exit_loop;

    case FED_DOLLAR:
      goto exit_loop;

    /*****  String edit descriptors  *****/
    case FED_STR:
    case FED_KANJI_STRING:
      i = g->fmt_base[g->fmt_pos++]; /* string length */
      errflag = fw_write_item((char *)&(g->fmt_base[g->fmt_pos]), i);
      if (errflag)
        return ERR_FLAG;
      g->fmt_pos += (i + 3) >> 2;
      break;

    case FED_ERROR:
      return ERR_FLAG;

    case FED_Ee:
      return __fortio_error(FIO_EEDITDSCR);

    default:
      /*****  Data edit desciptors: terminate processing  *****/
      goto exit_loop;
    } /* end switch */
  }   /* end while(TRUE) */

exit_loop:
  i = fw_write_record();
  if (i != 0)
    return __fortio_error(i);

  return 0;
} /*  end fw_end_nonadvance()  */

/* --------------------------------------------------------------------- */

static INT
fw_get_fmtcode(void)
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
    repeatcount = fw_get_val(g);
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
      (void) __fortio_error(FIO_EPNEST);
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
fw_get_val(G *g)
{
  INT flag = g->fmt_base[g->fmt_pos];
  INT val = g->fmt_base[g->fmt_pos + 1];

  g->fmt_pos += 2;

  if (flag != 0) { /* must call function to value */
    int (*fp)();
    fp = (int((*)()))(long)val;
    val = (*fp)();
  }

  return val;
}

static char *reserve_buffer(int width)
{
  G *g = gbl;
  long newpos = g->curr_pos + width;

  if (fw_check_size(newpos) == 0) {
    char *p = g->rec_buff + g->curr_pos;
    g->curr_pos = newpos;
    g->record_written = FALSE;
    if (newpos > g->max_pos)
      g->max_pos = newpos;
    return p;
  }

  /* __fortio_error() has been called */
  return NULL;
}

static bool call_format_double(int *result, int width, int format_char,
                               int fraction_digits, int exponent_digits,
                               int ESN_mode, int scale_factor,
                               bool explicit_plus, bool comma_radix,
                               bool elide_leading_spaces,
                               bool elide_trailing_spaces, int rounding_mode,
                               double x)
{
  static int use_this_code_path = -1; /* unknown */
  static int no_minus_zero = -1; /* unknown */

  struct formatting_control control;

  /* First call initializations */
  if (use_this_code_path == -1)
    use_this_code_path = __fortio_new_fp_formatter();
  if (no_minus_zero == -1)
    no_minus_zero = __fortio_no_minus_zero();

  *result = 0;
  if (!use_this_code_path)
    return FALSE;

  control.rounding = rounding_mode;
  control.format_char = format_char;
  control.fraction_digits = fraction_digits;
  control.exponent_digits = exponent_digits;
  control.scale_factor = scale_factor; /* 1 for ES */
  control.plus_sign = explicit_plus ? '+' : '\0';
  control.point_char = comma_radix ? ',' : '.';
  control.ESN_format = ESN_mode;
  control.no_minus_zero = no_minus_zero;

  if (elide_leading_spaces || elide_trailing_spaces || width > 256) {
    /* Format into a buffer, chop spaces, and copy.  Eschew alloca(). */
    char stack_buffer[256];
    char *emit = stack_buffer;
    char *malloced_buffer = NULL;
    char *pos;
    memset(stack_buffer, ' ', 256);
    if ((size_t)width > sizeof(stack_buffer) &&
        !(emit = malloced_buffer = malloc(width))) {
      *result = __fortio_error(FIO_ENOMEM);
    } else {
      __fortio_format_double(emit, width, &control, x);
      if (elide_leading_spaces) {
        while (*emit == ' ' && width > 1) {
          ++emit;
          --width;
        }
      }
      if (elide_trailing_spaces) {
        width = 0;
        pos = emit;
        while (*pos != ' ' && *pos != '\0') {
          ++pos;
          ++width;
        }
      }
      *result = fw_write_item(emit, width);
      if (malloced_buffer)
        free(malloced_buffer);
    }
  } else {
    /* Format right into g->rec_buff, no copy */
    char *emit = reserve_buffer(width);
    if (!emit) {
      /* fw_check_size() failed, __fortio_error() was called */
      *result = ERR_FLAG;
    } else {
      __fortio_format_double(emit, width, &control, x);
    }
  }

  return TRUE;
}

#ifdef TARGET_SUPPORTS_QUADFP
#define FORMAT_G0 1
#define FORMAT_G0_D 2
#define BUFFER_SIZE 256
static bool call_format_quad(int *result, int width, int format_char,
                             int fraction_digits, int exponent_digits,
                             int ESN_mode, int scale_factor,
                             bool explicit_plus, bool comma_radix,
                             bool elide_leading_spaces,
                             bool elide_trailing_spaces, int rounding_mode,
                             float128_t x)
{
  static int use_this_code_path = -1; /* unknown */
  static int no_minus_zero = -1; /* unknown */

  struct formatting_control control;

  /* First call initializations */
  if (use_this_code_path == -1)
    use_this_code_path = __fortio_new_fp_formatter();
  if (no_minus_zero == -1)
    no_minus_zero = __fortio_no_minus_zero();

  *result = 0;
  if (!use_this_code_path)
    return FALSE;
  /* deal with format F0 */
  if (format_char == 'F' && width == 0) {
    control.format_F0 = 1;
    width = G_REAL16_W + fraction_digits;
  /* deal with format G0 and G0.d */
  } else if (format_char == 'G' && width == 0) {
    if (fraction_digits == 0) {
      control.format_G0 = FORMAT_G0;
      fraction_digits = G_REAL16_D;
    } else
      control.format_G0 = FORMAT_G0_D;
    width = G_REAL16_W + fraction_digits;
  } else {
    control.format_G0 = control.format_F0 = 0;
  }
  control.rounding = rounding_mode;
  control.format_char = format_char;
  control.fraction_digits = fraction_digits;
  control.exponent_digits = exponent_digits;
  control.scale_factor = scale_factor; /* 1 for ES */
  control.plus_sign = explicit_plus ? '+' : '\0';
  control.point_char = comma_radix ? ',' : '.';
  control.ESN_format = ESN_mode;
  control.no_minus_zero = no_minus_zero;

  if (elide_leading_spaces || elide_trailing_spaces || width > BUFFER_SIZE) {
    /* Format into a buffer, chop spaces, and copy.  Eschew alloca(). */
    char stack_buffer[BUFFER_SIZE];
    char *emit = stack_buffer;
    char *malloced_buffer = NULL;
    char *pos = NULL;
    memset(stack_buffer, ' ', BUFFER_SIZE);
    if (width > sizeof stack_buffer &&
        !(emit = malloced_buffer = malloc(((unsigned long)width)))) {
      *result = __fortio_error(FIO_ENOMEM);
    } else {
      __fortio_format_quad(emit, width, &control, x);
      if (elide_leading_spaces && (emit != NULL)) {
        while (*emit == ' ' && width > 1) {
          ++emit;
          --width;
        }
      }
      if (elide_trailing_spaces) {
        width = 0;
        pos = emit;
        while(*pos != ' ' && *pos != '\0') {
          ++pos;
          ++width;
        }
      }
      *result = fw_write_item(emit, width);
      if (malloced_buffer != NULL)
        free(malloced_buffer);
    }
  } else {
    /* Format right into g->rec_buff, no copy */
    char *emit = reserve_buffer(width);
    if (emit == NULL) {
      /* fw_check_size() failed, __fortio_error() was called */
      *result = ERR_FLAG;
    } else {
      __fortio_format_quad(emit, width, &control, x);
    }
  }

  return TRUE;
}
#endif

/* ------------------------------------------------------------------- */
#ifdef TARGET_SUPPORTS_QUADFP
#define IFORT_R16_EXPONENT 3
#endif
static int
fw_writenum(int code, char *item, int type)
{
  __BIGINT_T ival;
  __BIGREAL_T dval;
#undef IS_INT
  DBLINT64 i8val;
#define IS_INT(t) (t == __BIGINT || t == __INT8)
  int ty;
  int w, m, d, e;
  char *p;
  G *g = gbl;
  bool e_flag; /* Ew.dEe, Gw.dEe */
  bool is_logical;
  bool dc_flag;
  bool elide_leading_spaces, elide_trailing_spaces;
  union {
    __BIGINT_T i;
    __INT4_T i4;
    __REAL4_T r4;
    __REAL8_T r8;
    DBLINT64 i8v;
    __INT8_T i8;
    __BIGREAL_T d;
    __REAL16_T r16;
  } crc;
  int result;

  is_logical = FALSE;
  switch (type) {
  case __INT1:
    ival = *(__INT1_T *)item;
    ty = __BIGINT;
    w = 7;
    break;
  case __LOG1:
    ival = *(__LOG1_T *)item;
    ty = __BIGINT;
    w = 7;
    is_logical = TRUE;
    break;
  case __INT2:
    ival = *(__INT2_T *)item;
    ty = __BIGINT;
    w = 7;
    break;
  case __LOG2:
    ival = *(__LOG2_T *)item;
    ty = __BIGINT;
    w = 7;
    is_logical = TRUE;
    break;
  case __INT4:
    ival = *(__INT4_T *)item;
    ty = __BIGINT;
    w = 12;
    break;
  case __LOG4:
    ival = *(__LOG4_T *)item;
    ty = __BIGINT;
    w = 12;
    is_logical = TRUE;
    break;
  case __LOG8:
    i8val[0] = ((__INT4_T *)item)[0];
    i8val[1] = ((__INT4_T *)item)[1];
    ty = __INT8;
    w = 24;
    is_logical = TRUE;
    break;
  case __INT8:
    i8val[0] = ((__INT4_T *)item)[0];
    i8val[1] = ((__INT4_T *)item)[1];
    ty = __INT8;
    w = 24;
    break;
  case __WORD4:
    ival = *(__WORD4_T *)item;
    ty = __BIGINT;
    w = 12;
    break;
  case __REAL4:
    dval = __fortio_chk_f((__REAL4_T *)item);
    ty = __REAL4;
    w = REAL4_W;
    d = REAL4_D;
    e = REAL4_E;
    break;
  case __REAL8:
    dval = *(__REAL8_T *)item;
    ty = __REAL8;
    w = REAL8_W;
    d = REAL8_D;
    if (dval == 0 || (dval > 1e-100 && dval < 1e100) ||
        (dval > -1e100 && dval < -1e-100)) {
      e = 2;
    } else {
      e = REAL8_E;
    }
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case __REAL16:
    dval = *(__REAL16_T *)item;
    ty = __REAL16;
    w = G_REAL16_W;
    d = G_REAL16_D;
    e = REAL16_E;
    break;
#endif
  default:
    goto fmt_mismatch;
  }
  /* If it is set, don't reset it, g->decimal could be set
   * by control edit descriptor via DC,DP
   */
  if (g->fcb) {
    if (g->decimal == 0)
      g->decimal = g->fcb->decimal;
    if (g->sign == 0)
      g->sign = g->fcb->sign;
    if (g->round == 0)
      g->round = g->fcb->round;
  }
  if (g->decimal == FIO_COMMA)
    dc_flag = TRUE;
  else
    dc_flag = FALSE;
  if (g->sign == FIO_PLUS)
    g->plus_flag = TRUE;
  else
    g->plus_flag = FALSE;

  switch (code) {
  case FED_G0_d: /* G0.d */
    fw_get_val(g); /* w is not 0 for G0.d */
    d = fw_get_val(g);
    if (d && IS_INT(ty)) { /* G0.d for integer error */
      goto fmt_mismatch;
    }
    if (d && (d >= w - 5)) {
      w = d + 10;
    }
    e_flag = TRUE;
    FLANG_FALLTHROUGH;
  case FED_G0: /* G0 */
    if (code == FED_G0 && dval == 0) {
      d = 0; /* d = 0 for zeros */
    }
    if (IS_INT(ty)) { /* I0 for integer data */
      w = 0;
      m = 1;
      if (is_logical) { /* L1 for logical data */
        w = 1;
        goto L_shared;
      }
      goto I0_shared;
    }
    elide_leading_spaces = TRUE;
    elide_trailing_spaces = TRUE;
    e_flag = TRUE;
    goto g_shared;

  case FED_Gw_d:
    w = fw_get_val(g);
    d = fw_get_val(g);
    e_flag = FALSE;
    if (g->fmt_base[g->fmt_pos] == FED_Ee) {
      g->fmt_pos++;
      e = fw_get_val(g);
      e_flag = TRUE;
    }
    if (IS_INT(ty)) {
      if (is_logical) {
        if (w > 1) /* blank pad if necessary */
          g->curr_pos += (w - 1);
        goto L_shared;
      }
      goto I_shared;
    }
    elide_leading_spaces = FALSE;
    elide_trailing_spaces = FALSE;
    goto g_shared;

  case FED_G:
    if (IS_INT(ty)) {
      if (is_logical) {
        g->curr_pos++; /* add 1 blank pad */
        goto L_shared;
      }
      goto I_shared;
    }
    e_flag = FALSE;
    elide_leading_spaces = FALSE;
    elide_trailing_spaces = FALSE;
  g_shared:
    if (ty != __REAL4 && ty != __REAL8 && ty != __REAL16)
      goto fmt_mismatch;
#ifdef TARGET_SUPPORTS_QUADFP
    if (ty == __REAL16) {
      if (code == FED_G0 || code == FED_G0_d) {
        w = 0;
        if (code == FED_G0)
	        d = 0;
        e = 0;
        call_format_quad(&result, w, 'G', d, e, '\0',
                         g->scale_factor, g->plus_flag, dc_flag,
                         elide_leading_spaces, elide_trailing_spaces,
                         g->round, dval);
      } else {
        int e1 = 0;
        /* compatible with ifort */
        if (code == FED_G)
          e1 = IFORT_R16_EXPONENT;
        call_format_quad(&result, w, 'G', d, e_flag ? e : e1, '\0',
                         g->scale_factor, g->plus_flag, dc_flag,
                         elide_leading_spaces, elide_trailing_spaces,
                         g->round, dval);
      }
      return result;
    }
#endif
    if (call_format_double(&result, w, 'G', d, e_flag ? e : 0, '\0',
                           g->scale_factor, g->plus_flag, dc_flag,
                           elide_leading_spaces, elide_trailing_spaces,
                           g->round, dval))
      return result;
    p = __fortio_fmt_g(dval, w, d, e, g->scale_factor, ty, g->plus_flag, e_flag,
                      dc_flag, g->round, FALSE);
    return fw_write_item(p, w);

  case FED_I:
    if (!IS_INT(ty)) {
      if (__fortio_check_format())
        goto fmt_mismatch;
      switch (ty) {
      case __REAL4:
        crc.r4 = dval;
        ival = crc.i4;
        ty = __BIGINT;
        w = 12;
        break;
      case __REAL8:
      case __REAL16:
        /* integer*16 is not supported, so we convert real*16 to integer*8. */
        crc.r8 = dval;
        i8val[0] = crc.i8v[0];
        i8val[1] = crc.i8v[1];
        ty = __INT8;
        w = 24;
        break;
      }
    }
  I_shared:
    if (ty == __INT8)
      p = __fortio_fmt_i8(i8val, w, 1, g->plus_flag);
    else
      p = __fortio_fmt_i(ival, w, 1, g->plus_flag);
    return fw_write_item(p, w);

  case FED_Iw_m:
    if (!IS_INT(ty)) {
      if (__fortio_check_format())
        goto fmt_mismatch;
      switch (ty) {
      case __REAL4:
        crc.r4 = dval;
        ival = crc.i4;
        ty = __BIGINT;
        break;
      case __REAL8:
      case __REAL16:
        /* integer*16 is not supported, so we convert real*16 to integer*8. */
        crc.r8 = dval;
        i8val[0] = crc.i8v[0];
        i8val[1] = crc.i8v[1];
        ty = __INT8;
        break;
      }
    }
    w = fw_get_val(g);
    m = fw_get_val(g);
    I0_shared:
    if (ty == __INT8) {
      if (w == 0) {
        /* compute a w which is the minimal value to represent
         * the item.
         */
        w = 21;
        p = __fortio_fmt_i8(i8val, w, m, g->plus_flag);
        while (*p == ' ') {
          p++;
          w--;
        }
      } else
        p = __fortio_fmt_i8(i8val, w, m, g->plus_flag);
    } else
    {
      if (w == 0) {
        /* compute a w which is the minimal value to represent
         * the item.
         */
        w = 12;
        p = __fortio_fmt_i(ival, w, m, g->plus_flag);
        while (*p == ' ') {
          p++;
          w--;
        }
      } else
        p = __fortio_fmt_i(ival, w, m, g->plus_flag);
    }
    return fw_write_item(p, w);

  case FED_L:
    g->curr_pos++; /* add 1 blank pad */
    goto L_shared;

  case FED_Lw:
    w = fw_get_val(g);
    if (w > 1) /* blank pad if necessary */
      g->curr_pos += (w - 1);
  L_shared:
    if (!IS_INT(ty)) {
      if (__fortio_check_format())
        goto fmt_mismatch;
      switch (ty) {
      case __REAL4:
        crc.r4 = dval;
        ival = crc.i4;
        ty = __BIGINT;
        break;
      case __REAL8:
      case __REAL16:
        /* integer*16 is not supported, so we convert real*16 to integer*8. */
        crc.r8 = dval;
        i8val[0] = crc.i8v[0];
        i8val[1] = crc.i8v[1];
        ty = __INT8;
        break;
      }
    }
    if (ty == __INT8)
      ival = I64_LSH(i8val);
    if ((ival & GET_FIO_CNFG_TRUE_MASK) == 0)
      return fw_write_item("F", 1);
    return fw_write_item("T", 1);

  case FED_Fw_d:
    w = fw_get_val(g);
    d = fw_get_val(g);
    switch (ty) {
    case __INT4:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i4 = ival;
      dval = crc.r4;
      e = 2;
      ty = __REAL4;
      break;
    case __INT8:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i8v[0] = i8val[0];
      crc.i8v[1] = i8val[1];
      dval = crc.r8;
      e = 2;
      ty = __REAL8;
      break;
    }
    if (w == 0) {
#ifdef TARGET_SUPPORTS_QUADFP
      if (ty == __REAL16) {
        call_format_quad(&result, w, 'F', d, 0, '\0', g->scale_factor,
                         g->plus_flag, dc_flag, TRUE, FALSE, g->round,
                         dval);
        return result;
      }
#endif
      /* compute a w which is the minimal value to represent
       * the item.
       */
      w = BIGREAL_W + d;
      if (call_format_double(&result, w, 'F', d, 0, '\0', g->scale_factor,
                             g->plus_flag, dc_flag, TRUE, FALSE, g->round,
                             dval))
        return result;
      p = __fortio_fmt_f(dval, w, d, g->scale_factor, g->plus_flag, dc_flag,
                        g->round);
      while (*p == ' ') {
        p++;
        w--;
      }
      return fw_write_item(p, w);
    }
#ifdef TARGET_SUPPORTS_QUADFP
    if (ty == __REAL16) {
      call_format_quad(&result, w, 'F', d, 0, '\0', g->scale_factor,
                       g->plus_flag, dc_flag, FALSE, FALSE, g->round, dval);
      return result;
    }
#endif
    if (call_format_double(&result, w, 'F', d, 0, '\0', g->scale_factor,
                           g->plus_flag, dc_flag, FALSE, FALSE, g->round, dval))
      return result;
    goto f_shared;

  case FED_F:
    switch (ty) {
    case __INT4:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i4 = ival;
      dval = crc.r4;
      w = REAL4_W;
      d = REAL4_D;
      ty = __REAL4;
      break;
    case __INT8:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i8v[0] = i8val[0];
      crc.i8v[1] = i8val[1];
      dval = crc.r8;
      w = REAL8_W;
      d = REAL8_D;
      ty = __REAL8;
      break;
    }
#ifdef TARGET_SUPPORTS_QUADFP
    if (ty == __REAL16) {
      call_format_quad(&result, w, 'F', d, 0, '\0', g->scale_factor,
                       g->plus_flag, dc_flag, FALSE, FALSE, g->round, dval);
      return result;
    }
#endif
    if (call_format_double(&result, w, 'F', d, 0, '\0', g->scale_factor,
                           g->plus_flag, dc_flag, FALSE, FALSE, g->round, dval))
      return result;
  f_shared:
    p = __fortio_fmt_f(dval, w, d, g->scale_factor, g->plus_flag, dc_flag,
                      g->round);
    return fw_write_item(p, w);
  case FED_ENw_d:
  case FED_ESw_d:
  case FED_Ew_d:
    w = fw_get_val(g);
    d = fw_get_val(g);
    e_flag = FALSE;
    if (g->fmt_base[g->fmt_pos] == FED_Ee) {
      g->fmt_pos++;
      e = fw_get_val(g);
      e_flag = TRUE;
    }
    switch (ty) {
    case __INT4:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i4 = ival;
      dval = crc.r4;
      if (!e_flag)
        e = 2;
      ty = __REAL4;
      break;
    case __INT8:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i8v[0] = i8val[0];
      crc.i8v[1] = i8val[1];
      dval = crc.r8;
      if (!e_flag)
        e = 2;
      ty = __REAL8;
      break;
    }
#ifdef TARGET_SUPPORTS_QUADFP
    if (ty == __REAL16) {
      call_format_quad(&result, w, 'E', d, e_flag ? e : 0,
                       code == FED_ESw_d ? 'S' : code == FED_ENw_d ? 'N' : '\0',
                       g->scale_factor, g->plus_flag, dc_flag, FALSE, FALSE,
                       g->round, dval);
      return result;
    }
#endif
    if (call_format_double(&result, w, 'E', d, e_flag ? e : 0,
                           code == FED_ESw_d ? 'S' :
                             code == FED_ENw_d ? 'N' : '\0',
                           g->scale_factor, g->plus_flag, dc_flag,
                           FALSE, FALSE, g->round, dval))
      return result;
    goto e_shared;

  case FED_E:
    e_flag = FALSE;
    switch (ty) {
    case __INT4:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i4 = ival;
      dval = crc.r4;
      w = REAL4_W;
      d = REAL4_D;
      e = 2;
      ty = __REAL4;
      break;
    case __INT8:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i8v[0] = i8val[0];
      crc.i8v[1] = i8val[1];
      dval = crc.r8;
      w = REAL8_W;
      d = REAL8_D;
      e = 2;
      ty = __REAL8;
      break;
    }
#ifdef TARGET_SUPPORTS_QUADFP
    if (ty == __REAL16) {
      /* exponent compatible with ifort */
      call_format_quad(&result, w, 'E', d, IFORT_R16_EXPONENT, '\0', g->scale_factor,
                       g->plus_flag, dc_flag, FALSE, FALSE, g->round, dval);
      return result;
    }
#endif
    if (call_format_double(&result, w, 'E', d, 0, '\0', g->scale_factor,
                           g->plus_flag, dc_flag, FALSE, FALSE, g->round, dval))
      return result;
  e_shared:
    p = __fortio_fmt_e(dval, w, d, e, g->scale_factor, ty, g->plus_flag, e_flag,
                      dc_flag, code, g->round);
    return fw_write_item(p, w);
  case FED_Dw_d:
    w = fw_get_val(g);
    d = fw_get_val(g);
    switch (ty) {
    case __INT4:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i4 = ival;
      dval = crc.r4;
      ty = __REAL4;
      break;
    case __INT8:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i8v[0] = i8val[0];
      crc.i8v[1] = i8val[1];
      dval = crc.r8;
      ty = __REAL8;
      break;
    }
    goto d_shared;

  case FED_D:
    switch (ty) {
    case __INT4:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i4 = ival;
      dval = crc.r4;
      w = REAL4_W;
      d = REAL4_D;
      ty = __REAL4;
      break;
    case __INT8:
      if (__fortio_check_format())
        goto fmt_mismatch;
      crc.i8v[0] = i8val[0];
      crc.i8v[1] = i8val[1];
      dval = crc.r8;
      w = REAL8_W;
      d = REAL8_D;
      ty = __REAL8;
      break;
    }
  d_shared:
#ifdef TARGET_SUPPORTS_QUADFP
    if (ty == __REAL16) {
      int e1 = 0;
      if (code == FED_D)
        e1 = IFORT_R16_EXPONENT;
      call_format_quad(&result, w, 'D', d, e1, '\0', g->scale_factor,
                       g->plus_flag, dc_flag, FALSE, FALSE, g->round, dval);
      return result;
    }
#endif
    if (call_format_double(&result, w, 'D', d, 0, '\0', g->scale_factor,
                           g->plus_flag, dc_flag, FALSE, FALSE, g->round, dval))
      return result;
    p = __fortio_fmt_d(dval, w, d, g->scale_factor, ty, g->plus_flag, g->round);
    return fw_write_item(p, w);

  } /* end of switch */

fmt_mismatch:
  return __fortio_error(FIO_EMISMATCH);
}

/* ------------------------------------------------------------------- */

/*  local static variables for octal/hex conversion:  */

static int OZbase;
static char hextab[17] = "0123456789ABCDEF";
static char *OZbuff;
static int bits_left;
static int bits; /* 0, 1 or 2 left over bits */
static char *buff_pos;

static __CLEN_T fw_OZconv_init(__CLEN_T);
static void fw_OZbyte(unsigned int);
static void fw_Bbyte(unsigned int);

static int
fw_OZwritenum(int code, char *item, int type, int item_length)
{
  char *p;
  G *g = gbl;
  int w, m, k, offset;

  OZbase = 16;
  if (code == FED_O || code == FED_Ow_m)
    OZbase = 8;

  if (type == __STR) {
    w = fw_OZconv_init(item_length);
    for (p = item; p < item + item_length; p++)
      fw_OZbyte(*p);
    item_length = w;
  } else {
    item_length = fw_OZconv_init(FIO_TYPE_SIZE(type));
    switch (FIO_TYPE_SIZE(type)) {
    case 1:
      fw_OZbyte(*item);
      w = 7;
      break;

    case 2:
      fw_OZbyte(*item);
      fw_OZbyte(*(item + 1));
      w = 7;
      break;

    case 4:
      fw_OZbyte(*item);
      fw_OZbyte(*(item + 1));
      fw_OZbyte(*(item + 2));
      fw_OZbyte(*(item + 3));
      w = 12;
      break;

    case 8:
      fw_OZbyte(*item);
      fw_OZbyte(*(item + 1));
      fw_OZbyte(*(item + 2));
      fw_OZbyte(*(item + 3));
      fw_OZbyte(*(item + 4));
      fw_OZbyte(*(item + 5));
      fw_OZbyte(*(item + 6));
      fw_OZbyte(*(item + 7));
      w = 23;
      break;

    case 16:
      fw_OZbyte(*item);
      fw_OZbyte(*(item + 1));
      fw_OZbyte(*(item + 2));
      fw_OZbyte(*(item + 3));
      fw_OZbyte(*(item + 4));
      fw_OZbyte(*(item + 5));
      fw_OZbyte(*(item + 6));
      fw_OZbyte(*(item + 7));
      fw_OZbyte(*(item + 8));
      fw_OZbyte(*(item + 9));
      fw_OZbyte(*(item + 10));
      fw_OZbyte(*(item + 11));
      fw_OZbyte(*(item + 12));
      fw_OZbyte(*(item + 13));
      fw_OZbyte(*(item + 14));
      fw_OZbyte(*(item + 15));
      w = 44;
      break;
    default:
      assert(FALSE);
      break;
    } /* switch */
  }

  m = 1;
  if (code == FED_Zw_m || code == FED_Ow_m) {
    w = fw_get_val(g);
    m = fw_get_val(g);
    if (w < 0)
      w = 0;
    if (m > w)
      m = w;
  }

  offset = 0;
  if (m < item_length) { /* delete leading zeros */
    for (k = 0; OZbuff[k] == '0' && item_length - k > m; k++)
      OZbuff[k] = ' ';
    item_length -= k;
    offset = k;
  }

  if (w >= item_length) {
    /*  set k to max of m and item_length:  */
    k = item_length;
    if (m > k)
      k = m;
    if (w > k) /* blank pad if necc.  */
      g->curr_pos += (w - k);
    assert(w >= m);
    for (; m > item_length; m--) /* add leading zeros if necc. */
      if (fw_write_item("0", 1) != 0)
        return ERR_FLAG;

    return fw_write_item(OZbuff + offset, item_length);
  } else {
    for (k = 0; k < w; k++)
      if (fw_write_item("*", 1) != 0)
        return ERR_FLAG;
  }

  return 0;
}

/* ------------------------------------------------------------------- */

static int
fw_Bwritenum(char *item, int type, __CLEN_T item_length)
{
  char *p;
  G *g = gbl;
  __CLEN_T m, w, k, offset;

  OZbase = 2;

  if (type == __STR) {
    w = fw_OZconv_init(item_length);
    for (p = item; p < item + item_length; p++)
      fw_Bbyte(*p);
    item_length = w;
  } else {
    item_length = fw_OZconv_init(FIO_TYPE_SIZE(type));
    switch (FIO_TYPE_SIZE(type)) {
    case 1:
      fw_Bbyte(*item);
      break;

    case 2:
      fw_Bbyte(*item);
      fw_Bbyte(*(item + 1));
      break;

    case 4:
      fw_Bbyte(*item);
      fw_Bbyte(*(item + 1));
      fw_Bbyte(*(item + 2));
      fw_Bbyte(*(item + 3));
      break;

    case 8:
      fw_Bbyte(*item);
      fw_Bbyte(*(item + 1));
      fw_Bbyte(*(item + 2));
      fw_Bbyte(*(item + 3));
      fw_Bbyte(*(item + 4));
      fw_Bbyte(*(item + 5));
      fw_Bbyte(*(item + 6));
      fw_Bbyte(*(item + 7));
      break;

    case 16:
      fw_Bbyte(*item);
      fw_Bbyte(*(item + 1));
      fw_Bbyte(*(item + 2));
      fw_Bbyte(*(item + 3));
      fw_Bbyte(*(item + 4));
      fw_Bbyte(*(item + 5));
      fw_Bbyte(*(item + 6));
      fw_Bbyte(*(item + 7));
      fw_Bbyte(*(item + 8));
      fw_Bbyte(*(item + 9));
      fw_Bbyte(*(item + 10));
      fw_Bbyte(*(item + 11));
      fw_Bbyte(*(item + 12));
      fw_Bbyte(*(item + 13));
      fw_Bbyte(*(item + 14));
      fw_Bbyte(*(item + 15));
      break;

    default:
      assert(FALSE);
      break;
    } /* switch */
  }

  m = 1;
  w = fw_get_val(g);
  m = fw_get_val(g);
  if (w < 0)
    w = 0;
  if (m > w)
    m = w;

  offset = 0;
  if (m < item_length) { /* delete leading zeros */
    for (k = 0; OZbuff[k] == '0' && item_length - k > m; k++)
      OZbuff[k] = ' ';
    item_length -= k;
    offset = k;
  }

  if (w >= item_length) {
    /*  set k to max of m and item_length:  */
    k = item_length;
    if (m > k)
      k = m;
    if (w > k) /* blank pad if necc.  */
      g->curr_pos += (w - k);
    assert(w >= m);
    for (; m > item_length; m--) /* add leading zeros if necc. */
      if (fw_write_item("0", 1) != 0)
        return ERR_FLAG;

    return fw_write_item(OZbuff + offset, item_length);
  } else {
    for (k = 0; k < w; k++)
      if (fw_write_item("*", 1) != 0)
        return ERR_FLAG;
  }

  return 0;
}

static void
fw_Bbyte(unsigned int c)
{
  int i;

  for (i = 0; i < 8; i++) {
    *buff_pos-- = '0' + (c & 1);
    c >>= 1;
  }
}

/* ------------------------------------------------------------------- */

static __CLEN_T
fw_OZconv_init(__CLEN_T len)
{
  static __CLEN_T buff_len = 0;

  if (OZbase == 16)
    len += len;
  else if (OZbase == 2)
    len *= 8;
  else
    len = ((len * 8) + 2) / 3;

  if (buff_len < len) {
    if (buff_len != 0)
      free(OZbuff);
    buff_len = len + 8;
    OZbuff = malloc(buff_len);
  }

  buff_pos = OZbuff + len - 1; /* start at right end of buffer */
  bits_left = 0;
  bits = 0;
  return len;
}

static void
fw_OZbyte(unsigned int c)
{
  if (OZbase == 16) {
    *buff_pos = hextab[c & 0xF];
    *(buff_pos - 1) = hextab[(c >> 4) & 0xF];
    buff_pos -= 2;
  } else {
    int bitcount;
    c = ((c & 0xFF) << bits_left) | bits;
    bitcount = 8 + bits_left;
    while (bitcount >= 3) { /* do one octal digit per iteration  */
      *(buff_pos--) = (c & 0x7) + '0';
      c >>= 3;
      bitcount -= 3;
    }
    bits_left = bitcount;
    bits = c;

    /* write one byte ahead ... */
    if (buff_pos >= OZbuff)
      *buff_pos = (c & 0x7) + '0';
  }

}

/* ------------------------------------------------------------------- */

/**  \return ERR_FLAG or 0 */
static int
fw_write_item(const char *p, int len)
{
  G *g = gbl;
  char *q;
  int newpos;

  newpos = len + g->curr_pos;

  assert(len > -1 && p != NULL);

  if (fw_check_size(newpos) != 0)
    return ERR_FLAG;

  q = &(g->rec_buff[g->curr_pos]);
  g->curr_pos = newpos;
  g->record_written = FALSE;
  if (newpos > g->max_pos)
    g->max_pos = newpos;

  if (len > 0)
    memcpy(q, p, len);

  return 0;
}

/* -------------------------------------------------------------------- */

static int
fw_check_size(long len)
{
  G *g = gbl;

  if (len > g->rec_len) {
    if (g->rec_len < 0) {
      assert(g->internal_file);
      g->rec_len = -g->rec_len;
      if (len > g->rec_len)
        return __fortio_error(FIO_ETOOBIG);
      (void) memset(g->rec_buff, ' ', g->rec_len);
    } else if (g->internal_file || (g->fcb)->acc == FIO_DIRECT)
      return __fortio_error(FIO_ETOOBIG);
    else {
      assert(g->obuff == g->rec_buff);
      assert(len > g->obuff_len);
      len += INIT_BUFF_LEN;
      g->obuff = realloc(g->obuff, len);
      if (g->obuff == NULL)
        return __fortio_error(FIO_ENOMEM);
      g->rec_buff = g->obuff;
      memset(g->obuff + g->rec_len, ' ', len - g->rec_len);
      g->rec_len = g->obuff_len = len;
    }
  }

  return 0;
}

/* ------------------------------------------------------------------- */

static int
fw_write_record(void)
{
  G *g = gbl;

  if (g->internal_file) {
    g->num_internal_recs--;
    if (g->num_internal_recs < 0)
      return FIO_ETOOFAR;

    /* note, negative rec_len indicates empty record:  */
    if (g->rec_len > 0) {
      g->rec_buff += g->rec_len; /* point to next record */
      g->rec_len = -g->rec_len;
    } else {
      (void) memset(g->rec_buff, ' ', -g->rec_len);
      g->rec_buff -= g->rec_len; /* point to next record */
    }
  } else { /* external file */
    FIO_FCB *f = g->fcb;

    if (f->acc == FIO_DIRECT) {
      if ((long)FWRITE(g->rec_buff, 1, g->rec_len, f->fp) != g->rec_len)
        return __io_errno();
    } else { /* sequential write */
      if (g->nonadvance) {
        if (g->curr_pos >= g->max_pos) {
          g->max_pos = g->curr_pos;
          fw_check_size(g->max_pos);
          if ((long)FWRITE(g->rec_buff, 1, g->max_pos, f->fp) != g->max_pos)
            return __io_errno();
        } else if (g->curr_pos < g->max_pos) {
          long len = g->max_pos - g->curr_pos;

          if ((long)FWRITE(g->rec_buff, 1, g->curr_pos, f->fp) != g->curr_pos)
            return __io_errno();
          g->fcb->skip = len;
          g->fcb->skip_buff = malloc(len);
          memcpy(g->fcb->skip_buff, &g->rec_buff[g->curr_pos], len);
        }
        f->nonadvance = TRUE; /* do it later */
      } else {
        if ((long)FWRITE(g->rec_buff, 1, g->max_pos, f->fp) != g->max_pos)
          return __io_errno();
        f->nonadvance = FALSE; /* do it now */
        if (!(g->suppress_crlf)) {
/* append carriage return */
#if defined(_WIN64)
          if (__io_binary_mode(f->fp))
            __io_fputc('\r', f->fp);
#endif
          /*                    if (g->max_pos > 0)*/
          __io_fputc('\n', f->fp);
          if (__io_ferror(f->fp))
            return __io_errno();
        } else if (fflush(f->fp) != 0)
          return __io_errno();
      }
    }
    /* set used portion of record buffer back to blanks:  */
    if (g->max_pos > 0)
      (void) memset(g->rec_buff, ' ', g->max_pos);
    g->record_written = TRUE;
    ++(f->nextrec);
  }

  g->curr_pos = 0;
  g->max_pos = 0;

  return 0;
}

/* ------------------------------------------------------------------ */

static int
_f90io_fmtw_end(void)
{
  G *g = gbl;
  FIO_FCB *f;
  int reterr = 0;

  if (fioFcbTbls.error)
    reterr = ERR_FLAG;
  else {

    if (gbl->same_fcb && !g->internal_file && g->record_written != 0 &&
        g->curr_pos != 0 && g->max_pos != 0)
      return 0;

    assert(fioFcbTbls.eof == 0);
    if (!g->nonadvance) {
      if (!gbl->same_fcb || gbl->internal_file)
        reterr = fw_write((char *)0, -1, -1);
    } else {
      reterr = fw_end_nonadvance();
    }
  }
  if (!reterr && !g->internal_file) {
    if (!g->record_written) {
      if (!gbl->same_fcb)
        reterr = fw_write_record();
      if (reterr)
        return __fortio_error(reterr);
    }
    f = g->fcb;
    f->nextrec--;
    if (f->acc == FIO_DIRECT) {
      if (f->nextrec - 1 > f->maxrec)
        f->maxrec = f->nextrec - 1;
    }
  }
  if (g->internal_file) {
    if (g->rec_len > 0) {        /* < 0 => empty record */
      g->rec_buff += g->rec_len; /* point to next record */
    }
  }

  return reterr;
}

__INT_T
ENTF90IO(FMTW_END, fmtw_end)()
{
  G *g = gbl;
  int ioproc;
  int len, s = 0;

  ioproc = GET_DIST_IOPROC;
  if ((GET_DIST_LCPU == ioproc) || LOCAL_MODE) {
    s = _f90io_fmtw_end();
    if (g->internal_file)
      len = g->rec_buff - g->internal_unit;
  }
  if ((!LOCAL_MODE) && (g->internal_file)) {
    DIST_RBCSTL(ioproc, &len, 1, 1, __CINT, sizeof(int));
    DIST_RBCSTL(ioproc, g->internal_unit, 1, 1, __CHAR, len);
  }

  save_samefcb();
  free_gbl();
  restore_gbl();
  __fortio_fmtend();
  __fortio_errend03();
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(FMTW_END, fmtw_end)()
{
  int s = 0;

  s = _f90io_fmtw_end();
  save_samefcb();
  free_gbl();
  restore_gbl();
  __fortio_fmtend();
  __fortio_errend03();

  return s;
}

/* --------------------------------------------------------------------- */
/*
 *  Opportunistic by-value write routines
 */
__INT_T
ENTF90IO(SC_FMT_WRITE,sc_fmt_write)(
    int  item,		/* scalar data to transfer */
    int  type)		/* data type (as defined in pghpft.h) */
{

  return __f90io_fmt_write(type, 1, 0, (char *)&item, 0);
}

__INT_T
ENTF90IO(SC_I_FMT_WRITE, sc_i_fmt_write)(int item, int type)
{
  return __f90io_fmt_write(type, 1, 0, (char *)&item, 0);
}

__INT_T
ENTF90IO(SC_L_FMT_WRITE, sc_l_fmt_write)(long long item, int type)
{
  return __f90io_fmt_write(type, 1, 0, (char *)&item, 0);
}

__INT_T
ENTF90IO(SC_F_FMT_WRITE, sc_f_fmt_write)(float item, int type)
{
  return __f90io_fmt_write(type, 1, 0, (char *)&item, 0);
}

__INT_T
ENTF90IO(SC_D_FMT_WRITE, sc_d_fmt_write)(double item, int type)
{
  return __f90io_fmt_write(type, 1, 0, (char *)&item, 0);
}

#ifdef TARGET_SUPPORTS_QUADFP
__INT_T ENTF90IO(SC_Q_FMT_WRITE, sc_q_fmt_write)(float128_t item, int type)
{
  return __f90io_fmt_write(type, 1, 0, (char *)&item, 0);
}
#endif

__INT_T
ENTF90IO(SC_CF_FMT_WRITE, sc_cf_fmt_write)(float real, float imag, int type)
{
  int err;
  err = __f90io_fmt_write(__REAL4, 1, 0, (char *)&real, 0);
  if (err)
    return err;
  return __f90io_fmt_write(__REAL4, 1, 0, (char *)&imag, 0);
}

__INT_T
ENTF90IO(SC_CD_FMT_WRITE, sc_cd_fmt_write)(double real, double imag, int type)
{
  int err;
  err = __f90io_fmt_write(__REAL8, 1, 0, (char *)&real, 0);
  if (err)
    return err;
  return __f90io_fmt_write(__REAL8, 1, 0, (char *)&imag, 0);
}

#ifdef TARGET_SUPPORTS_QUADFP
__INT_T ENTF90IO(SC_CQ_FMT_WRITE, sc_cq_fmt_write)(float128_t real, float128_t imag, int type)
{
  int err;
  err = __f90io_fmt_write(__REAL16, 1, 0, (char *)&real, 0);
  if (err)
    return err;
  return __f90io_fmt_write(__REAL16, 1, 0, (char *)&imag, 0);
}
#endif

/* --------------------------------------------------------------------- */
#define CHAR_ONLY 1
#define CHAR_AND_VLIST 2

__INT_T
ENTF90IO(DTS_FMTW,dts_fmtw)(char** cptr, void** iptr, INT * len, F90_Desc* sd, int* flag)
{
  INT code, k, first;
  G *g = gbl;
  int i, errflag;
  __INT_T ubnd = 0;
  __INT8_T **tptr8 = (__INT8_T **)iptr;
  INT **tptr4 = (INT **)iptr;

  while (TRUE) {
    code = fw_get_fmtcode();
    switch (code) {
    case FED_END:
      if (!g->repeat_flag && !fioFcbTbls.error) {
        i = fw_write_record();
        if (i != 0)
          return __fortio_error(i);
      }
      g->fmt_pos = g->fmt_base[g->fmt_pos];
      break;
    case FED_T:
      i = fw_get_val(g);
      if (i < 1) {
        i = 0;
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      }

      g->curr_pos = i - 1;
      break;
    case FED_TL:
      i = fw_get_val(g);

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
      i = fw_get_val(g);
      if (i < 1) {
        i = 0;
        return __fortio_error(FIO_ETAB_VALUE_OUT_OF_RANGE);
      }
      g->curr_pos += i;
      break;
    case FED_DT:
      goto exit_loop;
    case FED_STR:
      i = g->fmt_base[g->fmt_pos++]; /* string length */
      errflag = fw_write_item((char *)&(g->fmt_base[g->fmt_pos]), i);
      if (errflag)
        return ERR_FLAG;
      g->fmt_pos += (i + 3) >> 2;
      break;
    default:
      break;
    }
  }
exit_loop:

  /* *cptr always points to something, encoding must put at least "DT" there */

  /* get DT_ value */
  k = fw_get_val(g);
  switch (k) {
  case CHAR_ONLY:
    k = fw_get_val(g); /* length of this DT....*/
    *len = k;
    *cptr = (char *)&(g->fmt_base[g->fmt_pos]);
    *iptr = NULL;
    g->fmt_pos += (k + 3) >> 2;
    if (sd) {
      if (*flag == 3 || *flag == 2) {
        get_vlist_desc_i8(sd, ubnd);
      } else {
        get_vlist_desc(sd, ubnd);
      }
    }
    break;
  case CHAR_AND_VLIST:
    k = fw_get_val(g);
    *len = k;
    *cptr = (char *)&(g->fmt_base[g->fmt_pos]);
    g->fmt_pos += (k + 3) >> 2;
    k = fw_get_val(g); /* how many item is the vlist */
    first =
        fw_get_val(g); /* is this vlist has been modified by for loop below?*/

    /* flag=1 or flag=3, iptr is i8, we need to copy to  */
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

/* Make the value of an intent(out) user-defined derived-type I/O procedure
 * iostat argument available to the associated parent I/O statement.
 */
void
ENTF90IO(DTS_STAT, dts_stat)(int iostat)
{
  if (!iostat)
    return;

  /* Set an internal error flag to suppress end of record positioning.
   * It may be necessary to do additional processing with this iostat
   * value, such as splitting off EOR and EOF cases, or propagating the
   * value to additional places.
   */
  fioFcbTbls.error = TRUE;
}
