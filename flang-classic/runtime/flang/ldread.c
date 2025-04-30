/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief List-directed read module.
 */

#include "global.h"
#include "format.h"
#include "list_io.h"
#include <string.h>

/* define a few things for run-time tracing */
static int dbgflag;
#undef DBGBIT
#define DBGBIT(v) (LOCAL_DEBUG && (dbgflag & v))
/*
 * dbgbit values:
 * 0x01  ldr
 * 0x02  read_record
 * 0x04  ldr_end
 * 0x08  get_token
 */

int read_record_internal(void);
static int read_record(void);
static char *alloc_rbuf(int, bool);
static int skip_record(void);

static FIO_FCB *fcb;  /* fcb of external file */
static bool accessed; /* file has been read */
static unsigned int byte_cnt;  /* number of bytes read */
static int n_irecs;   /* number of internal file records */
static bool internal_file;
static int rec_len;

static int gbl_dtype; /* data type of item (global to local funcs) */

#define RBUF_SIZE 256
static char rbuf[RBUF_SIZE + 1];
static unsigned rbuf_size = RBUF_SIZE;

static char *rbufp = rbuf; /* ptr to read buffer */
static char *currc;        /* current pointer in buffer */

static char *in_recp; /* internal i/o record (user's space) */

struct struct_G {
  short blank_zero; /* FIO_ ZERO or NULL */
  short pad;        /* FIO_ YES or NULL */
  short decimal;    /* COMMA, POINT, NONE */
  short round;      /* FIO_ UP, DOWN, ZERO, NEAREST, COMPATIBLE,
                     *      PROCESSOR_DEFINED, NONE
                     */
  FIO_FCB *fcb;
  bool accessed;
  int byte_cnt;
  int n_irecs;
  bool internal_file;
  int rec_len;
  int gbl_dtype;
  char rbuf[RBUF_SIZE + 1];
  unsigned rbuf_size;
  char *rbufp;
  char *currc;
  char *in_recp;

  AVAL tknval;
  int tkntyp;
  int scan_err;
  int repeat_cnt;
  int prev_tkntyp;
  bool comma_seen;
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

/*  get_token declarations, etc.  */

#define TK_ERROR 1
#define TK_NULL 2
#define TK_SLASH 3
#define TK_VAL 4
#define TK_VALS 5

static void shared_init(void);
static void get_token(void);
static void get_number(void);
static void get_cmplx(void);
static void get_infinity(void);
static void get_nan(void);
static __BIGREAL_T to_bigreal(AVAL *);
static void get_qstr(int);
static void get_junk(void);
static bool skip_spaces(void);
static bool find_char(int);

static AVAL tknval; /* TK_VAL value returned by get_token */
static int tkntyp;
static int scan_err;

/*  Initial state for a READ statement  */
static int repeat_cnt;
static int prev_tkntyp;
static bool comma_seen;

static void
save_gbl()
{
  if (gbl_avl) {
    gbl->fcb = fcb;
    gbl->accessed = accessed;
    gbl->byte_cnt = byte_cnt;
    gbl->n_irecs = n_irecs;
    gbl->internal_file = internal_file;
    gbl->rec_len = rec_len;
    gbl->gbl_dtype = gbl_dtype;
    if (rbuf_size > gbl->rbuf_size) {
      gbl->rbufp = malloc(rbuf_size);
      gbl->rbuf_size = rbuf_size;
    } else {
      gbl->rbufp = gbl->rbuf;
      gbl->rbuf_size = RBUF_SIZE;
    }
    memcpy(gbl->rbufp, rbufp, rbuf_size);
    if (currc) {
      assert(currc == (rbufp + (currc - rbufp)));
      gbl->currc = gbl->rbufp + (currc - rbufp);
    } else {
      gbl->currc = NULL;
    }
    gbl->in_recp = in_recp;

    /* may need to revisit this for derived type io */
    gbl->tknval = tknval;
    gbl->tkntyp = tkntyp;
    gbl->scan_err = scan_err;
    gbl->repeat_cnt = repeat_cnt;
    gbl->prev_tkntyp = prev_tkntyp;
    gbl->comma_seen = comma_seen;
  }
}

static void
restore_gbl()
{
  if (gbl_avl) {
    fcb = gbl->fcb;
    accessed = gbl->accessed;
    byte_cnt = gbl->byte_cnt;
    n_irecs = gbl->n_irecs;
    internal_file = gbl->internal_file;
    rec_len = gbl->rec_len;
    gbl_dtype = gbl->gbl_dtype;
    if (gbl->rbuf_size > rbuf_size) {
      if (rbufp != rbuf)
        rbufp = realloc(rbufp, gbl->rbuf_size);
      else
        rbufp = malloc(gbl->rbuf_size);
    } else {
      rbufp = rbuf;
    }
    rbuf_size = gbl->rbuf_size;
    memcpy(rbufp, gbl->rbufp, rbuf_size);
    if (gbl->currc) {
      assert(gbl->currc == (gbl->rbufp + (gbl->currc - gbl->rbufp)));
      currc = rbufp + (gbl->currc - gbl->rbufp);
    } else {
      currc = NULL;
    }

    in_recp = gbl->in_recp;

    /* may need to revisit this for derived type io */
    tknval = gbl->tknval;
    tkntyp = gbl->tkntyp;
    scan_err = gbl->scan_err;
    repeat_cnt = gbl->repeat_cnt;
    prev_tkntyp = gbl->prev_tkntyp;
    comma_seen = gbl->comma_seen;
  }
}

static void
save_samefcb()
{
  G *tmp_gbl;
  tmp_gbl = gbl->same_fcb;
  if (tmp_gbl) {
    tmp_gbl = &gbl_head[gbl->same_fcb_idx];
    tmp_gbl->accessed = accessed;
    tmp_gbl->byte_cnt = byte_cnt;
    tmp_gbl->repeat_cnt = repeat_cnt;
    tmp_gbl->prev_tkntyp = prev_tkntyp;
    tmp_gbl->n_irecs = n_irecs;
    tmp_gbl->internal_file = internal_file;
    tmp_gbl->rec_len = rec_len;
    tmp_gbl->gbl_dtype = gbl_dtype;
    tmp_gbl->in_recp = in_recp;
    if (rbuf_size > tmp_gbl->rbuf_size) {
      if (tmp_gbl->rbuf != tmp_gbl->rbufp)
        tmp_gbl->rbufp = realloc(tmp_gbl->rbufp, rbuf_size);
      else
        tmp_gbl->rbufp = malloc(rbuf_size);
      tmp_gbl->rbuf_size = rbuf_size;
    } else {
      tmp_gbl->rbufp = tmp_gbl->rbuf;
    }
    memcpy(tmp_gbl->rbufp, rbufp, rbuf_size);
    if (currc) {
      assert(currc == (rbufp + (currc - rbufp)));
      tmp_gbl->currc = tmp_gbl->rbufp + (currc - rbufp);
    } else {
      tmp_gbl->currc = NULL;
    }

    tmp_gbl->blank_zero = gbl->blank_zero;
    tmp_gbl->pad = gbl->pad;
    tmp_gbl->decimal = gbl->decimal;
    tmp_gbl->round = gbl->round;
    tmp_gbl->internal_file = internal_file;
  }
}

static void
allocate_new_gbl()
{
  G *tmp_gbl;
  int gsize = sizeof(G);
  if (gbl_avl >= gbl_size) {
    if (gbl_size == GBL_SIZE) {
      gbl_size = gbl_size + GBL_SIZE;
      tmp_gbl = (G *)malloc(gsize * gbl_size);
      memcpy(tmp_gbl, gbl_head, gsize * gbl_avl);
      memset(tmp_gbl + gbl_avl, 0, gsize * GBL_SIZE);
      gbl_head = tmp_gbl;
    } else {
      gbl_size = gbl_size + GBL_SIZE;
      gbl_head = (G *)realloc(gbl_head, gsize * gbl_size);
      memset(gbl_head + gbl_avl, 0, gsize * GBL_SIZE);
    }
  }
  gbl = &gbl_head[gbl_avl];
  if (gbl->rbufp != gbl->rbuf) {
    free(gbl->rbufp);
  }
  memset(gbl, 0, gsize);
  gbl->rbufp = gbl->rbuf;
  gbl->rbuf_size = RBUF_SIZE;
  ++gbl_avl;
}

static void
free_gbl()
{
  --gbl_avl;
  if (gbl_avl <= 0)
    gbl_avl = 0;
  if (gbl_avl == 0)
    gbl = &gbl_head[gbl_avl];
  else
    gbl = &gbl_head[gbl_avl - 1];
}

/* ***************************************/
/* list-directed external file read init */
/* ***************************************/

static int
_f90io_ldr_init(__INT_T *unit,   /* unit number */
               __INT_T *rec,    /* record number for direct access I/O */
               __INT_T *bitv,   /* same as for ENTF90IO(open_) */
               __INT_T *iostat) /* same as for ENTF90IO(open_) */
{

  int i;
  G *tmp_gbl;
  save_gbl();
  __fortio_errinit03(*unit, *bitv, iostat, "list-directed read");
  allocate_new_gbl();
  fcb = __fortio_rwinit(*unit, FIO_FORMATTED, rec, 0 /*read*/);

  if (fcb == NULL) {
    if (fioFcbTbls.eof)
      return EOF_FLAG;
    /* TBD - does there need to be fioFcbTbls.eor */
    return ERR_FLAG;
  }

  rec_len = fcb->reclen;
  internal_file = FALSE;

  gbl->decimal = fcb->decimal;

  /* check if recursive io on same external file */
  tmp_gbl = NULL;
  if (gbl_avl > 1) {
    for (i = gbl_avl - 2; i >= 0; --i) {
      if (gbl_head[i].fcb == fcb) {
        tmp_gbl = &gbl_head[i];
        break;
      }
    }
  }
  if (tmp_gbl) {
    gbl->same_fcb = tmp_gbl;
    gbl->same_fcb_idx = i;
    gbl->blank_zero = tmp_gbl->blank_zero;
    gbl->pad = tmp_gbl->pad;
    gbl->decimal = tmp_gbl->decimal;
    gbl->round = tmp_gbl->round;

    accessed = tmp_gbl->accessed;
    byte_cnt = tmp_gbl->byte_cnt;
    prev_tkntyp = tmp_gbl->prev_tkntyp;
    repeat_cnt = tmp_gbl->repeat_cnt;
    n_irecs = tmp_gbl->n_irecs;
    rec_len = tmp_gbl->rec_len;
    gbl_dtype = tmp_gbl->gbl_dtype;
    in_recp = tmp_gbl->in_recp;
    internal_file = tmp_gbl->internal_file;
    if (tmp_gbl->rbuf_size > rbuf_size) {
      if (rbufp != rbuf)
        rbufp = realloc(rbufp, tmp_gbl->rbuf_size);
      else
        rbufp = malloc(tmp_gbl->rbuf_size);
      rbuf_size = tmp_gbl->rbuf_size;
    } else {
      rbufp = rbuf;
    }
    memcpy(rbufp, tmp_gbl->rbufp, tmp_gbl->rbuf_size);
    if (tmp_gbl->currc) {
      currc = rbufp + (tmp_gbl->currc - tmp_gbl->rbufp);
    } else {
      currc = NULL;
    }
    comma_seen = FALSE;
    return 0;
  } else {
    gbl->same_fcb = tmp_gbl;
    gbl->same_fcb_idx = 0;
    fcb->skip = 0;
  }

  shared_init();
  return 0;
}

__INT_T
ENTF90IO(LDR_INIT, ldr_init)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat) /* same as for ENTF90IO(open_) */
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_ldr_init(unit, rec, bitv, iostat);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTF90IO(LDR_INIT03A, ldr_init03a)
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
    if (ISPRESENTC(decimal) && s == 0) {
      if (__fortio_eq_str(CADR(decimal), CLEN(decimal), "COMMA")) {
        gbl->decimal = FIO_COMMA;
      } else if (__fortio_eq_str(CADR(decimal), CLEN(decimal), "POINT")) {
        gbl->decimal = FIO_POINT;
      } else
        s = __fortio_error(FIO_ESPEC);
    }
    if (ISPRESENTC(pad) && s == 0) {
      if (__fortio_eq_str(CADR(pad), CLEN(pad), "YES"))
        gbl->pad = FIO_YES;
      else if (__fortio_eq_str(CADR(pad), CLEN(pad), "NO"))
        gbl->pad = FIO_NO;
      else
        s = __fortio_error(FIO_ESPEC);
    }
    if (ISPRESENTC(round) && s == 0) {
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
ENTF90IO(LDR_INIT03, ldr_init03)
(__INT_T *istat, DCHAR(blank), DCHAR(decimal), DCHAR(pad),
 DCHAR(round) DCLEN(blank) DCLEN(decimal) DCLEN(pad) DCLEN(round))
{
  return ENTF90IO(LDR_INIT03A, ldr_init03a) (istat, CADR(blank), CADR(decimal),
                                 CADR(pad), CADR(round), (__CLEN_T)CLEN(blank),
               		  (__CLEN_T)CLEN(decimal), (__CLEN_T)CLEN(pad),
	               	  (__CLEN_T)CLEN(round));
}

__INT_T
ENTCRF90IO(LDR_INIT, ldr_init)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat) /* same as for ENTF90IO(open_) */
{
  int s = 0;
  s = _f90io_ldr_init(unit, rec, bitv, iostat);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}

/* ***********************************************************************/
/* list-directed internal file read init                                 */
/* ***********************************************************************/

static int
_f90io_ldr_intern_init(
    char *cunit,      /* pointer to variable or array to read from */
    __INT_T *rec_num, /* number of records in internal file.
                       * 0 if the file is an assumed size
                       * character array */
    __INT_T *bitv,    /* same as for ENTF90IO(open_) */
    __INT_T *iostat,  /* same as for ENTF90IO(open_) */
    __CLEN_T cunit_siz)
{
  save_gbl();
  __fortio_errinit03(-99, *bitv, iostat, "list-directed read");

  allocate_new_gbl();

  internal_file = TRUE;
  in_recp = cunit;
  n_irecs = *rec_num;
  rec_len = cunit_siz;

  shared_init();
  return 0;
}

__INT_T
ENTF90IO(LDR_INTERN_INITA, ldr_intern_inita)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat   /* same as for ENTF90IO(open_) */
 DCLEN64(cunit))
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_ldr_intern_init(CADR(cunit), rec_num, bitv, iostat, CLEN(cunit));
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(LDR_INTERN_INIT, ldr_intern_init)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat   /* same as for ENTF90IO(open_) */
 DCLEN(cunit))
{
  return ENTF90IO(LDR_INTERN_INITA, ldr_intern_inita) (CADR(cunit), rec_num,
                                   bitv, iostat, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTCRF90IO(LDR_INTERN_INITA, ldr_intern_inita)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat   /* same as for ENTF90IO(open_) */
 DCLEN64(cunit))
{
  return _f90io_ldr_intern_init(CADR(cunit), rec_num, bitv, iostat, CLEN(cunit));
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(LDR_INTERN_INIT, ldr_intern_init)
(DCHAR(cunit),     /* pointer to variable or array to read from */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat   /* same as for ENTF90IO(open_) */
 DCLEN(cunit))
{
  return ENTCRF90IO(LDR_INTERN_INITA, ldr_intern_inita) (CADR(cunit), rec_num,
                                      bitv, iostat, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTF90IO(LDR_INTERN_INITE, ldr_intern_inite)
(char **cunit,     /* variable containing address to read from */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *len)
{
  /* DECODE initialization */
  int s = 0;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_ldr_intern_init(*cunit, rec_num, bitv, iostat, *len);
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(LDR_INTERN_INITE, ldr_intern_inite)
(char **cunit,     /* variable containing address to read from */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *len)
{
  /* DECODE initialization */
  return _f90io_ldr_intern_init(*cunit, rec_num, bitv, iostat, *len);
}

static void
shared_init(void)
{
  accessed = FALSE;
  byte_cnt = 0;
  repeat_cnt = 0;
  prev_tkntyp = 0;
  comma_seen = FALSE; /* read_record sets it TRUE */
}

/* **************************************************************************/

int
__f90io_ldr(int type,    /* data type (as defined in pghpft.h) */
            long length, /* # items of type to read */
            int stride,  /* distance in bytes between items */
            char *item,  /* where to transfer data to */
            __CLEN_T itemlen)
{
  int ret_err = 0; /* return error (for iostat) */
  long item_num;
  char *tmpitem;

  if (fioFcbTbls.error) {
    ret_err = ERR_FLAG;
    goto ldr_err;
  }
  if (fioFcbTbls.eof) {
    ret_err = EOF_FLAG;
    goto ldr_err;
  }

  if (DBGBIT(0x1)) {
    __io_printf("\n__fortio_ldr: ");
    __io_printf("item=%p, ", item);
    __io_printf("type=%d, ", type);
    __io_printf("length=%ld, ", length);
    __io_printf("stride=%d\n", stride);
  }

  if (length <= 0)
    /* no items to read */
    return 0;

  if (prev_tkntyp == TK_SLASH)
    return 0;
  if (byte_cnt == 0 && (ret_err = read_record()) != 0) {
    ret_err = __fortio_error(ret_err);
    goto ldr_err;
  }

  /* main loop is driven by number (length) of items to be read */

  tmpitem = item;
  gbl_dtype = type;
  for (item_num = 0; item_num < length; item_num++, tmpitem += stride) {
    get_token();
    if (tkntyp == TK_SLASH)
      return 0;
    if (tkntyp == TK_ERROR) {
      ret_err = __fortio_error(scan_err);
      goto ldr_err;
    }
    if (tkntyp == TK_NULL)
      continue;

    if (tkntyp == TK_VALS) {
      tkntyp = TK_VAL;
      if (gbl_dtype != __STR && gbl_dtype != __NCHAR) {
        ret_err = __fortio_error(FIO_EERR_DATA_CONVERSION);
        goto ldr_err;
      }
    }
    scan_err = __fortio_assign(tmpitem, type, itemlen, &tknval);
    if (scan_err) {
      ret_err = __fortio_error(scan_err);
      goto ldr_err;
    }
  }
  return 0;

ldr_err:
  free_gbl();
  restore_gbl();
  __fortio_errend03();
  return (ret_err);
}

__INT_T
ENTF90IO(LDRA, ldra)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data to */
 DCLEN64(item))
{
  int typ;
  int cnt, ioproc, str;
  __CLEN_T len;
  char *adr;
  int s = 0;

  typ = *type;
  cnt = *count;
  str = *stride;
  adr = CADR(item);
  len = (typ == __STR) ? CLEN(item) : GET_DIST_SIZE_OF(typ);

#if defined(DEBUG)
  if ((str / len) * len != str)
    __fort_abort("f90io_ldr: stride not a multiple of item length");
#endif
  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_ldr(typ, cnt, str, adr, len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, adr, cnt, str / len, typ, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(LDR, ldr)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data to */
 DCLEN(item))
{
  return ENTF90IO(LDRA, ldra) (type, count, stride, CADR(item), (__CLEN_T)CLEN(item));
}

/* same as ldr, but item may be array - for ldr, the compiler
 * scalarizes.
 */
__INT_T
ENTF90IO(LDR_AA, ldr_aa)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data to */
 DCLEN64(item))
{
  int typ;
  int cnt, ioproc, str;
  __CLEN_T len;
  char *adr;
  int s = 0;

  typ = *type;
  cnt = *count;
  str = *stride;
  adr = CADR(item);
  len = (typ == __STR) ? CLEN(item) : GET_DIST_SIZE_OF(typ);

#if defined(DEBUG)
  if ((str / len) * len != str)
    __fort_abort("f90io_ldr_a: stride not a multiple of item length");
#endif
  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_ldr(typ, cnt, str, adr, len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, adr, cnt, str / len, typ, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(LDR_A, ldr_a)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data to */
 DCLEN(item))
{
  return ENTF90IO(LDR_AA, ldr_aa) (type, count, stride, CADR(item), (__CLEN_T)CLEN(item));
}

__INT_T
ENTF90IO(LDR64_AA, ldr64_aa)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT8_T *count, /* # items of type to read */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data to */
 DCLEN64(item))
{
  int typ;
  long cnt;
  int ioproc, str;
  __CLEN_T len;
  char *adr;
  int s = 0;

  typ = *type;
  cnt = *count;
  str = *stride;
  adr = CADR(item);
  len = (typ == __STR) ? CLEN(item) : GET_DIST_SIZE_OF(typ);

#if defined(DEBUG)
  if ((str / len) * len != str)
    __fort_abort("f90io_ldr_a: stride not a multiple of item length");
#endif
  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_ldr(typ, cnt, str, adr, len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, adr, cnt, str / len, typ, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(LDR64_A, ldr64_a)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT8_T *count, /* # items of type to read */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data to */
 DCLEN(item))
{
  return ENTF90IO(LDR64_AA, ldr64_aa) (type, count, stride, CADR(item),
                                       (__CLEN_T)CLEN(item));
}

__INT_T
ENTCRF90IO(LDRA, ldra)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data to */
 DCLEN64(item))
{
  int typ;
  int cnt, str;
  __CLEN_T len;
  char *adr;

  typ = *type;
  cnt = *count;
  str = *stride;
  adr = CADR(item);
  len = (typ == __STR) ? CLEN(item) : GET_DIST_SIZE_OF(typ);

#if defined(DEBUG)
  if ((str / len) * len != str)
    __fort_abort("__f90io_ldr: stride not a multiple of item length");
#endif
  return __f90io_ldr(typ, cnt, str, adr, len);
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(LDR, ldr)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *count,  /* # items of type to read */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data to */
 DCLEN(item))
{
  return ENTCRF90IO(LDRA, ldra) (type, count, stride, CADR(item), (__CLEN_T)CLEN(item));
}

/* **************************************************************************/
/* ***  get_token and support routines  *****/
/* **************************************************************************/

#undef BEGINS_NUM
#undef ISDELIMITER

#define BEGINS_NUM(c) (ISDIGIT(c) || (c) == '.')
#define ISDELIMITER(c)                                                         \
  ((c) == ',' || (c) == ' ' || (c) == '\t' || (c) == '/' || (c == '\n'))

static int is_repeat_count(char *);

static void
get_token()
{
  char ch;

  if (DBGBIT(0x8))
    __io_printf("get_token enter: repeat_cnt:%d\n", repeat_cnt);
  if (repeat_cnt) {
    repeat_cnt--;
    return;
  }
  scan_err = 0;
  tkntyp = 0;
  if (gbl_dtype == __STR || gbl_dtype == __NCHAR)
    /* current item is type character */
    do {
      switch (*currc++) {
      case '\n':
        /* read another record */
        if ((scan_err = read_record()) != 0)
          tkntyp = TK_ERROR;
        break;

      /* whitespace => delimiters */
      case ' ':
      case '\t':
        break;

      case ';':
        if (gbl->decimal != FIO_COMMA)
          break;
        if (comma_seen) {
          /* consecutive commas implies null value */
          tkntyp = TK_NULL;
          goto multiple_commas;
        }
        /* otherwise, it's just a delimiter */
        comma_seen = TRUE;
        break;
      case ',':
        if (gbl->decimal == FIO_COMMA)
          break;
        if (comma_seen) {
          /* consecutive commas implies null value */
          tkntyp = TK_NULL;
          goto multiple_commas;
        }
        /* otherwise, it's just a delimiter */
        comma_seen = TRUE;
        break;

      case '/':
        tkntyp = TK_SLASH;
        break;

      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        currc--;
        if (is_repeat_count(currc)) {
          int rc;
          get_number();
          rc = tknval.val.i;
          ch = *++currc;
          if (ISDELIMITER(ch))
            tkntyp = TK_NULL;
          else
            get_token();
          repeat_cnt = rc - 1;
        } else
          get_junk();
        break;

      case '\'':
        get_qstr('\'');
        break;

      case '\"':
        get_qstr('\"');
        break;

      default:
        currc--;
        get_junk();
        break;
      }

    } while (tkntyp == 0); /* end current item is type character */
  else
    /* current item is not type character */
    do {
      switch (*currc++) {
      case '\n':
        /* read another record */
        if ((scan_err = read_record()) != 0)
          tkntyp = TK_ERROR;
        break;

      /* whitespace => delimiters */
      case ' ':
      case '\t':
        break;
      case ';':
        if (gbl->decimal != FIO_COMMA)
          break;
        if (comma_seen) {
          /* consecutive commas implies null value */
          tkntyp = TK_NULL;
          goto multiple_commas;
        }
        /* otherwise, it's just a delimiter */
        comma_seen = TRUE;
        break;

      case ',':
        if (gbl->decimal == FIO_COMMA) {
          ch = *currc;
          if (ISDIGIT(ch)) {
            currc--;
            get_number();
            break;
          }
          /* check for logical */
          if (ch == 'T' || ch == 't') {
            currc++;
            goto have_true;
          }
          if (ch == 'F' || ch == 'f') {
            currc++;
            goto have_false;
          }
          currc--;
          get_number(); /* VMS extension (only if numeric or log */
          break;
        } else {
          if (comma_seen) {
            /* consecutive commas implies null value */
            tkntyp = TK_NULL;
            goto multiple_commas;
          }
          /* otherwise, it's just a delimiter */
          comma_seen = TRUE;
          break;
        }

      case '/':
        tkntyp = TK_SLASH;
        break;

      case '+':
      case '-':
        ch = *currc;
        if (BEGINS_NUM(ch)) {
          currc--;
          get_number();
          break;
        }
        /* Could also support NaN here too??? */
        if (ch == 'i' || ch == 'I') {
          currc--;
          get_infinity();
          break;
        }
        currc--;
        get_number(); /* VMS extension (only if numeric or log */
        break;

      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        currc--;
        get_number();
        /* check if repeat count */
        if (tkntyp == TK_VAL && tknval.dtype == __BIGINT && tknval.val.i != 0 &&
            *currc == '*') {
          int rc;
          rc = tknval.val.i;
          ch = *++currc;
          if (ISDELIMITER(ch))
            tkntyp = TK_NULL;
          else
            get_token();
          repeat_cnt = rc - 1;
        }
        break;

      case '.':
        ch = *currc;
        if (ISDIGIT(ch)) {
          currc--;
          get_number();
          break;
        }
        /* check for logical */
        if (ch == 'T' || ch == 't') {
          currc++;
          goto have_true;
        }
        if (ch == 'F' || ch == 'f') {
          currc++;
          goto have_false;
        }
        currc--;
        get_number(); /* VMS extension (only if numeric or log */
        break;

      case '(':
        get_cmplx();
        break;

      case 't':
      case 'T':
      have_true:
        tknval.val.i = FTN_TRUE;
        goto have_logical;

      case 'f':
      case 'F':
      have_false:
        tknval.val.i = FTN_FALSE;

      have_logical:
        while (TRUE) {
          ch = *currc;
          if (ISDELIMITER(ch))
            break;
          ++currc;
        }
        tkntyp = TK_VAL;
        tknval.dtype = __BIGLOG;
        break;

      case '\'':
        get_qstr('\'');
        break;

      case '\"':
        get_qstr('\"');
        break;

      case 'e':
      case 'E':
      case 'd':
      case 'D':
        currc--;
        get_number(); /* VMS extension (only if numeric or log */
        break;

      case 'i':
      case 'I':
        currc--;
        get_infinity();
        break;
      case 'n':
      case 'N':
        currc--;
        get_nan();
        break;

      default:
        tkntyp = TK_ERROR;
        scan_err = FIO_ELEX; /* unknown token */
        break;
      }

    } while (tkntyp == 0); /* end current item is not type character */
                           /*
                            * if a value is found, we did not see a comma.
                            */
  comma_seen = FALSE;

multiple_commas:

  prev_tkntyp = tkntyp;
  if (DBGBIT(0x8))
    __io_printf("get_token: new token %d\n", tkntyp);
}

/** \brief Given that a string begins with a digit, is it a repeat count?
 *
 * If a string of digits is immediately followed by '*', then the
 * digit string is a repeat count.
 */
static int
is_repeat_count(char *p)
{
  char *q;
  char ch;

  q = p + 1;
  while (TRUE) {
    ch = *q;
    if (!ISDIGIT(ch))
      break;
    q++;
  }
  if ((q - p) == 1 && *p == '0')
    return 0;
  if (ch == '*')
    return 1;
  return 0;
}

/** \brief
 * Extract integer, real, or double precision constant token:
 */
static void
get_number(void)
{
  int ret_err;
  int type;
  union {
    __BIGINT_T i;
    __BIGREAL_T d;
    __INT8_T i8v;
  } val;
  int len;

  if (gbl->decimal == FIO_COMMA)
    ret_err = __fortio_getnum(currc, &type, &val, &len, TRUE);
  else
    ret_err = __fortio_getnum(currc, &type, &val, &len, FALSE);
  currc += len;
  if (ret_err) {
    scan_err = ret_err;
    tkntyp = TK_ERROR;
    return;
  }
  if (type == 1) {
    tknval.dtype = __BIGREAL;
    tknval.val.d = val.d;
  }
  else if (type == 2) {
    tknval.dtype = __INT8;
    tknval.val.i8v = val.i8v;
  }
  else if (type == 3) {
    /* Degenerate VMS REAL, allow integer value 0 to be returned only if
     * a REAL type was expected.  Otherwise its an error.
     */
    if (!REAL_ALLOWED(gbl_dtype)) {
      scan_err = FIO_EERR_DATA_CONVERSION;
      tkntyp = TK_ERROR;
      return;
    }
    tknval.dtype = __BIGINT;
    tknval.val.i = val.i;
  } else {
    tknval.dtype = __BIGINT;
    tknval.val.i = val.i;
  }
  tkntyp = TK_VAL;

}

/** \brief
 * A left paren has been found.  Create a complex constant.
 * currc locates character after '('.
 */
static void
get_cmplx(void)
{
  static AVAL cmplx[2] = {{__BIGREAL, {0}}, {__BIGREAL, {0}}};

  get_token();
  if (tkntyp != TK_VAL || tknval.dtype == __STR || tknval.dtype == __NCHAR)
    goto cmplx_err;
  cmplx[0].val.d = to_bigreal(&tknval);
  if (gbl->decimal == FIO_COMMA) {
    if (!find_char(';')) /* leaves ptr at after ';' */
      goto cmplx_err;
  } else {
    if (!find_char(',')) /* leaves ptr at after ',' */
      goto cmplx_err;
  }
  get_token();
  if (tkntyp != TK_VAL || tknval.dtype == __STR || tknval.dtype == __NCHAR)
    goto cmplx_err;
  cmplx[1].val.d = to_bigreal(&tknval);
  tknval.dtype = __BIGCPLX;
  tknval.val.cmplx = cmplx;
  if (!find_char(')'))
    goto cmplx_err;
  tkntyp = TK_VAL;
  return;

cmplx_err:
  scan_err = FIO_ELEX; /* unknown token */
  tkntyp = TK_ERROR;
}

/** \brief
 * An 'I' has been found.  Is it an infinity?  Valid are +inf, -inf, inf,
 * +infinity, -infinity, infinity
 */
static void
get_infinity(void)
{
  union ieee ieee_v;
  char c;
  c = *currc;
  ieee_v.i[0] = 0x0;
  ieee_v.i[1] = 0x7ff00000;
  if (c == '-') {
    ieee_v.v.s = 1;
    currc++;
  } else if (c == '+') {
    ieee_v.v.s = 0;
    currc++;
  } else {
    ieee_v.v.s = 0;
  }
  c = *currc++;
  if (c == 'i' || c == 'I') {
    c = *currc++;
    if (c == 'n' || c == 'N') {
      c = *currc++;
      if (c == 'f' || c == 'F') {
        c = *currc++;
        if (ISDELIMITER(c)) {
          currc--;
          tknval.dtype = __BIGREAL;
          tknval.val.d = ieee_v.d;
          tkntyp = TK_VAL;
          return;
        }
        if (c == 'i' || c == 'I') {
          c = *currc++;
          if (c == 'n' || c == 'N') {
            c = *currc++;
            if (c == 'i' || c == 'I') {
              c = *currc++;
              if (c == 't' || c == 'T') {
                c = *currc++;
                if (c == 'y' || c == 'Y') {
                  c = *currc;
                  if (ISDELIMITER(c)) {
                    tknval.dtype = __BIGREAL;
                    tknval.val.d = ieee_v.d;
                    tkntyp = TK_VAL;
                    return;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  scan_err = FIO_ELEX; /* unknown token */
  tkntyp = TK_ERROR;
}

/** \brief
 * An 'N' has been found.  Is it a NaN?  Valid are nan and nan(ddd...)
 */
static void
get_nan(void)
{
  union ieee ieee_v;
  char c;
  unsigned int nval, ntval, nshplaces;
  c = *currc++;
  ieee_v.i[0] = 0x0;
  ieee_v.i[1] = 0x7ff00000;
  if (c == 'n' || c == 'N') {
    c = *currc++;
    if (c == 'a' || c == 'A') {
      c = *currc++;
      if (c == 'n' || c == 'N') {
        c = *currc++;
        if (c == '(') {
          ieee_v.i[0] = 0x0;
          ieee_v.v.hm = 0x0;
          ieee_v.v.e = 2047;
          ntval = 0;
          c = *currc++;
          nshplaces = 48;
          while (TRUE) {
            if ((c >= '0') && (c <= '9'))
              nval = c - '0';
            else if ((c >= 'a') && (c <= 'f'))
              nval = c - 'a' + 10;
            else if ((c >= 'A') && (c <= 'F'))
              nval = c - 'A' + 10;
            else if (c == ')')
              break;
            else
              goto conv_nan_error;
            if (nshplaces > 28)
              ieee_v.v.hm |= nval << (nshplaces - 32);
            else
              ieee_v.v.lm |= nval << nshplaces;
            ntval += nval;
            c = *currc++;
            if (nshplaces)
              nshplaces -= 4;
          }
          if (c == ')') {
            /* Set this quiet */
            if (ntval == 0)
              ieee_v.v.hm |= 0x80000;
            c = *currc;
          }
        } else {
          ieee_v.v.hm |= 0x80000; /* quiet */
        }
        if (ISDELIMITER(c)) {
          tknval.dtype = __BIGREAL;
          tknval.val.d = ieee_v.d;
          tkntyp = TK_VAL;
          return;
        }
      }
    }
  }
conv_nan_error:
  scan_err = FIO_ELEX; /* unknown token */
  tkntyp = TK_ERROR;
}

static __BIGREAL_T
to_bigreal(AVAL *valp)
{
  if (valp->dtype == __BIGREAL)
    return valp->val.d;
  if (valp->dtype == __INT8 || valp->dtype == __LOG8)
    return valp->val.d;
  assert(valp->dtype == __BIGINT || valp->dtype == __BIGLOG);
  return (__BIGREAL_T)valp->val.i;
}

/*  stuff for returning a string token */

static char chval[128];
static int chval_size = sizeof(chval);
static char *chvalp = chval;

/** \brief
 * A quote has been seen (' or ").  Create a character constant.
 */
static void
get_qstr(int quote)
{
  int len;
  char ch;

  len = 0;
  while (TRUE) {
    ch = *currc++;
    if (ch == '\n') {
      if (read_record()) {
        scan_err = FIO_ELEX; /* unknown token */
        tkntyp = TK_ERROR;
        return;
      }
      continue;
    }
    if (ch == quote) {
      if (*currc != quote)
        break;
      currc++;
    }
    if (len >= chval_size) {
      chval_size += 128;
      if (chvalp == chval) {
        chvalp = malloc(chval_size);
        (void) memcpy(chvalp, chval, len);
      } else
        chvalp = realloc(chvalp, chval_size);
    }
    chvalp[len++] = ch;
  }
  /* ************** HAND CHECK ****************/
  tknval.val.c.len = len;
  tknval.val.c.str = chvalp;
  tknval.dtype = gbl_dtype;
  tkntyp = TK_VALS;
}

/** \brief
 * Non-quoted strings in list-directed input.
 */
static void
get_junk(void)
{
  int len;
  char ch;

  len = 0;
  while (TRUE) {
    ch = *currc++;
    if (ch == '\\' && *currc == '\n') {
      if (read_record()) {
        scan_err = FIO_ELEX; /* unknown token */
        tkntyp = TK_ERROR;
        return;
      }
      continue;
    }
    if (ISDELIMITER(ch)) {
      currc--;
      break;
    }
    if (len >= chval_size) {
      chval_size += 128;
      if (chvalp == chval) {
        chvalp = malloc(chval_size);
        (void) memcpy(chvalp, chval, len);
      } else
        chvalp = realloc(chvalp, chval_size);
    }
    chvalp[len++] = ch;
  }
  /* ************** HAND CHECK ****************/
  tknval.val.c.len = len;
  tknval.val.c.str = chvalp;
  tknval.dtype = gbl_dtype;
  tkntyp = TK_VAL;
}

static bool
skip_spaces(void)
/* eat spaces, read new record if necessary */
{
  while (TRUE) {
    while (*currc == ' ' || *currc == '\t')
      currc++;
    if (*currc != '\n')
      break;
    scan_err = read_record();
    if (scan_err)
      return FALSE;
  }
  return TRUE;
}

static bool
find_char(int ch)
/* find a given character, skip leading spaces, read new record if necessary */
{
  if (!skip_spaces())
    return FALSE;
  if (*currc == ch) {
    currc++;
    return TRUE;
  }
  return FALSE;
}

/* ********************/
/*    read  support   */
/* ********************/

static int
read_record(void)
{
  if (internal_file) {
    if (n_irecs == 0)
      return read_record_internal();
    if (accessed)
      in_recp += rec_len;
    n_irecs--;

    byte_cnt = rec_len;
    if (byte_cnt >= rbuf_size)
      (void) alloc_rbuf(byte_cnt, FALSE);
    (void) memcpy(rbufp, in_recp, byte_cnt);
    accessed = TRUE;
  } else {
    if (fcb->pread) {
      int ch;
      char *p, *f;

      p = rbufp;
      f = fcb->pread;
      byte_cnt = 0;

      while (TRUE) {
        if (byte_cnt >= rbuf_size)
          p = alloc_rbuf(byte_cnt, TRUE);
        ch = *f++;
        if (ch == EOF) {
          if (__io_feof(fcb->fp)) {
            if (byte_cnt)
              break;
            return FIO_EEOF;
          }
          return __io_errno();
        }
        if (ch == '\r' && EOR_CRLF) {
          ch = *f++;
          if (ch == '\n')
            break;
          --f;
          ch = '\r';
        }
        if (ch == '\n')
          break;
        byte_cnt++;
        *p++ = ch;
      }
      fcb->pread = 0;
    } else {

      fcb->nextrec++;
      if (fcb->acc == FIO_DIRECT) {
        byte_cnt = rec_len;
        if (byte_cnt >= rbuf_size)
          (void) alloc_rbuf(byte_cnt, FALSE);
        if (fcb->nextrec > fcb->maxrec + 1)
          return FIO_EDREAD; /* attempt to read non-existent rec */
        if (__io_fread(rbufp, byte_cnt, 1, fcb->fp) != 1)
          return __io_errno();
      } else {
        /* sequential read */
        int ch;
        char *p;

        p = rbufp;
        byte_cnt = 0;

        while (TRUE) {
          if (byte_cnt >= rbuf_size)
            p = alloc_rbuf(byte_cnt, TRUE);
          ch = __io_fgetc(fcb->fp);
          if (ch == EOF) {
            if (__io_feof(fcb->fp)) {
              if (byte_cnt)
                break;
              return FIO_EEOF;
            }
            return __io_errno();
          }
          if (ch == '\r' && EOR_CRLF) {
            ch = __io_fgetc(fcb->fp);
            if (ch == '\n')
              break;
            __io_ungetc(ch, fcb->fp);
            ch = '\r';
          }
          if (ch == '\n')
            break;
          byte_cnt++;
          *p++ = ch;
        }
      }
    }
  }
  rbufp[byte_cnt] = '\n';
  if (!internal_file) {
    if (byte_cnt > 1)
      fcb->pback = &(rbufp[byte_cnt - 1]);
    else
      fcb->pback = &(rbufp[byte_cnt]);
  }
  currc = rbufp;
  comma_seen = TRUE;
  if (DBGBIT(0x2)) {
    __io_printf("read_rec: byte_cnt %d\n", byte_cnt);
    __io_printf("#%.*s#\n", byte_cnt, rbufp);
  }

  return 0;
}

static char *
alloc_rbuf(int size, bool copy)
{
  int old_size;

  old_size = rbuf_size;
  rbuf_size = size + 128;
  if (rbufp == rbuf) {
    rbufp = malloc(rbuf_size);
    if (copy)
      (void) memcpy(rbufp, rbuf, old_size);
  } else
    rbufp = realloc(rbufp, rbuf_size);
  return rbufp + size;
}

static int
skip_record(void)
{
  if (internal_file) {
    if (n_irecs == 0)
      return FIO_EEOF;
    n_irecs--;
    return 0;
  }

  /* external file:  check for errors */
  if (gbl->same_fcb) /* don't check for recursive io */
    return 0;

  fcb->nextrec++;
  if (fcb->acc == FIO_DIRECT) {
    if (fcb->nextrec > fcb->maxrec + 1)
      return FIO_EDREAD; /* attempt to read non-existent rec */
    if (__io_fseek(fcb->fp, (seekoffx_t)rec_len, SEEK_CUR) != 0)
      return __io_errno();
    fcb->coherent = 0;
  } else {
    /* sequential read */
    int ch;
    int bt = 0;

    while (TRUE) {
      ch = __io_fgetc(fcb->fp);
      if (ch == EOF) {
        if (__io_feof(fcb->fp)) {
          if (bt)
            break;
          return FIO_EEOF;
        }
        return __io_errno();
      }
#if defined(_WIN64)
      if (ch == '\r') {
        ch = __io_fgetc(fcb->fp);
        if (ch == '\n')
          break;
        __io_ungetc(ch, fcb->fp);
        ch = '\r';
      }
#endif
      if (ch == '\n')
        break;
      bt++;
    }
  }

  return 0;
}

/* **************************************************************************/

static int
_f90io_ldr_end(void)
{
  int ret_err = 0;

  if (DBGBIT(0x4))
    __io_printf("ENTER: f90io_ldr_end\n");

  if (fioFcbTbls.error) {
    return ERR_FLAG;
  }
  if (fioFcbTbls.eof) {
    return EOF_FLAG;
  }
  if (gbl->same_fcb)
    return 0;

  if (byte_cnt == 0)
    ret_err = skip_record();
  if (ret_err)
    ret_err = __fortio_error(ret_err);
  return ret_err;
}

__INT_T
ENTF90IO(LDR_END, ldr_end)()
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_ldr_end();

  save_samefcb();
  free_gbl();
  restore_gbl();
  __fortio_errend03();

  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(LDR_END, ldr_end)()
{
  int s = 0;

  s = _f90io_ldr_end();

  save_samefcb();
  free_gbl();
  restore_gbl();
  __fortio_errend03();
  return s;
}
