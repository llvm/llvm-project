/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Lst-directed write module.
 */

#include "global.h"
#include "format.h"
#include "list_io.h"
#include <string.h>

/* define a few things for run-time tracing */
static int dbgflag;
#undef DBGBIT
#define DBGBIT(v) (LOCAL_DEBUG && (dbgflag & v))

static FIO_FCB *fcb; /* fcb of external file */

static char *in_recp; /* internal i/o record (user's space) */
static char *in_curp; /* current position in internal i/o record */

static bool record_written; /* only used for writes to an external file */
static int byte_cnt;
static int rec_len;
static int n_irecs;         /* number of records in internal file */
static bool write_called;   /* __f90io_ldw called at least once (extern file)*/
static bool internal_file;  /* TRUE if writing to internal file */
static char *internal_unit; /* base address of internal file buffer */
static char delim;          /* delimiter character if DELIM was specified */

static int last_type; /* last data type written */

struct struct_G {
  short decimal; /* COMMA, POINT, NONE */
  short sign;    /* FIO_ PLUS, SUPPRESS, PROCESSOR_DEFINED,
                  *      NONE
                  */
  short round;   /* FIO_ UP, DOWN, ZERO, etc.
                  *
                  */
  FIO_FCB *fcb;  /* Other static variable */
  char *in_recp;
  char *in_curp;
  bool record_written;
  int byte_cnt;
  int rec_len;
  int n_irecs;
  bool write_called;
  bool internal_file;
  char *internal_unit;
  char delim;
  int last_type;
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

/* local functions */

static int write_item(const char *, int);
static int write_record(void);

static void
save_gbl()
{
  if (gbl_avl) {
    gbl->fcb = fcb;
    gbl->in_recp = in_recp;
    gbl->in_curp = in_curp;
    gbl->record_written = record_written;
    gbl->byte_cnt = byte_cnt;
    gbl->rec_len = rec_len;
    gbl->n_irecs = n_irecs;
    gbl->write_called = write_called;
    gbl->delim = delim;
    gbl->last_type = last_type;
  }
}

static void
save_samefcb()
{
  G *tmp_gbl;
  tmp_gbl = gbl->same_fcb;
  if (tmp_gbl) {
    tmp_gbl = &gbl_head[gbl->same_fcb_idx];
    tmp_gbl->in_recp = in_recp;
    tmp_gbl->in_curp = in_curp;
    tmp_gbl->record_written = record_written;
    tmp_gbl->byte_cnt = byte_cnt;
    tmp_gbl->rec_len = rec_len;
    tmp_gbl->n_irecs = n_irecs;
    tmp_gbl->write_called = write_called;
    tmp_gbl->delim = delim;
    tmp_gbl->last_type = last_type;
  }
}

static void
restore_gbl()
{
  if (gbl_avl) {
    fcb = gbl->fcb;
    in_recp = gbl->in_recp;
    in_curp = gbl->in_curp;
    record_written = gbl->record_written;
    byte_cnt = gbl->byte_cnt;
    rec_len = gbl->rec_len;
    n_irecs = gbl->n_irecs;
    write_called = gbl->write_called;
    internal_file = gbl->internal_file;
    internal_unit = gbl->internal_unit;
    delim = gbl->delim;
    last_type = gbl->last_type;
  }
}

static void
allocate_new_gbl()
{
  G *tmp_gbl;
  int gsize = sizeof(G);
  if (gbl_avl >= gbl_size) {
    if (gbl_size == GBL_SIZE) {
      gbl_size = gbl_size + 15;
      tmp_gbl = (G *)malloc(gsize * gbl_size);
      memcpy(tmp_gbl, gbl_head, gsize * gbl_avl);
      gbl_head = tmp_gbl;
    } else {
      gbl_size = gbl_size + 15;
      gbl_head = (G *)realloc(gbl_head, gsize * gbl_size);
    }
  }
  gbl = &gbl_head[gbl_avl];
  memset(gbl, 0, gsize);
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

/* **************************************************/
/* list-directed external file write initialization */
/* **************************************************/

static int
_f90io_ldw_init(__INT_T *unit,   /* unit number */
               __INT_T *rec,    /* record number for direct access I/O */
               __INT_T *bitv,   /* same as for ENTF90IO(open_) */
               __INT_T *iostat) /* same as for ENTF90IO(open_) */
{
  G *tmp_gbl;
  int i;
  save_gbl();

  __fortio_errinit03(*unit, *bitv, iostat, "list-directed write");

  allocate_new_gbl();
  fcb = __fortio_rwinit(*unit, FIO_FORMATTED, rec, 1 /*write*/);
  if (fcb == NULL)
    return ERR_FLAG;
  fcb->skip = 0;

  rec_len = (int)fcb->reclen;
  byte_cnt = 0;
  record_written = FALSE;
  write_called = FALSE;

  if (fcb->delim == FIO_APOSTROPHE) {
    delim = '\'';
  } else if (fcb->delim == FIO_QUOTE) {
    delim = '\"';
  } else {
    delim = 0;
  }

  /* check if same file */
  tmp_gbl = NULL;
  i = 0;
  if (gbl_avl > 1) {
    for (i = gbl_avl - 2; i >= 0; --i) {
      if (gbl_head[i].fcb == fcb) {
        tmp_gbl = &gbl_head[i];
        break;
      }
    }
  }
  if (tmp_gbl) {
    in_recp = tmp_gbl->in_recp;
    in_curp = tmp_gbl->in_curp;
    record_written = tmp_gbl->record_written;
    byte_cnt = tmp_gbl->byte_cnt;
    rec_len = tmp_gbl->rec_len;
    n_irecs = tmp_gbl->n_irecs;
    write_called = tmp_gbl->write_called;
    delim = tmp_gbl->delim;
    last_type = tmp_gbl->last_type;
    gbl->same_fcb = tmp_gbl;
    gbl->same_fcb_idx = i;
  } else {
    gbl->same_fcb = tmp_gbl;
    gbl->same_fcb_idx = 0;
    last_type = __NONE;
  }

  gbl->decimal = fcb->decimal;
  gbl->sign = fcb->sign;
  gbl->round = fcb->round;

  return 0;
}

__INT_T
ENTF90IO(LDW_INIT, ldw_init)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat) /* same as for ENTF90IO(open_) */
{
  int s = 0;
  internal_file = FALSE;
  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_ldw_init(unit, rec, bitv, iostat);
  gbl->internal_file = FALSE;
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTF90IO(PRINT_INIT, print_init)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat) /* same as for ENTF90IO(open_) */
{
  int s = 0;
  internal_file = FALSE;
  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_ldw_init(unit, rec, bitv, iostat);
  gbl->internal_file = FALSE;
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(LDW_INIT, ldw_init)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat) /* same as for ENTF90IO(open_) */
{
  int s = 0;
  internal_file = FALSE;
  s = _f90io_ldw_init(unit, rec, bitv, iostat);
  gbl->internal_file = FALSE;
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}

__INT_T
ENTCRF90IO(PRINT_INIT, print_init)
(__INT_T *unit,   /* unit number */
 __INT_T *rec,    /* record number for direct access I/O */
 __INT_T *bitv,   /* same as for ENTF90IO(open_) */
 __INT_T *iostat) /* same as for ENTF90IO(open_) */
{
  int s = 0;
  internal_file = FALSE;
  s = _f90io_ldw_init(unit, rec, bitv, iostat);
  gbl->internal_file = FALSE;
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}

__INT_T
ENTF90IO(LDW_INIT03A, ldw_init03a)
(__INT_T *istat, DCHAR(decimal), DCHAR(delim),
 DCHAR(sign) DCLEN64(decimal) DCLEN64(delim) DCLEN64(sign))
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
      } else {
        s = __fortio_error(FIO_ESPEC);
        goto init03_end;
      }
    }
    if (ISPRESENTC(delim)) {
      if (__fortio_eq_str(CADR(delim), CLEN(delim), "APOSTROPHE")) {
        delim = '\'';
      } else if (__fortio_eq_str(CADR(delim), CLEN(delim), "QUOTE")) {
        delim = '\"';
      } else if (__fortio_eq_str(CADR(delim), CLEN(delim), "NONE")) {
        delim = 0;
      } else {
        s = __fortio_error(FIO_ESPEC);
        goto init03_end;
      }
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
  }
init03_end:
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(LDW_INIT03, ldw_init03)
(__INT_T *istat, DCHAR(decimal), DCHAR(delim),
 DCHAR(sign) DCLEN(decimal) DCLEN(delim) DCLEN(sign))
{
  return ENTF90IO(LDW_INIT03A, ldw_init03a) (istat, CADR(decimal), CADR(delim),
                              CADR(sign), (__CLEN_T)CLEN(decimal),
			      (__CLEN_T)CLEN(delim), (__CLEN_T)CLEN(sign));
}

/* **************************************************/
/* list-directed internal file write initialization */
/* **************************************************/

static int
_f90io_ldw_intern_init(char *cunit,      /* pointer to variable or array to
                                         * write into */
                      __INT_T *rec_num, /* number of records in internal file.
                                         * 0 if the file is an assumed size
                                         * character * array */
                      __INT_T *bitv,    /* same as for ENTF90IO(open_) */
                      __INT_T *iostat,  /* same as for ENTF90IO(open_) */
                      __CLEN_T cunit_len)
{
  save_gbl();
  __fortio_errinit(-99, *bitv, iostat, "internal list-directed write");

  allocate_new_gbl();

  /* set newly static variable */
  rec_len = cunit_len;
  byte_cnt = 0;
  in_curp = in_recp = cunit;
  n_irecs = *rec_num;
  delim = 0;
  last_type = __NONE;

  /*  set first record to blanks; obviates need for checking if first time
      and if no items were written. */

  (void) memset(in_recp, ' ', rec_len);

  return 0;
}

__INT_T
ENTF90IO(LDW_INTERN_INITA, ldw_intern_inita)
(DCHAR(cunit),     /* pointer to variable or array to
                    * write into */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat   /* same as for ENTF90IO(open_) */
 DCLEN64(cunit))
{
  __INT_T s = 0;

  internal_file = TRUE;
  internal_unit = CADR(cunit);
  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_ldw_intern_init(CADR(cunit), rec_num, bitv, iostat, CLEN(cunit));
  gbl->internal_file = internal_file;
  gbl->internal_unit = internal_unit;
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(LDW_INTERN_INIT, ldw_intern_init)
(DCHAR(cunit),     /* pointer to variable or array to
                    * write into */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat   /* same as for ENTF90IO(open_) */
 DCLEN(cunit))
{
  return ENTF90IO(LDW_INTERN_INITA, ldw_intern_inita) (CADR(cunit), rec_num,
                                    bitv, iostat, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTCRF90IO(LDW_INTERN_INITA, ldw_intern_inita)
(DCHAR(cunit),     /* pointer to variable or array to
                    * write into */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat   /* same as for ENTF90IO(open_) */
 DCLEN64(cunit))
{
  __INT_T s;
  internal_file = TRUE;
  internal_unit = CADR(cunit);
  s = _f90io_ldw_intern_init(CADR(cunit), rec_num, bitv, iostat, CLEN(cunit));
  gbl->internal_file = internal_file;
  gbl->internal_unit = internal_unit;
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(LDW_INTERN_INIT, ldw_intern_init)
(DCHAR(cunit),     /* pointer to variable or array to
                    * write into */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat   /* same as for ENTF90IO(open_) */
 DCLEN(cunit))
{
  return ENTCRF90IO(LDW_INTERN_INITA, ldw_intern_inita) (CADR(cunit), rec_num,
                                      bitv, iostat, (__CLEN_T)CLEN(cunit));
}

__INT_T
ENTF90IO(LDW_INTERN_INITE, ldw_intern_inite)
(char **cunit,     /* variable containing address to write into */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *len)     /* size of 'cunit' */
{
  /* ENCODE initialization */
  __INT_T s = 0;

  internal_file = TRUE;
  internal_unit = *cunit;
  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_ldw_intern_init(*cunit, rec_num, bitv, iostat, *len);
  gbl->internal_file = internal_file;
  gbl->internal_unit = internal_unit;
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(LDW_INTERN_INITE, ldw_intern_inite)
(char **cunit,     /* variable containing address to write into */
 __INT_T *rec_num, /* number of records in internal file.
                    * 0 if the file is an assumed size
                    * character * array */
 __INT_T *bitv,    /* same as for ENTF90IO(open_) */
 __INT_T *iostat,  /* same as for ENTF90IO(open_) */
 __INT_T *len)     /* size of 'cunit' */
{
  __INT_T s = 0;

  /* ENCODE initialization */
  internal_file = TRUE;
  internal_unit = *cunit;
  s = _f90io_ldw_intern_init(*cunit, rec_num, bitv, iostat, *len);
  gbl->internal_file = internal_file;
  gbl->internal_unit = internal_unit;
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return s;
}

/* *************************/
/*   list-directed write   */
/* *************************/

extern char __f90io_conv_buf[];

int
__f90io_ldw(int type,    /* data type (as defined in pghpft.h) */
            long length, /* # items of type to write. May be <= 0 */
            int stride,  /* distance in bytes between items */
            char *item,  /* where to transfer data from */
            __CLEN_T item_length)
{
  long item_num; /* loop index for main item write loop */
  char *tmpitem;
  bool plus_sign;
  int ret_err = 0;

  if (DBGBIT(0x1)) {
    __io_printf("parameters to ENTF90IO(ldw) are:\n");
    __io_printf("item = %p\n", item);
    __io_printf("type = %d\n", type);
    __io_printf("length = %ld\n", length);
    __io_printf("stride = %d\n", stride);
  }

  if (fioFcbTbls.error) {
    ret_err = ERR_FLAG;
    goto ldw_error;
  }

  /* main loop is driven by number of items to be written */

  tmpitem = item;
  if (gbl->sign == FIO_PLUS)
    plus_sign = TRUE;
  else
    plus_sign = FALSE;
  for (item_num = 0; item_num < length; item_num++, tmpitem += stride) {
    int width;
    char *p;

    write_called = TRUE;

    if (gbl->decimal == FIO_COMMA)
      p = __fortio_default_convert(tmpitem, type, item_length, &width, TRUE,
                                  plus_sign, gbl->round);
    else
      p = __fortio_default_convert(tmpitem, type, item_length, &width, FALSE,
                                  plus_sign, gbl->round);
    if (Is_complex(type) && byte_cnt > 0) {
      /*	complex is a bit strange since blanks are removed from
          the beginning and end of the constant.  A blank is needed
          at the beginning. */
      ret_err = write_item(" ", 1);
      if (ret_err) {
        ret_err = __fortio_error(ret_err);
        goto ldw_error;
      }
    }

    if (byte_cnt && (delim || (type != __STR) || (last_type != __STR))) {
      ret_err = write_item(" ", 1);
      if (ret_err) {
        ret_err = __fortio_error(ret_err);
        goto ldw_error;
      }
    }

    if (type == __STR && delim) {
      /* add delimiters and double any contained delimiters */
      char *newp, *p0, *p1;
      p0 = p;
      width += 2; /* count added delimiters */
      while (*p0) {
        if (*p0++ == delim)
          ++width;
      }
      newp = p1 = malloc(width + 1); /* plus null terminator */
      p0 = p;
      *p1++ = delim;
      while (*p0) {
        if ((*p1++ = *p0++) == delim)
          *p1++ = delim;
      }
      *p1++ = delim;
      *p1 = '\0';
      if (p != __f90io_conv_buf)
        free(p);
      p = newp;
    }
    ret_err = write_item(p, width);
    if (ret_err) {
      ret_err = __fortio_error(ret_err);
      goto ldw_error;
    }
    last_type = type;
  }
  return 0;

ldw_error:
  free_gbl();
  restore_gbl();
  __fortio_errend03();
  return (ret_err);
}

__INT_T
ENTF90IO(LDWA, ldwa)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to write. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN64(item))
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_ldw(*type, *length, *stride, CADR(item),
                    (*type == __STR) ? CLEN(item) : 0);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(LDW, ldw)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to write. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN(item))
{
  return ENTF90IO(LDWA, ldwa) (type, length, stride, CADR(item), (__CLEN_T)CLEN(item));
}

/* same as ldw, but item may be array - for ldw, the compiler
 * scalarizes.
 */
__INT_T
ENTF90IO(LDW_AA, ldw_aa)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to write. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN64(item))
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_ldw(*type, *length, *stride, CADR(item),
                    (*type == __STR) ? CLEN(item) : 0);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(LDW_A, ldw_a)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to write. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN(item))
{
  return ENTF90IO(LDW_AA, ldw_aa) (type, length, stride, CADR(item), (__CLEN_T)CLEN(item));
}

__INT_T
ENTF90IO(LDW64_AA, ldw64_aa)
(__INT_T *type,    /* data type (as defined in pghpft.h) */
 __INT8_T *length, /* # items of type to write. May be <= 0 */
 __INT_T *stride,  /* distance in bytes between items */
 DCHAR(item)       /* where to transfer data from */
 DCLEN64(item))
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_ldw(*type, *length, *stride, CADR(item),
                    (*type == __STR) ? CLEN(item) : 0);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(LDW64_A, ldw64_a)
(__INT_T *type,    /* data type (as defined in pghpft.h) */
 __INT8_T *length, /* # items of type to write. May be <= 0 */
 __INT_T *stride,  /* distance in bytes between items */
 DCHAR(item)       /* where to transfer data from */
 DCLEN(item))
{
  return ENTF90IO(LDW64_AA, ldw64_aa) (type, length, stride, CADR(item), (__CLEN_T)CLEN(item));
}

__INT_T
ENTCRF90IO(LDWA, ldwa)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to write. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN64(item))
{
  return __f90io_ldw(*type, *length, *stride, CADR(item),
                     (*type == __STR) ? CLEN(item) : 0);
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(LDW, ldw)
(__INT_T *type,   /* data type (as defined in pghpft.h) */
 __INT_T *length, /* # items of type to write. May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to transfer data from */
 DCLEN(item))
{
  return
ENTCRF90IO(LDWA, ldwa) (type, length, stride, CADR(item), (__CLEN_T)CLEN(item));
}

/* --------------------------------------------------------------------- */

static int
write_item(const char *p, int len)
{
  int newlen;
  int ret_err;

  if (DBGBIT(0x1))
    __io_printf("write_item #%s#, len %d\n", p, len);

  record_written = FALSE;

  /*	compute the number of bytes written AFTER this item is written.
      NOTE that ByteCnt is set after the item is written since we may split
      lines.  */

  newlen = byte_cnt + len;

  /*  for internal i/o in_recp/in_curp is a pointer to user's space */
  if (internal_file) {
    if (byte_cnt == 0) { /* prepend a blank to a new record */
      newlen++;
      in_curp++;
    }
    if (newlen > rec_len) {
      if (byte_cnt == 0)
        return FIO_ETOOBIG;
      n_irecs--;
      if (n_irecs <= 0) /* write after last internal record */
        return FIO_ETOOFAR;

      in_recp += rec_len;
      (void) memset(in_recp, ' ', rec_len);
      newlen = len + 1;
      in_curp = in_recp + 1;
    }
    (void) memcpy(in_curp, p, len);
    in_curp += len;
  } else {               /* external file */
    if (byte_cnt == 0) { /* prepend a blank to a new record */
      if (FWRITE(" ", 1, 1, fcb->fp) != 1)
        return __io_errno();
      newlen++;
    }
    if (fcb->acc == FIO_DIRECT) {
      if (newlen > rec_len)
        return FIO_ETOOBIG;
      if (len && FWRITE(p, len, 1, fcb->fp) != 1)
        return __io_errno();
    } else { /* sequential write */
             /*	split lines if necessary; watch for the case where a long
                 character item is the first item for the record.  */

      if (byte_cnt && ((fcb->reclen && newlen > fcb->reclen) ||
                       (!fcb->reclen && newlen > 79))) {
        ret_err = write_record();
        if (ret_err)
          return ret_err;
        if (FWRITE(" ", 1, 1, fcb->fp) != 1)
          return __io_errno();
        newlen = len + 1;
        record_written = FALSE;
      }
      if (len && FWRITE(p, len, 1, fcb->fp) != 1)
        return __io_errno();
    }
  }

  byte_cnt = newlen;
  return 0;
}

/* ---------------------------------------------------------------------- */

static int
write_record(void)
{
  if (DBGBIT(0x1))
    __io_printf("ENTER: write_record\n");

  /* external file: check for errors */
  assert(!internal_file);
  assert(!fioFcbTbls.eof) assert(!fioFcbTbls.error)

      if (record_written) return 0;

  if (fcb->acc == FIO_DIRECT) {
    if (rec_len > byte_cnt) {
#define BL_BUF bl_buf
#define BL_BUFSZ sizeof(bl_buf)
      static char bl_buf[] = "                "; /* 16 blanks */
      int pad;
      int j, n;
      pad = rec_len - byte_cnt;
      n = pad / BL_BUFSZ;
      for (j = 0; j < n; j++)
        if (FWRITE(BL_BUF, BL_BUFSZ, 1, fcb->fp) != 1)
          return __io_errno();

      if ((j = pad - (n * BL_BUFSZ)) != 0)
        if (FWRITE(BL_BUF, j, 1, fcb->fp) != 1)
          return __io_errno();
    }
  } else { /* sequential write: append carriage return */
#if defined(_WIN64)
    if (__io_binary_mode(fcb->fp))
      if (FWRITE("\r", 1, 1, fcb->fp) != 1)
        return __io_errno();
#endif
    if (FWRITE("\n", 1, 1, fcb->fp) != 1)
      return __io_errno();
  }
  ++(fcb->nextrec);

  byte_cnt = 0;
  last_type = __NONE;
  record_written = TRUE;
  return 0;
}

/* -------------------------------------------------------------------- */

static int
_f90io_ldw_end()
{

  if (internal_file && in_curp != in_recp)
    in_recp += rec_len; /* update internal file pointer */

  if (fioFcbTbls.error)
    return ERR_FLAG;

  if (!internal_file) {
    int ret_err;
    /* if this is a child io, or recursive io that has io caller
     * don't print it out, let the parent io or caller print
     * to a file so that it all prints in one line(spec:9.5.3.7.1).
     */
    if (gbl->same_fcb)
      return 0;
    if (!write_called) {
      if (fcb->nonadvance) {
        fcb->nonadvance = FALSE;
      } else {
        if (FWRITE(" ", 1, 1, fcb->fp) != 1)
          return __fortio_error(__io_errno());
        byte_cnt = 1;
        record_written = FALSE;
      }
    }
    ret_err = write_record();
    if (ret_err)
      return __fortio_error(ret_err);

    fcb->nextrec--;
    if (fcb->acc == FIO_DIRECT) {
      /* this write statement may have increased maximum record num: */
      if (fcb->nextrec - 1 > fcb->maxrec)
        fcb->maxrec = fcb->nextrec - 1;
    }
  }

  return 0;
}

__INT_T
ENTF90IO(LDW_END, ldw_end)()
{
  int ioproc, len;
  int s = 0;

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc) {
    s = _f90io_ldw_end();
    if (internal_file)
      len = in_recp - internal_unit;
  }
  if (!LOCAL_MODE && internal_file) {
    DIST_RBCSTL(ioproc, &len, 1, 1, __CINT, sizeof(int));
    DIST_RBCSTL(ioproc, internal_unit, 1, 1, __CHAR, len);
  }

  save_samefcb();
  free_gbl();
  restore_gbl();
  __fortio_errend03();

  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(LDW_END, ldw_end)()
{
  int s = 0;
  s = _f90io_ldw_end();

  save_samefcb();
  free_gbl();
  restore_gbl();
  __fortio_errend03();

  return s;
}

/* ----------------------------------------------------------------------- */

#if LOCAL_DEBUG
void
ENTF90IO(LDW_DEBUG, ldw_debug)(__INT_T v)
{
  if (v)
    dbgflag |= v;
  else
    dbgflag = 0;
}
#endif

#if LOCAL_DEBUG
void
ENTCRF90IO(LDW_DEBUG, ldw_debug)(__INT_T v)
{
  if (v)
    dbgflag |= v;
  else
    dbgflag = 0;
}
#endif

/* --------------------------------------------------------------------- */
/*
 *  Opportunistic by-value write routines 
 *
 */
__INT_T
ENTF90IO(SC_LDW,sc_ldw)(
    int   item,         /* scalar data to transfer */
    int   type)         /* data type (as defined in pghpft.h) */
{

  return __f90io_ldw(type, 1, 0, (char *)&item, 0);
}

__INT_T
ENTF90IO(SC_I_LDW, sc_i_ldw)(int item, int type)
{
  return __f90io_ldw(type, 1, 0, (char *)&item, 0);
}

__INT_T
ENTF90IO(SC_L_LDW, sc_l_ldw)(long long item, int type)
{
  return __f90io_ldw(type, 1, 0, (char *)&item, 0);
}

__INT_T
ENTF90IO(SC_F_LDW, sc_f_ldw)(float item, int type)
{
  return __f90io_ldw(type, 1, 0, (char *)&item, 0);
}

__INT_T
ENTF90IO(SC_D_LDW, sc_d_ldw)(double item, int type)
{
  return __f90io_ldw(type, 1, 0, (char *)&item, 0);
}

__INT_T
ENTF90IO(SC_Q_LDW, sc_q_ldw)(float128_t item, int type)
{
  return __f90io_ldw(type, 1, 0, (char *)&item, 0);
}

__INT_T
ENTF90IO(SC_CF_LDW, sc_cf_ldw)(float real, float imag, int type)
{
  struct {
    float real;
    float imag;
  } dum;
  dum.real = real;
  dum.imag = imag;
  return __f90io_ldw(type, 1, 0, (char *)&dum, 0);
}

__INT_T
ENTF90IO(SC_CD_LDW, sc_cd_ldw)(double real, double imag, int type)
{
  struct {
    double real;
    double imag;
  } dum;
  dum.real = real;
  dum.imag = imag;
  return __f90io_ldw(type, 1, 0, (char *)&dum, 0);
}

#ifdef TARGET_SUPPORTS_QUADFP
__INT_T
ENTF90IO(SC_CQ_LDW, sc_cq_ldw)(float128_t real, float128_t imag, int type)
{
  struct {
    float128_t real;
    float128_t imag;
  } dum;
  dum.real = real;
  dum.imag = imag;
  return __f90io_ldw(type, 1, 0, (char *)&dum, 0);
}
#endif

__INT_T
ENTF90IO(SC_CH_LDW, sc_ch_ldw)(char *item, int type, int len)
{
  return __f90io_ldw(type, 1, 0, item, len);
}
