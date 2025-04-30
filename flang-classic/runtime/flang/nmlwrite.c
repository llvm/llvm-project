/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Support for namelist write statements.
 */

#include <ctype.h>
#include <string.h>
#include "global.h"
#include "format.h"
#include "nml.h"

/*** define a few things for run-time tracing ***/
static int dbgflag;
#undef DBGBIT
#define DBGBIT(v) (LOCAL_DEBUG && (dbgflag & v))

static FIO_FCB *f;

static char *in_recp; /* internal i/o record (user's space) */
static char *in_curp; /* current position in internal i/o record */

static int byte_cnt;
static int rec_len;
static int n_irecs;         /* number of records in internal file */
static bool internal_file;  /* TRUE if writing to internal file */
static char *internal_unit; /* base address of internal file buffer */
static char delim;
static bool need_comma;
static int skip;

typedef struct {
  short decimal; /* COMMA, POINT, NONE */
  short sign;    /* FIO_ PLUS, SUPPRESS, PROCESSOR_DEFINED,
                  *      NONE
                  */
  short round;   /* FIO_ UP, DOWN, etc. */

  FIO_FCB *f;

  char *in_recp; /* internal i/o record (user's space) */
  char *in_curp; /* current position in internal i/o record */

  int byte_cnt;
  int rec_len;
  int n_irecs;         /* number of records in internal file */
  bool internal_file;  /* TRUE if writing to internal file */
  char *internal_unit; /* base address of internal file buffer */
  char delim;
  bool need_comma;
  int skip;
  int same_fcb_idx;

  __INT_T *unit;   /* used in user defined io */
  __INT_T *iostat; /* used in user defined io */
} G;

static G static_gbl[GBL_SIZE];
static G *gbl = &static_gbl[0];

static int emit_eol(void);
static int write_nml_val(NML_DESC **, NML_DESC *, char *);
static int write_item(const char *, int);
static int write_char(int);
static int eval(int, char *, NML_DESC *, NML_DESC **);
static int eval_dtio(int, char *, NML_DESC *, NML_DESC **);
static int I8(eval_sb)(NML_DESC **, NML_DESC *, char *, int);
static int I8(eval_dtio_sb)(NML_DESC **, NML_DESC *, char *, int);
static int dtio_write_scalar(NML_DESC **, NML_DESC *, char *, int);

static SB sb;

/* ---------------------------------------------------------------- */

static int
_f90io_nmlw_init(__INT_T *unit, /* unit number */
                __INT_T *rec, /* record number for direct access I/O;
                               * rec not used, but JUST IN CASE
                               */
                __INT_T *bitv,   /* same as for ENTF90IO(open_) */
                __INT_T *iostat) /* same as for ENTF90IO(open_) */
{
  __fortio_errinit03(*unit, *bitv, iostat, "namelist write");

  f = __fortio_rwinit(*unit, FIO_FORMATTED, rec, 1 /*write*/);
  if (f == NULL)
    return ERR_FLAG;
  f->skip = 0;

  if (f->delim == FIO_APOSTROPHE) {
    delim = '\'';
  } else if (f->delim == FIO_QUOTE) {
    delim = '\"';
  } else {
    delim = 0;
  }
  gbl->decimal = f->decimal;
  gbl->sign = f->sign;
  gbl->round = f->round;
  gbl->unit = unit;
  gbl->iostat = iostat;

  return 0;
}

/** \brief Initialize for namelist write to an external file
 *
 * \param unit - unit number
 * \param rec - record number for direct access I/O; rec not used, but JUST IN CASE
 * \param bitv - same as for ENTF90IO(open_)
 * \param iostat - same as for ENTF90IO(open_)
 */
int
ENTF90IO(NMLW_INIT, nmlw_init)(__INT_T *unit,
                               __INT_T *rec,
                               __INT_T *bitv,
                               __INT_T *iostat)
{
  int s = 0;

  internal_file = FALSE;
  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_nmlw_init(unit, rec, bitv, iostat);
  return DIST_STATUS_BCST(s);
}

int
ENTF90IO(NMLW_INIT03A, nmlw_init03a)(__INT_T *istat,
                                   DCHAR(decimal),
                                   DCHAR(delim),
                                   DCHAR(sign)
                                   DCLEN64(decimal)
                                   DCLEN64(delim)
                                   DCLEN64(sign))
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
    if (ISPRESENTC(delim)) {
      if (__fortio_eq_str(CADR(delim), CLEN(delim), "APOSTROPHE"))
        delim = '\'';
      else if (__fortio_eq_str(CADR(delim), CLEN(delim), "QUOTE"))
        delim = '\"';
      else if (__fortio_eq_str(CADR(delim), CLEN(delim), "NONE"))
        delim = 0;
      else
        return __fortio_error(FIO_ESPEC);
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
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
int
ENTF90IO(NMLW_INIT03, nmlw_init03)(__INT_T *istat,
                                   DCHAR(decimal),
                                   DCHAR(delim),
                                   DCHAR(sign)
                                   DCLEN(decimal)
                                   DCLEN(delim)
                                   DCLEN(sign))
{
  return ENTF90IO(NMLW_INIT03A, nmlw_init03a)(istat, CADR(decimal), CADR(delim),
                                CADR(sign), (__CLEN_T)CLEN(decimal),
				(__CLEN_T)CLEN(delim), (__CLEN_T)CLEN(sign));
}

int
ENTCRF90IO(NMLW_INIT, nmlw_init)(__INT_T *unit, /* unit number */
                                 __INT_T *rec,  /* record number for direct
                                                 * access I/O; * rec not used,
                                                 * but JUST IN CASE
                                                 */
                                 __INT_T *bitv,   /*same as for ENTF90IO(open_) */
                                 __INT_T *iostat) /*same as for ENTF90IO(open_) */
{
  return _f90io_nmlw_init(unit, rec, bitv, iostat);
}

static int
_f90io_nmlw_intern_init(char *cunit,      /* pointer to variable or array to
                                          * write into */
                       __INT_T *rec_num, /* number of records in internal file.
                                          * 0 if the file is an assumed size
                                          * character * array */
                       __INT_T *bitv,    /* same as for ENTF90IO(open_) */
                       __INT_T *iostat,  /* same as for ENTF90IO(open_) */
                       __CLEN_T cunit_len)
{
  static FIO_FCB dumfcb;

  __fortio_errinit03(-99, *bitv, iostat, "internal namelist write");
  rec_len = cunit_len;
  byte_cnt = 0;
  in_curp = in_recp = cunit;
  n_irecs = *rec_num;
  delim = 0;

  f = &dumfcb; /* so the f-> refs don't have to be guarded */

  return 0;
}

/** \brief Internal file namelist write initialization
 *
 * \param rec_num - number of records in internal file; 0 if the file is an assumed size character array 
 * \param bitv - same as for ENTF90IO(open_)
 * \param iostat - same as for ENTF90IO(open_)
 */
int
ENTF90IO(NMLW_INTERN_INITA, nmlw_intern_inita)(
    DCHAR(cunit),     /* pointer to variable or array to
                       * write into */
    __INT_T *rec_num, /* number of records in internal file.
                       * 0 if the file is an assumed size
                       * character * array */
    __INT_T *bitv,    /* same as for ENTF90IO(open_) */
    __INT_T *iostat   /* same as for ENTF90IO(open_) */
    DCLEN64(cunit))
{
  int s = 0;

  internal_file = TRUE;
  internal_unit = CADR(cunit);
  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_nmlw_intern_init(CADR(cunit), rec_num, bitv, iostat, CLEN(cunit));
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
int
ENTF90IO(NMLW_INTERN_INIT, nmlw_intern_init)(
    DCHAR(cunit),     /* pointer to variable or array to
                       * write into */
    __INT_T *rec_num, /* number of records in internal file.
                       * 0 if the file is an assumed size
                       * character * array */
    __INT_T *bitv,    /* same as for ENTF90IO(open_) */
    __INT_T *iostat   /* same as for ENTF90IO(open_) */
    DCLEN(cunit))
{
  return ENTF90IO(NMLW_INTERN_INITA, nmlw_intern_inita)(CADR(cunit), rec_num,
                                     bitv, iostat, (__CLEN_T)CLEN(cunit));
}

int
ENTCRF90IO(NMLW_INTERN_INITA, nmlw_intern_inita)(
    DCHAR(cunit),     /* pointer to variable or array to
                       * write into */
    __INT_T *rec_num, /* number of records in internal file.
                       * 0 if the file is an assumed size
                       * character * array */
    __INT_T *bitv,    /* same as for ENTF90IO(open_) */
    __INT_T *iostat   /* same as for ENTF90IO(open_) */
    DCLEN64(cunit))
{
  internal_file = TRUE;
  internal_unit = CADR(cunit);
  return _f90io_nmlw_intern_init(CADR(cunit), rec_num, bitv, iostat,
                                CLEN(cunit));
}
/* 32 bit CLEN version */
int
ENTCRF90IO(NMLW_INTERN_INIT, nmlw_intern_init)(
    DCHAR(cunit),     /* pointer to variable or array to
                       * write into */
    __INT_T *rec_num, /* number of records in internal file.
                       * 0 if the file is an assumed size
                       * character * array */
    __INT_T *bitv,    /* same as for ENTF90IO(open_) */
    __INT_T *iostat   /* same as for ENTF90IO(open_) */
    DCLEN(cunit))
{
  return ENTCRF90IO(NMLW_INTERN_INITA, nmlw_intern_inita)(CADR(cunit), rec_num,
                                       bitv, iostat, (__CLEN_T)CLEN(cunit));
}

static int
emit_eol(void)
{
#if defined(_WIN64)
  int ret_err;
#endif

  if (!internal_file) {
#if defined(_WIN64)
    if (__io_binary_mode(f->fp)) {
      ret_err = write_char('\r');
      if (ret_err)
        return ret_err;
    }
#endif
    return write_char('\n');
  }
  n_irecs--;
  if (n_irecs < 0) /* write after last internal record */
    return FIO_ETOOFAR;
  /*
   * blankfill the internal file record
   */
  if (rec_len > byte_cnt)
    memset(in_curp, ' ', rec_len - byte_cnt);
  in_recp += rec_len; /* update internal file pointer */
  in_curp = in_recp;
  byte_cnt = 0;
  return 0;
}

static void
I8(fillup_sb)(int v, NML_DESC *descp, char *loc_addr)
{
  int i;
  F90_Desc *sd = get_descriptor(descp);
  DECL_DIM_PTRS(acd);

  sb.v = v;
  sb.ndims = *(__POINT_T *)((char *)descp + sizeof(NML_DESC));
  sb.elemsz = I8(siz_of)(descp);
  for (i = 0; i < sb.ndims; ++i) {
    SET_DIM_PTRS(acd, sd, i);
    sb.idx[i] = F90_DPTR_LBOUND_G(acd);
    sb.sect[i].lwb = F90_DPTR_LBOUND_G(acd);
    sb.sect[i].upb = F90_DPTR_EXTENT_G(acd);
    sb.sect[i].stride = F90_DPTR_SSTRIDE_G(acd);
    sb.mult[i] = F90_DPTR_LSTRIDE_G(acd);
    sb.lwb[i] = F90_DPTR_LBOUND_G(acd);
  }
  sb.loc_addr = loc_addr;
}

static int
write_nml_val(NML_DESC **NextDescp, NML_DESC *descp, char *loc_addr)
{
  int num_consts;
  __POINT_T *desc_dims, new_ndims;
  __POINT_T actual_ndims;
  int j, k;
  char *p;
  int len;
  int ret_err;
  NML_DESC *next_descp;

  num_consts = 1;
  desc_dims = (__POINT_T *)((char *)descp + sizeof(NML_DESC));
  if (descp->ndims == -1 || descp->ndims == -2) {
    new_ndims = *(__POINT_T *)((char *)descp + sizeof(__POINT_T));
    num_consts = nelems_of(descp);
  } else {
    num_consts = nelems_of(descp);
  }

  /*  compute number of bytes to add to reach next descriptor: */
  if (descp->ndims == -1 || descp->ndims == -2)
    k = sizeof(NML_DESC) + (sizeof(__POINT_T) * 2);
  else {
    ACTUAL_NDIMS(actual_ndims);
    k = sizeof(NML_DESC) + (actual_ndims * sizeof(__POINT_T) * 2);
  }
  next_descp = (NML_DESC *)((char *)descp + k);

  /*  print each constant for this derived type:  */

  if (descp->type == __DERIVED) {
    NML_DESC *start_descp;
    char *start_addr;
    char *mem_addr;

    start_descp = next_descp;
    start_addr = loc_addr;
    for (k = 0; k < num_consts; k++) {
      next_descp = start_descp;
      while (TRUE) {
        if (next_descp->nlen == 0) { /* end of members */
          next_descp = (NML_DESC *)((char *)next_descp + sizeof(__POINT_T));
          break;
        }
        mem_addr = start_addr + (long)next_descp->addr;

        ret_err = write_nml_val(&next_descp, next_descp, mem_addr);
        if (ret_err)
          return ret_err;
      }
      start_addr += descp->len;
    }
    *NextDescp = next_descp;
    return 0;
  }

  /*  print each constant for this variable/array:  */

  for (k = 0; k < num_consts; k++) {
    if (need_comma) { /*  put out leading blanks:  */
      /*  put commas after each constant except for very, very last: */
      if (gbl->decimal == FIO_COMMA) {
        ret_err = write_char(';');
      } else {
        ret_err = write_char(',');
      }
      if (ret_err)
        return __fortio_error(ret_err);
      ret_err = emit_eol();
      if (ret_err)
        return __fortio_error(ret_err);
      f->nextrec++;
      for (j = 0; j < skip; j++) {
        ret_err = write_char(' ');
        if (ret_err)
          return __fortio_error(ret_err);
      }
    }
    if (descp->len > 0) { /* CHARACTER variable */
      int c;

      if (descp->type == __STR) {
        if (delim) {
          ret_err = write_char(delim);
          if (ret_err)
            return __fortio_error(ret_err);
        }
        for (j = 0; j < descp->len; j++) {
          c = *loc_addr++;
          ret_err = write_char(c);
          if (ret_err)
            return __fortio_error(ret_err);
          if (delim && c == delim) {
            ret_err = write_char(c); /* double delimiter character */
            if (ret_err)
              return __fortio_error(ret_err);
          }
        }
        if (delim) {
          ret_err = write_char(delim);
          if (ret_err)
            return __fortio_error(ret_err);
        }
      }
    } else {
      bool plus_sign;
      if (gbl->sign == FIO_PLUS)
        plus_sign = TRUE;
      else
        plus_sign = FALSE;
      if (gbl->decimal == FIO_COMMA)
        p = __fortio_default_convert(loc_addr, descp->type, 0, &len, TRUE,
                                    plus_sign, gbl->round);
      else
        p = __fortio_default_convert(loc_addr, descp->type, 0, &len, FALSE,
                                    plus_sign, gbl->round);
      ret_err = write_item(p, len);
      if (ret_err)
        return __fortio_error(ret_err);

      loc_addr += FIO_TYPE_SIZE(descp->type);
    }
    need_comma = TRUE;
  }

  *NextDescp = next_descp;
  return 0;
}

static int
write_item(const char *p, int len)
{
  int newlen;

  if (DBGBIT(0x1))
    __io_printf("write_item #%s#, len %d\n", p, len);

  if (!internal_file) {
    if (len && FWRITE(p, len, 1, f->fp) != 1)
      return __io_errno();
    return 0;
  }
  /**  for internal i/o in_recp/in_curp is a pointer to user's space **/
  newlen = byte_cnt + len;
  if (newlen > rec_len) {
    /*
     * f2003 10.10.2 L9:  The processor may begin new records as necessary.
     * However, except for complex constants and character values, the end
     * of a record shall not occur within a constant, character value, or
     * name, and blanks shall not appear within a constant, character value,
     * or name.
     */
    if (byte_cnt == 0 || len > rec_len)
      return FIO_ETOOBIG;
    n_irecs--;
    if (n_irecs <= 0) /* write after last internal record */
      return FIO_ETOOFAR;
    /*
     * blankfill the internal file record
     */
    if (rec_len > byte_cnt)
      memset(in_curp, ' ', rec_len - byte_cnt);
    in_recp += rec_len;
    newlen = len;
    in_curp = in_recp;
  }
  (void) memcpy(in_curp, p, len);
  in_curp += len;
  byte_cnt = newlen;
  return 0;
}

static int
write_char(int ch)
{
  char bf[1];
  bf[0] = ch;
  return write_item(bf, 1);
}

/** \brief
 * Recursively compute the index space given a set of subscripts for n
 * dimensions. The evaluation begins by iterating over the last dimension,
 * recursively evaluating the subscripts of the next (to the left) for
 * each iteration.  For a given dimension d's index, subscripts to the left
 * are recursively computed.  When the first dimension is reached, the address
 * of the element represented by the current subscript values is passed to
 * the 'eval' function.
 */
static int
I8(eval_sb)(NML_DESC **NextDescp, NML_DESC *descp, char *loc_addr, int d)
{
  int err, k;
  char *new_addr = NULL;
  NML_DESC *next_descp;
  __POINT_T *desc_dims;
  __POINT_T actual_ndims;

  /*  compute number of bytes to add to reach next descriptor: */
  if (descp->ndims == -1 || descp->ndims == -2)
    k = sizeof(NML_DESC) + (sizeof(__POINT_T) * 2);
  else {
    ACTUAL_NDIMS(actual_ndims);
    k = sizeof(NML_DESC) + (actual_ndims * sizeof(__POINT_T) * 2);
  }

  next_descp = (NML_DESC *)((char *)descp + k);

  if (descp->ndims == -1 || descp->ndims == -2) {
    desc_dims = (__POINT_T *)((char *)descp + sizeof(NML_DESC));
    if (*desc_dims == 0) {
      /* $p contains an address of array/scalar */
      new_addr = *(char **)sb.loc_addr;
      err = write_nml_val(&next_descp, descp, new_addr);
      if (err)
        return err;
      *NextDescp = next_descp;
      return 0;
    }
  }

  if (d == 0) {
    /*
     * Reached the first dimension; iterate over the first dimension,
     * compute the address of each element, and pass each address to
     * 'eval'.
     */
    F90_Desc *sd = get_descriptor(descp);
    for (sb.idx[0] = sb.sect[0].lwb; sb.idx[0] <= sb.sect[0].upb;
         sb.idx[0] += sb.sect[0].stride) {
      new_addr = I8(__fort_local_address)((*(char **)sb.loc_addr), sd,
                                         (__INT_T *)&sb.idx[0]);
      err = write_nml_val(&next_descp, descp, new_addr);
      if (err)
        return err;
    }
    *NextDescp = next_descp;
    return 0;
  }

  /* Iterate over the current dimension, and recursively evaluate all
   * subscripts in the dimensions to the left.
   */
  for (sb.idx[d] = sb.sect[d].lwb; sb.idx[d] <= sb.sect[d].upb;
       sb.idx[d] += sb.sect[d].stride) {
    err = I8(eval_sb)(&next_descp, descp, new_addr, d - 1);
    if (err)
      return err;
  }
  *NextDescp = next_descp;
  return 0;
}

/** \brief
 * Recursively compute the index space given a set of subscripts for n
 * dimensions. The evaluation begins by iterating over the last dimension,
 * recursively evaluating the subscripts of the next (to the left) for
 * each iteration.  For a given dimension d's index, subscripts to the left
 * are recursively computed.  When the first dimension is reached, the address
 * of the element represented by the current subscript values is passed to
 * the 'eval' function.
 */
static int
I8(eval_dtio_sb)(NML_DESC **NextDescp, NML_DESC *descp, char *loc_addr, int d)
{
  int err, k;
  char *new_addr = NULL;
  NML_DESC *next_descp;
  __POINT_T *desc_dims;
  __POINT_T actual_ndims;

  /*  compute number of bytes to add to reach next descriptor: */
  if (descp->ndims == -2)
    k = sizeof(NML_DESC) + (sizeof(__POINT_T) * 2);
  else if (descp->ndims >= MAX_DIM) {
    ACTUAL_NDIMS(actual_ndims);
    k = sizeof(NML_DESC) + (actual_ndims * sizeof(__POINT_T) * 2);
  } else {
#if DEBUG
    printf("Error: eval_dtio_sb \n");
    return ERR_FLAG;
#endif
  }

  /*    next_descp = (NML_DESC *)((char*) descp + k);*/
  next_descp = (NML_DESC *)((char *)descp);

  if (descp->ndims == -2) {
    desc_dims = (__POINT_T *)((char *)descp + sizeof(NML_DESC));
    if (*desc_dims == 0) {
      /* $p contains an address of array/scalar */
      new_addr = *(char **)sb.loc_addr;
      err = dtio_write_scalar(&next_descp, descp, new_addr, descp->len);
      if (err)
        return err;
      *NextDescp = next_descp;
      return 0;
    }
  }

  if (d == 0) {
    /*
     * Reached the first dimension; iterate over the first dimension,
     * compute the address of each element, and pass each address to
     * 'eval'.
     */
    F90_Desc *sd = get_descriptor(descp);
    for (sb.idx[0] = sb.sect[0].lwb; sb.idx[0] <= sb.sect[0].upb;
         sb.idx[0] += sb.sect[0].stride) {
      new_addr = I8(__fort_local_address)((*(char **)sb.loc_addr), sd,
                                         (__INT_T *)&sb.idx[0]);
      err = dtio_write_scalar(&next_descp, descp, new_addr, descp->len);
      if (err)
        return err;
    }
    *NextDescp = next_descp;
    return 0;
  }

  /* Iterate over the current dimension, and recursively evaluate all
   * subscripts in the dimensions to the left.
   */
  for (sb.idx[d] = sb.sect[d].lwb; sb.idx[d] <= sb.sect[d].upb;
       sb.idx[d] += sb.sect[d].stride) {
    err = I8(eval_dtio_sb)(&next_descp, descp, new_addr, d - 1);
    if (err)
      return err;
  }
  *NextDescp = next_descp;
  return 0;
}

static int
eval(int v, char *loc_addr, NML_DESC *descp, NML_DESC **nextdescp)
{
  char *new_addr;
  new_addr = loc_addr;
  return I8(eval_sb)(nextdescp, descp, loc_addr, sb.ndims - 1);
}

static int
eval_dtio(int v, char *loc_addr, NML_DESC *descp, NML_DESC **nextdescp)
{
  char *new_addr;
  new_addr = loc_addr;
  return I8(eval_dtio_sb)(nextdescp, descp, loc_addr, sb.ndims - 1);
}

static int
_f90io_nml_write(NML_GROUP *nmldesc) /* namelist group descriptor */
{
  int i;
  NML_DESC *descp;
  char tbuf[64]; /* buffer to convert symbol names to upper */
  int n;
  int ret_err;

  if (fioFcbTbls.error)
    return ERR_FLAG;

  /*
   * f2003 10.10.2.2 L33: Except for coninuation of delimited character
   * sequences, each output record begins with a blank character.
   */
  /* ------ write group name line:  */

  for (n = 0; n < (int)nmldesc->nlen; n++) {
    tbuf[n] = toupper(nmldesc->group[n]);
  }
  ret_err = write_item(" &", 2);
  if (ret_err)
    return __fortio_error(ret_err);
  ret_err = write_item(tbuf, nmldesc->nlen);
  if (ret_err)
    return __fortio_error(ret_err);
  ret_err = emit_eol();
  if (ret_err)
    return __fortio_error(ret_err);
  f->nextrec++;

  /* ------ cycle through namelist entities */

  /*  point to first descriptor:  */
  descp = (NML_DESC *)((char *)nmldesc + sizeof(NML_GROUP));

  for (i = 0; i < nmldesc->ndesc; i++) {
    if (i) {
      if (gbl->decimal == FIO_COMMA)
        ret_err = write_char(';');
      else
        ret_err = write_char(',');
      if (ret_err)
        return __fortio_error(ret_err);
      ret_err = emit_eol();
      if (ret_err)
        return __fortio_error(ret_err);
      f->nextrec++;
    }
    need_comma = FALSE;
    /* write entity name followed by " = " */
    for (n = 0; n < (int)descp->nlen; n++) {
      tbuf[n] = toupper(descp->sym[n]);
    }
    ret_err = write_char(' ');
    if (ret_err)
      return __fortio_error(ret_err);
    ret_err = write_item(tbuf, descp->nlen);
    if (ret_err)
      return __fortio_error(ret_err);
    ret_err = write_item(" = ", 3);
    if (ret_err)
      return __fortio_error(ret_err);
    skip = descp->nlen + 4;
    if (descp->ndims == -2) { /* has defined io */
      I8(fillup_sb)(0, descp, descp->addr);
      eval_dtio(0, descp->addr, descp, &descp);
    } else if (descp->ndims == -1) {
      I8(fillup_sb)(0, descp, descp->addr);
      eval(0, descp->addr, descp, &descp);
    } else if (descp->ndims > MAX_DIM) { /* array defined io, dims-30 */
      ret_err = dtio_write_scalar(&descp, descp, descp->addr, descp->len);
    } else if (descp->ndims == MAX_DIM) { /* scalar defined io */
      /* call used defined io */
      ret_err = dtio_write_scalar(&descp, descp, descp->addr, descp->len);
    } else {
      ret_err = write_nml_val(&descp, descp, descp->addr);
    }
    if (ret_err)
      return ret_err;
  }
  ret_err = emit_eol();
  if (ret_err)
    return __fortio_error(ret_err);
  f->nextrec++;

  /*  write "$end" line:  */
  ret_err = write_item(" /", 2);
  if (ret_err)
    return __fortio_error(ret_err);
  ret_err = emit_eol();
  if (ret_err)
    return __fortio_error(ret_err);
  /* f->nextrec++; (nextrec incremented in rwinit, so omit here) */

  return 0;
}

int
ENTF90IO(NML_WRITE, nml_write)(__INT_T *unit, /* unit number */
                               __INT_T *bitv,     /* same as for ENTF90IO(open) */
                               __INT_T *iostat,   /* same as for ENTF90IO(open) */
                               NML_GROUP *nmldesc) /* namelist group descr */
{
  int s = 0;

  internal_file = FALSE;
  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC) {
    s = _f90io_nmlw_init(unit, 0, bitv, iostat);
    if (!s)
      s = _f90io_nml_write(nmldesc);
  }
  return DIST_STATUS_BCST(s);
}

int
ENTCRF90IO(NML_WRITE, nml_write)(__INT_T *unit, /* unit number */
                                 __INT_T *bitv, /* same as for ENTF90IO(open) */
                                 __INT_T *iostat, /* same as for ENTF90IO(open) */
                                 NML_GROUP *nmldesc) /* namelist group descr */
{
  int s;

  internal_file = FALSE;
  s = _f90io_nmlw_init(unit, 0, bitv, iostat);
  if (!s)
    s = _f90io_nml_write(nmldesc);
  return s;
}

/** \brief Write a namelist group
 * \param nmldesc - namelist group descriptor
 */
int
ENTF90IO(NMLW, nmlw)(NML_GROUP *nmldesc)
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC) {
    s = _f90io_nml_write(nmldesc);
  }
  return DIST_STATUS_BCST(s);
}

int
ENTCRF90IO(NMLW, nmlw)(NML_GROUP *nmldesc) /* namelist group descriptor */
{
  int s = 0;
  s = _f90io_nml_write(nmldesc);
  return s;
}

/* -------------------------------------------------------------------- */

static int
_f90io_nmlw_end()
{
  gbl->decimal = 0;
  gbl->sign = 0;
  gbl->round = 0;
  if (!gbl->same_fcb_idx) {
    gbl->unit = 0;
    gbl->iostat = 0;
  }

  if (fioFcbTbls.error)
    return ERR_FLAG;

  return 0;
}

/** \brief Terminates a WRITE statement
 */
int
ENTF90IO(NMLW_END, nmlw_end)()
{
  int ioproc, len;
  int s = 0;

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc) {
    s = _f90io_nmlw_end();
    if (internal_file)
      len = in_recp - internal_unit;
  }
  if (!LOCAL_MODE && internal_file) {
    DIST_RBCSTL(ioproc, &len, 1, 1, __CINT, sizeof(int));
    DIST_RBCSTL(ioproc, internal_unit, 1, 1, __CHAR, len);
  }
  __fortio_errend03();
  return DIST_STATUS_BCST(s);
}

int
ENTCRF90IO(NMLW_END, nmlw_end)()
{
  int s = 0;
  s = _f90io_nmlw_end();
  __fortio_errend03();
  return s;
}

/* -------------------------------------------------------------------- */
static int
dtio_write_scalar(NML_DESC **NextDescp, NML_DESC *descp, char *loc_addr,
                  int dtvsize)
{
  static __INT_T internal_unit = -1;
  __INT_T tmp_iostat = 0;
  __INT_T *iostat;
  __INT_T *unit;
  void (*dtio)(char *, INT *, const char *, INT *, INT *, char *, F90_Desc *,
               F90_Desc *, __CLEN_T, __CLEN_T);
  char *dtv;
  F90_Desc *dtv_sd;
  F90_Desc *vlist_sd;
  INT *vlist;
  NML_DESC *next_descp;
  NML_DESC *start_descp;
  __CLEN_T iotypelen = 8;
  __CLEN_T iomsglen = 250;
  static char iomsg[250];
  int k, num_consts, ret_err, j;
  const char *iotype = "NAMELIST";
  char *start_addr;
  __POINT_T *desc_dims, new_ndims;
  __POINT_T actual_ndims;

  /* if this is array */
  num_consts = 1;
  desc_dims = (__POINT_T *)((char *)descp + sizeof(NML_DESC));
  if (descp->ndims == -1 || descp->ndims == -2) {
    new_ndims = *(__POINT_T *)((char *)descp + sizeof(__POINT_T));
    num_consts = nelems_of(descp);
  } else {
    num_consts = nelems_of(descp);
  }

  actual_ndims = 0;
  if (descp->ndims == -2) {
    k = sizeof(NML_DESC) + (sizeof(__POINT_T) * 2);
  } else if (descp->ndims == MAX_DIM) {
    ACTUAL_NDIMS(actual_ndims);
    k = sizeof(NML_DESC) + (actual_ndims * sizeof(__POINT_T) * 2);
  } else if (descp->ndims > MAX_DIM) {
    ACTUAL_NDIMS(actual_ndims);
    k = sizeof(NML_DESC) + (actual_ndims * sizeof(__POINT_T) * 2);
  } else {
#if DEBUG
    printf("ERROR unexpected ndims:%d\n", (int)descp->ndims);
#endif
    return ERR_FLAG;
  }

  /* next_descp is now at the start of the defined io arguments */
  next_descp = (NML_DESC *)((char *)descp + k);

  /* after above, next_descp is now at -98, beginning of dinit define io
   * arguments */

  if (descp->type != __DERIVED) {
#if DEBUG
    printf("ERROR unexpected dtype, expecting derived type\n");
#endif
    return ERR_FLAG;
  }

  /* move to user defined io read*/
  desc_dims = (__POINT_T *)((char *)next_descp + sizeof(__POINT_T));
  next_descp = (NML_DESC *)desc_dims;

  /* write routine */
  desc_dims = (__POINT_T *)((char *)next_descp + sizeof(__POINT_T));
  next_descp = (NML_DESC *)desc_dims;
  dtio = (void *)*(char **)((char *)desc_dims);
#if DEBUG
  if ((INT *)dtio == 0) {
    printf("ERROR: unable find user defined io write routine \n");
  }

#endif

  /* dtv */
  desc_dims = (__POINT_T *)((char *)next_descp + sizeof(__POINT_T));
  next_descp = (NML_DESC *)desc_dims;
  dtv = (char *)*(char **)((char *)desc_dims);
  start_addr = (char *)dtv;

  /* dtv$sd */
  desc_dims = (__POINT_T *)((char *)next_descp + sizeof(__POINT_T));
  next_descp = (NML_DESC *)desc_dims;
  dtv_sd = (F90_Desc *)*(char **)((char *)desc_dims);

  /* vlist */
  desc_dims = (__POINT_T *)((char *)next_descp + sizeof(__POINT_T));
  next_descp = (NML_DESC *)desc_dims;
  vlist = (INT *)*(char **)((char *)desc_dims);

  /* vlist$sd */
  desc_dims = (__POINT_T *)((char *)next_descp + sizeof(__POINT_T));
  next_descp = (NML_DESC *)desc_dims;
  vlist_sd = (F90_Desc *)*(char **)((char *)desc_dims);

  /* move to next descriptor */
  desc_dims = (__POINT_T *)((char *)next_descp + sizeof(__POINT_T));
  next_descp = (NML_DESC *)desc_dims;

  start_descp = next_descp;
  start_addr = loc_addr;
  if (gbl->unit)
    unit = gbl->unit;
  else
    unit = &internal_unit;

  if (gbl->iostat)
    iostat = gbl->iostat;
  else
    iostat = &tmp_iostat;

  for (k = 0; k < num_consts; k++) {
    if (need_comma) { /*  put out leading blanks:  */
      /*  put commas after each constant except for very, very last: */
      if (gbl->decimal == FIO_COMMA) {
        ret_err = write_char(';');
      } else {
        ret_err = write_char(',');
      }
      if (ret_err)
        return __fortio_error(ret_err);
      ret_err = emit_eol();
      if (ret_err)
        return __fortio_error(ret_err);
      f->nextrec++;
      for (j = 0; j < skip; j++) {
        ret_err = write_char(' ');
        if (ret_err)
          return __fortio_error(ret_err);
      }
    }

    (*dtio)(start_addr, unit, iotype, vlist, iostat, iomsg, dtv_sd, vlist_sd,
            iotypelen, iomsglen);
    if (*iostat != 0)
      return *iostat;
    start_addr = start_addr + descp->len;
    need_comma = TRUE;
  }
  *NextDescp = next_descp;
  return 0;
}
