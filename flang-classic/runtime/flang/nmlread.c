/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Implement namelist read statement.
 */

#include "global.h"
#include "format.h"
#include "nml.h"
#include <string.h>

/* define a few things for run-time tracing */
static int dbgflag;
#undef DBGBIT
#define DBGBIT(v) (LOCAL_DEBUG && (dbgflag & v))

#define MAX_TOKEN_LEN 300

#define TK_IDENT 1
#define TK_COMMA 2
#define TK_EQUALS 3
#define TK_CONST 4
#define TK_ENDGROUP 5
#define TK_SKIP 6
#define TK_LPAREN 7
#define TK_RPAREN 8
#define TK_COLON 9
#define TK_PERCENT 10
#define TK_SEMICOLON 11

#define IS_SPACE(c) ((c) == ' ' || (c) == '\n' || (c) == '\t' || (c) == '\r')

#define VRF_ID 0
#define VRF_ELEMENT 1
#define VRF_SECTION 2
#define VRF_MEMBER 3

static TRI tri;

/* Record the presence of a substring in a reference */
static struct {
  bool present;
  __BIGINT_T start;
  __BIGINT_T end;
} substring;

#define TRI_SECT(i, j) tri.base[(i) + (j)]
#define TRI_LWB(i, j) TRI_SECT(i, j).lwb
#define TRI_UPB(i, j) TRI_SECT(i, j).upb
#define TRI_STRIDE(i, j) TRI_SECT(i, j).stride

typedef struct {
  int type;      /* VRF_... */
  int subscript; /* nonzero, locates a set of TRIPLES */
  NML_DESC *descp;
  char *addr;
} VRF;

static struct {
  int size;
  int avl;
  VRF *base;
} vrf;

static int vrf_cur;

#define VRF_TYPE(i) vrf.base[i].type
#define VRF_SUBSCRIPT(i) vrf.base[i].subscript
#define VRF_DESCP(i) vrf.base[i].descp
#define VRF_ADDR(i) vrf.base[i].addr

static FIO_FCB *f;
static bool accessed; /* file has been read */
static unsigned int byte_cnt;  /* number of bytes read */
static int n_irecs;   /* number of internal file records */
static bool internal_file;
static int rec_len;
static int token;
static char token_buff[MAX_TOKEN_LEN + 1];
static int live_token;
static AVAL constval;
static AVAL cmplxval[2];
static bool lparen_is_token;
static bool comma_is_token;
static FILE *gblfp;

#define RBUF_SIZE 256
static char rbuf[RBUF_SIZE + 1];
static unsigned rbuf_size = RBUF_SIZE;

static char *rbufp = rbuf; /* ptr to read buffer */
static char *currc;        /* current pointer in buffer */

static char *in_recp; /* internal i/o record (user's space) */

typedef struct {
  short blank_zero; /* FIO_ ZERO or NULL */
  short pad;        /* FIO_ YES or NULL */
  short decimal;    /* COMMA, POINT, NONE */
  short round;      /* FIO_ UP, DOWN, ZERO, NEAREST, COMPATIBLE,
                     *      PROCESSOR_DEFINED, NONE
                     */
  int same_fcb_idx;
  FIO_FCB *gblfp;
  FIO_FCB *f;
  char *currc;
  char *rbufp;
  char rbuf[RBUF_SIZE + 1];
  char *in_recp;
  bool comman_is_token;
  bool lparen_is_token;
  int live_token;
  INT tokenval;
  char token_buff[MAX_TOKEN_LEN + 1];
  int rec_len;
  bool internal_file;
  int token;
  int n_irecs;
  int byte_cnt;
  bool accessed;
  int vrf_cur;
  __INT_T *unit;
  __INT_T *iostat;

} G;

static G static_gbl[GBL_SIZE];
static G *gbl = &static_gbl[0];

static void shared_init(void);
static NML_DESC *skip_to_next(NML_DESC *);
static NML_DESC *skip_dtio_datainit(NML_DESC *);
static int find_group(char *, int);
static int get_token(void);
static int do_parse(NML_GROUP *);
static int parse_ref(NML_DESC *);
static int add_vrf(int, NML_DESC *);
static int I8(parse_subscripts)(NML_DESC *);
static int I8(parse_substring)(NML_DESC *);
static int add_triple(int);
static int assign_values(void);
static int eval_ptr(int, char *);
static void I8(fillup_sb)(int, NML_DESC *, char *);
static int dtio_read_scalar(NML_DESC *, char *);

static bool comma_live;
static int eval(int, char *);
static int I8(eval_dtio_sb)(int d);
static int assign(NML_DESC *, char *, char **, bool, bool);
static int dtio_assign(NML_DESC *, char *, char **, bool, bool);

#undef GET_TOKEN
#define GET_TOKEN(i)       \
  if ((i = get_token()))   \
  return i

#undef NML_ERROR
#define NML_ERROR(e) (__fortio_error(e))

static int read_record(void);
static char *alloc_rbuf(int, bool);
static SB sb;

/* ------------------------------------------------------------------- */

/** \param unit unit number
 *  \param rec record number for direct access I/O; rec not used, but 
 *         JUST IN CASE
 *  \param bitv same as for ENTF90IO(open_)
 *  \param iostat same as for ENTF90IO(open_)
 */
static int
_f90io_nmlr_init(__INT_T *unit,
                __INT_T *rec, 
                __INT_T *bitv,   
                __INT_T *iostat) 
{
  __fortio_errinit03(*unit, *bitv, iostat, "namelist read");

  /* -------  perform error checking and initialization of unit:  */

  f = __fortio_rwinit(*unit, FIO_FORMATTED, rec, 0 /*read*/);
  if (f == NULL) {
    if (fioFcbTbls.eof)
      return EOF_FLAG;
    /* TBD - does there need to be fioFcbTbls.eor */
    return ERR_FLAG;
  }

  f->skip = 0;
  gblfp = f->fp;
  internal_file = FALSE;
  gbl->decimal = f->decimal;
  gbl->unit = unit;
  gbl->iostat = iostat;

  shared_init();
  return 0;
}

static void
shared_init(void)
{
  accessed = FALSE;
  byte_cnt = 0;
}

/** \brief
 * Initialize for namelist read to an external file
 *
 * \param unit - unit number
 * \param rec - record number for direct access I/O; not used, but JUST IN CASE
 * \param bitv - same as for ENTF90IO(open_)
 * \param iostat - same as for ENTF90IO(open_)
 */
int
ENTF90IO(NMLR_INIT, nmlr_init)(__INT_T *unit, __INT_T *rec, __INT_T *bitv,
                               __INT_T *iostat)
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_nmlr_init(unit, rec, bitv, iostat);
  return DIST_STATUS_BCST(s);
}

int
ENTF90IO(NMLR_INIT03A, nmlr_init03a)(__INT_T *istat, DCHAR(blank),
                                   DCHAR(decimal), DCHAR(pad),
                                   DCHAR(round) DCLEN64(blank) DCLEN64(decimal)
                                       DCLEN64(pad) DCLEN64(round))
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
int
ENTF90IO(NMLR_INIT03, nmlr_init03)(__INT_T *istat, DCHAR(blank),
                                   DCHAR(decimal), DCHAR(pad),
                                   DCHAR(round) DCLEN(blank) DCLEN(decimal)
                                       DCLEN(pad) DCLEN(round))
{
  return ENTF90IO(NMLR_INIT03A, nmlr_init03a)(istat, CADR(blank), CADR(decimal),
                               CADR(pad), CADR(round), (__CLEN_T)CLEN(blank),
			       (__CLEN_T)CLEN(decimal), (__CLEN_T)CLEN(pad),
			       (__CLEN_T)CLEN(round));
}

/** \brief
 *
 * \param unit unit number
 * \param rec record number for direct access I/O
 * \param bitv same as for ENTF90IO(open_)
 * \param iostat same as for ENTF90IO(open_)
 */
int
ENTCRF90IO(NMLR_INIT, nmlr_init)(
           __INT_T *unit,   
           __INT_T *rec,    
           __INT_T *bitv,  
           __INT_T *iostat) 
{
  return _f90io_nmlr_init(unit, rec, bitv, iostat);
}

/** \brief
 * 
 * \param cunit ptr to var or array to read from
 * \param rec_num number of records in internal file. 0 if the file is an
 *        assumed size character array
 * \param bitv same as for ENTF90IO(open_)
 * \param iostat same as for ENTF90IO(open_)
 * \param cunit_siz size of \p cunit
 */
int
I8(_f90io_nmlr_intern_init)( char *cunit, 
                            __INT_T *rec_num, 
                             __INT_T *bitv,    
                             __INT_T *iostat,  
                             __CLEN_T cunit_siz)
{
  static FIO_FCB dumfcb;

  __fortio_errinit03(-99, *bitv, iostat, "namelist read");

  f = &dumfcb; /* so the f-> refs don't have to be guarded */
  internal_file = TRUE;
  in_recp = cunit;
  n_irecs = *rec_num;
  rec_len = cunit_siz;

  shared_init();
  return 0;
}

/** \brief Internal file namelist read initialization
 * 
 * \param cunit is a pointer to variable or array to read from
 * \param rec_num - number of records in internal file; 0 if the file
 *   is an assumed size character array
 * \param bitv - same as for ENTF90IO(open_)
 * \param iostat - same as for ENTF90IO(open_)
 */
int
ENTF90IO(LDR_INTERN_INITA, nmlr_intern_inita)(
         DCHAR(cunit),     
         __INT_T *rec_num,
         __INT_T *bitv,
         __INT_T *iostat
         DCLEN64(cunit))
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = I8(_f90io_nmlr_intern_init)(CADR(cunit), rec_num, bitv, iostat,
                                   CLEN(cunit));
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
int
ENTF90IO(LDR_INTERN_INIT, nmlr_intern_init)(
         DCHAR(cunit),     
         __INT_T *rec_num,
         __INT_T *bitv,
         __INT_T *iostat
         DCLEN(cunit))
{
  return ENTF90IO(LDR_INTERN_INITA, nmlr_intern_inita)(CADR(cunit), rec_num,
                                   bitv, iostat, (__CLEN_T)CLEN(cunit));
}

/** \param rec_num number of records in internal file. 0 if the file is an 
 *         assumed size character array
 * \param bitv same as for ENTF90IO(open_)
 * \param iostat same as for ENTF90IO(open_)
 */
int
ENTCRF90IO(LDR_INTERN_INITA, nmlr_intern_inita)(
           DCHAR(cunit),     
           __INT_T *rec_num, 
           __INT_T *bitv,   
           __INT_T *iostat   
           DCLEN64(cunit))
{
  return I8(_f90io_nmlr_intern_init)(CADR(cunit), rec_num, bitv, iostat,
                                    CLEN(cunit));
}
/* 32 bit CLEN version */
int
ENTCRF90IO(LDR_INTERN_INIT, nmlr_intern_init)(
           DCHAR(cunit),     
           __INT_T *rec_num, 
           __INT_T *bitv,   
           __INT_T *iostat   
           DCLEN(cunit))
{
  return ENTCRF90IO(LDR_INTERN_INITA, nmlr_intern_inita)(CADR(cunit), rec_num,
                                      bitv, iostat, (__CLEN_T)CLEN(cunit));
}

/** \param nmldesc namelist group descriptor */
static int
_f90io_nml_read(NML_GROUP *nmldesc) 
{
  int err;

  /* first check for errors: */
  if (fioFcbTbls.eof)
    return EOF_FLAG;
  if (fioFcbTbls.error)
    return ERR_FLAG;

  assert(nmldesc);

  err = find_group(nmldesc->group, nmldesc->nlen);
  if (err != 0)
    return err; /*  error or eof condition  */

  /* -------- file is now positioned immediately after group name:  */

  live_token = 0;
  vrf.size = 32;
  vrf.base = (VRF *)malloc(sizeof(VRF) * vrf.size);
  tri.size = 32;
  tri.base = (TRIPLE *)malloc(sizeof(TRIPLE) * tri.size);

  /* at this point we should call a routine that will fill
   * the lower/upper/stride for array for array pointer/allocatable
   * nmldesc should contain enough information
   */

  while (TRUE) { /*  loop once for each namelist group item */
    err = do_parse(nmldesc);
    if (err == -1) /* end of group token encountered */
      break;
    if (err != 0)
      goto return_err; /*  error or end of file  */
    err = assign_values();
    if (err != 0)
      goto return_err; /*  error or end of file  */
  }
  err = 0;
return_err:
  free(vrf.base);
  free(tri.base);
  return err;
}

/** \brief transfer data to other processes 
  * \param nmldesc namelist group descriptor 
  *
  */
static void
xfer(NML_GROUP *nmldesc)
{
  int num_consts;
  int i;
  NML_DESC *descp;
  int pn;

  /* ------ cycle through namelist entities */

  /*  point to first descriptor:  */
  descp = (NML_DESC *)((char *)nmldesc + sizeof(NML_GROUP));

  i = 0;
  while (TRUE) {

    /* count up number of constants ( == 1 unless array):  */
    num_consts = nelems_of(descp);

    /* transfer data */

    pn = (descp->len > 0 ? descp->len * num_consts : num_consts);
    if (!LOCAL_MODE) {
      if (descp->type != __DERIVED)
        DIST_RBCST(GET_DIST_IOPROC, descp->addr, pn, 1, descp->type);
      else
        DIST_RBCST(GET_DIST_IOPROC, descp->addr, pn, 1, __STR);
    }

    i++;
    if (i >= nmldesc->ndesc)
      break;

    descp = skip_to_next(descp);
  }
}

static NML_DESC *
skip_to_next(NML_DESC *descp)
{
  NML_DESC *next_descp;
  int k;
  __POINT_T actual_ndims;

  /*  compute number of bytes to add to reach next descriptor: */
  ACTUAL_NDIMS(actual_ndims);
  if (actual_ndims >= 0)
    k = sizeof(NML_DESC) + (actual_ndims * sizeof(__POINT_T) * 2);
  else
    k = sizeof(NML_DESC) + (sizeof(__POINT_T) * 2);
  next_descp = (NML_DESC *)((char *)descp + k);

  if (descp->ndims == -2 || descp->ndims >= MAX_DIM) {
    return skip_dtio_datainit(next_descp);
  } else if (descp->type == __DERIVED) {
    int level = 0;
    /* skip over all members and the members of any contained
     * derived types.
     */
    while (TRUE) {
      if (next_descp->nlen) {
        if (next_descp->type == __DERIVED)
          level++;
      } else {
        next_descp = (NML_DESC *)((char *)next_descp + sizeof(__POINT_T));
        if (level <= 0)
          break;
        level--;
        continue;
      }
      actual_ndims = next_descp->ndims >= MAX_DIM ? next_descp->ndims - 30
                                                  : next_descp->ndims;
      if (actual_ndims >= 0) {
        k = sizeof(NML_DESC) + (actual_ndims * sizeof(__POINT_T) * 2);
      } else {
        k = sizeof(NML_DESC) + (sizeof(__POINT_T) * 2);
      }
      next_descp = (NML_DESC *)((char *)next_descp + k);
    }
  }
  return next_descp;
}

/** \brief 
 *
 *  \param unit unit number
 *  \param bitv same as for ENTF90IO(open)
 *  \param iostat same as for ENTF90IO(open)
 *  \param nmldesc namelist group descr
 */
int
ENTF90IO(NML_READ, nml_read)( __INT_T *unit, 
                              __INT_T *bitv,      
                              __INT_T *iostat,    
                              NML_GROUP *nmldesc)
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC) {
    s = _f90io_nmlr_init(unit, 0, bitv, iostat);
    if (!s)
      s = _f90io_nml_read(nmldesc);
  }
  xfer(nmldesc);
  return DIST_STATUS_BCST(s);
}

/** \param unit unit number
 *  \param bitv same as for ENTF90IO(open)
 *  \param iostat same as for ENTF90IO(open)
 *  \param nmldesc) namelist group descr
 */
int
ENTCRF90IO(NML_READ, nml_read)(__INT_T *unit, 
                               __INT_T *bitv, 
                               __INT_T *iostat, 
                               NML_GROUP *nmldesc) 
{
  int s;

  s = _f90io_nmlr_init(unit, 0, bitv, iostat);
  if (!s)
    s = _f90io_nml_read(nmldesc);
  return s;
}

/** \brief read a namelist group
 *
 * \param nmldesc - namelist group descriptor
 */
int
ENTF90IO(NMLR, nmlr)(NML_GROUP *nmldesc) 
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC) {
    s = _f90io_nml_read(nmldesc);
  }
  xfer(nmldesc);
  return DIST_STATUS_BCST(s);
}

/** \brief
  *
  * \param  nmlr namelist group descriptor 
  */
int
ENTCRF90IO(NMLR, nmlr)(NML_GROUP *nmldesc) 
{
  int s;

  s = _f90io_nml_read(nmldesc);
  return s;
}

/* ----------------------------------------------------------------------- */

/** \brief  search file for line which begins with '&<groupname>'  */
static int
find_group(char *str, int nlen)
{
  int c;
  int i;
  int ret_err;

  while (TRUE) {
    ret_err = read_record();
    if (ret_err) {
      if (ret_err == FIO_EEOF)
        return __fortio_eoferr(FIO_ENOGROUP);
      return __fortio_error(ret_err);
    }
    while ((c = *currc++) == ' ')
      ;
    if (c != '$' && c != '&')
      continue; /* eat record */

    for (i = 0; i < nlen; i++) { /*  compare letters of group name  */
      c = *currc++;
      if (c >= 'A' && c <= 'Z')
        c = c + ('a' - 'A');
      if (str[i] != c)
        goto eat_record;
    }
    c = *currc++;
    if (IS_SPACE(c)) { /* group name matched */
      currc--;
      break;
    }
  eat_record:;
  }
  return 0;
}

/* ----------------------------------------------------------------- */

static int
get_token(void)
{
  static int recur = 0;
  int i, c;
  char delim;
  int ret_err;

  if (live_token) { /* token exists from previous call to get_token */
    assert(live_token > 0);
    live_token--;
    return 0;
  }

/*  scan past white space:  */
  while (TRUE) {
    c = *currc++;
    if (c == '\n') {
      ret_err = read_record();
      if (ret_err) {
        if (ret_err == FIO_EEOF)
          return __fortio_eoferr(FIO_ENMLEOF);
        return __fortio_error(ret_err);
      }
      continue;
    }
    if (IS_SPACE(c))
      continue;
    if (c == '!') {
      /* comment; skip to end of line:*/
      ret_err = read_record();
      if (ret_err) {
        if (ret_err == FIO_EEOF)
          return __fortio_eoferr(FIO_ENMLEOF);
        return __fortio_error(ret_err);
      }
      continue;
    }
    break;
  }

  /*  switch based on first character of token:  */

  switch (c) {
  case 'A':
  case 'B':
  case 'C':
  case 'D':
  case 'E':
  case 'F':
  case 'G':
  case 'H':
  case 'I':
  case 'J':
  case 'K':
  case 'L':
  case 'M':
  case 'N':
  case 'O':
  case 'P':
  case 'Q':
  case 'R':
  case 'S':
  case 'T':
  case 'U':
  case 'V':
  case 'W':
  case 'X':
  case 'Y':
  case 'Z':
    c = c + ('a' - 'A');
    FLANG_FALLTHROUGH;
  case 'a':
  case 'b':
  case 'c':
  case 'd':
  case 'e':
  case 'f':
  case 'g':
  case 'h':
  case 'i':
  case 'j':
  case 'k':
  case 'l':
  case 'm':
  case 'n':
  case 'o':
  case 'p':
  case 'q':
  case 'r':
  case 's':
  case 't':
  case 'u':
  case 'v':
  case 'w':
  case 'x':
  case 'y':
  case 'z':
  case '_':
    for (i = 0; i < MAX_TOKEN_LEN;) { /*  copy ident into buffer */
      token_buff[i] = c;
      c = *currc++;
      if (c >= 'A' && c <= 'Z') /*  convert to lower case  */
        c = c + ('a' - 'A');
      i++;
      if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' ||
            c == '_' || c == '$'))
        break;
    }
    token_buff[i] = '\0';

    if (token_buff[0] == 't' || token_buff[0] == 'f') {
      /*  determine if this is a logical constant or an identifier: */
      while (TRUE) {
        if (c == '=' || c == '(' || c == '%')
          goto return_ident;
        if (!IS_SPACE(c))
          break;
        if (c == '\n') {
          ret_err = read_record();
          if (ret_err) {
            if (ret_err == FIO_EEOF)
              return __fortio_eoferr(FIO_ENMLEOF);
            return __fortio_error(ret_err);
          }
        }
        c = *currc++;
      }

      /*  token is logical constant:  */
      constval.dtype = __BIGLOG;
      constval.val.i = FTN_FALSE;
      if (token_buff[0] == 't')
        constval.val.i = FTN_TRUE;
      currc--;
      token = TK_CONST;
      break;
    }

  return_ident:
    currc--;
    token = TK_IDENT;
    break;

  case '\'':
  case '\"':
    constval.dtype = __STR;
    token = TK_CONST;                 /*  string constant */
    delim = c;                        /*  delim matches first */
    for (i = 0; i < MAX_TOKEN_LEN;) { /*  copy string into buffer */
      c = *currc++;
      if (c == delim) {
        c = *currc++;
        if (c != delim) {
          currc--; /* put back char following string */
          break;   /* exit loop */
        }
      } else {
        if (c == '\r' && EOR_CRLF) {
          c = *currc++;
          if (c != '\n') {
            currc--;
            c = '\r';
          }
        }
        if (c == '\n') {
          /*
           * do not discard 1st character of a rew record;
           * WAS (void) __io_fgetc(fp)
           */
          ret_err = read_record();
          if (ret_err) {
            if (ret_err == FIO_EEOF)
              return __fortio_eoferr(FIO_ENMLEOF);
            return __fortio_error(ret_err);
          }
          continue; /* ignore end of line char */
        }
      }
      token_buff[i++] = c;
    }
    token_buff[i + 1] = '\0';
    constval.val.c.len = i;
    constval.val.c.str = token_buff;
    break;

  case '(':
    if (lparen_is_token) {
      token = TK_LPAREN;
      break;
    }

    /*  else return a complex constant.  Call get_token recursively to
        process constant -which must be of form: ( TK_CONST , TK_CONST ) */

    if (recur > 1) /* error if get_token is being called recursively */
      return __fortio_error(FIO_ESYNTAX);
    recur = 2;

    GET_TOKEN(i);
    if (token != TK_CONST || constval.dtype == __STR ||
        constval.dtype == __NCHAR)
      return __fortio_error(FIO_ESYNTAX);
    cmplxval[0] = constval;

    GET_TOKEN(i);
    if (gbl->decimal == FIO_COMMA) {
      if (token != TK_SEMICOLON)
        return __fortio_error(FIO_ESYNTAX);
    } else {
      if (token != TK_COMMA)
        return __fortio_error(FIO_ESYNTAX);
    }
    GET_TOKEN(i);
    if (token != TK_CONST || constval.dtype == __STR ||
        constval.dtype == __NCHAR)
      return __fortio_error(FIO_ESYNTAX);
    cmplxval[1] = constval;

    GET_TOKEN(i);
    if (token != TK_RPAREN)
      return __fortio_error(FIO_ESYNTAX);

    recur = 0;
    token = TK_CONST;
    constval.dtype = __BIGCPLX;
    constval.val.cmplx = cmplxval;
    break;

  case ')':
    token = TK_RPAREN;
    break;

  case ',':
    if (comma_is_token) {
      token = TK_COMMA;
      break;
    }
    if (gbl->decimal != FIO_COMMA) {
      token = TK_COMMA;
      break;
    }
    c = *currc++;
    if (c == 't' || c == 'T' || c == 'f' || c == 'F') {
      token = TK_CONST;
      constval.dtype = __BIGLOG;
      constval.val.i = FTN_FALSE;
      if (c == 't' || c == 'T')
        constval.val.i = FTN_TRUE;
      /*  read and discard remaining alphabetic characters in token: */
      do {
        c = *currc++;
      } while ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == ',' ||
               c == '_' || c == '$');
      currc--;
      break;
    }

    /*  else, treat this ',' as beginning of numeric constant:  */
    currc--;
    c = ',';

    i = 0;
    goto do_numeric_token;

  case ':':
    token = TK_COLON;
    break;
  case ';':
    if (gbl->decimal == FIO_COMMA)
      token = TK_SEMICOLON;
    break;
  case '%':
    token = TK_PERCENT;
    break;

  case '=':
    token = TK_EQUALS;
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
    token_buff[0] = c;
    for (i = 1; i < MAX_TOKEN_LEN; i++) {
      c = *currc++;
      ;
      if (c < '0' || c > '9')
        break;
      token_buff[i] = c;
    }
    token_buff[i] = '\0';

    if (c != '*') /* no repeat count present */
      goto do_numeric_token;

    if (c == '*') { /* REPEAT COUNT */
      long k = atol(token_buff);
      if (recur)
        return __fortio_error(FIO_ELEX); /* unknown token */
      assert(live_token == 0);

      /*  check character after '*' for blank, comma or e.o.g.:  */
      c = *currc++;
      currc--;
      if (c == ',' || IS_SPACE(c) || c == '$' || c == '&') {
        token = TK_SKIP;
        live_token = k - 1;
      } else {
        recur = 1;
        GET_TOKEN(i);
        recur = 0;
        if (token != TK_CONST)
          return __fortio_error(FIO_ESYNTAX); /* syntax error */
        live_token = k - 1;
      }
    }
    break;
  case '.':
    c = *currc++;
    if (c == 't' || c == 'T' || c == 'f' || c == 'F') {
      token = TK_CONST;
      constval.dtype = __BIGLOG;
      constval.val.i = FTN_FALSE;
      if (c == 't' || c == 'T')
        constval.val.i = FTN_TRUE;
      /*  read and discard remaining alphabetic characters in token: */
      do {
        c = *currc++;
      } while ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '.' ||
               c == '_' || c == '$');
      currc--;
      break;
    }

    /*  else, treat this '.' as beginning of numeric constant:  */
    currc--;
    c = '.';
    FLANG_FALLTHROUGH;
  case '+':
  case '-':
    i = 0;
  do_numeric_token:
    /*  scan for as long as we see characters which may be part of
        a numeric constant:  */
    for (; i < MAX_TOKEN_LEN; i++) { /*  copy number into buffer */
      char decimal = '.';
      if (gbl->decimal == FIO_COMMA)
        decimal = ',';
      if (!((c >= '0' && c <= '9') || c == decimal || c == 'e' || c == 'E' ||
            c == 'd' || c == 'D' || c == 'q' || c == 'Q' || c == '-' || c == '+'))
        break;
      token_buff[i] = c;
      c = *currc++;
    }
    token_buff[i] = '\0';
    currc--;
    {
      int type; /* 0 - integer    1 - __BIGREAL */
      union {
        __BIGINT_T i;
        __BIGREAL_T d;
        __INT8_T i8v;
      } val;
      int len, errcode;

      if (gbl->decimal == FIO_COMMA)
        errcode = __fortio_getnum(token_buff, &type, &val, &len, TRUE);
      else
        errcode = __fortio_getnum(token_buff, &type, &val, &len, FALSE);
      if (errcode != 0)
        return __fortio_error(errcode);
      if (len != i)                     /*  token not entirely used up  */
        return __fortio_error(FIO_ELEX); /* unknown token */
      token = TK_CONST;
      if (type == 0) {
        constval.dtype = __BIGINT;
        constval.val.i = val.i;
      }
      else if (type == 2) {
        constval.dtype = __INT8;
        constval.val.i8v = val.i8v;
      }
      else if (type == 3) {
        if (!REAL_ALLOWED(VRF_DESCP(vrf_cur)->type)) {
          return __fortio_error(FIO_EERR_DATA_CONVERSION);
        } else {
          constval.dtype = __BIGINT;
          constval.val.i = val.i;
        }
      } else {
        constval.dtype = __BIGREAL;
        constval.val.d = val.d;
      }
    }
    break;

  case '/': /* f90 */
  case '$': /* f77 */
  case '&': /* extension */
    token = TK_ENDGROUP;
    break;

  default:                          /* no possible legal token:  */
    return __fortio_error(FIO_ELEX); /* unknown token */
  }

  return 0;
}

/* ----------------------------------------------------------------- */

static int
do_parse(NML_GROUP *nmldesc)
{
  int i;
  int err;
  NML_DESC *descp;

  /* The syntax for a reference is:
   *  <ref>     ::= <id> |
   *                <ref> % <id>
   *                <ref> ( <ss list> )
   *  <ss list> ::= <ss list> , <ss> |
   *                <ss>
   *  <ss>      ::= <ct> |
   *                <opt ct> : <opt ct> <opt stride>
   *  <opt ct>  ::= |
   *                <ct>
   *  <opt stride> := |
   *                  : <ct>
   */

  /* Begin by retrieving an identifier */
  GET_TOKEN(i);
  if (token != TK_IDENT) {
    if (token == TK_ENDGROUP)
      return -1;
    return __fortio_error(FIO_ENONAME); /* syntax error */
  }

  /* find the matching namelist item descriptors */

  /*  point to the first item descriptor:  */
  descp = (NML_DESC *)((char *)nmldesc + sizeof(NML_GROUP));

  i = 0;
  while (TRUE) {
    if ((long)strlen(token_buff) == descp->nlen &&
        strncmp(descp->sym, token_buff, (int)descp->nlen) == 0)
      break;

    i++;
    if (i >= nmldesc->ndesc)
      break;

    descp = skip_to_next(descp);
  }
  if (i == nmldesc->ndesc) /* match not found */
    return NML_ERROR(FIO_ENOTMEM);

  /* Setup for the main parsing loop: */
  vrf.avl = 0;
  tri.avl = 1;
  lparen_is_token = TRUE; /* want get_token to recognize '(' as a token */
  substring.present = FALSE;
  vrf_cur = add_vrf(VRF_ID, descp);

  if (descp->ndims == -1 || descp->ndims == -2)
    I8(fillup_sb)(vrf_cur, descp, descp->addr);

  err = parse_ref(descp);

  /* set pread */
  if (descp->ndims == -2 || descp->ndims >= 30) {
    if (!internal_file)
      f->pread = currc;
  }

  /* Cleanup after the main parsing loop */
  lparen_is_token = FALSE;

  return err;
}

static NML_DESC *
skip_dtio_datainit(NML_DESC *descp)
{
  __POINT_T *dtio_desc;
  NML_DESC *next_descp;

  /*read*/
  dtio_desc = (__POINT_T *)((char *)descp + sizeof(__POINT_T));
  /*write*/
  dtio_desc = (__POINT_T *)((char *)dtio_desc + sizeof(__POINT_T));
  /*dtv*/
  dtio_desc = (__POINT_T *)((char *)dtio_desc + sizeof(__POINT_T));
  /*dtv$sd*/
  dtio_desc = (__POINT_T *)((char *)dtio_desc + sizeof(__POINT_T));
  /*past vlist*/
  dtio_desc = (__POINT_T *)((char *)dtio_desc + sizeof(__POINT_T));
  /*past vlist$sd*/
  dtio_desc = (__POINT_T *)((char *)dtio_desc + sizeof(__POINT_T));
  /*next descriptor*/
  dtio_desc = (__POINT_T *)((char *)dtio_desc + sizeof(__POINT_T));

  next_descp = (NML_DESC *)dtio_desc;
  return next_descp;
}

static int
parse_ref(NML_DESC *gdescp)
{
  int i, k;
  NML_DESC *descp, *next_descp;
  __POINT_T new_ndims;
  __POINT_T actual_ndims;

  descp = gdescp;

/*  Enter a parsing loop searching for subobject designators */

ref_loop:
  GET_TOKEN(i);
  ACTUAL_NDIMS(actual_ndims);
  switch (token) {
  case TK_EQUALS:
    return 0;

  case TK_PERCENT:
    GET_TOKEN(i);
    if (token != TK_IDENT)
      return NML_ERROR(FIO_ESYNTAX);
    if (descp->type != __DERIVED)
      return NML_ERROR(FIO_ESYNTAX);

    /* -- scan item descriptors to find one with name that matches:  */
    /*  compute number of bytes to add to reach next descriptor: */
    if (actual_ndims >= 0)
      k = sizeof(NML_DESC) + (actual_ndims * sizeof(__POINT_T) * 2);
    else
      k = sizeof(NML_DESC) + (sizeof(__POINT_T) * 2);
    next_descp = (NML_DESC *)((char *)descp + k);
    if (descp->ndims >= 30) {
      /* need to skip dtio data init section */
      next_descp = skip_dtio_datainit(descp);
    }
    while (TRUE) {
      if (next_descp->nlen == 0) {
        return NML_ERROR(FIO_ESYNTAX);
      }
      if ((long)strlen(token_buff) == next_descp->nlen &&
          strncmp(next_descp->sym, token_buff, next_descp->nlen) == 0)
        break;
      next_descp = skip_to_next(next_descp);
    }
    descp = next_descp;
    vrf_cur = add_vrf(VRF_MEMBER, descp);
    break;

  case TK_LPAREN:
    new_ndims = *(__POINT_T *)((char *)descp + sizeof(NML_DESC));
    if (actual_ndims > 0 &&
        (VRF_TYPE(vrf_cur) == VRF_ID || VRF_TYPE(vrf_cur) == VRF_MEMBER)) {
      i = I8(parse_subscripts)(descp);
      if (i)
        return i;
    } else if ((descp->ndims == -1 || descp->ndims == -2) && new_ndims > 0) {
      i = I8(parse_subscripts)(descp);
      if (i)
        return i;

    } else if (descp->type == __STR) {
      i = I8(parse_substring)(descp);
      if (i)
        return i;
    }
    else
      return NML_ERROR(FIO_ESYNTAX);
    break;
  default:
    return NML_ERROR(FIO_ESYNTAX);
  }
  goto ref_loop;
}

static int
add_vrf(int type, NML_DESC *descp)
{
  int i;
  i = vrf.avl++;
  VRF_TYPE(i) = type;
  VRF_DESCP(i) = descp;
  VRF_SUBSCRIPT(i) = 0;
  VRF_ADDR(i) = descp->addr;
  return i;
}

static int
I8(parse_subscripts)(NML_DESC *descp)
{
  int i, k, v;
  __POINT_T *desc_dims; /* base pointer to 2-dim descriptor array */
  __BIGINT_T val, upb, stride;
  bool is_section;
  __POINT_T new_ndims = 0;
  __POINT_T actual_ndims = 0;
  F90_Desc *sd;
  DECL_DIM_PTRS(acd);

  ACTUAL_NDIMS(actual_ndims);
  desc_dims = (__POINT_T *)((char *)descp + sizeof(NML_DESC));
  k = -1;
  is_section = FALSE;
  new_ndims = actual_ndims;
  if (descp->ndims == -1 || descp->ndims == -2) {
    new_ndims = *(__POINT_T *)((char *)descp + sizeof(NML_DESC));
    sd = get_descriptor(descp);
  }
  v = add_triple(new_ndims);

  do {
    k++;
    if (k >= new_ndims)
      goto subscript_error; /* too many subscripts */
                            /* examine first token for the current dimension */
    comma_is_token = TRUE;
    GET_TOKEN(i);
    if (descp->ndims == -1 || descp->ndims == -2)
      SET_DIM_PTRS(acd, sd, k);
    if (token == TK_CONST && constval.dtype == __BIGINT) {
      val = constval.val.i;

      if (descp->ndims == -1 || descp->ndims == -2) {
        if (val < F90_DPTR_LBOUND_G(acd) || val > F90_DPTR_EXTENT_G(acd))
          goto subscript_error; /* subscr out of range */
      } else if (val < desc_dims[2 * k] || val > desc_dims[2 * k + 1]) {
        goto subscript_error; /* subscr out of range */
      }

      /* Is this constant a subscipt or the lower bound of a section? */

      GET_TOKEN(i);
      if (token == TK_COMMA || token == TK_RPAREN) {
        /* the constant is a subscript */
        TRI_LWB(v, k) = val;
        TRI_UPB(v, k) = val;
        TRI_STRIDE(v, k) = 1;
        if (descp->ndims == -1 || descp->ndims == -2) {
          sb.sect[k].lwb = val;
          sb.sect[k].upb = val;
          sb.sect[k].stride = 1;
        }
        continue;
      }

      /* Expect to see a : */

      if (token != TK_COLON)
        goto subscript_error;
    } else if (token == TK_COLON) {
      if (descp->ndims == -1 || descp->ndims == -2)
        val = F90_DPTR_LBOUND_G(acd);
      else
        val = desc_dims[2 * k]; /* default lower bound */
    } else
      goto subscript_error;

    /* So far, we've parsed
     *    <c> :
     * or
     *    :
     * in which case we now have a lower bound of a section - determine
     * the upper bound.
     */
    is_section = TRUE;
    if (descp->ndims == -1 || descp->ndims == -2) {
      upb = F90_DPTR_EXTENT_G(acd);
    } else {
      upb = desc_dims[2 * k + 1]; /* default upper bound */
    }
    stride = 1; /* default stride */
    GET_TOKEN(i);
    if (token == TK_CONST && constval.dtype == __BIGINT) {
      upb = constval.val.i;
      if (descp->ndims == -1 || descp->ndims == -2) {
        if (upb < F90_DPTR_LBOUND_G(acd) || upb > F90_DPTR_EXTENT_G(acd))
          goto subscript_error; /* subscr out of range */
      } else if (upb < desc_dims[2 * k] || upb > desc_dims[2 * k + 1])
        goto subscript_error; /* subscr out of range */

      /* Found <c> as the upper bound; check for stride. */

      GET_TOKEN(i);
      if (token == TK_COLON) {
        /* expect a constant */
        GET_TOKEN(i);
        if (token != TK_CONST || constval.dtype != __BIGINT)
          goto subscript_error;
        stride = constval.val.i;
        if (stride < 0)
          goto subscript_error; /* subscr out of range */
        GET_TOKEN(i);
      }
    } else if (token != TK_COMMA && token != TK_RPAREN)
      goto subscript_error;

    TRI_LWB(v, k) = val;
    TRI_UPB(v, k) = upb;
    TRI_STRIDE(v, k) = stride;
    if (descp->ndims == -1 || descp->ndims == -2) {
      sb.sect[k].lwb = val;
      sb.sect[k].upb = upb;
      sb.sect[k].stride = stride;
    }

  } while (token != TK_RPAREN);
  comma_is_token = FALSE;

  if (descp->ndims == -1 || descp->ndims == -2) {
    VRF_SUBSCRIPT(vrf_cur) = v;
    if (k != new_ndims - 1)
      goto subscript_error;
  } else if (k != actual_ndims - 1)
    goto subscript_error;

  if (is_section)
    vrf_cur = add_vrf(VRF_SECTION, descp);
  else
    vrf_cur = add_vrf(VRF_ELEMENT, descp);
  VRF_SUBSCRIPT(vrf_cur) = v;

  return 0; /* no errors encountered */

subscript_error:
  return NML_ERROR(FIO_ESUBSC);
}

static int
I8(parse_substring)(NML_DESC *descp)
{
  int i;
  __BIGINT_T val, end;
  F90_Desc *sd;

  val = 1; /* default starting value */
  GET_TOKEN(i);
  if (token == TK_CONST && constval.dtype == __BIGINT) {
    val = constval.val.i;
    GET_TOKEN(i); /* expect a : */
  }
  if (token != TK_COLON) /* illegal substring spec */
    return NML_ERROR(FIO_ESUBSC);

  /* At this point, ":" of substring descriptor has been read and
   * 'val' contains the value of the starting position.
   */
  if (descp->ndims == -1 || descp->ndims == -2) {
    sd = get_descriptor(descp);
    end = descp->len = F90_LEN_G(sd); /* deferred char */
  } else
    end = descp->len; /* default end position */

  GET_TOKEN(i);
  if (token == TK_CONST && constval.dtype == __BIGINT) {
    end = constval.val.i;
    GET_TOKEN(i); /* expect right paren */
  }
  if (token != TK_RPAREN) /* check for closing paren */
    return NML_ERROR(FIO_ESUBSC);

  substring.present = TRUE;
  substring.start = val;
  substring.end = end;

  return 0; /* no errors encountered */
}

static int
add_triple(int n)
{
  int i;
  i = tri.avl;
  tri.avl += n;
  return i;
}

static int
assign_values(void)
{
  int i;
  int err;

  comma_live = TRUE;
  err = eval(0, NULL);

  /* Have processed a name-value pair; check the next token */
  GET_TOKEN(i);
  if (token == TK_CONST) {
    live_token = 1;
    return NML_ERROR(FIO_ETOOM);
  }
  if ((token == TK_COMMA && gbl->decimal != FIO_COMMA) ||
      (token == TK_SEMICOLON && gbl->decimal == FIO_COMMA)) {
    /* cleanup get_token - eat the separating ',' */
    GET_TOKEN(i);
    if (token == TK_CONST) {
      live_token = 1;
      return NML_ERROR(FIO_ETOOM);
    }
  }
  /* other token - 'put token back' and exit loop.  */
  assert(live_token == 0);
  live_token = 1;

  return err;
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

/*
 * Recursively compute the index space given a set of subscripts for n
 * dimensions. The evaluation begins by iterating over the last dimension,
 * recursively evaluating the subscripts of the next (to the left) for
 * each iteration.  For a given dimension d's index, subscripts to the left
 * are recursively computed.  When the first dimension is reached, the address
 * of the element represented by the current subscript values is passed to
 * the 'eval' function.
 */
static int
I8(eval_sb)(int d)
{
  int j, err;
  __BIGINT_T offset;
  char *new_addr;
  NML_DESC *descp;
  __POINT_T *desc_dims;
  F90_Desc *sd;

  descp = VRF_DESCP(sb.v);

  if (descp->ndims == -1 || descp->ndims == -2) {
    desc_dims = (__POINT_T *)((char *)descp + sizeof(NML_DESC));
    if (*desc_dims == 0) {
      /* $p contains an address of array/scalar */
      new_addr = *(char **)sb.loc_addr;
      err = eval_ptr(sb.v, new_addr);
      if (err)
        return err;
      return 0;
    }
  }
  if (d == 0) {
    /*
     * Reached the first dimension; iterate over the first dimension,
     * compute the address of each element, and pass each address to
     * 'eval'.
     */
    sd = get_descriptor(descp);
    for (sb.idx[0] = sb.sect[0].lwb; sb.idx[0] <= sb.sect[0].upb;
         sb.idx[0] += sb.sect[0].stride) {
      offset = 0;
      if (descp->ndims == -1 || descp->ndims == -2) {
        new_addr = I8(__fort_local_address)((*(char **)sb.loc_addr), sd,
                                           (__INT_T *)&sb.idx[0]);
        err = eval_ptr(sb.v, new_addr);
      } else {
        for (j = 0; j < sb.ndims; j++) {
          offset += (sb.idx[j] - sb.lwb[j]) * sb.mult[j];
        }
        offset *= sb.elemsz;
        new_addr = sb.loc_addr + offset;
        err = eval(sb.v + 1, new_addr);
      }
      if (err)
        return err;
    }
    return 0;
  }

  /* Iterate over the current dimension, and recursively evaluate all
   * subscripts in the dimensions to the left.
   */
  for (sb.idx[d] = sb.sect[d].lwb; sb.idx[d] <= sb.sect[d].upb;
       sb.idx[d] += sb.sect[d].stride) {
    err = I8(eval_sb)(d - 1);
    if (err)
      return err;
  }
  return 0;
}

/** \brief
 * Recursively compute the index space given a set of subscripts for n
 * dimensions.
 *
 *  The evaluation begins by iterating over the last dimension,
 * recursively evaluating the subscripts of the next (to the left) for
 * each iteration.  For a given dimension d's index, subscripts to the left
 * are recursively computed.  When the first dimension is reached, the address
 * of the element represented by the current subscript values is passed to
 * the 'eval' function.
 */
static int
I8(eval_dtio_sb)(int d)
{
  int j, err;
  __BIGINT_T offset;
  char *new_addr;
  NML_DESC *descp;
  __POINT_T *desc_dims;
  F90_Desc *sd;

  descp = VRF_DESCP(sb.v);

  if (descp->ndims == -2) {
    desc_dims = (__POINT_T *)((char *)descp + sizeof(NML_DESC));
    if (*desc_dims == 0) {
      /* $p contains an address of array/scalar */
      new_addr = *(char **)sb.loc_addr;
      err = eval_ptr(sb.v, new_addr);
      if (err)
        return err;
      return 0;
    }
  } else if (descp->ndims == -1) {
#ifdef DEBUG
    printf("unexpect ndims in eval_dtio_sb\n");
    return ERR_FLAG;
#endif
  }
  if (d == 0) {
    /*
     * Reached the first dimension; iterate over the first dimension,
     * compute the address of each element, and pass each address to
     * 'eval'.
     */
    sd = get_descriptor(descp);
    for (sb.idx[0] = sb.sect[0].lwb; sb.idx[0] <= sb.sect[0].upb;
         sb.idx[0] += sb.sect[0].stride) {
      offset = 0;
      if (descp->ndims == -2) {
        new_addr = I8(__fort_local_address)((*(char **)sb.loc_addr), sd,
                                           (__INT_T *)&sb.idx[0]);
        err = eval_ptr(sb.v, new_addr);
      } else {
        for (j = 0; j < sb.ndims; j++) {
          offset += (sb.idx[j] - sb.lwb[j]) * sb.mult[j];
        }
        offset *= sb.elemsz;
        new_addr = sb.loc_addr + offset;
        err = eval(sb.v + 1, new_addr);
      }
      if (err)
        return err;
    }
    return 0;
  }

  /* Iterate over the current dimension, and recursively evaluate all
   * subscripts in the dimensions to the left.
   */
  for (sb.idx[d] = sb.sect[d].lwb; sb.idx[d] <= sb.sect[d].upb;
       sb.idx[d] += sb.sect[d].stride) {
    err = I8(eval_dtio_sb)(d - 1);
    if (err)
      return err;
  }
  return 0;
}

static int
eval_ptr(int v, char *loc_addr)
{

  switch (VRF_TYPE(v)) {
  case VRF_ELEMENT:
    /* subscripted reference, assign() stores a scalar but there
     * may be additional values.
     */
    return assign(VRF_DESCP(v), loc_addr, NULL, FALSE, TRUE);
  case VRF_SECTION:
    /* for this type, we're already iterating over an index space;
     * just have assign store a scalar.
     */
    return assign(VRF_DESCP(v), loc_addr, NULL, FALSE, FALSE);
  default:
    break;
  }
  return assign(VRF_DESCP(v), loc_addr, NULL, TRUE, FALSE);
}

static int
eval(int v, char *loc_addr)
{
  NML_DESC *descp;
  int i, k;
  __BIGINT_T offset, mm;
  char *new_addr;
  __POINT_T *desc_dims; /* base pointer to 2-dim descriptor array */
  __POINT_T actual_ndims;

  if (v > vrf_cur) {

    descp = VRF_DESCP(v - 1);
    if (descp->ndims == -1)
      return I8(eval_sb)(sb.ndims - 1);
    else if (descp->ndims == -2)
      return I8(eval_dtio_sb)(sb.ndims - 1);
    switch (VRF_TYPE(v - 1)) {
    case VRF_ELEMENT:
      /* subscripted reference, assign() stores a scalar but there
       * may be additional values.
       */
      return assign(VRF_DESCP(v - 1), loc_addr, NULL, FALSE, TRUE);
    case VRF_SECTION:
      /* for this type, we're already iterating over an index space;
       * just have assign store a scalar.
       */
      return assign(VRF_DESCP(v - 1), loc_addr, NULL, FALSE, FALSE);
    default:
      break;
    }
    return assign(VRF_DESCP(v - 1), loc_addr, NULL, TRUE, FALSE);
  }

  new_addr = loc_addr;
  descp = VRF_DESCP(v);
  ACTUAL_NDIMS(actual_ndims);
  switch (VRF_TYPE(v)) {
  case VRF_ID:
    if (descp->ndims == -1)   /* scalar pointer - getting $p */
      new_addr = *(char **)descp->addr;
    else
      new_addr = descp->addr;
    break;

  case VRF_ELEMENT:
    k = VRF_SUBSCRIPT(v);
    desc_dims = (__POINT_T *)((char *)descp + sizeof(NML_DESC));
    if (descp->ndims != -1 && descp->ndims != -2) {
      offset = TRI_LWB(k, 0) - desc_dims[0];
      mm = 1; /*  multiplier for each dimension */
      for (i = 1; i < actual_ndims; i++) {
        mm *= desc_dims[2 * (i - 1) + 1] - desc_dims[2 * (i - 1)] + 1;
        offset += (TRI_LWB(k, i) - desc_dims[2 * i]) * mm;
      }
      offset *= I8(siz_of)(descp);
      new_addr += offset;
    }
    break;

  case VRF_SECTION:
    k = VRF_SUBSCRIPT(v);

    desc_dims = (__POINT_T *)((char *)descp + sizeof(NML_DESC));
    /*
     * Copy the section information into the looping sect structure.
     * Compute the multipliers for each dimension.
     */
    if (descp->ndims != -1 && descp->ndims != -2) {
      sb.mult[0] = 1;
      sb.sect[0] = TRI_SECT(k, 0);
      sb.lwb[0] = desc_dims[0];
      sb.ndims = actual_ndims;
      sb.loc_addr = loc_addr;
      for (i = 1; i < actual_ndims; i++) {
        sb.lwb[i] = desc_dims[2 * i];
        sb.mult[i] = sb.mult[i - 1] *
                     (desc_dims[2 * (i - 1) + 1] - desc_dims[2 * (i - 1)] + 1);
        sb.sect[i] = TRI_SECT(k, i);
      }
    }
    sb.v = v;
    sb.elemsz = I8(siz_of)(descp);
    if (descp->ndims == -2) {
      return I8(eval_dtio_sb)(sb.ndims - 1);
    } else if (descp->ndims >= MAX_DIM) {
      return I8(eval_dtio_sb)(sb.ndims - 1);
    }
    return I8(eval_sb)(sb.ndims - 1);

  case VRF_MEMBER:
    new_addr = loc_addr + (long)descp->addr;
    break;
  }

  return eval(v + 1, new_addr);
}

static int
assign(NML_DESC *descp, char *loc_addr, char **p_next_addr, bool chkarr,
       bool is_subscripted)
{
  int i, k;
  int length;
  int err;
  char *new_addr;
  NML_DESC *new_descp;

  if (descp->ndims == -2 || descp->ndims >= MAX_DIM) {
    return dtio_assign(descp, loc_addr, p_next_addr, chkarr, is_subscripted);
  }

  if (p_next_addr)
    *p_next_addr = NULL;
  if (chkarr && descp->ndims > 0) {
    __BIGINT_T elemsz;
    int nitems;
    char *last_addr;
    char *next_addr;

    /* Compute the size of each array element */
    elemsz = I8(siz_of)(descp);
    /*
     * Compute the number of items in the array.  Loop on the number
     * of items in the array, assigning to the elements of the array
     * in lexical order; the address of the next element is just the
     * sum of the previous address and the element size.
     */
    nitems = nelems_of(descp);
    if (nitems > 0) {
      new_addr = loc_addr;
      last_addr = loc_addr + (nitems - 1) * elemsz;
      next_addr = NULL;
      while (TRUE) {
        err = assign(descp, new_addr, &next_addr, FALSE, FALSE);
        if (err)
          return err;
        if (next_addr && next_addr > new_addr)
          new_addr = next_addr;
        else
          new_addr += elemsz;
        if (new_addr > last_addr)
          break;
      }
    }
    return 0;
  } else if (chkarr && (descp->ndims == -1 || descp->ndims == -2)) {
    __BIGINT_T elemsz;
    int nitems;
    char *last_addr;
    char *next_addr;

    /* Compute the size of each array element */
    elemsz = I8(siz_of)(descp);
    /*
     * Compute the number of items in the array.  Loop on the number
     * of items in the array, assigning to the elements of the array
     * in lexical order; the address of the next element is just the
     * sum of the previous address and the element size.
     */
    nitems = nelems_of(descp);
    if (nitems > 0) {
      new_addr = loc_addr;
      last_addr = loc_addr + (nitems - 1) * elemsz;
      next_addr = NULL;
      while (TRUE) {
        err = assign(descp, new_addr, &next_addr, FALSE, FALSE);
        if (err)
          return err;
        if (next_addr && next_addr > new_addr)
          new_addr = next_addr;
        else
          new_addr += elemsz;
        if (new_addr > last_addr)
          break;
      }
    }
    return 0;
  }

  if (descp->type == __DERIVED) {
    /*
     * Loop on the members of the derived type.  First, compute the
     * number of bytes to add to reach the descriptor of the first
     * member.
     */
    if (descp->ndims >= 0)
      k = sizeof(NML_DESC) + (descp->ndims * sizeof(__POINT_T) * 2);
    else
      k = sizeof(NML_DESC) + (sizeof(__POINT_T) * 2);
    new_descp = (NML_DESC *)((char *)descp + k);
    new_addr = loc_addr;
    while (TRUE) {
      if (new_descp->nlen == 0) {
        break;
      }
      new_addr = loc_addr + (long)new_descp->addr;
      err = assign(new_descp, new_addr, NULL, TRUE, FALSE);
      if (err)
        return err;
      new_descp = skip_to_next(new_descp);
    }
    return 0;
  }

  /*  Just store into a scalar  */

  length = descp->len; /* Need to update the length of deferchar */
  while (TRUE) {
    GET_TOKEN(i);
    switch (token) {
    case TK_CONST:
      if (!substring.present)
        err = __fortio_assign(loc_addr, descp->type, length, &constval);
      else {
        /* Given that the substring is present, it's an assertion that
         * the data type is a string.
         */
        new_addr = loc_addr + substring.start * FIO_TYPE_SIZE(descp->type);
        length = substring.end - substring.start + 1;
        err = __fortio_assign(new_addr, descp->type, length, &constval);
      }
      if (err != 0)
        return NML_ERROR(err);
      comma_live = FALSE;
      /*
       * If the name represents an array,
       * increment address just in case there are more values,
       * in which case the values will be stored in the next
       * element-order address; e.g.
       *    A(1,1) = 1 2 3
       * stores
       *    A(1,1) = 1
       *    A(2,1) = 2
       *    A(3,1) = 3
       */
      loc_addr += I8(siz_of)(descp);
      if (!is_subscripted)
        goto exit_loop;
      break;
    case TK_SEMICOLON:
      if (gbl->decimal == FIO_COMMA) {
        if (comma_live)
          loc_addr += I8(siz_of)(descp);
        comma_live = TRUE;
      }
      break;

    case TK_COMMA:
      if (comma_live)
        loc_addr += I8(siz_of)(descp);
      comma_live = TRUE;
      break;

    case TK_SKIP: /*  '<repeat_count>*' null values  */
      comma_live = FALSE;
      loc_addr += I8(siz_of)(descp);
      break;

    default:
      /* other token - 'put token back' and exit loop.  */
      assert(live_token == 0);
      live_token = 1;
      goto exit_loop;
    }
  }
exit_loop:
  if (p_next_addr)
    *p_next_addr = loc_addr;
  return 0;
}

static int
dtio_assign(NML_DESC *descp, char *loc_addr, char **p_next_addr, bool chkarr,
            bool is_subscripted)
{
  int i;
  int err;
  char *new_addr;
  __POINT_T actual_ndims;
  ACTUAL_NDIMS(actual_ndims);

  if (p_next_addr)
    *p_next_addr = NULL;
  if (chkarr && actual_ndims > 0) {
    __BIGINT_T elemsz;
    int nitems;
    char *last_addr;
    char *next_addr;

    /* Compute the size of each array element */
    elemsz = I8(siz_of)(descp);
    /*
     * Compute the number of items in the array.  Loop on the number
     * of items in the array, assigning to the elements of the array
     * in lexical order; the address of the next element is just the
     * sum of the previous address and the element size.
     */
    nitems = nelems_of(descp);
    if (nitems > 0) {
      new_addr = loc_addr;
      last_addr = loc_addr + (nitems - 1) * elemsz;
      next_addr = NULL;
      while (TRUE) {
        err = dtio_assign(descp, new_addr, &next_addr, FALSE, FALSE);
        if (err)
          return err;
        if (next_addr && next_addr > new_addr)
          new_addr = next_addr;
        else
          new_addr += elemsz;
        if (new_addr > last_addr)
          break;
      }
    }
    return 0;
  } else if (chkarr && (descp->ndims == -2)) {
    __BIGINT_T elemsz;
    int nitems;
    char *last_addr;
    char *next_addr;

    /* Compute the size of each array element */
    elemsz = I8(siz_of)(descp);
    /*
     * Compute the number of items in the array.  Loop on the number
     * of items in the array, assigning to the elements of the array
     * in lexical order; the address of the next element is just the
     * sum of the previous address and the element size.
     */
    nitems = nelems_of(descp);
    if (nitems > 0) {
      new_addr = loc_addr;
      last_addr = loc_addr + (nitems - 1) * elemsz;
      next_addr = NULL;
      while (TRUE) {
        err = dtio_assign(descp, new_addr, &next_addr, FALSE, FALSE);
        if (err)
          return err;
        if (next_addr && next_addr > new_addr)
          new_addr = next_addr;
        else
          new_addr += elemsz;
        if (new_addr > last_addr)
          break;
      }
    }
    return 0;
  }

  while (TRUE) {
    /* call dtio here */
    err = dtio_read_scalar(descp, loc_addr);
    if (err)
      return err;
    comma_live = FALSE;
    loc_addr += I8(siz_of)(descp);
    GET_TOKEN(i);

    switch (token) {
    case TK_SEMICOLON:
      if (gbl->decimal == FIO_COMMA) {
        if (comma_live)
          loc_addr += I8(siz_of)(descp);
        comma_live = TRUE;
      }
      if (!is_subscripted)
        goto exit_loop;
      break;

    case TK_COMMA:
      if (comma_live)
        loc_addr += I8(siz_of)(descp);
      comma_live = TRUE;
      if (!is_subscripted)
        goto exit_loop;
      break;

    case TK_SKIP: /*  '<repeat_count>*' null values  */
      comma_live = FALSE;
      loc_addr += I8(siz_of)(descp);
      if (!is_subscripted)
        goto exit_loop;
      break;

    default:
      /* other token - 'put token back' and exit loop.  */
      /*	    comma_live = FALSE;*/
      if (!is_subscripted) {
        goto exit_loop;
      } else {
        assert(live_token == 0);
        live_token = 1;
        goto exit_loop;
      }
    }
  }
exit_loop:
  if (p_next_addr)
    *p_next_addr = loc_addr;
  return 0;
}

/* -------------------------------------------------------------------- */

static int
_f90io_nmlr_end()
{
  gbl->decimal = 0;
  if (!gbl->same_fcb_idx) {
    gbl->unit = 0;
    gbl->iostat = 0;
  }

  /* first check for errors: */
  if (fioFcbTbls.eof)
    return EOF_FLAG;
  if (fioFcbTbls.error)
    return ERR_FLAG;

  return 0;
}

/** \brief Read a namelist group
 */
int
ENTF90IO(NMLR_END, nmlr_end)()
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_nmlr_end();
  __fortio_errend03();
  return DIST_STATUS_BCST(s);
}

int
ENTCRF90IO(NMLR_END, nmlr_end)()
{
  int s = 0;
  s = _f90io_nmlr_end();
  __fortio_errend03();
  return s;
}

/* ********************/
/*    read  support   */
/* ********************/

static int
read_record(void)
{
  if (internal_file) {
    if (n_irecs == 0)
      return FIO_EEOF;
    if (accessed)
      in_recp += rec_len;
    n_irecs--;

    byte_cnt = rec_len;
    if (byte_cnt >= rbuf_size)
      (void) alloc_rbuf(byte_cnt, FALSE);
    (void) memcpy(rbufp, in_recp, byte_cnt);
    accessed = TRUE;
  } else {
    /* sequential read */
    int ch;
    char *p;

    f->nextrec++;
    p = rbufp;
    byte_cnt = 0;

    while (TRUE) {
      if (byte_cnt >= rbuf_size)
        p = alloc_rbuf(byte_cnt, TRUE);
      ch = __io_fgetc(f->fp);
      if (ch == EOF) {
        if (__io_feof(f->fp)) {
          if (byte_cnt)
            break;
          return FIO_EEOF;
        }
        return __io_errno();
      }
      if (ch == '\r' && EOR_CRLF) {
        ch = __io_fgetc(f->fp);
        if (ch == '\n')
          break;
        __io_ungetc(ch, f->fp);
        ch = '\r';
      }
      if (ch == '\n')
        break;
      byte_cnt++;
      *p++ = ch;
    }
  }
  rbufp[byte_cnt] = '\n';
  currc = rbufp;
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
dtio_read_scalar(NML_DESC *descp, char *loc_addr)
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
  int k, num_consts;
  const char *iotype = "NAMELIST";
  char *start_addr;
  __POINT_T *desc_dims, new_ndims;
  __POINT_T actual_ndims;

  /* if this is array */
  num_consts = 1;
  desc_dims = (__POINT_T *)((char *)descp + sizeof(NML_DESC));
  if (descp->ndims == -2) {
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
  dtio = (void *)*(char **)((char *)desc_dims);
#if DEBUG
  if ((INT *)dtio == 0) {
    printf("ERROR: unable find user defined io read routine \n");
  }
#endif

  /* skip write routine */
  desc_dims = (__POINT_T *)((char *)next_descp + sizeof(__POINT_T));
  next_descp = (NML_DESC *)desc_dims;

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

  (*dtio)(start_addr, unit, iotype, vlist, iostat, iomsg, dtv_sd, vlist_sd,
          iotypelen, iomsglen);
  if (*iostat != 0)
    return *iostat;
  start_addr = start_addr + descp->len;
  if (!internal_file && f->pback) {
    currc = f->pback;
    f->pback = 0;
  }
  return 0;
}
