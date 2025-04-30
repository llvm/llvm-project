/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Initialization and error handling functions for Fortran I/O
 */

#include <errno.h>
#include <string.h> /* for declarations of memcpy and memset */
#include "global.h"

typedef struct {
  char *name;
  __CLEN_T len;
  int lineno;
} src_info_struct;

static src_info_struct src_info;

static int current_unit;
static INT *iostat_ptr;
static int iobitv;
static const char *err_str = "?";
char *envar_fortranopt;

static char *iomsg; /* pointer for optional IOMSG area */
static __CLEN_T iomsgl;  /* length of above */

typedef struct {
  INT *enctab;
} f90fmt;

typedef struct {
  src_info_struct src_info;
  int current_unit;
  bool newunit;
  INT *iostat_ptr;
  int iobitv;
  const char *err_str;
  char *envar_fortranopt;
  char *iomsg;
  __CLEN_T iomsgl;

  /* fioFcbTbls stuff */
  FIO_FCB *fcbs;
  INT *enctab;
  char *fname;
  int fnamelen;
  bool error;
  bool eof;
  bool pos_present;
  seekoffx_t pos;

} fioerror;

#define GBL_SIZE 15
static int gbl_size = 15;
static int gbl_avl = 0;
static fioerror static_gbl[GBL_SIZE];
static fioerror *gbl = &static_gbl[0];
static fioerror *gbl_head = &static_gbl[0];

static int fmtgbl_size = 15;
static int fmtgbl_avl = 0;
static f90fmt static_fmtgbl[GBL_SIZE];
static f90fmt *fmtgbl = &static_fmtgbl[0];
static f90fmt *fmtgbl_head = &static_fmtgbl[0];

static void ioerrinfo(FIO_FCB *);
static void __fortio_init(void);

#include "fort_vars.h"
extern void  f90_compiled();

/* --------------------------------------------------------------------- */
void
set_gbl_newunit(bool newunit)
{
  gbl->newunit = newunit;
}

bool
get_gbl_newunit()
{
  return gbl->newunit;
}

/* --------------------------------------------------------------- */
static void
save_gbl()
{
  if (gbl_avl) {
    gbl->iostat_ptr = iostat_ptr;
    gbl->err_str = err_str;
    gbl->current_unit = current_unit;
    gbl->iobitv = iobitv;
    gbl->envar_fortranopt = envar_fortranopt;

    gbl->error = fioFcbTbls.error;
    gbl->eof = fioFcbTbls.eof;
    gbl->pos_present = fioFcbTbls.pos_present;
    gbl->pos = fioFcbTbls.pos;
    gbl->fname = fioFcbTbls.fname;
    gbl->fnamelen = fioFcbTbls.fnamelen;
  }
}

static void
restore_gbl()
{
  if (gbl_avl) {
    iostat_ptr = gbl->iostat_ptr;
    err_str = gbl->err_str;
    current_unit = gbl->current_unit;
    iobitv = gbl->iobitv;
    envar_fortranopt = gbl->envar_fortranopt;
    iomsg = gbl->iomsg;
    iomsgl = gbl->iomsgl;
    src_info.name = gbl->src_info.name;
    src_info.len = gbl->src_info.len;
    src_info.lineno = gbl->src_info.lineno;

    if (gbl->current_unit != current_unit) {
      fioFcbTbls.error = gbl->error;
      fioFcbTbls.eof = gbl->eof;
    } else {
      /* may need to recursively check current_unit with other gbl->current_unit
       * if it is a match, then save fioFcbTbls.error/eof to that gbl? F2008?
       */
    }
    fioFcbTbls.pos_present = gbl->pos_present;
    fioFcbTbls.pos = gbl->pos;
    fioFcbTbls.fname = gbl->fname;
    fioFcbTbls.fnamelen = gbl->fnamelen;
  }
}

static void
free_gbl()
{
  --gbl_avl;
  if (gbl_avl <= 0)
    gbl_avl = 0;
  if (gbl_avl == 0)
    gbl = &gbl_head[0];
  else
    gbl = &gbl_head[gbl_avl - 1];
}

static void
allocate_new_gbl()
{
  fioerror *tmp_gbl;
  if (gbl_avl >= gbl_size) {
    if (gbl_size == GBL_SIZE) {
      gbl_size = gbl_size + 15;
      tmp_gbl = (fioerror *)malloc(sizeof(fioerror) * gbl_size);
      memcpy(tmp_gbl, gbl_head, sizeof(fioerror) * gbl_avl);
      gbl_head = tmp_gbl;
    } else {
      gbl_size = gbl_size + 15;
      gbl_head = (fioerror *)realloc(gbl_head, sizeof(fioerror) * gbl_size);
    }
  }
  gbl = &gbl_head[gbl_avl];
  memset(gbl, 0, sizeof(fioerror));
  ++gbl_avl;
}

static void
allocate_new_fmtgbl()
{
  f90fmt *tmp_gbl;
  if (fmtgbl_avl >= fmtgbl_size) {
    if (fmtgbl_size == GBL_SIZE) {
      fmtgbl_size = fmtgbl_size + 15;
      tmp_gbl = (f90fmt *)malloc(sizeof(f90fmt) * fmtgbl_size);
      memcpy(tmp_gbl, fmtgbl_head, sizeof(f90fmt) * fmtgbl_avl);
      fmtgbl_head = tmp_gbl;
    } else {
      fmtgbl_size = fmtgbl_size + 15;
      fmtgbl_head =
          (f90fmt *)realloc(fmtgbl_head, sizeof(f90fmt) * fmtgbl_size);
    }
  }
  fmtgbl = &fmtgbl_head[fmtgbl_avl];
  memset(fmtgbl, 0, sizeof(f90fmt));
  ++fmtgbl_avl;
}

static void
free_fmtgbl()
{
  --fmtgbl_avl;
  if (fmtgbl_avl <= 0)
    fmtgbl_avl = 0;
  if (fmtgbl_avl == 0)
    fmtgbl = &fmtgbl_head[0];
  else
    fmtgbl = &fmtgbl_head[fmtgbl_avl - 1];
}

static void
restore_fmtgbl()
{
  if (fmtgbl_avl) {
    fioFcbTbls.enctab = fmtgbl->enctab;
  }
}

static void
save_fmtgbl()
{
  if (fmtgbl_avl) {
    fmtgbl->enctab = fioFcbTbls.enctab;
  }
}

/* --------------------------------------------------------------- */

extern void
__fortio_errinit(__INT_T unit, __INT_T bitv, __INT_T *iostat, const char *str)
{
  if (fioFcbTbls.fcbs == NULL)
    __fortio_init();

  fioFcbTbls.error = FALSE;
  fioFcbTbls.eof = FALSE;
  fioFcbTbls.fname = NULL;

  current_unit = unit;
  iobitv = bitv;
  if (iobitv & FIO_BITV_IOSTAT) {
    iostat_ptr = iostat;
    *iostat_ptr = 0;
  } else {
    iostat_ptr = NULL;
  }

  /* save str for error messages  ... */
  err_str = str;

}

extern void
__fortio_errinit03(__INT_T unit, __INT_T bitv, __INT_T *iostat, const char *str)
{
  if (fioFcbTbls.fcbs == NULL)
    __fortio_init();

  save_gbl();

  fioFcbTbls.error = FALSE;
  fioFcbTbls.eof = FALSE;
  fioFcbTbls.fname = NULL;

  current_unit = unit;
  iobitv = bitv;
  if (iobitv & FIO_BITV_IOSTAT) {
    iostat_ptr = iostat;
    *iostat_ptr = 0;
  } else {
    iostat_ptr = NULL;
  }

  /* save str for error messages  ... */
  err_str = str;
}
extern void
__fortio_errend03()
/* restore the previous value of previous status of io error.*/
{
  free_gbl();
  restore_gbl();
}

extern void
__fortio_fmtinit()
/* restore the previous value of previous status of enctab.*/
{
  save_fmtgbl();
  allocate_new_fmtgbl();
}

extern void
__fortio_fmtend(void)
/* restore the previous value of enctab.*/
{
  free_fmtgbl();
  restore_fmtgbl();
}

/* --------------------------------------------------------------- */

/*  define text for error messages:  */

#define X(str) str,

static const char *errtxt[] = {
    X("xxx")                                           /* 200 */
    X("illegal value for specifier")                   /* ESPEC 201 */
    X("conflicting specifiers")                        /* ECOMPAT 202 */
    X("record length must be specified")               /* ERECLEN 203 */
    X("illegal use of a read-only file")               /* EREADONLY 204 */
    X("'SCRATCH' and 'SAVE'/'KEEP' both specified")    /* EDISPOSE 205 */
    X("attempt to open a named file as 'SCRATCH'")     /* ESCRATCH 206 */
    X("file is already connected to another unit")     /* EOPENED 207 */
    X("'NEW' specified for file which already exists") /* EEXIST 208 */
    X("'OLD' specified for file which does not exist") /* ENOEXIST 209 */

    X("dynamic memory allocation failed")                  /* ENOMEM 210 */
    X("invalid file name")                                 /* ENAME 211 */
    X("invalid unit number")                               /* EUNIT 212 */
    X("RECL cannot be present")                            /* ERECL 213 */
    X("READ not allowed for write-only file")              /* EWRITEONLY 214 */
    X("formatted/unformatted file conflict")               /* EFORM 215 */
    X("www")                                               /* 216 */
    X("attempt to read past end of file")                  /* EEOF 217 */
    X("attempt to read (nonadvancing) past end of record") /* EEOR 218 */
    X("attempt to read/write past end of record")          /* ETOOBIG 219 */

    X("write after last internal record")                  /* ETOOFAR 220 */
    X("syntax error in format string")                     /* EFSYNTAX 221	*/
    X("unbalanced parentheses in format string")           /* EPAREN 222 */
    X("illegal P, T or B edit descriptor - value missing") /* EPT 223 */
    X("illegal Hollerith or character string in format")   /* ESTRING 224 */
    X("lexical error-- unknown token type")                /* ELEX 225 */
    X("unrecognized edit descriptor letter in format")     /* ELETTER 226 */
    X("ccc")                                               /* 227 */
    X("end of file reached without finding group")         /* ENOGROUP 228 */
    X("end of file reached while processing group")        /* ENMLEOF 229 */

    X("scale factor not in range -128 to 127") /* ESCALEF 230 */
    X("error on data conversion")              /* EERR_DATA_CONVERSION231 */
    X("fff")                                   /* 232 */
    X("too many constants to initialize group item")  /* ETOOM 233 */
    X("invalid edit descriptor")                      /* EEDITDSCR 234 */
    X("edit descriptor does not match item type")     /* EMISMATCH 235 */
    X("formatted record longer than 2000 characters") /* EBIGREC 236 */
    X("quad precision type unsupported")              /* EQUAD 237 */
    X("tab value out of range")             /* ETAB_VALUE_OUT_OF_RANGE 238 */
    X("entity name is not member of group") /* ENOTMEM 239 */
    X("no initial left parenthesis in format string") /* ELPAREN 240 */
    X("unexpected end of format string")              /* EENDFMT 241 */
    X("illegal operation on direct access file")      /* EDIRECT 242 */
    X("format parentheses nesting depth too great")   /* EPNEST 243 */
    X("syntax error - entity name expected")          /* ENONAME 244 */
    X("syntax error within group definition")         /* ESYNTAX 245 */
    X("infinite format scan for edit descriptor") /* EINFINITE_REVERSION 246 */
    X("ggg")                                      /* 247 */
    X("illegal subscript or substring specification")      /* ESUBSC 248 */
    X("error in format - illegal E, F, G or D descriptor") /* EFGD 249 */

    X("error in format - number missing after '.', '-', or '+'") /* EDOT 250 */
    X("illegal character in format string")                      /* ECHAR 251 */
    X("operation attempted after end of file")               /* EEOFERR 252 */
    X("attempt to read non-existent record (direct access)") /* EDREAD 253 */
    X("illegal repeat count in format")                      /* EREPCNT 254 */
    X("illegal asynchronous I/O operation")                  /* EASYNC  255 */
    X("POS can only be specified for a 'STREAM' file")       /* EPOS    256 */
    X("POS value must be positive")                          /* EPOSV   257 */
    X("NEWUNIT requires FILE or STATUS=SCRATCH")             /* ENEWUNIT 258 */
};

/*  include Kanji error message text:  */

#include "kanjidf.h"

/* ------------------------------------------------------------------ */

int
__fortio_error(int errval)
{
  FIO_FCB *fdesc;
  const char *eoln, *txt;
  int retval;

  assert(errval > 0);
  retval = ERR_FLAG;

  if (errval == FIO_EEOF) /* handle end-of-file separately */
    return __fortio_eoferr(FIO_EEOF);
  if (errval == FIO_EEOFERR) /* handle end-of-file separately */
    return __fortio_eoferr(FIO_EEOFERR);

  if (errval == FIO_EEOR) /* handle end-of-record separately */
    return __fortio_eorerr(FIO_EEOR);

  fdesc = __fortio_find_unit(current_unit);

  if (iobitv == FIO_BITV_NONE || iobitv == FIO_BITV_EOF) {
/* Abort if:
 * 1.  no specifier, or
 * 2.  just the END= was specified.
 */
    eoln = "\n";
    if (errval >= FIO_ERROR_OFFSET) {
      txt = __fortio_errmsg(errval);
      if (current_unit == -99) /* internal file */
        __io_fprintf(__io_stderr(), "FIO-F-%d/%s/internal file/%s.%s",
                       errval, err_str, txt, eoln);
      else
        __io_fprintf(__io_stderr(), "FIO-F-%d/%s/unit=%d/%s.%s", errval,
                       err_str, current_unit, txt, eoln);
    } else {
      __io_perror("FIO/stdio");
      __io_fprintf(__io_stderr(), "FIO-F-/%s/unit=%d/%s - %d.%s", err_str,
                     current_unit, "error code returned by host stdio", errval,
                     eoln);
    }
    ioerrinfo(fdesc);
    __fort_abort((char *)0);
  }

  /*  At this point, at least one of {IOSTAT,ERR,END,EOR} was specified.  */

  if (iobitv & FIO_BITV_IOSTAT)
    *iostat_ptr = errval;

  if (iobitv & FIO_BITV_ERR) {
    retval = ERR_FLAG;
  }

  if (iobitv & FIO_BITV_IOMSG) {
    strncpy(iomsg, __fortio_errmsg(errval), iomsgl);
  }

  fioFcbTbls.error = TRUE;
  if (fdesc && fdesc->fp && fdesc->acc == FIO_DIRECT) {
    /* leave file in consistent state:  */
    fdesc->nextrec = 1;
    __io_fseek(fdesc->fp, 0L, SEEK_SET);
  }

  if ((iobitv & FIO_BITV_EOR) && (errval == FIO_ETOOBIG)) {
    retval = EOR_FLAG;
  }

  return retval;
}

/* ------------------------------------------------------------------ */

/* FIXME: this routine is a duplicate of
 *   runtime/lib/pgftn/error.h:__fio_errmsg
 */
extern const char *
__fortio_errmsg(int errval)
{
  const char *txt;
  static char buf[128];
  if (errval == 0) {
    buf[0] = ' ';
    buf[1] = '\0';
    txt = buf;
  } else if (errval >= FIO_ERROR_OFFSET) {
    if (errval - FIO_ERROR_OFFSET >= sizeof(errtxt) / sizeof(errtxt[0])) {
      sprintf(buf, "get_iostat_msg: iostat value %d is out of range", errval);
      txt = buf;
    } else if ((txt = getenv("LANG")) && strcmp(txt, "japan") == 0)
      txt = kanjitxt[errval - FIO_ERROR_OFFSET];
    else {
      txt = errtxt[errval - FIO_ERROR_OFFSET];
    }
  } else
    txt = strerror(errval);
  return txt;
}

/* Return 0 when it's internal file and iobitv = 0 */
int
read_record_internal()
{
  if (iobitv == FIO_BITV_NONE && current_unit == -99) {
    return 0;
  } else {
    return FIO_EEOF;
  }
}

int
__fortio_eoferr(int errval)
{
  FIO_FCB *fdesc;
  const char *eoln, *txt;

  assert(errval > FIO_ERROR_OFFSET);

  fdesc = __fortio_find_unit(current_unit);
  assert(fdesc == NULL || fdesc->acc != FIO_DIRECT);

  if (iobitv == FIO_BITV_NONE ||
      (iobitv & (FIO_BITV_IOSTAT | FIO_BITV_EOF)) == 0) {
/* Abort if:
 * 1.  no specifier, or
 * 2.  neither iostat nor eof were specified.
 */
    eoln = "\n";
    txt = __fortio_errmsg(errval);

    if (current_unit == -99) /* internal file */
      __io_fprintf(__io_stderr(), "FIO-F-%d/%s/internal file/%s.%s",
                     errval, err_str, txt, eoln);
    else
      __io_fprintf(__io_stderr(), "FIO-F-%d/%s/unit=%d/%s.%s", errval,
                     err_str, current_unit, txt, eoln);
    ioerrinfo(fdesc);
    __fort_abort((char *)0);
  }

  /*  At this point, end-of-file occurred and IOSTAT, END, or both, was
   *  specified.
   */
  if (iobitv & FIO_BITV_IOSTAT)
    *iostat_ptr = -1;
  if (iobitv & FIO_BITV_IOMSG) {
    /*        tmp = __fortio_errmsg(errval);
            strncpy(iomsg, tmp, iomsgl);*/
    strncpy(iomsg, __fortio_errmsg(errval), iomsgl);
  }

  fioFcbTbls.eof = TRUE;
  if (fdesc) { /* indicate that 'eof record' has been read */
    fdesc->eof_flag = TRUE;
  }
  return EOF_FLAG;
}

/** \brief end-of-record error when a nonadvancing read */
int
__fortio_eorerr(int errval)
{
  FIO_FCB *fdesc;
  const char *eoln, *txt;

  assert(errval > FIO_ERROR_OFFSET);

  fdesc = __fortio_find_unit(current_unit);
  assert(fdesc == NULL || fdesc->acc != FIO_DIRECT);

  if (iobitv == FIO_BITV_NONE ||
      (iobitv & (FIO_BITV_IOSTAT | FIO_BITV_EOR)) == 0) {
    /* Abort if:
     * 1.  no specifier, or
     * 2.  neither iostat nor eor were specified.
     */
    eoln = "\n";
    txt = __fortio_errmsg(errval);

    if (current_unit == -99) /* internal file */
      __io_fprintf(__io_stderr(), "FIO-F-%d/%s/internal file/%s.%s",
                     errval, err_str, txt, eoln);
    else
      __io_fprintf(__io_stderr(), "FIO-F-%d/%s/unit=%d/%s.%s", errval,
                     err_str, current_unit, txt, eoln);
    ioerrinfo(fdesc);
    __fort_abort((char *)0);
  }

  /*  At this point, end-of-file occurred and IOSTAT, EOR, or both, was
   *  specified.
   */
  if (iobitv & FIO_BITV_IOSTAT)
    *iostat_ptr = -2;
  fioFcbTbls.error = TRUE; /* TBD - does there need to be fioFcbTbls.eor */
  return EOR_FLAG;
}

/* ------------------------------------------------------------------- */

static void
ioerrinfo(FIO_FCB *fdesc)
{
  const char *eoln;
  FILE *fp; /* stderr */

  fp = __io_stderr();
  eoln = "\n";
  if (fdesc != NULL) {
    __io_fprintf(fp, " File name = '");
    if (fdesc->name != NULL)
      __io_fprintf(fp, "%s", fdesc->name);

    if (fdesc->form == FIO_FORMATTED) {
      __io_fprintf(fp, "',    formatted, ");
    } else {
      __io_fprintf(fp, "',    unformatted, ");
    }

    if (fdesc->acc == FIO_DIRECT) {
      __io_fprintf(fp, "direct access  ");
    } else if (fdesc->acc == FIO_STREAM) {
      __io_fprintf(fp, "stream access  ");
    } else {
      __io_fprintf(fp, "sequential access  ");
    }
    if (fdesc->asyptr != (void *)0) {
      if (fdesc->asy_rw) {
        fprintf(fp, "async/active  ");
      } else {
        fprintf(fp, "async  ");
      }
    }
    __io_fprintf(fp, " record = %ld%s", fdesc->nextrec - 1, eoln);
  } else if (fioFcbTbls.fname != NULL)
    __io_fprintf(fp, " File name = %.*s%s", fioFcbTbls.fnamelen, fioFcbTbls.fname,
                   eoln);

  __io_fprintf(fp, " In source file %.*s,", src_info.len, src_info.name);
  __io_fprintf(fp, " at line number %d%s", src_info.lineno, eoln);
}

/* ---------------------------------------------------------------- */

static void
set_src_info()
{
  allocate_new_gbl();
  gbl->src_info.lineno = src_info.lineno;
  gbl->src_info.name = src_info.name;
  gbl->src_info.len = src_info.len;
  gbl->pos_present = fioFcbTbls.pos_present;
}

void ENTF90IO(SRC_INFOA, src_info03a)(
    __INT_T *lineno, /* line number of i/o stmt in source file */
    DCHAR(name)      /* name of source file */
    DCLEN64(name))
{
  src_info.lineno = *lineno;
  src_info.name = CADR(name);
  src_info.len = CLEN(name);
  fioFcbTbls.pos_present = FALSE;
  set_src_info();
}
/* 32 bit CLEN version */
void ENTF90IO(SRC_INFO, src_info03)(
    __INT_T *lineno, /* line number of i/o stmt in source file */
    DCHAR(name)      /* name of source file */
    DCLEN(name))
{
  ENTF90IO(SRC_INFOA, src_info03a)(lineno, CADR(name), (__CLEN_T)CLEN(name));
}

void ENTF90IO(SRC_INFOXA, src_infox03a)(
    __INT_T lineno, /* line number of i/o stmt in source file */
    DCHAR(name)     /* name of source file */
    DCLEN64(name))
{
  src_info.lineno = lineno;
  src_info.name = CADR(name);
  src_info.len = CLEN(name);
  fioFcbTbls.pos_present = FALSE;
  set_src_info();
}
/* 32 bit CLEN version */
void ENTF90IO(SRC_INFOX, src_infox03)(
    __INT_T lineno, /* line number of i/o stmt in source file */
    DCHAR(name)     /* name of source file */
    DCLEN(name))
{
  ENTF90IO(SRC_INFOXA, src_infox03a)(lineno, CADR(name), (__CLEN_T)CLEN(name));
}

void ENTCRF90IO(SRC_INFOA, src_info03a)(
    __INT_T *lineno, /* line number of i/o stmt in source file */
    DCHAR(name)      /* name of source file */
    DCLEN64(name))
{
  src_info.lineno = *lineno;
  src_info.name = CADR(name);
  src_info.len = CLEN(name);
  set_src_info();
}
/* 32 bit CLEN version */
void ENTCRF90IO(SRC_INFO, src_info03)(
    __INT_T *lineno, /* line number of i/o stmt in source file */
    DCHAR(name)      /* name of source file */
    DCLEN(name))
{
  ENTCRF90IO(SRC_INFOA, src_info03a)(lineno, CADR(name), (__CLEN_T)CLEN(name));
}

void ENTCRF90IO(SRC_INFOXA, src_infox03a)(
    __INT_T lineno, /* line number of i/o stmt in source file */
    DCHAR(name)     /* name of source file */
    DCLEN64(name))
{
  src_info.lineno = lineno;
  src_info.name = CADR(name);
  src_info.len = CLEN(name);
  set_src_info();
}
/* 32 bit CLEN version */
void ENTCRF90IO(SRC_INFOX, src_infox03)(
    __INT_T lineno, /* line number of i/o stmt in source file */
    DCHAR(name)     /* name of source file */
    DCLEN(name))
{
  ENTCRF90IO(SRC_INFOXA, src_infox03a)(lineno, CADR(name), (__CLEN_T)CLEN(name));
}

void ENTF90IO(SRC_INFOA, src_infoa)(
    __INT_T *lineno, /* line number of i/o stmt in source file */
    DCHAR(name)      /* name of source file */
    DCLEN64(name))
{
  src_info.lineno = *lineno;
  src_info.name = CADR(name);
  src_info.len = CLEN(name);
  fioFcbTbls.pos_present = FALSE;
}
/* 32 bit CLEN version */
void ENTF90IO(SRC_INFO, src_info)(
    __INT_T *lineno, /* line number of i/o stmt in source file */
    DCHAR(name)      /* name of source file */
    DCLEN(name))
{
  ENTF90IO(SRC_INFOA, src_infoa)(lineno, CADR(name), (__CLEN_T)CLEN(name));
}

void ENTF90IO(SRC_INFOXA, src_infoxa)(
    __INT_T lineno, /* line number of i/o stmt in source file */
    DCHAR(name)     /* name of source file */
    DCLEN64(name))
{
  src_info.lineno = lineno;
  src_info.name = CADR(name);
  src_info.len = CLEN(name);
  fioFcbTbls.pos_present = FALSE;
}
/* 32 bit CLEN version */
void ENTF90IO(SRC_INFOX, src_infox)(
    __INT_T lineno, /* line number of i/o stmt in source file */
    DCHAR(name)     /* name of source file */
    DCLEN(name))
{
  ENTF90IO(SRC_INFOXA, src_infoxa)(lineno, CADR(name), (__CLEN_T)CLEN(name));
}

void ENTCRF90IO(SRC_INFOA, src_infoa)(
    __INT_T *lineno, /* line number of i/o stmt in source file */
    DCHAR(name)      /* name of source file */
    DCLEN64(name))
{
  src_info.lineno = *lineno;
  src_info.name = CADR(name);
  src_info.len = CLEN(name);
}
/* 32 bit CLEN version */
void ENTCRF90IO(SRC_INFO, src_info)(
    __INT_T *lineno, /* line number of i/o stmt in source file */
    DCHAR(name)      /* name of source file */
    DCLEN(name))
{
  ENTCRF90IO(SRC_INFOA, src_infoa)(lineno, CADR(name), (__CLEN_T)CLEN(name));
}

void ENTCRF90IO(SRC_INFOXA, src_infoxa)(
    __INT_T lineno, /* line number of i/o stmt in source file */
    DCHAR(name)     /* name of source file */
    DCLEN64(name))
{
  src_info.lineno = lineno;
  src_info.name = CADR(name);
  src_info.len = CLEN(name);
}
/* 32 bit CLEN version */
void ENTCRF90IO(SRC_INFOX, src_infox)(
    __INT_T lineno, /* line number of i/o stmt in source file */
    DCHAR(name)     /* name of source file */
    DCLEN(name))
{
  ENTCRF90IO(SRC_INFOXA, src_infoxa)(lineno, CADR(name), (__CLEN_T)CLEN(name));
}

/* ---------------------------------------------------------------- */

static void
set_iomsg()
{
  gbl->iomsg = iomsg;
  gbl->iomsgl = iomsgl;
}

void ENTF90IO(IOMSGA, iomsga)(DCHAR(msg) DCLEN64(msg))
{
  iomsg = CADR(msg);
  iomsgl = CLEN(msg);
  set_iomsg();
}
/* 32 bit CLEN version */
void ENTF90IO(IOMSG, iomsg)(DCHAR(msg) DCLEN(msg))
{
  ENTF90IO(IOMSGA, iomsga)(CADR(msg), (__CLEN_T)CLEN(msg));
}

void ENTCRF90IO(IOMSGA, iomsga)(DCHAR(msg) DCLEN64(msg))
{
  iomsg = CADR(msg);
  iomsgl = CLEN(msg);
  set_iomsg();
}
/* 32 bit CLEN version */
void ENTCRF90IO(IOMSG, iomsg)(DCHAR(msg) DCLEN(msg))
{
  ENTCRF90IO(IOMSGA, iomsga)(CADR(msg), (__CLEN_T)CLEN(msg));
}

/* ------------------------------------------------------------------- */

#if !defined(TARGET_WIN)
#define WIN_SET_BINARY(f)
#else
#define WIN_SET_BINARY(f) win_set_binary(f)
static void
win_set_binary(FIO_FCB *f)
{
  FILE *fil;

  fil = f->fp;
  if (!__fort_isatty(__fort_getfd(fil))) {
    __io_setmode_binary(fil);
  }
}
#endif

/* ***  FORTRANOPT settings  *****/
static int check_format = 1; /* format checking enabled */
static int crlf = 0;         /* crlf does not denote end-of-line */
static int legacy_large_rec_fmt = 0; /* are legacy large unf records used */
static int no_minus_zero = 0; /* -0 allowed in formatted 0 */
static int new_fp_formatter = TRUE;

/** \brief  initialize Fortran I/O system.  Specifically, initialize
    preconnected units:  */
static void
__fortio_init(void)
{
  FIO_FCB *f;

  assert(fioFcbTbls.fcbs == NULL);

  /* preconnect stdin as unit -5 for * unit specifier */
  f = __fortio_alloc_fcb();

  f->fp = __io_stdin();
  f->unit = -5;
  f->name = strdup("stdin ");
  f->reclen = 0;
  f->wordlen = 1;
  f->nextrec = 1;
  f->status = FIO_OLD;
  f->dispose = FIO_KEEP;
  f->acc = FIO_SEQUENTIAL;
  f->action = FIO_READ;
  f->blank = FIO_NULL;
  f->form = FIO_FORMATTED;
  f->coherent = 0;
  f->skip = 0;
  f->eof_flag = FALSE;
  f->eor_flag = FALSE;
  f->named = TRUE;
  f->pad = FIO_YES;
  f->stdunit = TRUE;
  f->truncflag = FALSE;
  f->nonadvance = FALSE;
  f->ispipe = FALSE;
  f->asy_rw = 0; /* init async flags */
  f->asyptr = (void *)0;
  f->pread = 0;
  f->pback = 0;
  WIN_SET_BINARY(f);

  /* preconnect stdout as unit -6 for * unit specifier */
  f = __fortio_alloc_fcb();

  f->fp = __io_stdout();
  f->unit = -6;
  f->name = strdup("stdout ");
  f->reclen = 0;
  f->wordlen = 1;
  f->nextrec = 1;
  f->status = FIO_OLD;
  f->dispose = FIO_KEEP;
  f->acc = FIO_SEQUENTIAL;
  f->action = FIO_WRITE;
  f->blank = FIO_NULL;
  f->delim = FIO_NONE;
  f->form = FIO_FORMATTED;
  f->coherent = 0;
  f->skip = 0;
  f->eof_flag = FALSE;
  f->eor_flag = FALSE;
  f->named = TRUE;
  f->stdunit = TRUE;
  f->truncflag = FALSE;
  f->nonadvance = FALSE;
  f->ispipe = FALSE;
  f->asy_rw = 0; /* init async flags */
  f->asyptr = (void *)0;
  f->pread = 0;
  f->pback = 0;
  WIN_SET_BINARY(f);

  /* preconnect stdin as unit 5 */
  f = __fortio_alloc_fcb();

  f->fp = __io_stdin();
  f->unit = 5;
  f->name = strdup("stdin ");
  f->reclen = 0;
  f->wordlen = 1;
  f->nextrec = 1;
  f->status = FIO_OLD;
  f->dispose = FIO_KEEP;
  f->acc = FIO_SEQUENTIAL;
  f->action = FIO_READ;
  f->blank = FIO_NULL;
  f->form = FIO_FORMATTED;
  f->coherent = 0;
  f->skip = 0;
  f->eof_flag = FALSE;
  f->eor_flag = FALSE;
  f->named = TRUE;
  f->pad = FIO_YES;
  f->stdunit = TRUE;
  f->truncflag = FALSE;
  f->nonadvance = FALSE;
  f->ispipe = FALSE;
  f->asy_rw = 0; /* init async flags */
  f->asyptr = (void *)0;
  f->pread = 0;
  f->pback = 0;
  WIN_SET_BINARY(f);

  /* preconnect stdout as unit 6 */
  f = __fortio_alloc_fcb();

  f->fp = __io_stdout();
  f->unit = 6;
  f->name = strdup("stdout ");
  f->reclen = 0;
  f->wordlen = 1;
  f->nextrec = 1;
  f->status = FIO_OLD;
  f->dispose = FIO_KEEP;
  f->acc = FIO_SEQUENTIAL;
  f->action = FIO_WRITE;
  f->blank = FIO_NULL;
  f->delim = FIO_NONE;
  f->form = FIO_FORMATTED;
  f->coherent = 0;
  f->skip = 0;
  f->eof_flag = FALSE;
  f->eor_flag = FALSE;
  f->named = TRUE;
  f->stdunit = TRUE;
  f->truncflag = FALSE;
  f->nonadvance = FALSE;
  f->ispipe = FALSE;
  f->asy_rw = 0; /* init async flags */
  f->asyptr = (void *)0;
  f->pread = 0;
  f->pback = 0;
  WIN_SET_BINARY(f);

  /* preconnect stderr as unit 0 */
  f = __fortio_alloc_fcb();

  f->fp = __io_stderr();
  f->unit = 0;
  f->name = strdup("stderr ");
  f->reclen = 0;
  f->wordlen = 1;
  f->nextrec = 1;
  f->status = FIO_OLD;
  f->dispose = FIO_KEEP;
  f->acc = FIO_SEQUENTIAL;
  f->action = FIO_WRITE;
  f->blank = FIO_NULL;
  f->delim = FIO_NONE;
  f->form = FIO_FORMATTED;
  f->coherent = 0;
  f->skip = 0;
  f->eof_flag = FALSE;
  f->eor_flag = FALSE;
  f->named = TRUE;
  f->stdunit = TRUE;
  f->truncflag = FALSE;
  f->nonadvance = FALSE;
  f->ispipe = FALSE;
  f->asy_rw = 0; /* init async flags */
  f->pread = 0;
  f->pback = 0;
  f->asyptr = (void *)0;

  /* check environment variables */

  envar_fortranopt = __fort_getenv("FORTRANOPT");
  if (envar_fortranopt) {
    if (strstr(envar_fortranopt, "format_relaxed")) {
      check_format = 0;
    }
    if (strstr(envar_fortranopt, "crlf")) {
      crlf = 1;
    }
    if (strstr(envar_fortranopt, "pgi_legacy_large_rec_fmt")) {
      legacy_large_rec_fmt = 1;
    }
    if (strstr(envar_fortranopt, "no_minus_zero")) {
      no_minus_zero = 1;
    }
    if (strstr(envar_fortranopt, "no_new_fp_formatter") ||
        strstr(envar_fortranopt, "old_fp_formatter")) {
      new_fp_formatter = 0;
    } else if (strstr(envar_fortranopt, "new_fp_formatter")) {
      new_fp_formatter = 1;
    }
  }
}

int
__fortio_check_format(void)
{
  return check_format;
}

int
__fortio_eor_crlf(void)
{
  return crlf;
}

int
f90_old_huge_rec_fmt(void)
{
  return legacy_large_rec_fmt;
}

int
__fortio_no_minus_zero(void)
{
  return no_minus_zero;
}

int
__fortio_new_fp_formatter(void)
{
  return new_fp_formatter;
}

static void
set_pos()
{
  gbl->pos = fioFcbTbls.pos;
  gbl->pos_present = fioFcbTbls.pos_present;
}

void ENTF90IO(IOMSG_, iomsg_)(char *p, int n)
{
  iomsg = p;
  iomsgl = n;
}

/* ---------------------------------------------------------------- */

void ENTF90IO(AUX_INIT, aux_init)(int mask, __INT8_T pos)
{
  /*
   * More initialization depending on the value of mask; the intent
   * is to have a routine that will initialize for new features
   * that's backward's compatible.  The routine is called after
   * the call to src_info and before the I/O-specific init routine.
   */
  if (mask & 0x1) {
    fioFcbTbls.pos_present = TRUE;
    fioFcbTbls.pos = pos;
  }
  set_pos();
}
