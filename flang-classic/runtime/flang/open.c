/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Implements the Fortran OPEN statment
 */

#include <stdarg.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "global.h"
#include "open_close.h"
#include "async.h"
#include <fcntl.h>

#if defined(_WIN64)
#define access _access
#define unlink _unlink
#endif

static FIO_FCB *Fcb; /* pointer to the file control block */

int next_newunit = -13;

/* --------------------------------------------------------------------- */
int
ENTF90IO(GET_NEWUNIT, get_newunit)()
{
  set_gbl_newunit(TRUE);
  return next_newunit--;
}

/* --------------------------------------------------------------------- */

/** \brief Function called from within fiolib to open a file.
 */
int
__fortio_open(int unit, int action_flag, int status_flag, int dispose_flag,
             int acc_flag, int blank_flag, int form_flag, int delim_flag,
             int pos_flag, int pad_flag, __INT8_T reclen, char *name,
             __CLEN_T namelen)
{
  char *q;
  const char *perms;
  char bfilename[MAX_NAMELEN + 1];
  char *filename;
  int long_name;
  FILE *lcl_fp; /* local file fp */
  FIO_FCB *f;   /* local file control block ptr */
  __CLEN_T i;
  int fd;

  if (ILLEGAL_UNIT(unit))
    return __fortio_error(FIO_EUNIT);

  lcl_fp = NULL;
  f = NULL;

  /* ------- if file name specified, delete trailing blanks, copy
              into filename buffer and null-terminate it:  */

  long_name = 0;
  filename = bfilename;

#undef EXIT_OPEN
#define EXIT_OPEN(ec)                                                          \
  {                                                                            \
    if (long_name)                                                             \
      free(filename);                                                          \
    return ec;                                                                 \
  }

  if (name != NULL) {
    /*  remove trailing blanks:  */
    while (namelen > 0 && name[namelen - 1] == ' ')
      --namelen;
    /*  remove preceding blanks:  */
    while (namelen > 0 && name[0] == ' ')
      name++, namelen--;
    if (namelen <= 0)
      return __fortio_error(FIO_ENAME);

    if (namelen > MAX_NAMELEN) {
      filename = malloc(namelen + 1);
      long_name = 1;
    }
    for (i = 0; i < namelen; i++)
      filename[i] = name[i];
    filename[namelen] = '\0';
#if defined(_WIN64)
    if (filename[0] == '/' && filename[1] == '/' && filename[3] == '/') {
      /* convert posix format to win32 format */
      filename[0] = filename[2]; /* drive letter */
      filename[1] = ':';
      filename[2] = '\\';
      strcpy(filename + 3, filename + 4);
      namelen--;
    }
#endif
    /*  check that file is not already connected to different unit: */
    for (f = fioFcbTbls.fcbs; f; f = f->next)
      if (f->named && strcmp(filename, f->name) == 0)
        if (unit != f->unit)
          EXIT_OPEN(__fortio_error(FIO_EOPENED))
  }

  /* ------- handle situation in which unit is already connected:  */

  f = __fortio_find_unit(unit);

  if (f != NULL) {
    if (name == NULL || strcmp(filename, f->name) == 0) {
      /*  case 1:  file to be connected is the same:  */
      /*  make sure no specifier other than BLANK is different than
          the one currently in effect */

      if (status_flag == FIO_SCRATCH && f->status != FIO_SCRATCH)
        EXIT_OPEN(__fortio_error(FIO_ECOMPAT))
      if (acc_flag != f->acc || form_flag != f->form)
        EXIT_OPEN(__fortio_error(FIO_ECOMPAT))
      if (acc_flag == FIO_DIRECT && reclen != (f->reclen / f->wordlen))
        EXIT_OPEN(__fortio_error(FIO_ECOMPAT))

      f->blank = blank_flag;
      if (pos_flag == FIO_REWIND) {
        __io_fseek(f->fp, (seekoffx_t)0L, SEEK_SET);
      } else if (pos_flag == FIO_APPEND) {
        __io_fseek(f->fp, (seekoffx_t)0L, SEEK_END);
      }
      f->reclen = reclen * f->wordlen;

      Fcb = f;     /* save pointer to the fcb for any augmented opens */
      EXIT_OPEN(0) /* no error occurred */
    } else {
      /*  case 2:  file to be connected is NOT the same:  */
      if (__fortio_close(f, 0 /*dispose flag*/) != 0)
        EXIT_OPEN(ERR_FLAG)
    }
  }

  /* ------- create default name if none specified:  */

  if (name == NULL) {
    /*  for unnamed unit, first check environment variable */
    sprintf(filename, "FOR%03d", unit);
    if ((q = __fort_getenv(filename)) != NULL)
      strcpy(filename, q);
    else if (status_flag != FIO_SCRATCH)
      sprintf(filename, GET_FIO_CNFG_DEFAULT_NAME, unit);
    else { /*  unnamed SCRATCH file:  */
      while (1) {
        __fortio_scratch_name(filename, unit);
        fd = open(filename, O_RDWR | O_CREAT | O_TRUNC | O_EXCL, 0666);
        if (fd == -1) {
          continue;
        }
        close(fd);
        break;
      }
    }
    namelen = strlen(filename);
  }

  /* ******************************************************************
      procede with opening of new file based on value of STATUS:
      *******************************************************************/

  if (status_flag == FIO_OLD) {
    /* if OLD and doesn't exist, then error */
    if (__fort_access(filename, 0) != 0)
      EXIT_OPEN(__fortio_error(FIO_ENOEXIST))

/*  open file for readonly or read/write:  */

    perms = "r+";
    if ((action_flag == FIO_READ) ||
        (lcl_fp = __io_fopen(filename, perms)) == NULL) {
      perms = "r";
      if ((lcl_fp = (__io_fopen(filename, perms))) == NULL)
        EXIT_OPEN(__fortio_error(__io_errno()))
    }
  } else if (status_flag == FIO_NEW) {
    /* if NEW and exists then error */
    if (__fort_access(filename, 0) == 0)
      EXIT_OPEN(__fortio_error(FIO_EEXIST))

    perms = "w+";
    if ((lcl_fp = __io_fopen(filename, perms)) == NULL)
      EXIT_OPEN(__fortio_error(__io_errno()))
  } else if (status_flag == FIO_REPLACE) {
/* if file does not exist, create a file;
 * if file exists, delete the file and create a new file.
 */
    perms = "w+";
    if ((lcl_fp = __io_fopen(filename, perms)) == NULL)
      EXIT_OPEN(__fortio_error(__io_errno()))
  } else if (status_flag == FIO_UNKNOWN) {
    i = 0;
    if (__fort_access(filename, 0) == 0) { /* file exists */
      perms = "r+";
      i = 1;
    } else /* file does not exist */
      perms = "w+";

    if ((lcl_fp = __io_fopen(filename, perms)) == NULL) {
      if (i == 0) /* file does not exist */
        EXIT_OPEN(__fortio_error(__io_errno()))
/*  try again with different mode:  */
      perms = "r";
      if ((lcl_fp = __io_fopen(filename, perms)) == NULL)
        EXIT_OPEN(__fortio_error(__io_errno()))
    }
  } else {
    assert(status_flag == FIO_SCRATCH);
    perms = "w+";
    if ((lcl_fp = __io_fopen(filename, perms)) == NULL) {
      EXIT_OPEN(__fortio_error(__io_errno()))
    }
    __fort_unlink(filename);
  }

  /* ****************************************************************
      allocate entry for file just opened and assign the
      characteristics to the file:
      ***************************************************************/

  f = __fortio_alloc_fcb();

  f->fp = lcl_fp;
  assert(lcl_fp != NULL);
  f->unit = unit;
  f->action = action_flag;
  f->status = FIO_OLD;
  if (status_flag == FIO_SCRATCH)
    f->status = FIO_SCRATCH;
  f->delim = delim_flag;
  f->dispose = dispose_flag;
  f->blank = blank_flag;
  f->form = form_flag;
  f->pad = pad_flag;
  f->pos = pos_flag;
  f->wordlen = 1; /* default */
  if (f->form == FIO_UNFORMATTED) {
    if (envar_fortranopt != NULL && strstr(envar_fortranopt, "vaxio") != NULL)
      f->wordlen = 4; /* WHAT */
  }
  f->reclen = reclen * f->wordlen;
  f->nextrec = 1;
  f->truncflag = FALSE;
  f->skip = 0;
  f->ispipe = FALSE;
  f->nonadvance = FALSE;
  f->pread = 0;
  f->pback = 0;

  if (acc_flag == FIO_DIRECT) {
    /*  compute number of records in direct access file:  */
    f->acc = FIO_DIRECT;
    f->maxrec = 0;
    if (status_flag == FIO_OLD || status_flag == FIO_UNKNOWN) {
      seekoffx_t len;
      if (__io_fseekx(lcl_fp, (seekoffx_t)0L, SEEK_END) != 0)
        goto free_fcb_err;
      len = (seekoffx_t)__io_ftellx(lcl_fp);
      f->maxrec = len / f->reclen;
      __io_fseek(lcl_fp, (seekoffx_t)0L,
                  SEEK_SET); /* re-position to beginning */
    }
  } else {
    if (acc_flag == FIO_STREAM)
      f->acc = FIO_STREAM;
    else
      f->acc = FIO_SEQUENTIAL;
    if ((status_flag == FIO_OLD || status_flag == FIO_UNKNOWN) &&
        pos_flag != FIO_APPEND)
      f->truncflag = TRUE;
    if (status_flag != FIO_SCRATCH && __fortio_ispipe(f->fp)) {
      f->truncflag = FALSE;
      f->ispipe = TRUE;
    } else if (pos_flag == FIO_APPEND) /* position file at end of file */
      if (__io_fseek(lcl_fp, (seekoffx_t)0L, SEEK_END) != 0)
        goto free_fcb_err;
  }

  if (status_flag != FIO_SCRATCH)
    f->named = TRUE;
  else
    f->named = FALSE;
  f->name = STASH(filename);
  f->coherent = 0;
  f->eof_flag = FALSE;
  f->eor_flag = FALSE;
  f->stdunit = FALSE;
  f->byte_swap = FALSE;
  f->native = FALSE;
  f->binary = FALSE;
  f->asy_rw = 0; /* init async flags */
  f->asyptr = (void *)0;
  f->decimal = FIO_POINT;
  f->encoding = FIO_DEFAULT;
  f->round = FIO_COMPATIBLE;
  f->sign = FIO_PROCESSOR_DEFINED;
  Fcb = f; /* save pointer to the fcb for any augmented opens */

  EXIT_OPEN(0) /* no error occurred */

free_fcb_err:
  __fortio_free_fcb(f); /*  free up FCB for later use  */
  EXIT_OPEN(__fortio_error(__io_errno()))
}

/* --------------------------------------------------------------------- */
/* internal open */

static int
f90_open(__INT_T *unit, __INT_T *bitv, char *acc_ptr, char *action_ptr,
         char *blank_ptr, char *delim_ptr, char *name_ptr, char *form_ptr,
         __INT_T *iostat, char *pad_ptr, char *pos_ptr, __INT8_T *reclen,
         char *status_ptr, char *dispose_ptr, __CLEN_T acc_siz, __CLEN_T action_siz,
         __CLEN_T blank_siz, __CLEN_T delim_siz, __CLEN_T name_siz, __CLEN_T form_siz, __CLEN_T pad_siz,
         __CLEN_T pos_siz, __CLEN_T status_siz, __CLEN_T dispose_siz)
{
  int acc_flag, action_flag, delim_flag, form_flag, blank_flag;
  int pad_flag, pos_flag, status_flag, dispose_flag;
  __INT8_T tmpreclen;
  int retv;
  bool binary;

  __fortio_errinit03(*unit, *bitv, iostat, "OPEN");

  if (name_ptr != NULL) {
    fioFcbTbls.fname = name_ptr;
    fioFcbTbls.fnamelen = name_siz;
  } else {
    fioFcbTbls.fname = NULL;
    fioFcbTbls.fnamelen = 0;
  }
  binary = FALSE;

  /* -------- check specifiers and set flags appropriately:  */

  /* ACCESS: */

  pos_flag = FIO_ASIS;       /* default to handle "APPEND" */
  acc_flag = FIO_SEQUENTIAL; /*  default value  */
  if (acc_ptr != NULL) {
    if (__fortio_eq_str(acc_ptr, acc_siz, "DIRECT")) {
      acc_flag = FIO_DIRECT;
    } else if (__fortio_eq_str(acc_ptr, acc_siz, "STREAM")) {
      acc_flag = FIO_STREAM;
    } else if (__fortio_eq_str(acc_ptr, acc_siz, "SEQUENTIAL")) {
      acc_flag = FIO_SEQUENTIAL;
    } else if (__fortio_eq_str(acc_ptr, acc_siz, "APPEND")) {
      pos_flag = FIO_APPEND;
    } else {
      return __fortio_error(FIO_ESPEC);
    }
  }

  /* ACTION: */

  action_flag = FIO_READWRITE; /*  default value  */
  if (action_ptr != NULL) {
    if (__fortio_eq_str(action_ptr, action_siz, "READ"))
      action_flag = FIO_READ;
    else if (__fortio_eq_str(action_ptr, action_siz, "WRITE"))
      action_flag = FIO_WRITE;
    else if (__fortio_eq_str(action_ptr, action_siz, "READWRITE"))
      action_flag = FIO_READWRITE;
    else
      return __fortio_error(FIO_ESPEC);
  }

  /* FORM: */

  if (form_ptr != NULL) {
    if (__fortio_eq_str(form_ptr, form_siz, "FORMATTED"))
      form_flag = FIO_FORMATTED;
    else if (__fortio_eq_str(form_ptr, form_siz, "UNFORMATTED"))
      form_flag = FIO_UNFORMATTED;
    else if (__fortio_eq_str(form_ptr, form_siz, "BINARY")) {
      form_flag = FIO_UNFORMATTED;
      binary = TRUE;
    } else
      return __fortio_error(FIO_ESPEC);
  } else if (acc_flag == FIO_DIRECT || acc_flag == FIO_STREAM)
    form_flag = FIO_UNFORMATTED;
  else
    form_flag = FIO_FORMATTED;

  /* DELIM: */

  delim_flag = FIO_NONE; /*  default value  */
  if (delim_ptr != NULL) {
    if (form_flag != FIO_FORMATTED)
      return __fortio_error(FIO_ECOMPAT);
    if (__fortio_eq_str(delim_ptr, delim_siz, "APOSTROPHE"))
      delim_flag = FIO_APOSTROPHE;
    else if (__fortio_eq_str(delim_ptr, delim_siz, "QUOTE"))
      delim_flag = FIO_QUOTE;
    else if (__fortio_eq_str(delim_ptr, delim_siz, "NONE"))
      delim_flag = FIO_NONE;
    else
      return __fortio_error(FIO_ESPEC);
  }

  /* BLANK: */

  blank_flag = FIO_NULL; /*  default value */
  if (blank_ptr != NULL) {
    /*  file must be connected for formatted I/O:  */
    if (form_flag != FIO_FORMATTED)
      return __fortio_error(FIO_ECOMPAT);
    if (__fortio_eq_str(blank_ptr, blank_siz, "ZERO"))
      blank_flag = FIO_ZERO;
    else if (!__fortio_eq_str(blank_ptr, blank_siz, "NULL"))
      return __fortio_error(FIO_ESPEC);
  }

  /* PAD: */

  pad_flag = FIO_YES; /* default */
  if (pad_ptr != NULL) {
    if (form_flag != FIO_FORMATTED)
      return __fortio_error(FIO_ECOMPAT);
    if (__fortio_eq_str(pad_ptr, pad_siz, "YES"))
      pad_flag = FIO_YES;
    else if (__fortio_eq_str(pad_ptr, pad_siz, "NO"))
      pad_flag = FIO_NO;
    else
      return __fortio_error(FIO_ESPEC);
  }

  /* POSITION: */

  /* moved to ACCESS to handle ACCESS=APPEND
      pos_flag = FIO_ASIS;
  */
  if (pos_ptr != NULL) {
    if (acc_flag != FIO_SEQUENTIAL && acc_flag != FIO_STREAM)
      return __fortio_error(FIO_ECOMPAT);
    if (__fortio_eq_str(pos_ptr, pos_siz, "ASIS"))
      pos_flag = FIO_ASIS;
    else if (__fortio_eq_str(pos_ptr, pos_siz, "REWIND"))
      pos_flag = FIO_REWIND;
    else if (__fortio_eq_str(pos_ptr, pos_siz, "APPEND"))
      pos_flag = FIO_APPEND;
    else
      return __fortio_error(FIO_ESPEC);
  }

  /* STATUS: */

  status_flag = FIO_UNKNOWN; /* default */
  if (status_ptr != NULL) {
    if (__fortio_eq_str(status_ptr, status_siz, "OLD"))
      status_flag = FIO_OLD;
    else if (__fortio_eq_str(status_ptr, status_siz, "NEW"))
      status_flag = FIO_NEW;
    else if (__fortio_eq_str(status_ptr, status_siz, "REPLACE"))
      status_flag = FIO_REPLACE;
    else if (__fortio_eq_str(status_ptr, status_siz, "UNKNOWN"))
      status_flag = FIO_UNKNOWN;
    else if (__fortio_eq_str(status_ptr, status_siz, "SCRATCH"))
      status_flag = FIO_SCRATCH;
    else
      return __fortio_error(FIO_ESPEC);
  }

  /* DISPOSE: */

  if (dispose_ptr != NULL) {
    if (__fortio_eq_str(dispose_ptr, dispose_siz, "KEEP"))
      dispose_flag = FIO_KEEP;
    else if (__fortio_eq_str(dispose_ptr, dispose_siz, "SAVE"))
      dispose_flag = FIO_KEEP;
    else if (__fortio_eq_str(dispose_ptr, dispose_siz, "DELETE"))
      dispose_flag = FIO_DELETE;
    else
      return __fortio_error(FIO_ESPEC);
  } else if (status_flag == FIO_SCRATCH)
    dispose_flag = FIO_DELETE;
  else
    dispose_flag = FIO_KEEP;

  /* ------------- Check for compatibility between specifiers:  */

  if (get_gbl_newunit() && name_ptr == NULL && status_flag != FIO_SCRATCH) {
    return __fortio_error(FIO_ENEWUNIT);
  }

  if (acc_flag == FIO_DIRECT) {
    tmpreclen = 0;
    if (reclen)
      tmpreclen = *reclen;
    if (tmpreclen < 1)
      return __fortio_error(FIO_ERECLEN);
  } else if (acc_flag == FIO_SEQUENTIAL && reclen) {
    tmpreclen = *reclen;
    if (tmpreclen < 1)
      return __fortio_error(FIO_ERECLEN);
  } else {
    /*  VMS allows RECL with non-DIRECT access.  Just ignore reclen.  */
    tmpreclen = 0;
    if (reclen) {
      if (acc_flag == FIO_STREAM)
        return __fortio_error(FIO_ERECL);
    }
  }

  if (status_flag == FIO_SCRATCH) {
    if (dispose_flag == FIO_KEEP)
      return __fortio_error(FIO_EDISPOSE);
    if (fioFcbTbls.fname != NULL)
      return __fortio_error(FIO_ESCRATCH);
  }

  if (action_flag == FIO_READ) {
    if (status_flag == FIO_SCRATCH || status_flag == FIO_REPLACE ||
        dispose_flag == FIO_DELETE)
      return __fortio_error(FIO_EREADONLY);
  }

  if ((acc_flag == FIO_STREAM) && (form_flag == FIO_UNFORMATTED))
    binary = TRUE;

  /* ---------  call __fortio_open to complete process of opening file:   */

  retv = __fortio_open(*unit, action_flag, status_flag, dispose_flag, acc_flag,
                      blank_flag, form_flag, delim_flag, pos_flag, pad_flag,
                      tmpreclen, fioFcbTbls.fname, fioFcbTbls.fnamelen);
  if (!retv && binary) {
    if (acc_flag == FIO_DIRECT)
      retv = __fortio_error(FIO_ESPEC);
    else
      Fcb->binary = TRUE;
  }

  return retv;
}

/* --------------------------------------------------------------------- */

/** \brief Called from user program; implements Fortran OPEN statement.
 */
__INT_T
ENTF90IO(OPENA, opena) (
  __INT_T *unit,   /* unit number */
  __INT_T *bitv,   /* determines action if error occurs */
  DCHAR(acc),      /* DIRECT, SEQUENTIAL, or NULL */
  DCHAR(action),   /* READ, WRITE, READWRITE, or NULL */
  DCHAR(blank),    /* ZERO or NULL */
  DCHAR(delim),    /* APOSTROPHE, QUOTE, NONE, or NULL */
  DCHAR(name),     /* file name */
  DCHAR(form),     /* FORMATTED, UNFORMATTED, or NULL */
  __INT_T *iostat, /* IOSTAT variable */
  DCHAR(pad),      /* YES, NO, or NULL */
  DCHAR(pos),      /* ASIS, REWIND, APPEND, or NULL */
  __INT_T *reclen, /* record length in bytes or words */
  DCHAR(status),   /* OLD, NEW, SCRATCH, REPLACE, or NULL */
  DCHAR(dispose)   /* KEEP, DELETE, SAVE, or NULL */
  DCLEN64(acc)       /* length of acc */
  DCLEN64(action)    /* length of action */
  DCLEN64(blank)     /* length of blank */
  DCLEN64(delim)     /* length of delim */
  DCLEN64(name)      /* length of name */
  DCLEN64(form)      /* length of form */
  DCLEN64(pad)       /* length of pad */
  DCLEN64(pos)       /* length of pos */
  DCLEN64(status)    /* length of status */
  DCLEN64(dispose))  /* length of dispose */
{
  char *acc_ptr;
  char *action_ptr;
  char *blank_ptr;
  char *delim_ptr;
  char *name_ptr;
  char *form_ptr;
  char *pad_ptr;
  char *pos_ptr;
  char *status_ptr;
  char *dispose_ptr;
  __CLEN_T acc_siz;
  __CLEN_T action_siz;
  __CLEN_T blank_siz;
  __CLEN_T delim_siz;
  __CLEN_T name_siz;
  __CLEN_T form_siz;
  __CLEN_T pad_siz;
  __CLEN_T pos_siz;
  __CLEN_T status_siz;
  __CLEN_T dispose_siz;
  __INT8_T newreclen;

  int s = 0;

  acc_ptr = (ISPRESENTC(acc) ? CADR(acc) : NULL);
  action_ptr = (ISPRESENTC(action) ? CADR(action) : NULL);
  blank_ptr = (ISPRESENTC(blank) ? CADR(blank) : NULL);
  delim_ptr = (ISPRESENTC(delim) ? CADR(delim) : NULL);
  name_ptr = (ISPRESENTC(name) ? CADR(name) : NULL);
  form_ptr = (ISPRESENTC(form) ? CADR(form) : NULL);
  pad_ptr = (ISPRESENTC(pad) ? CADR(pad) : NULL);
  pos_ptr = (ISPRESENTC(pos) ? CADR(pos) : NULL);
  status_ptr = (ISPRESENTC(status) ? CADR(status) : NULL);
  dispose_ptr = (ISPRESENTC(dispose) ? CADR(dispose) : NULL);
  acc_siz = CLEN(acc);
  action_siz = CLEN(action);
  blank_siz = CLEN(blank);
  delim_siz = CLEN(delim);
  name_siz = CLEN(name);
  form_siz = CLEN(form);
  pad_siz = CLEN(pad);
  pos_siz = CLEN(pos);
  status_siz = CLEN(status);
  dispose_siz = CLEN(dispose);

  __fort_status_init(bitv, iostat);
  newreclen = (__INT8_T)*reclen;
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = f90_open(unit, /* do real open (finally) */
                 bitv, acc_ptr, action_ptr, blank_ptr, delim_ptr, name_ptr,
                 form_ptr, iostat, pad_ptr, pos_ptr, &newreclen, status_ptr,
                 dispose_ptr, acc_siz, action_siz, blank_siz, delim_siz,
                 name_siz, form_siz, pad_siz, pos_siz, status_siz, dispose_siz);
  *reclen = (int)newreclen;
  __fortio_errend03();
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(OPEN, open) (
  __INT_T *unit,   /* unit number */
  __INT_T *bitv,   /* determines action if error occurs */
  DCHAR(acc),      /* DIRECT, SEQUENTIAL, or NULL */
  DCHAR(action),   /* READ, WRITE, READWRITE, or NULL */
  DCHAR(blank),    /* ZERO or NULL */
  DCHAR(delim),    /* APOSTROPHE, QUOTE, NONE, or NULL */
  DCHAR(name),     /* file name */
  DCHAR(form),     /* FORMATTED, UNFORMATTED, or NULL */
  __INT_T *iostat, /* IOSTAT variable */
  DCHAR(pad),      /* YES, NO, or NULL */
  DCHAR(pos),      /* ASIS, REWIND, APPEND, or NULL */
  __INT_T *reclen, /* record length in bytes or words */
  DCHAR(status),   /* OLD, NEW, SCRATCH, REPLACE, or NULL */
  DCHAR(dispose)   /* KEEP, DELETE, SAVE, or NULL */
  DCLEN(acc)       /* length of acc */
  DCLEN(action)    /* length of action */
  DCLEN(blank)     /* length of blank */
  DCLEN(delim)     /* length of delim */
  DCLEN(name)      /* length of name */
  DCLEN(form)      /* length of form */
  DCLEN(pad)       /* length of pad */
  DCLEN(pos)       /* length of pos */
  DCLEN(status)    /* length of status */
  DCLEN(dispose))  /* length of dispose */
{
  return ENTF90IO(OPENA, opena) (unit, bitv, CADR(acc), CADR(action),
		  CADR(blank), CADR(delim), CADR(name), CADR(form), iostat,
		  CADR(pad), CADR(pos), reclen, CADR(status), CADR(dispose),
		  (__CLEN_T)CLEN(acc), (__CLEN_T)CLEN(action),
		  (__CLEN_T)CLEN(blank), (__CLEN_T)CLEN(delim),
		  (__CLEN_T)CLEN(name), (__CLEN_T)CLEN(form),
		  (__CLEN_T)CLEN(pad), (__CLEN_T)CLEN(pos),
		  (__CLEN_T)CLEN(status), (__CLEN_T)CLEN(dispose));
}

/* --------------------------------------------------------------------- */

__INT_T
ENTF90IO(OPEN2003A, open2003a)(
  __INT_T *unit,    /* unit number */
  __INT_T *bitv,    /* determines action if error occurs */
  DCHAR(acc),       /* DIRECT, SEQUENTIAL, or NULL */
  DCHAR(action),    /* READ, WRITE, READWRITE, or NULL */
  DCHAR(blank),     /* ZERO or NULL */
  DCHAR(delim),     /* APOSTROPHE, QUOTE, NONE, or NULL */
  DCHAR(name),      /* file name */
  DCHAR(form),      /* FORMATTED, UNFORMATTED, or NULL */
  __INT_T *iostat,  /* IOSTAT variable */
  DCHAR(pad),       /* YES, NO, or NULL */
  DCHAR(pos),       /* ASIS, REWIND, APPEND, or NULL */
  __INT8_T *reclen, /* record length in bytes or words */
  DCHAR(status),    /* OLD, NEW, SCRATCH, REPLACE, or NULL */
  DCHAR(dispose)    /* KEEP, DELETE, SAVE, or NULL */
  DCLEN64(acc)        /* length of acc */
  DCLEN64(action)     /* length of action */
  DCLEN64(blank)      /* length of blank */
  DCLEN64(delim)      /* length of delim */
  DCLEN64(name)       /* length of name */
  DCLEN64(form)       /* length of form */
  DCLEN64(pad)        /* length of pad */
  DCLEN64(pos)        /* length of pos */
  DCLEN64(status)     /* length of status */
  DCLEN64(dispose))   /* length of dispose */
{
  char *acc_ptr;
  char *action_ptr;
  char *blank_ptr;
  char *delim_ptr;
  char *name_ptr;
  char *form_ptr;
  char *pad_ptr;
  char *pos_ptr;
  char *status_ptr;
  char *dispose_ptr;
  __CLEN_T acc_siz;
  __CLEN_T action_siz;
  __CLEN_T blank_siz;
  __CLEN_T delim_siz;
  __CLEN_T name_siz;
  __CLEN_T form_siz;
  __CLEN_T pad_siz;
  __CLEN_T pos_siz;
  __CLEN_T status_siz;
  __CLEN_T dispose_siz;

  int s = 0;

  acc_ptr = (ISPRESENTC(acc) ? CADR(acc) : NULL);
  action_ptr = (ISPRESENTC(action) ? CADR(action) : NULL);
  blank_ptr = (ISPRESENTC(blank) ? CADR(blank) : NULL);
  delim_ptr = (ISPRESENTC(delim) ? CADR(delim) : NULL);
  name_ptr = (ISPRESENTC(name) ? CADR(name) : NULL);
  form_ptr = (ISPRESENTC(form) ? CADR(form) : NULL);
  pad_ptr = (ISPRESENTC(pad) ? CADR(pad) : NULL);
  pos_ptr = (ISPRESENTC(pos) ? CADR(pos) : NULL);
  status_ptr = (ISPRESENTC(status) ? CADR(status) : NULL);
  dispose_ptr = (ISPRESENTC(dispose) ? CADR(dispose) : NULL);
  acc_siz = CLEN(acc);
  action_siz = CLEN(action);
  blank_siz = CLEN(blank);
  delim_siz = CLEN(delim);
  name_siz = CLEN(name);
  form_siz = CLEN(form);
  pad_siz = CLEN(pad);
  pos_siz = CLEN(pos);
  status_siz = CLEN(status);
  dispose_siz = CLEN(dispose);
  if (!ISPRESENT(reclen)) {
    reclen = NULL;
  }

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = f90_open(unit, /* do real open (finally) */
                 bitv, acc_ptr, action_ptr, blank_ptr, delim_ptr, name_ptr,
                 form_ptr, iostat, pad_ptr, pos_ptr, reclen, status_ptr,
                 dispose_ptr, acc_siz, action_siz, blank_siz, delim_siz,
                 name_siz, form_siz, pad_siz, pos_siz, status_siz, dispose_siz);
  __fortio_errend03();
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(OPEN2003, open2003)(
  __INT_T *unit,    /* unit number */
  __INT_T *bitv,    /* determines action if error occurs */
  DCHAR(acc),       /* DIRECT, SEQUENTIAL, or NULL */
  DCHAR(action),    /* READ, WRITE, READWRITE, or NULL */
  DCHAR(blank),     /* ZERO or NULL */
  DCHAR(delim),     /* APOSTROPHE, QUOTE, NONE, or NULL */
  DCHAR(name),      /* file name */
  DCHAR(form),      /* FORMATTED, UNFORMATTED, or NULL */
  __INT_T *iostat,  /* IOSTAT variable */
  DCHAR(pad),       /* YES, NO, or NULL */
  DCHAR(pos),       /* ASIS, REWIND, APPEND, or NULL */
  __INT8_T *reclen, /* record length in bytes or words */
  DCHAR(status),    /* OLD, NEW, SCRATCH, REPLACE, or NULL */
  DCHAR(dispose)    /* KEEP, DELETE, SAVE, or NULL */
  DCLEN(acc)        /* length of acc */
  DCLEN(action)     /* length of action */
  DCLEN(blank)      /* length of blank */
  DCLEN(delim)      /* length of delim */
  DCLEN(name)       /* length of name */
  DCLEN(form)       /* length of form */
  DCLEN(pad)        /* length of pad */
  DCLEN(pos)        /* length of pos */
  DCLEN(status)     /* length of status */
  DCLEN(dispose))   /* length of dispose */
{
  return ENTF90IO(OPEN2003A, open2003a)(unit, bitv, CADR(acc), CADR(action),
		  CADR(blank), CADR(delim), CADR(name), CADR(form), iostat,
		  CADR(pad), CADR(pos), reclen, CADR(status), CADR(dispose),
		  (__CLEN_T)CLEN(acc), (__CLEN_T)CLEN(action),
		  (__CLEN_T)CLEN(blank), (__CLEN_T)CLEN(delim),
		  (__CLEN_T)CLEN(name), (__CLEN_T)CLEN(form),
		  (__CLEN_T)CLEN(pad), (__CLEN_T)CLEN(pos),
		  (__CLEN_T)CLEN(status), (__CLEN_T)CLEN(dispose));
}

/** \brief Called from user program; augments the OPEN with CONVERT specifier.
 */
__INT_T
ENTF90IO(OPEN_CVTA, open_cvta)
(__INT_T *istat, /* status of OPEN */
 DCHAR(endian)   /* BIG_ENDIAN or LITTLE_ENDIAN */
 DCLEN64(endian))  /* length of endian */
{
  int s = *istat;

  if (s)
    return DIST_STATUS_BCST(s);

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC) {

    if (Fcb->form != FIO_UNFORMATTED)
      s = __fortio_error(FIO_ECOMPAT);
    else if (__fortio_eq_str(CADR(endian), CLEN(endian), "BIG_ENDIAN")) {
      Fcb->byte_swap = TRUE;
    } else if (__fortio_eq_str(CADR(endian), CLEN(endian), "LITTLE_ENDIAN")) {
      Fcb->native = TRUE;
    } else if (__fortio_eq_str(CADR(endian), CLEN(endian), "NATIVE")) {
      Fcb->native = TRUE;
    } else
      s = __fortio_error(FIO_ESPEC);
  }
  __fortio_errend03();
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(OPEN_CVT, open_cvt)
(__INT_T *istat, /* status of OPEN */
 DCHAR(endian)   /* BIG_ENDIAN or LITTLE_ENDIAN */
 DCLEN(endian))  /* length of endian */
{
  return ENTF90IO(OPEN_CVTA, open_cvta) (istat, CADR(endian), (__CLEN_T)CLEN(endian));
}

__INT_T
ENTF90IO(OPEN_SHAREA, open_sharea)
(__INT_T *istat, /* status of OPEN */
 DCHAR(shv)      /* BIG_ENDIAN or LITTLE_ENDIAN */
 DCLEN64(shv))     /* length */
{
  int s = *istat;

  if (s)
    return DIST_STATUS_BCST(s);

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC) {
    if (__fortio_eq_str(CADR(shv), CLEN(shv), "SHARED")) {
      ;
    }
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(OPEN_SHARE, open_share)
(__INT_T *istat, /* status of OPEN */
 DCHAR(shv)      /* BIG_ENDIAN or LITTLE_ENDIAN */
 DCLEN(shv))     /* length */
{
  return ENTF90IO(OPEN_SHAREA, open_sharea) (istat, CADR(shv), (__CLEN_T)CLEN(shv));
}

__INT_T
ENTCRF90IO(OPENA, opena)( 
  __INT_T *unit,   /* unit number */
  __INT_T *bitv,   /* determines action if error occurs */
  DCHAR(acc),      /* DIRECT, SEQUENTIAL, or NULL */
  DCHAR(action),   /* READ, WRITE, READWRITE, or NULL */
  DCHAR(blank),    /* ZERO or NULL */
  DCHAR(delim),    /* APOSTROPHE, QUOTE, NONE, or NULL */
  DCHAR(name),     /* file name */
  DCHAR(form),     /* FORMATTED, UNFORMATTED, or NULL */
  __INT_T *iostat, /* IOSTAT variable */
  DCHAR(pad),      /* YES, NO, or NULL */
  DCHAR(pos),      /* ASIS, REWIND, APPEND, or NULL */
  __INT_T *reclen, /* record length in bytes or words */
  DCHAR(status),   /* OLD, NEW, SCRATCH, REPLACE, or NULL */
  DCHAR(dispose)   /* KEEP, DELETE, SAVE, or NULL */
  DCLEN64(acc)       /* length of acc */
  DCLEN64(action)    /* length of action */
  DCLEN64(blank)     /* length of blank */
  DCLEN64(delim)     /* length of delim */
  DCLEN64(name)      /* length of name */
  DCLEN64(form)      /* length of form */
  DCLEN64(pad)       /* length of pad */
  DCLEN64(pos)       /* length of pos */
  DCLEN64(status)    /* length of status */
  DCLEN64(dispose))  /* length of dispose */
{
  char *acc_ptr;
  char *action_ptr;
  char *blank_ptr;
  char *delim_ptr;
  char *name_ptr;
  char *form_ptr;
  char *pad_ptr;
  char *pos_ptr;
  char *status_ptr;
  char *dispose_ptr;
  __CLEN_T acc_siz;
  __CLEN_T action_siz;
  __CLEN_T blank_siz;
  __CLEN_T delim_siz;
  __CLEN_T name_siz;
  __CLEN_T form_siz;
  __CLEN_T pad_siz;
  __CLEN_T pos_siz;
  __CLEN_T status_siz;
  __CLEN_T dispose_siz;
  int s = 0;
  __INT8_T newreclen;

  acc_ptr = (ISPRESENTC(acc) ? CADR(acc) : NULL);
  action_ptr = (ISPRESENTC(action) ? CADR(action) : NULL);
  blank_ptr = (ISPRESENTC(blank) ? CADR(blank) : NULL);
  delim_ptr = (ISPRESENTC(delim) ? CADR(delim) : NULL);
  name_ptr = (ISPRESENTC(name) ? CADR(name) : NULL);
  form_ptr = (ISPRESENTC(form) ? CADR(form) : NULL);
  pad_ptr = (ISPRESENTC(pad) ? CADR(pad) : NULL);
  pos_ptr = (ISPRESENTC(pos) ? CADR(pos) : NULL);
  status_ptr = (ISPRESENTC(status) ? CADR(status) : NULL);
  dispose_ptr = (ISPRESENTC(dispose) ? CADR(dispose) : NULL);
  acc_siz = CLEN(acc);
  action_siz = CLEN(action);
  blank_siz = CLEN(blank);
  delim_siz = CLEN(delim);
  name_siz = CLEN(name);
  form_siz = CLEN(form);
  pad_siz = CLEN(pad);
  pos_siz = CLEN(pos);
  status_siz = CLEN(status);
  dispose_siz = CLEN(dispose);

  newreclen = (__INT8_T)*reclen;
  s = f90_open(unit, /* do real open (finally) */
               bitv, acc_ptr, action_ptr, blank_ptr, delim_ptr, name_ptr,
               form_ptr, iostat, pad_ptr, pos_ptr, &newreclen, status_ptr,
               dispose_ptr, acc_siz, action_siz, blank_siz, delim_siz, name_siz,
               form_siz, pad_siz, pos_siz, status_siz, dispose_siz);
  __fortio_errend03();
  return s;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(OPEN, open)( 
  __INT_T *unit,   /* unit number */
  __INT_T *bitv,   /* determines action if error occurs */
  DCHAR(acc),      /* DIRECT, SEQUENTIAL, or NULL */
  DCHAR(action),   /* READ, WRITE, READWRITE, or NULL */
  DCHAR(blank),    /* ZERO or NULL */
  DCHAR(delim),    /* APOSTROPHE, QUOTE, NONE, or NULL */
  DCHAR(name),     /* file name */
  DCHAR(form),     /* FORMATTED, UNFORMATTED, or NULL */
  __INT_T *iostat, /* IOSTAT variable */
  DCHAR(pad),      /* YES, NO, or NULL */
  DCHAR(pos),      /* ASIS, REWIND, APPEND, or NULL */
  __INT_T *reclen, /* record length in bytes or words */
  DCHAR(status),   /* OLD, NEW, SCRATCH, REPLACE, or NULL */
  DCHAR(dispose)   /* KEEP, DELETE, SAVE, or NULL */
  DCLEN(acc)       /* length of acc */
  DCLEN(action)    /* length of action */
  DCLEN(blank)     /* length of blank */
  DCLEN(delim)     /* length of delim */
  DCLEN(name)      /* length of name */
  DCLEN(form)      /* length of form */
  DCLEN(pad)       /* length of pad */
  DCLEN(pos)       /* length of pos */
  DCLEN(status)    /* length of status */
  DCLEN(dispose))  /* length of dispose */
{
  return ENTCRF90IO(OPENA, opena)(unit, bitv, CADR(acc), CADR(action),
		    CADR(blank), CADR(delim), CADR(name), CADR(form), iostat,
		    CADR(pad), CADR(pos), reclen, CADR(status), CADR(dispose),
		    (__CLEN_T)CLEN(acc), (__CLEN_T)CLEN(action),
		    (__CLEN_T)CLEN(blank), (__CLEN_T)CLEN(delim),
		    (__CLEN_T)CLEN(name), (__CLEN_T)CLEN(form),
		    (__CLEN_T)CLEN(pad), (__CLEN_T)CLEN(pos),
		    (__CLEN_T)CLEN(status), (__CLEN_T)CLEN(dispose));
}

__INT_T
ENTCRF90IO(OPEN_CVTA, open_cvta)(
  __INT_T *istat, /* status of OPEN */
  DCHAR(endian)   /* BIG_ENDIAN or LITTLE_ENDIAN */
  DCLEN64(endian))  /* length of endian */
{
  if (*istat)
    return *istat;

  if (Fcb->form != FIO_UNFORMATTED)
    return __fortio_error(FIO_ECOMPAT);

  if (__fortio_eq_str(CADR(endian), CLEN(endian), "BIG_ENDIAN")) {
    Fcb->byte_swap = TRUE;
  } else if (__fortio_eq_str(CADR(endian), CLEN(endian), "LITTLE_ENDIAN")) {
    Fcb->native = TRUE;
  } else if (__fortio_eq_str(CADR(endian), CLEN(endian), "NATIVE")) {
    Fcb->native = TRUE;
  } else
    return __fortio_error(FIO_ESPEC);

  return 0;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(OPEN_CVT, open_cvt)(
  __INT_T *istat, /* status of OPEN */
  DCHAR(endian)   /* BIG_ENDIAN or LITTLE_ENDIAN */
  DCLEN(endian))  /* length of endian */
{
  return ENTCRF90IO(OPEN_CVTA, open_cvta)(istat, CADR(endian), (__CLEN_T)CLEN(endian));
}

__INT_T
ENTCRF90IO(OPEN_SHAREA, open_sharea)(
  __INT_T *istat, /* status of OPEN */
  DCHAR(shv)      /* BIG_ENDIAN or LITTLE_ENDIAN */
  DCLEN64(shv))     /* length */
{
  int s = *istat;

  if (s)
    return DIST_STATUS_BCST(s);

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC) {
    if (__fortio_eq_str(CADR(shv), CLEN(shv), "SHARED")) {
      ;
    }
  }
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(OPEN_SHARE, open_share)(
  __INT_T *istat, /* status of OPEN */
  DCHAR(shv)      /* BIG_ENDIAN or LITTLE_ENDIAN */
  DCLEN(shv))     /* length */
{
  return ENTCRF90IO(OPEN_SHAREA, open_sharea)(istat, CADR(shv), (__CLEN_T)CLEN(shv));
}

/* handle asyncronous open parameter, called after open */

int
ENTF90IO(OPEN_ASYNCA, open_asynca)(__INT_T *istat, DCHAR(asy) DCLEN64(asy))
{
  int retval;

  if (*istat)
    return *istat;

  if (!ISPRESENTC(asy)) {
    return 0;
  }
  if (__fortio_eq_str(CADR(asy), CLEN(asy), "YES")) {
    /* do nothing */
  } else if (__fortio_eq_str(CADR(asy), CLEN(asy), "NO")) {
    return 0;
  } else {
    return FIO_ESPEC;
  }

  /* enable asynchronous i/o */

  retval = 0;
#if !defined(TARGET_WIN_X8632) && !defined(TARGET_OSX)
  if ((Fcb->acc == FIO_STREAM || Fcb->acc == FIO_SEQUENTIAL
       || Fcb->acc == FIO_DIRECT)
      &&
      (!Fcb->byte_swap)) {
    if (Fio_asy_open(Fcb->fp, &Fcb->asyptr) == -1) {
      retval = __fortio_error(__io_errno());
    }
  }
#endif
  return (retval);
}
/* 32 bit CLEN version */
int
ENTF90IO(OPEN_ASYNC, open_async)(__INT_T *istat, DCHAR(asy) DCLEN(asy))
{
  return ENTF90IO(OPEN_ASYNCA, open_asynca)(istat, CADR(asy), (__CLEN_T)CLEN(asy));
}

__INT_T
ENTF90IO(OPEN03A, open03a)(
  __INT_T *istat, DCHAR(decimal), /* POINT, COMMA, or NULL */
  DCHAR(round),                   /* UP, DOWN, ZERO, NEAREST, COMPATIBLE,
                                     PROCESSOR_DEFINED,or NULL */
  DCHAR(sign),     /* PLUS, SUPPRESS, PROCESSOR_DEFINED, or NULL */
  DCHAR(encoding)  /* UTF-8, DEFAULT, or NULL */
  DCLEN64(decimal)   /* length of decimal */
  DCLEN64(round)     /* length of round */
  DCLEN64(sign)      /* length of sign */
  DCLEN64(encoding)) /* length of encoding */
{
  /*
   *  N O T E  --  For any 'Fcb' members which are defined in this routine,
   *  make sure that they are also initialized in __fortio_open().
   *  ENTF90IO(open)() is always called for OPEN, but ENTF90IO(open03) is
   *  called only if any of the selected '03' specifiers are present.
   */
  if (*istat)
    return *istat;

  if (Fcb->form != FIO_FORMATTED)
    return __fortio_error(FIO_ECOMPAT);

  Fcb->encoding = FIO_DEFAULT;
  if (ISPRESENTC(encoding)) {
    if (__fortio_eq_str(CADR(encoding), CLEN(encoding), "UTF-8"))
      Fcb->encoding = FIO_UTF_8;
    else if (__fortio_eq_str(CADR(encoding), CLEN(encoding), "DEFAULT"))
      Fcb->encoding = FIO_DEFAULT;
    else
      return __fortio_error(FIO_ESPEC);
  }

  Fcb->decimal = FIO_POINT;
  if (ISPRESENTC(decimal)) {
    if (__fortio_eq_str(CADR(decimal), CLEN(decimal), "COMMA"))
      Fcb->decimal = FIO_COMMA;
    else if (__fortio_eq_str(CADR(decimal), CLEN(decimal), "POINT"))
      Fcb->decimal = FIO_POINT;
    else
      return __fortio_error(FIO_ESPEC);
  }

  Fcb->round = FIO_COMPATIBLE; /* This must be default mode.
                                  What is our default mode?
                                  On 10/14/10 we think it is compatible */
  if (ISPRESENTC(round)) {
    if (__fortio_eq_str(CADR(round), CLEN(round), "UP"))
      Fcb->round = FIO_UP;
    else if (__fortio_eq_str(CADR(round), CLEN(round), "DOWN"))
      Fcb->round = FIO_DOWN;
    else if (__fortio_eq_str(CADR(round), CLEN(round), "ZERO"))
      Fcb->round = FIO_ZERO;
    else if (__fortio_eq_str(CADR(round), CLEN(round), "NEAREST"))
      Fcb->round = FIO_NEAREST;
    else if (__fortio_eq_str(CADR(round), CLEN(round), "COMPATIBLE"))
      Fcb->round = FIO_COMPATIBLE;
    else if (__fortio_eq_str(CADR(round), CLEN(round), "PROCESSOR_DEFINED"))
      Fcb->round = FIO_PROCESSOR_DEFINED;
    else
      return __fortio_error(FIO_ESPEC);
  }

  Fcb->sign = FIO_PROCESSOR_DEFINED;
  if (ISPRESENTC(sign)) {
    if (__fortio_eq_str(CADR(sign), CLEN(sign), "PLUS"))
      Fcb->sign = FIO_PLUS;
    else if (__fortio_eq_str(CADR(sign), CLEN(sign), "SUPPRESS"))
      Fcb->sign = FIO_SUPPRESS;
    else if (__fortio_eq_str(CADR(sign), CLEN(sign), "PROCESOR_DEFINED"))
      Fcb->sign = FIO_PROCESSOR_DEFINED;
    else
      return __fortio_error(FIO_ESPEC);
  }

  return 0;
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(OPEN03, open03)(
  __INT_T *istat, DCHAR(decimal), /* POINT, COMMA, or NULL */
  DCHAR(round),                   /* UP, DOWN, ZERO, NEAREST, COMPATIBLE,
                                     PROCESSOR_DEFINED,or NULL */
  DCHAR(sign),     /* PLUS, SUPPRESS, PROCESSOR_DEFINED, or NULL */
  DCHAR(encoding)  /* UTF-8, DEFAULT, or NULL */
  DCLEN(decimal)   /* length of decimal */
  DCLEN(round)     /* length of round */
  DCLEN(sign)      /* length of sign */
  DCLEN(encoding)) /* length of encoding */
{

  return ENTF90IO(OPEN03A, open03a)(istat, CADR(decimal), CADR(round),
		  CADR(sign), CADR(encoding), (__CLEN_T)CLEN(decimal),
		  (__CLEN_T)CLEN(round), (__CLEN_T)CLEN(sign),
		  (__CLEN_T)CLEN(encoding));
}
