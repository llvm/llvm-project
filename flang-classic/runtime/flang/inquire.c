/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Implements Fortran INQUIRE statement.
 */

#include <stdarg.h>
#include <string.h>
#include "global.h"
#include "async.h"

#if defined(_WIN64)
#define access _access
#endif

static FIO_FCB *f2; /* save fcb for inquire2 */

static void copystr(char *dst,       /* destination string, blank-filled */
                    int len,         /* length of destination space */
                    const char *src) /* null terminated source string  */
{
  char *end = dst + len;
  while (dst < end && *src != '\0')
    *dst++ = *src++;
  while (dst < end)
    *dst++ = ' ';
}

#define SINTN 64 /* max number of return ints */

/* internal inquire */

static int
inquire(__INT_T *unit, char *file_ptr, __INT_T *bitv, __INT_T *iostat,
        bool *exist, bool *opened, __INT8_T *number, bool *named,
        char *name_ptr, char *acc_ptr, char *sequential_ptr, char *direct_ptr,
        char *form_ptr, char *formatted_ptr, char *unformatted_ptr,
        __INT8_T *recl, __INT8_T *nextrec, char *blank_ptr, char *position_ptr,
        char *action_ptr, char *read_ptr, char *write0_ptr, char *readwrite_ptr,
        char *delim_ptr, char *pad_ptr, __INT_T *id, __INT_T *pending,
        __INT8_T *pos, __INT8_T *size, char *asynchronous_ptr,
        char *decimal_ptr, char *encoding_ptr, char *sign_ptr, char *stream_ptr,
        char *round_ptr, __CLEN_T file_siz, __CLEN_T name_siz, __CLEN_T acc_siz,
        __CLEN_T sequential_siz, __CLEN_T direct_siz, __CLEN_T form_siz, __CLEN_T formatted_siz,
        __CLEN_T unformatted_siz, __CLEN_T blank_siz, __CLEN_T position_siz, __CLEN_T action_siz,
        __CLEN_T read_siz, __CLEN_T write0_siz, __CLEN_T readwrite_siz, __CLEN_T delim_siz,
        __CLEN_T pad_siz, int asynchronous_siz, int decimal_siz, int encoding_siz,
        int sign_siz, int stream_siz, int round_siz)
{
  FIO_FCB *f;
  __CLEN_T i;
  const char *cp;
  __CLEN_T len, nleadb;

  __fortio_errinit03(*unit, *bitv, iostat, "INQUIRE");

  /*  determine whether file name was specified and strip off any trailing
      and leading blnks; also, strip any trailing null characters.
      All blank names have length 0 */

  if (file_ptr != NULL) { /*  inquire by file  */
    len = file_siz;
    while (len > 0 && (file_ptr[len - 1] == ' ' || file_ptr[len - 1] == '\0'))
      --len;
    nleadb = 0;
    while (len > 0 && file_ptr[nleadb] == ' ')
      ++nleadb, --len;
    if (len <= 0) {
      len = 0;
      f = NULL;
    } else {
      for (f = fioFcbTbls.fcbs; f; f = f->next)
        if (len == strlen(f->name) &&
            strncmp(file_ptr + nleadb, f->name, len) == 0)
          break;
    }
  } else { /*  inquire by unit  */
    if (ILLEGAL_UNIT(*unit)) {
      f = NULL;
      if (*unit == -1 && iostat) {
        *iostat = FIO_STAT_INTERNAL_UNIT;
      }
    } else {
      f = __fortio_find_unit(*unit);
    }
  }

  /* check for outstanding async i/o */

  if ((f != NULL) && f->asy_rw) { /* stop any async i/o */
    f->asy_rw = 0;
    if (Fio_asy_disable(f->asyptr) == -1) {
      return (__fortio_error(__io_errno()));
    }
  }

  f2 = f; /* save fcb for inquire2 */

  /*  now fill in those values that were requested by the user:  */

  if (acc_ptr != NULL) {
    if (f == NULL)
      cp = "UNDEFINED";
    else if (f->acc == FIO_DIRECT)
      cp = "DIRECT";
    else if (f->acc == FIO_STREAM)
      cp = "STREAM";
    else
      cp = "SEQUENTIAL";
    copystr(acc_ptr, acc_siz, cp);
  }
  if (action_ptr != NULL) {
    if (f == NULL)
      cp = "UNDEFINED";
    else if (f->action == FIO_READ)
      cp = "READ";
    else if (f->action == FIO_WRITE)
      cp = "WRITE";
    else
      cp = "READWRITE";
    copystr(action_ptr, action_siz, cp);
  }
  if (blank_ptr != NULL) {
    if (f == NULL || f->form != FIO_FORMATTED)
      cp = "UNDEFINED";
    else if (f->blank == FIO_NULL)
      cp = "NULL";
    else
      cp = "ZERO";
    copystr(blank_ptr, blank_siz, cp);
  }
  if (delim_ptr != NULL) {
    if (f == NULL)
      cp = "UNDEFINED";
    else if (f->form != FIO_FORMATTED)
      cp = "UNDEFINED";
    else if (f->delim == FIO_APOSTROPHE)
      cp = "APOSTROPHE";
    else if (f->delim == FIO_QUOTE)
      cp = "QUOTE";
    else
      cp = "NONE";
    copystr(delim_ptr, delim_siz, cp);
  }
  if (direct_ptr != NULL) {
    if (f == NULL)
      cp = "UNKNOWN";
    else if (f->acc == FIO_DIRECT)
      cp = "YES";
    else
      cp = "NO";
    copystr(direct_ptr, direct_siz, cp);
  }
  if (exist != (bool *)0) {
    if (file_ptr == NULL)
      *exist = FTN_TRUE; /* inquire by unit */
    else if (f != NULL)
      *exist = FTN_TRUE; /* connected */
    else {               /* inquire by file and not connected to unit */
      char btmpnam[MAX_NAMELEN + 1];
      char *tmpnam;
      int long_name;

      long_name = 0;
      tmpnam = btmpnam;
      if (len > MAX_NAMELEN) {
        tmpnam = malloc(len + 1);
        long_name = 1;
      }
      for (i = 0; i < len; ++i)
        tmpnam[i] = file_ptr[nleadb + i];
      tmpnam[len] = 0;
      if (__fort_access(tmpnam, 0) == 0)
        *exist = FTN_TRUE;
      else
        *exist = FTN_FALSE;
      if (long_name) {
        free(tmpnam);
      }
    }
  }
  if (form_ptr != NULL) {
    if (f == NULL)
      cp = "UNDEFINED";
    else if (f->form == FIO_FORMATTED)
      cp = "FORMATTED";
    else
      cp = "UNFORMATTED";
    copystr(form_ptr, form_siz, cp);
  }
  if (formatted_ptr != NULL) {
    if (f == NULL)
      cp = "UNKNOWN";
    else if (f->form == FIO_FORMATTED)
      cp = "YES";
    else
      cp = "NO";
    copystr(formatted_ptr, formatted_siz, cp);
  }
  if (name_ptr != NULL) {
    if (file_ptr != NULL && f == NULL) {
      /* tpr1933 - inquire by file and the file hasn't been opened --
       * just copy the file specifier without any leading and trailing
       * blanks to the name specifier.
       */
      char *pn, *pf;
      __CLEN_T cn, cf;
      pn = name_ptr;
      cn = name_siz;
      pf = file_ptr + nleadb;
      cf = len;
      while (cf-- > 0) {
        if (cn <= 0)
          break;
        *pn++ = *pf++;
        cn--;
      }
      while (cn-- > 0)
        *pn++ = ' ';
    } else if (f != NULL && f->named)
      copystr(name_ptr, name_siz, f->name);
  }
  if (named != (bool *)0) {
    if (f != NULL && f->named)
      *named = FTN_TRUE;
    else
      *named = FTN_FALSE;
  }
  if (nextrec != 0) {
    *nextrec = 0;
    if (f != NULL && f->acc == FIO_DIRECT)
      *nextrec = f->nextrec;
  }
  if (number != 0) {
    if (f != NULL)
      *number = f->unit;
    else
      *number = -1;
  }
  if (opened != (bool *)0) {
    if (f != NULL)
      *opened = FTN_TRUE;
    else
      *opened = FTN_FALSE;
  }
  if (pad_ptr != NULL) {
    if (f == NULL || f->pad == FIO_YES)
      cp = "YES";
    else
      cp = "NO";
    copystr(pad_ptr, pad_siz, cp);
  }
  if (position_ptr != NULL) {
    if (f == NULL)
      cp = "UNDEFINED";
    else if (f->pos == FIO_REWIND)
      cp = "REWIND";
    else if (f->pos == FIO_APPEND)
      cp = "APPEND";
    else
      cp = "ASIS";
    copystr(position_ptr, position_siz, cp);
  }
  if (read_ptr != NULL) {
    if (f == NULL)
      cp = "UNKNOWN";
    else if (f->action == FIO_READ || f->action == FIO_READWRITE)
      cp = "YES";
    else
      cp = "NO";
    copystr(read_ptr, read_siz, cp);
  }
  if (readwrite_ptr != NULL) {
    if (f == NULL)
      cp = "UNKNOWN";
    else if (f->action == FIO_READWRITE)
      cp = "YES";
    else
      cp = "NO";
    copystr(readwrite_ptr, readwrite_siz, cp);
  }
  if ((recl != 0) && f != NULL && f->acc == FIO_DIRECT)
    *recl = f->reclen / f->wordlen; /* internally, reclen is in bytes */
  if (sequential_ptr != NULL) {
    if (f == NULL)
      cp = "UNKNOWN";
    else if (f->acc == FIO_SEQUENTIAL)
      cp = "YES";
    else
      cp = "NO";
    copystr(sequential_ptr, sequential_siz, cp);
  }
  if (unformatted_ptr != NULL) {
    if (f == NULL)
      cp = "UNKNOWN";
    else if (f->form == FIO_FORMATTED)
      cp = "NO";
    else
      cp = "YES";
    copystr(unformatted_ptr, unformatted_siz, cp);
  }
  if (write0_ptr != NULL) {
    if (f == NULL)
      cp = "UNKNOWN";
    else if (f->action == FIO_WRITE || f->action == FIO_READWRITE)
      cp = "YES";
    else
      cp = "NO";
    copystr(write0_ptr, write0_siz, cp);
  }
  if (id) {
    *id = 0;
  }
  if (pending != NULL) {
    *pending = FTN_FALSE;
  }
  if (pos) {
    if (f != NULL)
      *pos = __io_ftellx(f->fp) + 1;
  }
  if (size) {
    FILE *lcl_fp;
    if (f != NULL) {
      seekoffx_t currpos;
      lcl_fp = f->fp;
      currpos = (seekoffx_t)__io_ftellx(f->fp);
      if (__io_fseek(f->fp, 0L, SEEK_END) != 0)
        return (__fortio_error(__io_errno()));
      *size = __io_ftellx(f->fp);
      __io_fseek(f->fp, currpos, SEEK_SET);
    } else if (file_ptr != NULL) { /* inquire by file and not connected */
      char btmpnam[MAX_NAMELEN + 1];
      char *tmpnam;
      int long_name;
      long_name = 0;
      tmpnam = btmpnam;
      if (len > MAX_NAMELEN) {
        tmpnam = malloc(len + 1);
        long_name = 1;
      }
      for (i = 0; i < len; ++i)
        tmpnam[i] = file_ptr[nleadb + i];
      tmpnam[len] = 0;
      if ((lcl_fp = __io_fopen(tmpnam, "rb")) == NULL)
        *size = -1;
      else if (__io_fseek(lcl_fp, 0L, SEEK_END) != 0)
        *size = -1;
      else
        *size = __io_ftellx(lcl_fp);
      if (lcl_fp)
        (void)__io_fclose(lcl_fp);
      if (long_name) {
        free(tmpnam);
      }
    } else
      *size = -1;
  }
  if (asynchronous_ptr) {
    if (f == NULL)
      cp = "UNDEFINED";
    else if (f->asyptr)
      cp = "YES";
    else
      cp = "NO";
    copystr(asynchronous_ptr, asynchronous_siz, cp);
  }
  if (decimal_ptr) {
    /* "DECIMAL" "POINT" "UNDEFINED" */
    if (f == NULL || f->form != FIO_FORMATTED)
      cp = "UNDEFINED";
    else if (f->decimal == FIO_COMMA)
      cp = "COMMA";
    else if (f->decimal == FIO_POINT)
      cp = "POINT";
    else
      cp = "POINT";
    copystr(decimal_ptr, decimal_siz, cp);
  }
  if (encoding_ptr) {
    /* "UTF-8" "DEFAULT" "UNDEFINED" "UNKNOWN" */
    if (f == NULL)
      cp = "UNKNOWN";
    else if (f->form != FIO_FORMATTED)
      cp = "UNDEFINED";
    else if (f->encoding == FIO_UTF_8)
      cp = "UTF-8";
    else
      cp = "DEFAULT";
    copystr(encoding_ptr, encoding_siz, cp);
  }
  if (sign_ptr) {
    /* "PLUS" "SUPPRESS" "PROCESSOR_DEFINED" "UNDEFINED"*/
    if (f == NULL || f->form != FIO_FORMATTED)
      cp = "UNDEFINED";
    else if (f->sign == FIO_PLUS)
      cp = "PLUS";
    else if (f->sign == FIO_SUPPRESS)
      cp = "SUPPRESS";
    else
      cp = "PROCESSOR_DEFINED";
    copystr(sign_ptr, sign_siz, cp);
  }
  if (stream_ptr) {
    if ((f == NULL) || (f->acc == FIO_STREAM))
      cp = "YES";
    else
      cp = "NO";
    copystr(stream_ptr, stream_siz, cp);
  }
  if (round_ptr) {
    switch (f->round) {
    case FIO_UP:
      cp = "UP";
      break;
    case FIO_DOWN:
      cp = "DOWN";
      break;
    case FIO_ZERO:
      cp = "ZERO";
      break;
    case FIO_NEAREST:
      cp = "NEAREST";
      break;
    case FIO_COMPATIBLE:
      cp = "COMPATIBLE";
      break;
    case FIO_PROCESSOR_DEFINED:
      cp = "PROCESSOR_DEFINED";
      break;
    default:
      cp = "UNDEFINED";
      break;
    }
    copystr(round_ptr, round_siz, cp);
  }

  return 0; /* no error occurred */
}

/* ----------------------------------------------------------------------- */

/* external inquire */

int ENTF90IO(INQUIREA, inquirea)(
    __INT_T *unit, DCHAR(file), __INT_T *bitv, __INT_T *iostat, bool *exist,
    bool *opened, __INT_T *number, bool *named, DCHAR(name), DCHAR(acc),
    DCHAR(sequential), DCHAR(direct), DCHAR(form), DCHAR(formatted),
    DCHAR(unformatted), __INT_T *recl, __INT_T *nextrec, DCHAR(blank),
    DCHAR(position), DCHAR(action), DCHAR(read), DCHAR(write0),
    DCHAR(readwrite), DCHAR(delim),
    DCHAR(pad) DCLEN64(file) DCLEN64(name) DCLEN64(acc) DCLEN64(sequential)
    DCLEN64(direct) DCLEN64(form) DCLEN64(formatted) DCLEN64(unformatted)
    DCLEN64(blank) DCLEN64(position) DCLEN64(action) DCLEN64(read) DCLEN64(write0)
    DCLEN64(readwrite) DCLEN64(delim) DCLEN64(pad))
{
  char *file_ptr;
  char *name_ptr;
  char *acc_ptr;
  char *sequential_ptr;
  char *direct_ptr;
  char *form_ptr;
  char *formatted_ptr;
  char *unformatted_ptr;
  char *blank_ptr;
  char *position_ptr;
  char *action_ptr;
  char *read_ptr;
  char *write0_ptr;
  char *readwrite_ptr;
  char *delim_ptr;
  char *pad_ptr;
  __CLEN_T file_siz;
  __CLEN_T name_siz;
  __CLEN_T acc_siz;
  __CLEN_T sequential_siz;
  __CLEN_T direct_siz;
  __CLEN_T form_siz;
  __CLEN_T formatted_siz;
  __CLEN_T unformatted_siz;
  __CLEN_T blank_siz;
  __CLEN_T position_siz;
  __CLEN_T action_siz;
  __CLEN_T read_siz;
  __CLEN_T write0_siz;
  __CLEN_T readwrite_siz;
  __CLEN_T delim_siz;
  __CLEN_T pad_siz;
  __INT8_T newnumber;
  __INT8_T newrecl;
  __INT8_T newnextrec;

  int s;
  int ioproc;
  int sint[SINTN], *sintp;
  char *schr = 0, *schrp;
  __CLEN_T schr_len = 0;

  file_ptr = (ISPRESENTC(file) ? CADR(file) : NULL);
  name_ptr = (ISPRESENTC(name) ? CADR(name) : NULL);
  acc_ptr = (ISPRESENTC(acc) ? CADR(acc) : NULL);
  sequential_ptr = (ISPRESENTC(sequential) ? CADR(sequential) : NULL);
  direct_ptr = (ISPRESENTC(direct) ? CADR(direct) : NULL);
  form_ptr = (ISPRESENTC(form) ? CADR(form) : NULL);
  formatted_ptr = (ISPRESENTC(formatted) ? CADR(formatted) : NULL);
  unformatted_ptr = (ISPRESENTC(unformatted) ? CADR(unformatted) : NULL);
  blank_ptr = (ISPRESENTC(blank) ? CADR(blank) : NULL);
  position_ptr = (ISPRESENTC(position) ? CADR(position) : NULL);
  action_ptr = (ISPRESENTC(action) ? CADR(action) : NULL);
  read_ptr = (ISPRESENTC(read) ? CADR(read) : NULL);
  write0_ptr = (ISPRESENTC(write0) ? CADR(write0) : NULL);
  readwrite_ptr = (ISPRESENTC(readwrite) ? CADR(readwrite) : NULL);
  delim_ptr = (ISPRESENTC(delim) ? CADR(delim) : NULL);
  pad_ptr = (ISPRESENTC(pad) ? CADR(pad) : NULL);
  file_siz = CLEN(file);
  schr_len += (ISPRESENTC(file) ? file_siz : 0);
  name_siz = CLEN(name);
  schr_len += (ISPRESENTC(name) ? name_siz : 0);
  acc_siz = CLEN(acc);
  schr_len += (ISPRESENTC(acc) ? acc_siz : 0);
  sequential_siz = CLEN(sequential);
  schr_len += (ISPRESENTC(sequential) ? sequential_siz : 0);
  direct_siz = CLEN(direct);
  schr_len += (ISPRESENTC(direct) ? direct_siz : 0);
  form_siz = CLEN(form);
  schr_len += (ISPRESENTC(form) ? form_siz : 0);
  formatted_siz = CLEN(formatted);
  schr_len += (ISPRESENTC(formatted) ? formatted_siz : 0);
  unformatted_siz = CLEN(unformatted);
  schr_len += (ISPRESENTC(unformatted) ? unformatted_siz : 0);
  blank_siz = CLEN(blank);
  schr_len += (ISPRESENTC(blank) ? blank_siz : 0);
  position_siz = CLEN(position);
  schr_len += (ISPRESENTC(position) ? position_siz : 0);
  action_siz = CLEN(action);
  schr_len += (ISPRESENTC(action) ? action_siz : 0);
  read_siz = CLEN(read);
  schr_len += (ISPRESENTC(read) ? read_siz : 0);
  write0_siz = CLEN(write0);
  schr_len += (ISPRESENTC(write0) ? write0_siz : 0);
  readwrite_siz = CLEN(readwrite);
  schr_len += (ISPRESENTC(readwrite) ? readwrite_siz : 0);
  delim_siz = CLEN(delim);
  schr_len += (ISPRESENTC(delim) ? delim_siz : 0);
  pad_siz = CLEN(pad);
  schr_len += (ISPRESENTC(pad) ? pad_siz : 0);

  if (schr_len)
    schr = (char *)__fort_malloc(sizeof(char) * schr_len);

  sintp = sint + 1;
  schrp = schr;

  /* non i/o processors */

  ioproc = GET_DIST_IOPROC;
  if ((GET_DIST_LCPU != ioproc) && (!LOCAL_MODE)) {

    DIST_RBCST(ioproc, sint, SINTN, 1, __CINT);
    DIST_RBCST(ioproc, schr, schr_len, 1, __CHAR);

    if (*bitv & FIO_BITV_IOSTAT) {
      *iostat = *sintp++;
    }
    if (ISPRESENTC(acc)) {
      memcpy(acc_ptr, schrp, acc_siz);
      schrp += acc_siz;
    }
    if (ISPRESENTC(action)) {
      memcpy(action_ptr, schrp, action_siz);
      schrp += action_siz;
    }
    if (ISPRESENTC(blank)) {
      memcpy(blank_ptr, schrp, blank_siz);
      schrp += blank_siz;
    }
    if (ISPRESENTC(delim)) {
      memcpy(delim_ptr, schrp, delim_siz);
      schrp += delim_siz;
    }
    if (ISPRESENTC(direct)) {
      memcpy(direct_ptr, schrp, direct_siz);
      schrp += direct_siz;
    }
    if (ISPRESENT(exist)) {
      *exist = *sintp++;
    }
    if (ISPRESENTC(form)) {
      memcpy(form_ptr, schrp, form_siz);
      schrp += form_siz;
    }
    if (ISPRESENTC(formatted)) {
      memcpy(formatted_ptr, schrp, formatted_siz);
      schrp += formatted_siz;
    }
    if (ISPRESENTC(name)) {
      memcpy(name_ptr, schrp, name_siz);
      schrp += name_siz;
    }
    if (ISPRESENT(named)) {
      *named = *sintp++;
    }
    if (ISPRESENT(nextrec)) {
      *nextrec = *sintp++;
    }
    if (ISPRESENT(number)) {
      *number = *sintp++;
    }
    if (ISPRESENT(opened)) {
      *opened = *sintp++;
    }
    if (ISPRESENTC(pad)) {
      memcpy(pad_ptr, schrp, pad_siz);
      schrp += pad_siz;
    }
    if (ISPRESENTC(position)) {
      memcpy(position_ptr, schrp, position_siz);
      schrp += position_siz;
    }
    if (ISPRESENTC(read)) {
      memcpy(read_ptr, schrp, read_siz);
      schrp += read_siz;
    }
    if (ISPRESENTC(readwrite)) {
      memcpy(readwrite_ptr, schrp, readwrite_siz);
      schrp += readwrite_siz;
    }
    if (ISPRESENT(recl)) {
      *recl = *sintp++;
    }
    if (ISPRESENTC(sequential)) {
      memcpy(sequential_ptr, schrp, sequential_siz);
      schrp += sequential_siz;
    }
    if (ISPRESENTC(unformatted)) {
      memcpy(unformatted_ptr, schrp, unformatted_siz);
      schrp += unformatted_siz;
    }
    if (ISPRESENTC(write0)) {
      memcpy(write0_ptr, schrp, write0_siz);
      schrp += write0_siz;
    }

    return (sint[0]);
  }

  /* i/o processor */

  s = inquire(unit, file_ptr, bitv, iostat, exist, opened, &newnumber, named,
              name_ptr, acc_ptr, sequential_ptr, direct_ptr, form_ptr,
              formatted_ptr, unformatted_ptr, &newrecl, &newnextrec, blank_ptr,
              position_ptr, action_ptr, read_ptr, write0_ptr, readwrite_ptr,
              delim_ptr, pad_ptr, 0, /* id */
              0,                     /* pending */
              0,                     /* pos */
              0,                     /* size */
              0,                     /* asynchronous_ptr */
              0,                     /* decimal_ptr */
              0,                     /* encoding_ptr */
              0,                     /* sign_ptr */
              0,                     /* stream_ptr */
              0,                     /* round_ptr */
              file_siz, name_siz, acc_siz, sequential_siz, direct_siz, form_siz,
              formatted_siz, unformatted_siz, blank_siz, position_siz,
              action_siz, read_siz, write0_siz, readwrite_siz, delim_siz,
              pad_siz, 0, /* asynchronous_siz */
              0,          /* decimal_siz */
              0,          /* encoding_siz */
              0,          /* sign_siz */
              0,          /* stream_siz */
              0           /* round_siz */
              );

  sint[0] = s;

  if (*bitv & FIO_BITV_IOSTAT) {
    *sintp++ = *iostat;
  }
  if (ISPRESENTC(acc)) {
    memcpy(schrp, CADR(acc), CLEN(acc));
    schrp += CLEN(acc);
  }
  if (ISPRESENTC(action)) {
    memcpy(schrp, CADR(action), CLEN(action));
    schrp += CLEN(action);
  }
  if (ISPRESENTC(blank)) {
    memcpy(schrp, CADR(blank), CLEN(blank));
    schrp += CLEN(blank);
  }
  if (ISPRESENTC(delim)) {
    memcpy(schrp, CADR(delim), CLEN(delim));
    schrp += CLEN(delim);
  }
  if (ISPRESENTC(direct)) {
    memcpy(schrp, CADR(direct), CLEN(direct));
    schrp += CLEN(direct);
  }
  if (ISPRESENT(exist)) {
    *sintp++ = *exist;
  }
  if (ISPRESENTC(form)) {
    memcpy(schrp, CADR(form), CLEN(form));
    schrp += CLEN(form);
  }
  if (ISPRESENTC(formatted)) {
    memcpy(schrp, CADR(formatted), CLEN(formatted));
    schrp += CLEN(formatted);
  }
  if (ISPRESENTC(name)) {
    memcpy(schrp, CADR(name), CLEN(name));
    schrp += CLEN(name);
  }
  if (ISPRESENT(named)) {
    *sintp++ = *named;
  }
  if (ISPRESENT(nextrec)) {
    *nextrec = (__INT_T)newnextrec;
    *sintp++ = *nextrec;
  }
  if (ISPRESENT(number)) {
    *number = (__INT_T)newnumber;
    *sintp++ = *number;
  }
  if (ISPRESENT(opened)) {
    *sintp++ = *opened;
  }
  if (ISPRESENTC(pad)) {
    memcpy(schrp, CADR(pad), CLEN(pad));
    schrp += CLEN(pad);
  }
  if (ISPRESENTC(position)) {
    memcpy(schrp, CADR(position), CLEN(position));
    schrp += CLEN(position);
  }
  if (ISPRESENTC(read)) {
    memcpy(schrp, CADR(read), CLEN(read));
    schrp += CLEN(read);
  }
  if (ISPRESENTC(readwrite)) {
    memcpy(schrp, CADR(readwrite), CLEN(readwrite));
    schrp += CLEN(readwrite);
  }
  if (ISPRESENT(recl)) {
    *recl = (__INT_T)newrecl;
    *sintp++ = *recl;
  }
  if (ISPRESENTC(sequential)) {
    memcpy(schrp, CADR(sequential), CLEN(sequential));
    schrp += CLEN(sequential);
  }
  if (ISPRESENTC(unformatted)) {
    memcpy(schrp, CADR(unformatted), CLEN(unformatted));
    schrp += CLEN(unformatted);
  }
  if (ISPRESENTC(write0)) {
    memcpy(schrp, CADR(write0), CLEN(write0));
    schrp += CLEN(write0);
  }

  if (!LOCAL_MODE) {
    DIST_RBCST(ioproc, sint, SINTN, 1, __CINT);
    DIST_RBCST(ioproc, schr, schr_len, 1, __CHAR);
  }

  if (schr)
    __fort_free(schr);

  __fortio_errend03();
  return (sint[0]);
}
/* 32 bit CLEN version */
int ENTF90IO(INQUIRE, inquire)(
    __INT_T *unit, DCHAR(file), __INT_T *bitv, __INT_T *iostat, bool *exist,
    bool *opened, __INT_T *number, bool *named, DCHAR(name), DCHAR(acc),
    DCHAR(sequential), DCHAR(direct), DCHAR(form), DCHAR(formatted),
    DCHAR(unformatted), __INT_T *recl, __INT_T *nextrec, DCHAR(blank),
    DCHAR(position), DCHAR(action), DCHAR(read), DCHAR(write0),
    DCHAR(readwrite), DCHAR(delim),
    DCHAR(pad) DCLEN(file) DCLEN(name) DCLEN(acc) DCLEN(sequential)
    DCLEN(direct) DCLEN(form) DCLEN(formatted) DCLEN(unformatted)
    DCLEN(blank) DCLEN(position) DCLEN(action) DCLEN(read) DCLEN(write0)
    DCLEN(readwrite) DCLEN(delim) DCLEN(pad))
{
  return ENTF90IO(INQUIREA, inquirea)(unit, CADR(file), bitv, iostat, exist,
                  opened, number, named, CADR(name), CADR(acc),
                  CADR(sequential), CADR(direct), CADR(form), CADR(formatted),
                  CADR(unformatted), recl, nextrec, CADR(blank),
                  CADR(position), CADR(action), CADR(read), CADR(write0),
                  CADR(readwrite), CADR(delim), CADR(pad), (__CLEN_T)CLEN(file),
                  (__CLEN_T)CLEN(name), (__CLEN_T)CLEN(acc),
                  (__CLEN_T)CLEN(sequential), (__CLEN_T)CLEN(direct),
                  (__CLEN_T)CLEN(form), (__CLEN_T)CLEN(formatted),
                  (__CLEN_T)CLEN(unformatted), (__CLEN_T)CLEN(blank),
                  (__CLEN_T)CLEN(position), (__CLEN_T)CLEN(action),
                  (__CLEN_T)CLEN(read), (__CLEN_T)CLEN(write0),
                  (__CLEN_T)CLEN(readwrite), (__CLEN_T)CLEN(delim),
                  (__CLEN_T)CLEN(pad));
}

/* new external inquire support INT8 for number, recl, nextrec */
int ENTF90IO(INQUIRE2003A, inquire2003a)(
    __INT_T *unit, DCHAR(file), __INT_T *bitv, __INT_T *iostat, bool *exist,
    bool *opened, __INT8_T *number, bool *named, DCHAR(name), DCHAR(acc),
    DCHAR(sequential), DCHAR(direct), DCHAR(form), DCHAR(formatted),
    DCHAR(unformatted), __INT8_T *recl, __INT8_T *nextrec, DCHAR(blank),
    DCHAR(position), DCHAR(action), DCHAR(read), DCHAR(write0),
    DCHAR(readwrite), DCHAR(delim),
    DCHAR(pad) DCLEN64(file) DCLEN64(name) DCLEN64(acc) DCLEN64(sequential)
    DCLEN64(direct) DCLEN64(form) DCLEN64(formatted) DCLEN64(unformatted)
    DCLEN64(blank) DCLEN64(position) DCLEN64(action) DCLEN64(read) DCLEN64(write0)
    DCLEN64(readwrite) DCLEN64(delim) DCLEN64(pad))
{
  char *file_ptr;
  char *name_ptr;
  char *acc_ptr;
  char *sequential_ptr;
  char *direct_ptr;
  char *form_ptr;
  char *formatted_ptr;
  char *unformatted_ptr;
  char *blank_ptr;
  char *position_ptr;
  char *action_ptr;
  char *read_ptr;
  char *write0_ptr;
  char *readwrite_ptr;
  char *delim_ptr;
  char *pad_ptr;
  __CLEN_T file_siz;
  __CLEN_T name_siz;
  __CLEN_T acc_siz;
  __CLEN_T sequential_siz;
  __CLEN_T direct_siz;
  __CLEN_T form_siz;
  __CLEN_T formatted_siz;
  __CLEN_T unformatted_siz;
  __CLEN_T blank_siz;
  __CLEN_T position_siz;
  __CLEN_T action_siz;
  __CLEN_T read_siz;
  __CLEN_T write0_siz;
  __CLEN_T readwrite_siz;
  __CLEN_T delim_siz;
  __CLEN_T pad_siz;
  __INT8_T newnumber;
  __INT8_T newrecl;
  __INT8_T newnextrec;

  int s;
  int ioproc;
  int sint[SINTN], *sintp;
  char *schr = 0, *schrp;
  __CLEN_T schr_len = 0;

  file_ptr = (ISPRESENTC(file) ? CADR(file) : NULL);
  name_ptr = (ISPRESENTC(name) ? CADR(name) : NULL);
  acc_ptr = (ISPRESENTC(acc) ? CADR(acc) : NULL);
  sequential_ptr = (ISPRESENTC(sequential) ? CADR(sequential) : NULL);
  direct_ptr = (ISPRESENTC(direct) ? CADR(direct) : NULL);
  form_ptr = (ISPRESENTC(form) ? CADR(form) : NULL);
  formatted_ptr = (ISPRESENTC(formatted) ? CADR(formatted) : NULL);
  unformatted_ptr = (ISPRESENTC(unformatted) ? CADR(unformatted) : NULL);
  blank_ptr = (ISPRESENTC(blank) ? CADR(blank) : NULL);
  position_ptr = (ISPRESENTC(position) ? CADR(position) : NULL);
  action_ptr = (ISPRESENTC(action) ? CADR(action) : NULL);
  read_ptr = (ISPRESENTC(read) ? CADR(read) : NULL);
  write0_ptr = (ISPRESENTC(write0) ? CADR(write0) : NULL);
  readwrite_ptr = (ISPRESENTC(readwrite) ? CADR(readwrite) : NULL);
  delim_ptr = (ISPRESENTC(delim) ? CADR(delim) : NULL);
  pad_ptr = (ISPRESENTC(pad) ? CADR(pad) : NULL);
  file_siz = CLEN(file);
  schr_len += (ISPRESENTC(file) ? file_siz : 0);
  name_siz = CLEN(name);
  schr_len += (ISPRESENTC(name) ? name_siz : 0);
  acc_siz = CLEN(acc);
  schr_len += (ISPRESENTC(acc) ? acc_siz : 0);
  sequential_siz = CLEN(sequential);
  schr_len += (ISPRESENTC(sequential) ? sequential_siz : 0);
  direct_siz = CLEN(direct);
  schr_len += (ISPRESENTC(direct) ? direct_siz : 0);
  form_siz = CLEN(form);
  schr_len += (ISPRESENTC(form) ? form_siz : 0);
  formatted_siz = CLEN(formatted);
  schr_len += (ISPRESENTC(formatted) ? formatted_siz : 0);
  unformatted_siz = CLEN(unformatted);
  schr_len += (ISPRESENTC(unformatted) ? unformatted_siz : 0);
  blank_siz = CLEN(blank);
  schr_len += (ISPRESENTC(blank) ? blank_siz : 0);
  position_siz = CLEN(position);
  schr_len += (ISPRESENTC(position) ? position_siz : 0);
  action_siz = CLEN(action);
  schr_len += (ISPRESENTC(action) ? action_siz : 0);
  read_siz = CLEN(read);
  schr_len += (ISPRESENTC(read) ? read_siz : 0);
  write0_siz = CLEN(write0);
  schr_len += (ISPRESENTC(write0) ? write0_siz : 0);
  readwrite_siz = CLEN(readwrite);
  schr_len += (ISPRESENTC(readwrite) ? readwrite_siz : 0);
  delim_siz = CLEN(delim);
  schr_len += (ISPRESENTC(delim) ? delim_siz : 0);
  pad_siz = CLEN(pad);
  schr_len += (ISPRESENTC(pad) ? pad_siz : 0);

  if (schr_len)
    schr = (char *)__fort_malloc(sizeof(char) * schr_len);

  sintp = sint + 1;
  schrp = schr;

  /* non i/o processors */

  ioproc = GET_DIST_IOPROC;
  if ((GET_DIST_LCPU != ioproc) && (!LOCAL_MODE)) {

    DIST_RBCST(ioproc, sint, SINTN, 1, __CINT);
    DIST_RBCST(ioproc, schr, schr_len, 1, __CHAR);

    if (*bitv & FIO_BITV_IOSTAT) {
      *iostat = *sintp++;
    }
    if (ISPRESENTC(acc)) {
      memcpy(acc_ptr, schrp, acc_siz);
      schrp += acc_siz;
    }
    if (ISPRESENTC(action)) {
      memcpy(action_ptr, schrp, action_siz);
      schrp += action_siz;
    }
    if (ISPRESENTC(blank)) {
      memcpy(blank_ptr, schrp, blank_siz);
      schrp += blank_siz;
    }
    if (ISPRESENTC(delim)) {
      memcpy(delim_ptr, schrp, delim_siz);
      schrp += delim_siz;
    }
    if (ISPRESENTC(direct)) {
      memcpy(direct_ptr, schrp, direct_siz);
      schrp += direct_siz;
    }
    if (ISPRESENT(exist)) {
      *exist = *sintp++;
    }
    if (ISPRESENTC(form)) {
      memcpy(form_ptr, schrp, form_siz);
      schrp += form_siz;
    }
    if (ISPRESENTC(formatted)) {
      memcpy(formatted_ptr, schrp, formatted_siz);
      schrp += formatted_siz;
    }
    if (ISPRESENTC(name)) {
      memcpy(name_ptr, schrp, name_siz);
      schrp += name_siz;
    }
    if (ISPRESENT(named)) {
      *named = *sintp++;
    }
    if (ISPRESENT(nextrec)) {
      *nextrec = *sintp++;
    }
    if (ISPRESENT(number)) {
      *number = *sintp++;
    }
    if (ISPRESENT(opened)) {
      *opened = *sintp++;
    }
    if (ISPRESENTC(pad)) {
      memcpy(pad_ptr, schrp, pad_siz);
      schrp += pad_siz;
    }
    if (ISPRESENTC(position)) {
      memcpy(position_ptr, schrp, position_siz);
      schrp += position_siz;
    }
    if (ISPRESENTC(read)) {
      memcpy(read_ptr, schrp, read_siz);
      schrp += read_siz;
    }
    if (ISPRESENTC(readwrite)) {
      memcpy(readwrite_ptr, schrp, readwrite_siz);
      schrp += readwrite_siz;
    }
    if (ISPRESENT(recl)) {
      *recl = *sintp++;
    }
    if (ISPRESENTC(sequential)) {
      memcpy(sequential_ptr, schrp, sequential_siz);
      schrp += sequential_siz;
    }
    if (ISPRESENTC(unformatted)) {
      memcpy(unformatted_ptr, schrp, unformatted_siz);
      schrp += unformatted_siz;
    }
    if (ISPRESENTC(write0)) {
      memcpy(write0_ptr, schrp, write0_siz);
      schrp += write0_siz;
    }

    return (sint[0]);
  }

  /* i/o processor */

  s = inquire(unit, file_ptr, bitv, iostat, exist, opened, &newnumber, named,
              name_ptr, acc_ptr, sequential_ptr, direct_ptr, form_ptr,
              formatted_ptr, unformatted_ptr, &newrecl, &newnextrec, blank_ptr,
              position_ptr, action_ptr, read_ptr, write0_ptr, readwrite_ptr,
              delim_ptr, pad_ptr, 0, /* id */
              0,                     /* pending */
              0,                     /* pos */
              0,                     /* size */
              0,                     /* asynchronous_ptr */
              0,                     /* decimal_ptr */
              0,                     /* encoding_ptr */
              0,                     /* sign_ptr */
              0,                     /* stream_ptr */
              0,                     /* round_ptr */
              file_siz, name_siz, acc_siz, sequential_siz, direct_siz, form_siz,
              formatted_siz, unformatted_siz, blank_siz, position_siz,
              action_siz, read_siz, write0_siz, readwrite_siz, delim_siz,
              pad_siz, 0, /* asynchronous_siz */
              0,          /* decimal_siz */
              0,          /* encoding_siz */
              0,          /* sign_siz */
              0,          /* stream_siz */
              0           /* round_siz */
              );

  sint[0] = s;

  if (*bitv & FIO_BITV_IOSTAT) {
    *sintp++ = *iostat;
  }
  if (ISPRESENTC(acc)) {
    memcpy(schrp, CADR(acc), CLEN(acc));
    schrp += CLEN(acc);
  }
  if (ISPRESENTC(action)) {
    memcpy(schrp, CADR(action), CLEN(action));
    schrp += CLEN(action);
  }
  if (ISPRESENTC(blank)) {
    memcpy(schrp, CADR(blank), CLEN(blank));
    schrp += CLEN(blank);
  }
  if (ISPRESENTC(delim)) {
    memcpy(schrp, CADR(delim), CLEN(delim));
    schrp += CLEN(delim);
  }
  if (ISPRESENTC(direct)) {
    memcpy(schrp, CADR(direct), CLEN(direct));
    schrp += CLEN(direct);
  }
  if (ISPRESENT(exist)) {
    *sintp++ = *exist;
  }
  if (ISPRESENTC(form)) {
    memcpy(schrp, CADR(form), CLEN(form));
    schrp += CLEN(form);
  }
  if (ISPRESENTC(formatted)) {
    memcpy(schrp, CADR(formatted), CLEN(formatted));
    schrp += CLEN(formatted);
  }
  if (ISPRESENTC(name)) {
    memcpy(schrp, CADR(name), CLEN(name));
    schrp += CLEN(name);
  }
  if (ISPRESENT(named)) {
    *sintp++ = *named;
  }
  if (ISPRESENT(nextrec)) {
    *nextrec = (__INT_T)newnextrec;
    *sintp++ = *nextrec;
  }
  if (ISPRESENT(number)) {
    *number = (__INT_T)newnumber;
    *sintp++ = *number;
  }
  if (ISPRESENT(opened)) {
    *sintp++ = *opened;
  }
  if (ISPRESENTC(pad)) {
    memcpy(schrp, CADR(pad), CLEN(pad));
    schrp += CLEN(pad);
  }
  if (ISPRESENTC(position)) {
    memcpy(schrp, CADR(position), CLEN(position));
    schrp += CLEN(position);
  }
  if (ISPRESENTC(read)) {
    memcpy(schrp, CADR(read), CLEN(read));
    schrp += CLEN(read);
  }
  if (ISPRESENTC(readwrite)) {
    memcpy(schrp, CADR(readwrite), CLEN(readwrite));
    schrp += CLEN(readwrite);
  }
  if (ISPRESENT(recl)) {
    *recl = (__INT_T)newrecl;
    *sintp++ = *recl;
  }
  if (ISPRESENTC(sequential)) {
    memcpy(schrp, CADR(sequential), CLEN(sequential));
    schrp += CLEN(sequential);
  }
  if (ISPRESENTC(unformatted)) {
    memcpy(schrp, CADR(unformatted), CLEN(unformatted));
    schrp += CLEN(unformatted);
  }
  if (ISPRESENTC(write0)) {
    memcpy(schrp, CADR(write0), CLEN(write0));
    schrp += CLEN(write0);
  }

  if (!LOCAL_MODE) {
    DIST_RBCST(ioproc, sint, SINTN, 1, __CINT);
    DIST_RBCST(ioproc, schr, schr_len, 1, __CHAR);
  }

  if (schr)
    __fort_free(schr);

  __fortio_errend03();
  return (sint[0]);
}
/* 32 bit CLEN version */
int ENTF90IO(INQUIRE2003, inquire2003)(
    __INT_T *unit, DCHAR(file), __INT_T *bitv, __INT_T *iostat, bool *exist,
    bool *opened, __INT8_T *number, bool *named, DCHAR(name), DCHAR(acc),
    DCHAR(sequential), DCHAR(direct), DCHAR(form), DCHAR(formatted),
    DCHAR(unformatted), __INT8_T *recl, __INT8_T *nextrec, DCHAR(blank),
    DCHAR(position), DCHAR(action), DCHAR(read), DCHAR(write0),
    DCHAR(readwrite), DCHAR(delim),
    DCHAR(pad) DCLEN(file) DCLEN(name) DCLEN(acc) DCLEN(sequential)
    DCLEN(direct) DCLEN(form) DCLEN(formatted) DCLEN(unformatted)
    DCLEN(blank) DCLEN(position) DCLEN(action) DCLEN(read) DCLEN(write0)
    DCLEN(readwrite) DCLEN(delim) DCLEN(pad))
{
  return ENTF90IO(INQUIRE2003A, inquire2003a)(unit, CADR(file), bitv, iostat,
                  exist, opened, number, named, CADR(name), CADR(acc),
                  CADR(sequential), CADR(direct), CADR(form), CADR(formatted),
                  CADR(unformatted), recl, nextrec, CADR(blank),
                  CADR(position), CADR(action), CADR(read), CADR(write0),
                  CADR(readwrite), CADR(delim), CADR(pad),
                  (__CLEN_T)CLEN(file), (__CLEN_T)CLEN(name),
                  (__CLEN_T)CLEN(acc), (__CLEN_T)CLEN(sequential),
                  (__CLEN_T)CLEN(direct), (__CLEN_T)CLEN(form),
                  (__CLEN_T)CLEN(formatted), (__CLEN_T)CLEN(unformatted),
                  (__CLEN_T)CLEN(blank), (__CLEN_T)CLEN(position),
                  (__CLEN_T)CLEN(action), (__CLEN_T)CLEN(read),
                  (__CLEN_T)CLEN(write0), (__CLEN_T)CLEN(readwrite),
                  (__CLEN_T)CLEN(delim), (__CLEN_T)CLEN(pad)); }

int ENTCRF90IO(INQUIREa, inquirea)(
    __INT_T *unit, DCHAR(file), __INT_T *bitv, __INT_T *iostat, bool *exist,
    bool *opened, __INT_T *number, bool *named, DCHAR(name), DCHAR(acc),
    DCHAR(sequential), DCHAR(direct), DCHAR(form), DCHAR(formatted),
    DCHAR(unformatted), __INT_T *recl, __INT_T *nextrec, DCHAR(blank),
    DCHAR(position), DCHAR(action), DCHAR(read), DCHAR(write0),
    DCHAR(readwrite), DCHAR(delim),
    DCHAR(pad) DCLEN64(file) DCLEN64(name) DCLEN64(acc) DCLEN64(sequential)
    DCLEN64(direct) DCLEN64(form) DCLEN64(formatted) DCLEN64(unformatted)
    DCLEN64(blank) DCLEN64(position) DCLEN64(action) DCLEN64(read) DCLEN64(write0)
    DCLEN64(readwrite) DCLEN64(delim) DCLEN64(pad))
{
  char *file_ptr;
  char *name_ptr;
  char *acc_ptr;
  char *sequential_ptr;
  char *direct_ptr;
  char *form_ptr;
  char *formatted_ptr;
  char *unformatted_ptr;
  char *blank_ptr;
  char *position_ptr;
  char *action_ptr;
  char *read_ptr;
  char *write0_ptr;
  char *readwrite_ptr;
  char *delim_ptr;
  char *pad_ptr;
  __CLEN_T file_siz;
  __CLEN_T name_siz;
  __CLEN_T acc_siz;
  __CLEN_T sequential_siz;
  __CLEN_T direct_siz;
  __CLEN_T form_siz;
  __CLEN_T formatted_siz;
  __CLEN_T unformatted_siz;
  __CLEN_T blank_siz;
  __CLEN_T position_siz;
  __CLEN_T action_siz;
  __CLEN_T read_siz;
  __CLEN_T write0_siz;
  __CLEN_T readwrite_siz;
  __CLEN_T delim_siz;
  __CLEN_T pad_siz;
  __INT8_T newnumber;
  __INT8_T newrecl;
  __INT8_T newnextrec;

  int s;

  file_ptr = (ISPRESENTC(file) ? CADR(file) : NULL);
  name_ptr = (ISPRESENTC(name) ? CADR(name) : NULL);
  acc_ptr = (ISPRESENTC(acc) ? CADR(acc) : NULL);
  sequential_ptr = (ISPRESENTC(sequential) ? CADR(sequential) : NULL);
  direct_ptr = (ISPRESENTC(direct) ? CADR(direct) : NULL);
  form_ptr = (ISPRESENTC(form) ? CADR(form) : NULL);
  formatted_ptr = (ISPRESENTC(formatted) ? CADR(formatted) : NULL);
  unformatted_ptr = (ISPRESENTC(unformatted) ? CADR(unformatted) : NULL);
  blank_ptr = (ISPRESENTC(blank) ? CADR(blank) : NULL);
  position_ptr = (ISPRESENTC(position) ? CADR(position) : NULL);
  action_ptr = (ISPRESENTC(action) ? CADR(action) : NULL);
  read_ptr = (ISPRESENTC(read) ? CADR(read) : NULL);
  write0_ptr = (ISPRESENTC(write0) ? CADR(write0) : NULL);
  readwrite_ptr = (ISPRESENTC(readwrite) ? CADR(readwrite) : NULL);
  delim_ptr = (ISPRESENTC(delim) ? CADR(delim) : NULL);
  pad_ptr = (ISPRESENTC(pad) ? CADR(pad) : NULL);
  file_siz = CLEN(file);
  name_siz = CLEN(name);
  acc_siz = CLEN(acc);
  sequential_siz = CLEN(sequential);
  direct_siz = CLEN(direct);
  form_siz = CLEN(form);
  formatted_siz = CLEN(formatted);
  unformatted_siz = CLEN(unformatted);
  blank_siz = CLEN(blank);
  position_siz = CLEN(position);
  action_siz = CLEN(action);
  read_siz = CLEN(read);
  write0_siz = CLEN(write0);
  readwrite_siz = CLEN(readwrite);
  delim_siz = CLEN(delim);
  pad_siz = CLEN(pad);

  s = inquire(unit, file_ptr, bitv, iostat, exist, opened, &newnumber, named,
              name_ptr, acc_ptr, sequential_ptr, direct_ptr, form_ptr,
              formatted_ptr, unformatted_ptr, &newrecl, &newnextrec, blank_ptr,
              position_ptr, action_ptr, read_ptr, write0_ptr, readwrite_ptr,
              delim_ptr, pad_ptr, 0, /* id */
              0,                     /* pending */
              0,                     /* pos */
              0,                     /* size */
              0,                     /* asynchronous_ptr */
              0,                     /* decimal_ptr */
              0,                     /* encoding_ptr */
              0,                     /* sign_ptr */
              0,                     /* stream_ptr */
              0,                     /* round_ptr */
              file_siz, name_siz, acc_siz, sequential_siz, direct_siz, form_siz,
              formatted_siz, unformatted_siz, blank_siz, position_siz,
              action_siz, read_siz, write0_siz, readwrite_siz, delim_siz,
              pad_siz, 0, /* asynchronous_siz */
              0,          /* decimal_siz */
              0,          /* encoding_siz */
              0,          /* sign_siz */
              0,          /* stream_siz */
              0           /* round_siz */
              );
  if (ISPRESENT(number))
    *number = (__INT_T)newnumber;
  if (ISPRESENT(recl))
    *recl = (__INT_T)newrecl;
  if (ISPRESENT(nextrec))
    *nextrec = (__INT_T)newnextrec;

  __fortio_errend03();
  return (s);
}
/* 32 bit CLEN version */
int ENTCRF90IO(INQUIRE, inquire)(
    __INT_T *unit, DCHAR(file), __INT_T *bitv, __INT_T *iostat, bool *exist,
    bool *opened, __INT_T *number, bool *named, DCHAR(name), DCHAR(acc),
    DCHAR(sequential), DCHAR(direct), DCHAR(form), DCHAR(formatted),
    DCHAR(unformatted), __INT_T *recl, __INT_T *nextrec, DCHAR(blank),
    DCHAR(position), DCHAR(action), DCHAR(read), DCHAR(write0),
    DCHAR(readwrite), DCHAR(delim),
    DCHAR(pad) DCLEN(file) DCLEN(name) DCLEN(acc) DCLEN(sequential)
    DCLEN(direct) DCLEN(form) DCLEN(formatted) DCLEN(unformatted)
    DCLEN(blank) DCLEN(position) DCLEN(action) DCLEN(read) DCLEN(write0)
    DCLEN(readwrite) DCLEN(delim) DCLEN(pad))
{
  return ENTCRF90IO(INQUIREA, inquirea)(unit, CADR(file), bitv, iostat, exist,
                    opened, number, named, CADR(name), CADR(acc),
                    CADR(sequential), CADR(direct), CADR(form), CADR(formatted),
                    CADR(unformatted), recl, nextrec, CADR(blank),
                    CADR(position), CADR(action), CADR(read), CADR(write0),
                    CADR(readwrite), CADR(delim), CADR(pad),
                    (__CLEN_T)CLEN(file), (__CLEN_T)CLEN(name),
                    (__CLEN_T)CLEN(acc), (__CLEN_T)CLEN(sequential),
                    (__CLEN_T)CLEN(direct), (__CLEN_T)CLEN(form),
                    (__CLEN_T)CLEN(formatted), (__CLEN_T)CLEN(unformatted),
                    (__CLEN_T)CLEN(blank), (__CLEN_T)CLEN(position),
                    (__CLEN_T)CLEN(action), (__CLEN_T)CLEN(read),
                    (__CLEN_T)CLEN(write0), (__CLEN_T)CLEN(readwrite),
                    (__CLEN_T)CLEN(delim), (__CLEN_T)CLEN(pad));
}

/* new inquire 2003 */
int ENTCRF90IO(INQUIRE2003A, inquire2003a)(
    __INT_T *unit, DCHAR(file), __INT_T *bitv, __INT_T *iostat, bool *exist,
    bool *opened, __INT8_T *number, bool *named, DCHAR(name), DCHAR(acc),
    DCHAR(sequential), DCHAR(direct), DCHAR(form), DCHAR(formatted),
    DCHAR(unformatted), __INT8_T *recl, __INT8_T *nextrec, DCHAR(blank),
    DCHAR(position), DCHAR(action), DCHAR(read), DCHAR(write0),
    DCHAR(readwrite), DCHAR(delim),
    DCHAR(pad) DCLEN64(file) DCLEN64(name) DCLEN64(acc) DCLEN64(sequential)
    DCLEN64(direct) DCLEN64(form) DCLEN64(formatted) DCLEN64(unformatted)
    DCLEN64(blank) DCLEN64(position) DCLEN64(action) DCLEN64(read) DCLEN64(write0)
    DCLEN64(readwrite) DCLEN64(delim) DCLEN64(pad))
{
  char *file_ptr;
  char *name_ptr;
  char *acc_ptr;
  char *sequential_ptr;
  char *direct_ptr;
  char *form_ptr;
  char *formatted_ptr;
  char *unformatted_ptr;
  char *blank_ptr;
  char *position_ptr;
  char *action_ptr;
  char *read_ptr;
  char *write0_ptr;
  char *readwrite_ptr;
  char *delim_ptr;
  char *pad_ptr;
  __CLEN_T file_siz;
  __CLEN_T name_siz;
  __CLEN_T acc_siz;
  __CLEN_T sequential_siz;
  __CLEN_T direct_siz;
  __CLEN_T form_siz;
  __CLEN_T formatted_siz;
  __CLEN_T unformatted_siz;
  __CLEN_T blank_siz;
  __CLEN_T position_siz;
  __CLEN_T action_siz;
  __CLEN_T read_siz;
  __CLEN_T write0_siz;
  __CLEN_T readwrite_siz;
  __CLEN_T delim_siz;
  __CLEN_T pad_siz;

  int s;

  file_ptr = (ISPRESENTC(file) ? CADR(file) : NULL);
  name_ptr = (ISPRESENTC(name) ? CADR(name) : NULL);
  acc_ptr = (ISPRESENTC(acc) ? CADR(acc) : NULL);
  sequential_ptr = (ISPRESENTC(sequential) ? CADR(sequential) : NULL);
  direct_ptr = (ISPRESENTC(direct) ? CADR(direct) : NULL);
  form_ptr = (ISPRESENTC(form) ? CADR(form) : NULL);
  formatted_ptr = (ISPRESENTC(formatted) ? CADR(formatted) : NULL);
  unformatted_ptr = (ISPRESENTC(unformatted) ? CADR(unformatted) : NULL);
  blank_ptr = (ISPRESENTC(blank) ? CADR(blank) : NULL);
  position_ptr = (ISPRESENTC(position) ? CADR(position) : NULL);
  action_ptr = (ISPRESENTC(action) ? CADR(action) : NULL);
  read_ptr = (ISPRESENTC(read) ? CADR(read) : NULL);
  write0_ptr = (ISPRESENTC(write0) ? CADR(write0) : NULL);
  readwrite_ptr = (ISPRESENTC(readwrite) ? CADR(readwrite) : NULL);
  delim_ptr = (ISPRESENTC(delim) ? CADR(delim) : NULL);
  pad_ptr = (ISPRESENTC(pad) ? CADR(pad) : NULL);
  file_siz = CLEN(file);
  name_siz = CLEN(name);
  acc_siz = CLEN(acc);
  sequential_siz = CLEN(sequential);
  direct_siz = CLEN(direct);
  form_siz = CLEN(form);
  formatted_siz = CLEN(formatted);
  unformatted_siz = CLEN(unformatted);
  blank_siz = CLEN(blank);
  position_siz = CLEN(position);
  action_siz = CLEN(action);
  read_siz = CLEN(read);
  write0_siz = CLEN(write0);
  readwrite_siz = CLEN(readwrite);
  delim_siz = CLEN(delim);
  pad_siz = CLEN(pad);

  s = inquire(unit, file_ptr, bitv, iostat, exist, opened, number, named,
              name_ptr, acc_ptr, sequential_ptr, direct_ptr, form_ptr,
              formatted_ptr, unformatted_ptr, recl, nextrec, blank_ptr,
              position_ptr, action_ptr, read_ptr, write0_ptr, readwrite_ptr,
              delim_ptr, pad_ptr, 0, /* id */
              0,                     /* pending */
              0,                     /* pos */
              0,                     /* size */
              0,                     /* asynchronous_ptr */
              0,                     /* decimal_ptr */
              0,                     /* encoding_ptr */
              0,                     /* sign_ptr */
              0,                     /* stream_ptr */
              0,                     /* round_ptr */
              file_siz, name_siz, acc_siz, sequential_siz, direct_siz, form_siz,
              formatted_siz, unformatted_siz, blank_siz, position_siz,
              action_siz, read_siz, write0_siz, readwrite_siz, delim_siz,
              pad_siz, 0, /* asynchronous_siz */
              0,          /* decimal_siz */
              0,          /* encoding_siz */
              0,          /* sign_siz */
              0,          /* stream_siz */
              0           /* round_siz */
              );

  __fortio_errend03();
  return (s);
}
/* 32 bit CLEN version */
int ENTCRF90IO(INQUIRE2003, inquire2003)(
    __INT_T *unit, DCHAR(file), __INT_T *bitv, __INT_T *iostat, bool *exist,
    bool *opened, __INT8_T *number, bool *named, DCHAR(name), DCHAR(acc),
    DCHAR(sequential), DCHAR(direct), DCHAR(form), DCHAR(formatted),
    DCHAR(unformatted), __INT8_T *recl, __INT8_T *nextrec, DCHAR(blank),
    DCHAR(position), DCHAR(action), DCHAR(read), DCHAR(write0),
    DCHAR(readwrite), DCHAR(delim),
    DCHAR(pad) DCLEN(file) DCLEN(name) DCLEN(acc) DCLEN(sequential)
    DCLEN(direct) DCLEN(form) DCLEN(formatted) DCLEN(unformatted)
    DCLEN(blank) DCLEN(position) DCLEN(action) DCLEN(read) DCLEN(write0)
    DCLEN(readwrite) DCLEN(delim) DCLEN(pad))
{
  return ENTCRF90IO(INQUIRE2003A, inquire2003a)(unit, CADR(file), bitv, iostat,
                    exist, opened, number, named, CADR(name), CADR(acc),
                    CADR(sequential), CADR(direct), CADR(form), CADR(formatted),
                    CADR(unformatted), recl, nextrec, CADR(blank),
                    CADR(position), CADR(action), CADR(read), CADR(write0),
                    CADR(readwrite), CADR(delim), CADR(pad),
                    (__CLEN_T)CLEN(file), (__CLEN_T)CLEN(name),
                    (__CLEN_T)CLEN(acc), (__CLEN_T)CLEN(sequential),
                    (__CLEN_T)CLEN(direct), (__CLEN_T)CLEN(form),
                    (__CLEN_T)CLEN(formatted), (__CLEN_T)CLEN(unformatted),
                    (__CLEN_T)CLEN(blank), (__CLEN_T)CLEN(position),
                    (__CLEN_T)CLEN(action), (__CLEN_T)CLEN(read),
                    (__CLEN_T)CLEN(write0), (__CLEN_T)CLEN(readwrite),
                    (__CLEN_T)CLEN(delim), (__CLEN_T)CLEN(pad));
}

/* additional inquire options for 2003 spec */

__INT_T
ENTF90IO(INQUIRE2A, inquire2a)
(__INT_T *istat, /* status of OPEN */
 __INT_T *id, __INT_T *pending, __INT8_T *pos, __INT_T *size,
 DCHAR(asynchronous), DCHAR(decimal), DCHAR(encoding), DCHAR(sign),
 DCHAR(stream) DCLEN64(asynchronous) DCLEN64(decimal) DCLEN64(encoding) DCLEN64(sign)
     DCLEN64(stream))
{
  FIO_FCB *f;
  const char *cp;

  if (*istat)
    return *istat;

  f = f2;

  if (id != NULL) {
    *id = 0;
  }
  if (pending != NULL) {
    *pending = FTN_FALSE;
  }
  if (pos != NULL) {
    if (f != NULL)
      *pos = __io_ftellx(f->fp) + 1;
  }
  if (size != NULL) {
    if (f != NULL) {
      seekoffx_t currpos;
      currpos = (seekoffx_t)__io_ftellx(f->fp);
      if (__io_fseek(f->fp, 0L, SEEK_END) != 0)
        return (__fortio_error(__io_errno()));
      *size = __io_ftellx(f->fp);
      __io_fseek(f->fp, currpos, SEEK_SET);
    } else
      *size = -1;
  }
  if (ISPRESENTC(asynchronous)) {
    if (f == NULL)
      cp = "UNDEFINED";
    else if (f->asyptr)
      cp = "YES";
    else
      cp = "NO";
    copystr(CADR(asynchronous), CLEN(asynchronous), cp);
  }
  if (ISPRESENTC(decimal)) {
    /* "DECIMAL" "POINT" "UNDEFINED" */
    if (f == NULL || f->form != FIO_FORMATTED)
      cp = "UNDEFINED";
    else if (f->decimal == FIO_COMMA)
      cp = "COMMA";
    else if (f->decimal == FIO_POINT)
      cp = "POINT";
    else
      cp = "POINT";
    copystr(CADR(decimal), CLEN(decimal), cp);
  }
  if (ISPRESENTC(encoding)) {
    /* "UTF-8" "DEFAULT" "UNDEFINED" "UNKNOWN" */
    if (f == NULL)
      cp = "UNKNOWN";
    else if (f->form != FIO_FORMATTED)
      cp = "UNDEFINED";
    else if (f->encoding == FIO_UTF_8)
      cp = "UTF-8";
    else
      cp = "DEFAULT";
    copystr(CADR(encoding), CLEN(encoding), cp);
  }
  if (ISPRESENTC(sign)) {
    /* "PLUS" "SUPPRESS" "PROCESSOR_DEFINED" */
    if (f == NULL || f->form != FIO_FORMATTED)
      cp = "UNDEFINED";
    else if (f->sign == FIO_PLUS)
      cp = "PLUS";
    else if (f->sign == FIO_SUPPRESS)
      cp = "SUPPRESS";
    else
      cp = "PROCESSOR_DEFINED";
    copystr(CADR(sign), CLEN(sign), cp);
  }
  if (ISPRESENTC(stream)) {
    if ((f == NULL) || (f->acc == FIO_STREAM))
      cp = "YES";
    else
      cp = "NO";
    copystr(CADR(stream), CLEN(stream), cp);
  }

  return (0);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(INQUIRE2, inquire2)
(__INT_T *istat, /* status of OPEN */
 __INT_T *id, __INT_T *pending, __INT8_T *pos, __INT_T *size,
 DCHAR(asynchronous), DCHAR(decimal), DCHAR(encoding), DCHAR(sign),
 DCHAR(stream) DCLEN(asynchronous) DCLEN(decimal) DCLEN(encoding) DCLEN(sign)
     DCLEN(stream))
{
  return ENTF90IO(INQUIRE2A, inquire2a) (istat, id, pending, pos, size,
                  CADR(asynchronous), CADR(decimal), CADR(encoding), CADR(sign),
                  CADR(stream), (__CLEN_T)CLEN(asynchronous),
                  (__CLEN_T)CLEN(decimal), (__CLEN_T)CLEN(encoding),
                  (__CLEN_T)CLEN(sign), (__CLEN_T)CLEN(stream));
}

/* f2003 version of inquire */

__INT_T
ENTF90IO(INQUIRE03A, inquire03a) (
  __INT_T *unit,
  DCHAR(file),
  __INT_T *bitv,
  __INT_T *iostat,
  bool *exist,
  bool *opened,
  __INT_T *number,
  bool *named,
  DCHAR(name),
  DCHAR(acc),
  DCHAR(sequential),
  DCHAR(direct),
  DCHAR(form),
  DCHAR(formatted),
  DCHAR(unformatted),
  __INT_T *recl,
  __INT_T *nextrec,
  DCHAR(blank),
  DCHAR(position),
  DCHAR(action),
  DCHAR(read),
  DCHAR(write0),
  DCHAR(readwrite),
  DCHAR(delim),
  DCHAR(pad),
  __INT_T  *id,
  __INT_T  *pending,
  __INT8_T *pos,
  __INT_T  *size,
  DCHAR(asynchronous),
  DCHAR(decimal),
  DCHAR(encoding),
  DCHAR(sign),
  DCHAR(stream),
  DCHAR(round)
  DCLEN64(file)
  DCLEN64(name)
  DCLEN64(acc)
  DCLEN64(sequential)
  DCLEN64(direct)
  DCLEN64(form)
  DCLEN64(formatted)
  DCLEN64(unformatted)
  DCLEN64(blank)
  DCLEN64(position)
  DCLEN64(action)
  DCLEN64(read)
  DCLEN64(write0)
  DCLEN64(readwrite)
  DCLEN64(delim)
  DCLEN64(pad)
  DCLEN64(asynchronous)
  DCLEN64(decimal)
  DCLEN64(encoding)
  DCLEN64(sign)
  DCLEN64(stream)
  DCLEN64(round)
  )
{
  char *file_ptr;
  char *name_ptr;
  char *acc_ptr;
  char *sequential_ptr;
  char *direct_ptr;
  char *form_ptr;
  char *formatted_ptr;
  char *unformatted_ptr;
  char *blank_ptr;
  char *position_ptr;
  char *action_ptr;
  char *read_ptr;
  char *write0_ptr;
  char *readwrite_ptr;
  char *delim_ptr;
  char *pad_ptr;
  char *asynchronous_ptr;
  char *decimal_ptr;
  char *encoding_ptr;
  char *sign_ptr;
  char *stream_ptr;
  char *round_ptr;

  __CLEN_T file_siz;
  __CLEN_T name_siz;
  __CLEN_T acc_siz;
  __CLEN_T sequential_siz;
  __CLEN_T direct_siz;
  __CLEN_T form_siz;
  __CLEN_T formatted_siz;
  __CLEN_T unformatted_siz;
  __CLEN_T blank_siz;
  __CLEN_T position_siz;
  __CLEN_T action_siz;
  __CLEN_T read_siz;
  __CLEN_T write0_siz;
  __CLEN_T readwrite_siz;
  __CLEN_T delim_siz;
  __CLEN_T pad_siz;
  __CLEN_T asynchronous_siz;
  __CLEN_T decimal_siz;
  __CLEN_T encoding_siz;
  __CLEN_T sign_siz;
  __CLEN_T stream_siz;
  __CLEN_T round_siz;
  __INT8_T newnumber;
  __INT8_T newrecl;
  __INT8_T newnextrec;
  __INT8_T newsize;

  int s;
  int ioproc;

  __INT8_T sint[SINTN], *sintp;
  char *schr = 0, *schrp;
  __CLEN_T schr_len = 0;

  file_ptr = (ISPRESENTC(file) ? CADR(file) : NULL);
  name_ptr = (ISPRESENTC(name) ? CADR(name) : NULL);
  acc_ptr = (ISPRESENTC(acc) ? CADR(acc) : NULL);
  sequential_ptr = (ISPRESENTC(sequential) ? CADR(sequential) : NULL);
  direct_ptr = (ISPRESENTC(direct) ? CADR(direct) : NULL);
  form_ptr = (ISPRESENTC(form) ? CADR(form) : NULL);
  formatted_ptr = (ISPRESENTC(formatted) ? CADR(formatted) : NULL);
  unformatted_ptr = (ISPRESENTC(unformatted) ? CADR(unformatted) : NULL);
  blank_ptr = (ISPRESENTC(blank) ? CADR(blank) : NULL);
  position_ptr = (ISPRESENTC(position) ? CADR(position) : NULL);
  action_ptr = (ISPRESENTC(action) ? CADR(action) : NULL);
  read_ptr = (ISPRESENTC(read) ? CADR(read) : NULL);
  write0_ptr = (ISPRESENTC(write0) ? CADR(write0) : NULL);
  readwrite_ptr = (ISPRESENTC(readwrite) ? CADR(readwrite) : NULL);
  delim_ptr = (ISPRESENTC(delim) ? CADR(delim) : NULL);
  pad_ptr = (ISPRESENTC(pad) ? CADR(pad) : NULL);
  asynchronous_ptr = (ISPRESENTC(asynchronous) ? CADR(asynchronous) : NULL);
  decimal_ptr = (ISPRESENTC(decimal) ? CADR(decimal) : NULL);
  encoding_ptr = (ISPRESENTC(encoding) ? CADR(encoding) : NULL);
  sign_ptr = (ISPRESENTC(sign) ? CADR(sign) : NULL);
  stream_ptr = (ISPRESENTC(stream) ? CADR(stream) : NULL);
  round_ptr = (ISPRESENTC(round) ? CADR(round) : NULL);

  file_siz = CLEN(file);
  schr_len += (ISPRESENTC(file) ? file_siz : 0);
  name_siz = CLEN(name);
  schr_len += (ISPRESENTC(name) ? name_siz : 0);
  acc_siz = CLEN(acc);
  schr_len += (ISPRESENTC(acc) ? acc_siz : 0);
  sequential_siz = CLEN(sequential);
  schr_len += (ISPRESENTC(sequential) ? sequential_siz : 0);
  direct_siz = CLEN(direct);
  schr_len += (ISPRESENTC(direct) ? direct_siz : 0);
  form_siz = CLEN(form);
  schr_len += (ISPRESENTC(form) ? form_siz : 0);
  formatted_siz = CLEN(formatted);
  schr_len += (ISPRESENTC(formatted) ? formatted_siz : 0);
  unformatted_siz = CLEN(unformatted);
  schr_len += (ISPRESENTC(unformatted) ? unformatted_siz : 0);
  blank_siz = CLEN(blank);
  schr_len += (ISPRESENTC(blank) ? blank_siz : 0);
  position_siz = CLEN(position);
  schr_len += (ISPRESENTC(position) ? position_siz : 0);
  action_siz = CLEN(action);
  schr_len += (ISPRESENTC(action) ? action_siz : 0);
  read_siz = CLEN(read);
  schr_len += (ISPRESENTC(read) ? read_siz : 0);
  write0_siz = CLEN(write0);
  schr_len += (ISPRESENTC(write0) ? write0_siz : 0);
  readwrite_siz = CLEN(readwrite);
  schr_len += (ISPRESENTC(readwrite) ? readwrite_siz : 0);
  delim_siz = CLEN(delim);
  schr_len += (ISPRESENTC(delim) ? delim_siz : 0);
  pad_siz = CLEN(pad);
  schr_len += (ISPRESENTC(pad) ? pad_siz : 0);
  asynchronous_siz = CLEN(asynchronous);
  schr_len += (ISPRESENTC(asynchronous) ? asynchronous_siz : 0);
  decimal_siz = CLEN(decimal);
  schr_len += (ISPRESENTC(decimal) ? decimal_siz : 0);
  encoding_siz = CLEN(encoding);
  schr_len += (ISPRESENTC(encoding) ? encoding_siz : 0);
  sign_siz = CLEN(sign);
  schr_len += (ISPRESENTC(sign) ? sign_siz : 0);
  stream_siz = CLEN(stream);
  schr_len += (ISPRESENTC(stream) ? stream_siz : 0);
  round_siz = CLEN(round);
  schr_len += (ISPRESENTC(round) ? round_siz : 0);

  if (schr_len)
    schr = (char *)__fort_malloc(sizeof(char) * schr_len);

  sintp = sint + 1;
  schrp = schr;

  /* non i/o processors */

  ioproc = GET_DIST_IOPROC;
  if ((GET_DIST_LCPU != ioproc) && (!LOCAL_MODE)) {

    DIST_RBCST(ioproc, sint, SINTN, 1, __CINT);
    DIST_RBCST(ioproc, schr, schr_len, 1, __CHAR);

    if (*bitv & FIO_BITV_IOSTAT) {
      *iostat = *sintp++;
    }
    if (ISPRESENTC(acc)) {
      memcpy(acc_ptr, schrp, acc_siz);
      schrp += acc_siz;
    }
    if (ISPRESENTC(action)) {
      memcpy(action_ptr, schrp, action_siz);
      schrp += action_siz;
    }
    if (ISPRESENTC(blank)) {
      memcpy(blank_ptr, schrp, blank_siz);
      schrp += blank_siz;
    }
    if (ISPRESENTC(delim)) {
      memcpy(delim_ptr, schrp, delim_siz);
      schrp += delim_siz;
    }
    if (ISPRESENTC(direct)) {
      memcpy(direct_ptr, schrp, direct_siz);
      schrp += direct_siz;
    }
    if (ISPRESENT(exist)) {
      *exist = *sintp++;
    }
    if (ISPRESENTC(form)) {
      memcpy(form_ptr, schrp, form_siz);
      schrp += form_siz;
    }
    if (ISPRESENTC(formatted)) {
      memcpy(formatted_ptr, schrp, formatted_siz);
      schrp += formatted_siz;
    }
    if (ISPRESENTC(name)) {
      memcpy(name_ptr, schrp, name_siz);
      schrp += name_siz;
    }
    if (ISPRESENT(named)) {
      *named = *sintp++;
    }
    if (ISPRESENT(nextrec)) {
      *nextrec = *sintp++;
    }
    if (ISPRESENT(number)) {
      *number = *sintp++;
    }
    if (ISPRESENT(opened)) {
      *opened = *sintp++;
    }
    if (ISPRESENTC(pad)) {
      memcpy(pad_ptr, schrp, pad_siz);
      schrp += pad_siz;
    }
    if (ISPRESENTC(position)) {
      memcpy(position_ptr, schrp, position_siz);
      schrp += position_siz;
    }
    if (ISPRESENTC(read)) {
      memcpy(read_ptr, schrp, read_siz);
      schrp += read_siz;
    }
    if (ISPRESENTC(readwrite)) {
      memcpy(readwrite_ptr, schrp, readwrite_siz);
      schrp += readwrite_siz;
    }
    if (ISPRESENT(recl)) {
      *recl = *sintp++;
    }
    if (ISPRESENTC(sequential)) {
      memcpy(sequential_ptr, schrp, sequential_siz);
      schrp += sequential_siz;
    }
    if (ISPRESENTC(unformatted)) {
      memcpy(unformatted_ptr, schrp, unformatted_siz);
      schrp += unformatted_siz;
    }
    if (ISPRESENTC(write0)) {
      memcpy(write0_ptr, schrp, write0_siz);
      schrp += write0_siz;
    }
    if (ISPRESENT(id)) {
      *id = *sintp++;
    }
    if (ISPRESENT(pending)) {
      *pending = *sintp++;
    }
    if (ISPRESENT(pos)) {
      *pos = *sintp++;
    }
    if (ISPRESENT(size)) {
      *size = *sintp++;
    }
    if (ISPRESENTC(asynchronous)) {
      memcpy(asynchronous_ptr, schrp, asynchronous_siz);
      schrp += asynchronous_siz;
    }
    if (ISPRESENTC(decimal)) {
      memcpy(decimal_ptr, schrp, decimal_siz);
      schrp += decimal_siz;
    }
    if (ISPRESENTC(encoding)) {
      memcpy(encoding_ptr, schrp, encoding_siz);
      schrp += encoding_siz;
    }
    if (ISPRESENTC(sign)) {
      memcpy(sign_ptr, schrp, sign_siz);
      schrp += sign_siz;
    }
    if (ISPRESENTC(stream)) {
      memcpy(stream_ptr, schrp, stream_siz);
      schrp += stream_siz;
    }
    if (ISPRESENTC(round)) {
      memcpy(round_ptr, schrp, round_siz);
      schrp += round_siz;
    }

    return (sint[0]);
  }

  /* i/o processor */

  s = inquire(unit, file_ptr, bitv, iostat, exist, opened, &newnumber, named,
              name_ptr, acc_ptr, sequential_ptr, direct_ptr, form_ptr,
              formatted_ptr, unformatted_ptr, &newrecl, &newnextrec, blank_ptr,
              position_ptr, action_ptr, read_ptr, write0_ptr, readwrite_ptr,
              delim_ptr, pad_ptr, id, pending, pos, &newsize, asynchronous_ptr,
              decimal_ptr, encoding_ptr, sign_ptr, stream_ptr, round_ptr,
              file_siz, name_siz, acc_siz, sequential_siz, direct_siz, form_siz,
              formatted_siz, unformatted_siz, blank_siz, position_siz,
              action_siz, read_siz, write0_siz, readwrite_siz, delim_siz,
              pad_siz, asynchronous_siz, decimal_siz, encoding_siz, sign_siz,
              stream_siz, round_siz);

  sint[0] = s;

  if (*bitv & FIO_BITV_IOSTAT) {
    *sintp++ = *iostat;
  }
  if (ISPRESENTC(acc)) {
    memcpy(schrp, CADR(acc), CLEN(acc));
    schrp += CLEN(acc);
  }
  if (ISPRESENTC(action)) {
    memcpy(schrp, CADR(action), CLEN(action));
    schrp += CLEN(action);
  }
  if (ISPRESENTC(blank)) {
    memcpy(schrp, CADR(blank), CLEN(blank));
    schrp += CLEN(blank);
  }
  if (ISPRESENTC(delim)) {
    memcpy(schrp, CADR(delim), CLEN(delim));
    schrp += CLEN(delim);
  }
  if (ISPRESENTC(direct)) {
    memcpy(schrp, CADR(direct), CLEN(direct));
    schrp += CLEN(direct);
  }
  if (ISPRESENT(exist)) {
    *sintp++ = *exist;
  }
  if (ISPRESENTC(form)) {
    memcpy(schrp, CADR(form), CLEN(form));
    schrp += CLEN(form);
  }
  if (ISPRESENTC(formatted)) {
    memcpy(schrp, CADR(formatted), CLEN(formatted));
    schrp += CLEN(formatted);
  }
  if (ISPRESENTC(name)) {
    memcpy(schrp, CADR(name), CLEN(name));
    schrp += CLEN(name);
  }
  if (ISPRESENT(named)) {
    *sintp++ = *named;
  }
  if (ISPRESENT(nextrec)) {
    *nextrec = (__INT_T)newnextrec;
    *sintp++ = *nextrec;
  }
  if (ISPRESENT(number)) {
    *number = (__INT_T)newnumber;
    *sintp++ = *number;
  }
  if (ISPRESENT(opened)) {
    *sintp++ = *opened;
  }
  if (ISPRESENTC(pad)) {
    memcpy(schrp, CADR(pad), CLEN(pad));
    schrp += CLEN(pad);
  }
  if (ISPRESENTC(position)) {
    memcpy(schrp, CADR(position), CLEN(position));
    schrp += CLEN(position);
  }
  if (ISPRESENTC(read)) {
    memcpy(schrp, CADR(read), CLEN(read));
    schrp += CLEN(read);
  }
  if (ISPRESENTC(readwrite)) {
    memcpy(schrp, CADR(readwrite), CLEN(readwrite));
    schrp += CLEN(readwrite);
  }
  if (ISPRESENT(recl)) {
    *recl = (__INT_T)newrecl;
    *sintp++ = *recl;
  }
  if (ISPRESENTC(sequential)) {
    memcpy(schrp, CADR(sequential), CLEN(sequential));
    schrp += CLEN(sequential);
  }
  if (ISPRESENTC(unformatted)) {
    memcpy(schrp, CADR(unformatted), CLEN(unformatted));
    schrp += CLEN(unformatted);
  }
  if (ISPRESENTC(write0)) {
    memcpy(schrp, CADR(write0), CLEN(write0));
    schrp += CLEN(write0);
  }
  if (ISPRESENT(id)) {
    *sintp++ = *id;
  }
  if (ISPRESENT(pending)) {
    *sintp++ = *pending;
  }
  if (ISPRESENT(pos)) {
    *sintp++ = *pos;
  }
  if (ISPRESENT(size)) {
    *size = (__INT_T)newsize;
    *sintp++ = *size;
  }
  if (ISPRESENTC(asynchronous)) {
    memcpy(schrp, CADR(asynchronous), CLEN(asynchronous));
    schrp += CLEN(asynchronous);
  }
  if (ISPRESENTC(decimal)) {
    memcpy(schrp, CADR(decimal), CLEN(decimal));
    schrp += CLEN(decimal);
  }
  if (ISPRESENTC(encoding)) {
    memcpy(schrp, CADR(encoding), CLEN(encoding));
    schrp += CLEN(encoding);
  }
  if (ISPRESENTC(sign)) {
    memcpy(schrp, CADR(sign), CLEN(sign));
    schrp += CLEN(sign);
  }
  if (ISPRESENTC(stream)) {
    memcpy(schrp, CADR(stream), CLEN(stream));
    schrp += CLEN(stream);
  }
  if (ISPRESENTC(round)) {
    memcpy(schrp, CADR(round), CLEN(round));
    schrp += CLEN(round);
  }

  if (!LOCAL_MODE) {
    DIST_RBCST(ioproc, sint, SINTN, 1, __INT8);
    DIST_RBCST(ioproc, schr, schr_len, 1, __CHAR);
  }

  if (schr)
    __fort_free(schr);

  __fortio_errend03();
  return (sint[0]);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(INQUIRE03, inquire03) (
  __INT_T *unit,
  DCHAR(file),
  __INT_T *bitv,
  __INT_T *iostat,
  bool *exist,
  bool *opened,
  __INT_T *number,
  bool *named,
  DCHAR(name),
  DCHAR(acc),
  DCHAR(sequential),
  DCHAR(direct),
  DCHAR(form),
  DCHAR(formatted),
  DCHAR(unformatted),
  __INT_T *recl,
  __INT_T *nextrec,
  DCHAR(blank),
  DCHAR(position),
  DCHAR(action),
  DCHAR(read),
  DCHAR(write0),
  DCHAR(readwrite),
  DCHAR(delim),
  DCHAR(pad),
  __INT_T  *id,
  __INT_T  *pending,
  __INT8_T *pos,
  __INT_T  *size,
  DCHAR(asynchronous),
  DCHAR(decimal),
  DCHAR(encoding),
  DCHAR(sign),
  DCHAR(stream),
  DCHAR(round)
  DCLEN(file)
  DCLEN(name)
  DCLEN(acc)
  DCLEN(sequential)
  DCLEN(direct)
  DCLEN(form)
  DCLEN(formatted)
  DCLEN(unformatted)
  DCLEN(blank)
  DCLEN(position)
  DCLEN(action)
  DCLEN(read)
  DCLEN(write0)
  DCLEN(readwrite)
  DCLEN(delim)
  DCLEN(pad)
  DCLEN(asynchronous)
  DCLEN(decimal)
  DCLEN(encoding)
  DCLEN(sign)
  DCLEN(stream)
  DCLEN(round)
  )
{

  return ENTF90IO(INQUIRE03A, inquire03a) (unit, CADR(file), bitv, iostat,
                  exist, opened, number, named, CADR(name), CADR(acc),
                  CADR(sequential), CADR(direct), CADR(form), CADR(formatted),
                  CADR(unformatted), recl, nextrec, CADR(blank),
                  CADR(position), CADR(action), CADR(read), CADR(write0),
                  CADR(readwrite), CADR(delim), CADR(pad), id, pending, pos,
                  size, CADR(asynchronous), CADR(decimal), CADR(encoding),
                  CADR(sign), CADR(stream), CADR(round), (__CLEN_T)CLEN(file),
                  (__CLEN_T)CLEN(name), (__CLEN_T)CLEN(acc),
                  (__CLEN_T)CLEN(sequential), (__CLEN_T)CLEN(direct),
                  (__CLEN_T)CLEN(form), (__CLEN_T)CLEN(formatted),
                  (__CLEN_T)CLEN(unformatted), (__CLEN_T)CLEN(blank),
                  (__CLEN_T)CLEN(position), (__CLEN_T)CLEN(action),
                  (__CLEN_T)CLEN(read), (__CLEN_T)CLEN(write0),
                  (__CLEN_T)CLEN(readwrite), (__CLEN_T)CLEN(delim),
                  (__CLEN_T)CLEN(pad), (__CLEN_T)CLEN(asynchronous),
                  (__CLEN_T)CLEN(decimal), (__CLEN_T)CLEN(encoding),
                  (__CLEN_T)CLEN(sign), (__CLEN_T)CLEN(stream),
                  (__CLEN_T)CLEN(round));
}

/* new f2003 version of inquire */

__INT_T
ENTF90IO(INQUIRE03_2A, inquire03_2a)
(__INT_T *unit, DCHAR(file), __INT_T *bitv, __INT_T *iostat, bool *exist,
 bool *opened, __INT8_T *number, bool *named, DCHAR(name), DCHAR(acc),
 DCHAR(sequential), DCHAR(direct), DCHAR(form), DCHAR(formatted),
 DCHAR(unformatted), __INT8_T *recl, __INT8_T *nextrec, DCHAR(blank),
 DCHAR(position), DCHAR(action), DCHAR(read), DCHAR(write0), DCHAR(readwrite),
 DCHAR(delim), DCHAR(pad), __INT_T *id, __INT_T *pending, __INT8_T *pos,
 __INT8_T *size, DCHAR(asynchronous), DCHAR(decimal), DCHAR(encoding),
 DCHAR(sign), DCHAR(stream),
 DCHAR(round) DCLEN64(file) DCLEN64(name) DCLEN64(acc) DCLEN64(sequential) DCLEN64(direct)
 DCLEN64(form) DCLEN64(formatted) DCLEN64(unformatted) DCLEN64(blank)
 DCLEN64(position) DCLEN64(action) DCLEN64(read) DCLEN64(write0)
 DCLEN64(readwrite) DCLEN64(delim) DCLEN64(pad) DCLEN64(asynchronous)
 DCLEN64(decimal) DCLEN64(encoding) DCLEN64(sign) DCLEN64(stream)
 DCLEN64(round))
{
  char *file_ptr;
  char *name_ptr;
  char *acc_ptr;
  char *sequential_ptr;
  char *direct_ptr;
  char *form_ptr;
  char *formatted_ptr;
  char *unformatted_ptr;
  char *blank_ptr;
  char *position_ptr;
  char *action_ptr;
  char *read_ptr;
  char *write0_ptr;
  char *readwrite_ptr;
  char *delim_ptr;
  char *pad_ptr;
  char *asynchronous_ptr;
  char *decimal_ptr;
  char *encoding_ptr;
  char *sign_ptr;
  char *stream_ptr;
  char *round_ptr;

  __CLEN_T file_siz;
  __CLEN_T name_siz;
  __CLEN_T acc_siz;
  __CLEN_T sequential_siz;
  __CLEN_T direct_siz;
  __CLEN_T form_siz;
  __CLEN_T formatted_siz;
  __CLEN_T unformatted_siz;
  __CLEN_T blank_siz;
  __CLEN_T position_siz;
  __CLEN_T action_siz;
  __CLEN_T read_siz;
  __CLEN_T write0_siz;
  __CLEN_T readwrite_siz;
  __CLEN_T delim_siz;
  __CLEN_T pad_siz;
  __CLEN_T asynchronous_siz;
  __CLEN_T decimal_siz;
  __CLEN_T encoding_siz;
  __CLEN_T sign_siz;
  __CLEN_T stream_siz;
  __CLEN_T round_siz;

  int s;
  int ioproc;

  __INT8_T sint[SINTN], *sintp;
  char *schr = 0, *schrp;
  __CLEN_T schr_len = 0;

  file_ptr = (ISPRESENTC(file) ? CADR(file) : NULL);
  name_ptr = (ISPRESENTC(name) ? CADR(name) : NULL);
  acc_ptr = (ISPRESENTC(acc) ? CADR(acc) : NULL);
  sequential_ptr = (ISPRESENTC(sequential) ? CADR(sequential) : NULL);
  direct_ptr = (ISPRESENTC(direct) ? CADR(direct) : NULL);
  form_ptr = (ISPRESENTC(form) ? CADR(form) : NULL);
  formatted_ptr = (ISPRESENTC(formatted) ? CADR(formatted) : NULL);
  unformatted_ptr = (ISPRESENTC(unformatted) ? CADR(unformatted) : NULL);
  blank_ptr = (ISPRESENTC(blank) ? CADR(blank) : NULL);
  position_ptr = (ISPRESENTC(position) ? CADR(position) : NULL);
  action_ptr = (ISPRESENTC(action) ? CADR(action) : NULL);
  read_ptr = (ISPRESENTC(read) ? CADR(read) : NULL);
  write0_ptr = (ISPRESENTC(write0) ? CADR(write0) : NULL);
  readwrite_ptr = (ISPRESENTC(readwrite) ? CADR(readwrite) : NULL);
  delim_ptr = (ISPRESENTC(delim) ? CADR(delim) : NULL);
  pad_ptr = (ISPRESENTC(pad) ? CADR(pad) : NULL);
  asynchronous_ptr = (ISPRESENTC(asynchronous) ? CADR(asynchronous) : NULL);
  decimal_ptr = (ISPRESENTC(decimal) ? CADR(decimal) : NULL);
  encoding_ptr = (ISPRESENTC(encoding) ? CADR(encoding) : NULL);
  sign_ptr = (ISPRESENTC(sign) ? CADR(sign) : NULL);
  stream_ptr = (ISPRESENTC(stream) ? CADR(stream) : NULL);
  round_ptr = (ISPRESENTC(round) ? CADR(round) : NULL);

  file_siz = CLEN(file);
  schr_len += (ISPRESENTC(file) ? file_siz : 0);
  name_siz = CLEN(name);
  schr_len += (ISPRESENTC(name) ? name_siz : 0);
  acc_siz = CLEN(acc);
  schr_len += (ISPRESENTC(acc) ? acc_siz : 0);
  sequential_siz = CLEN(sequential);
  schr_len += (ISPRESENTC(sequential) ? sequential_siz : 0);
  direct_siz = CLEN(direct);
  schr_len += (ISPRESENTC(direct) ? direct_siz : 0);
  form_siz = CLEN(form);
  schr_len += (ISPRESENTC(form) ? form_siz : 0);
  formatted_siz = CLEN(formatted);
  schr_len += (ISPRESENTC(formatted) ? formatted_siz : 0);
  unformatted_siz = CLEN(unformatted);
  schr_len += (ISPRESENTC(unformatted) ? unformatted_siz : 0);
  blank_siz = CLEN(blank);
  schr_len += (ISPRESENTC(blank) ? blank_siz : 0);
  position_siz = CLEN(position);
  schr_len += (ISPRESENTC(position) ? position_siz : 0);
  action_siz = CLEN(action);
  schr_len += (ISPRESENTC(action) ? action_siz : 0);
  read_siz = CLEN(read);
  schr_len += (ISPRESENTC(read) ? read_siz : 0);
  write0_siz = CLEN(write0);
  schr_len += (ISPRESENTC(write0) ? write0_siz : 0);
  readwrite_siz = CLEN(readwrite);
  schr_len += (ISPRESENTC(readwrite) ? readwrite_siz : 0);
  delim_siz = CLEN(delim);
  schr_len += (ISPRESENTC(delim) ? delim_siz : 0);
  pad_siz = CLEN(pad);
  schr_len += (ISPRESENTC(pad) ? pad_siz : 0);
  asynchronous_siz = CLEN(asynchronous);
  schr_len += (ISPRESENTC(asynchronous) ? asynchronous_siz : 0);
  decimal_siz = CLEN(decimal);
  schr_len += (ISPRESENTC(decimal) ? decimal_siz : 0);
  encoding_siz = CLEN(encoding);
  schr_len += (ISPRESENTC(encoding) ? encoding_siz : 0);
  sign_siz = CLEN(sign);
  schr_len += (ISPRESENTC(sign) ? sign_siz : 0);
  stream_siz = CLEN(stream);
  schr_len += (ISPRESENTC(stream) ? stream_siz : 0);
  round_siz = CLEN(round);
  schr_len += (ISPRESENTC(round) ? round_siz : 0);

  if (schr_len)
    schr = (char *)__fort_malloc(sizeof(char) * schr_len);

  sintp = sint + 1;
  schrp = schr;

  /* non i/o processors */

  ioproc = GET_DIST_IOPROC;
  if ((GET_DIST_LCPU != ioproc) && (!LOCAL_MODE)) {

    DIST_RBCST(ioproc, sint, SINTN, 1, __CINT);
    DIST_RBCST(ioproc, schr, schr_len, 1, __CHAR);

    if (*bitv & FIO_BITV_IOSTAT) {
      *iostat = *sintp++;
    }
    if (ISPRESENTC(acc)) {
      memcpy(acc_ptr, schrp, acc_siz);
      schrp += acc_siz;
    }
    if (ISPRESENTC(action)) {
      memcpy(action_ptr, schrp, action_siz);
      schrp += action_siz;
    }
    if (ISPRESENTC(blank)) {
      memcpy(blank_ptr, schrp, blank_siz);
      schrp += blank_siz;
    }
    if (ISPRESENTC(delim)) {
      memcpy(delim_ptr, schrp, delim_siz);
      schrp += delim_siz;
    }
    if (ISPRESENTC(direct)) {
      memcpy(direct_ptr, schrp, direct_siz);
      schrp += direct_siz;
    }
    if (ISPRESENT(exist)) {
      *exist = *sintp++;
    }
    if (ISPRESENTC(form)) {
      memcpy(form_ptr, schrp, form_siz);
      schrp += form_siz;
    }
    if (ISPRESENTC(formatted)) {
      memcpy(formatted_ptr, schrp, formatted_siz);
      schrp += formatted_siz;
    }
    if (ISPRESENTC(name)) {
      memcpy(name_ptr, schrp, name_siz);
      schrp += name_siz;
    }
    if (ISPRESENT(named)) {
      *named = *sintp++;
    }
    if (ISPRESENT(nextrec)) {
      *nextrec = *sintp++;
    }
    if (ISPRESENT(number)) {
      *number = *sintp++;
    }
    if (ISPRESENT(opened)) {
      *opened = *sintp++;
    }
    if (ISPRESENTC(pad)) {
      memcpy(pad_ptr, schrp, pad_siz);
      schrp += pad_siz;
    }
    if (ISPRESENTC(position)) {
      memcpy(position_ptr, schrp, position_siz);
      schrp += position_siz;
    }
    if (ISPRESENTC(read)) {
      memcpy(read_ptr, schrp, read_siz);
      schrp += read_siz;
    }
    if (ISPRESENTC(readwrite)) {
      memcpy(readwrite_ptr, schrp, readwrite_siz);
      schrp += readwrite_siz;
    }
    if (ISPRESENT(recl)) {
      *recl = *sintp++;
    }
    if (ISPRESENTC(sequential)) {
      memcpy(sequential_ptr, schrp, sequential_siz);
      schrp += sequential_siz;
    }
    if (ISPRESENTC(unformatted)) {
      memcpy(unformatted_ptr, schrp, unformatted_siz);
      schrp += unformatted_siz;
    }
    if (ISPRESENTC(write0)) {
      memcpy(write0_ptr, schrp, write0_siz);
      schrp += write0_siz;
    }
    if (ISPRESENT(id)) {
      *id = *sintp++;
    }
    if (ISPRESENT(pending)) {
      *pending = *sintp++;
    }
    if (ISPRESENT(pos)) {
      *pos = *sintp++;
    }
    if (ISPRESENT(size)) {
      *size = *sintp++;
    }
    if (ISPRESENTC(asynchronous)) {
      memcpy(asynchronous_ptr, schrp, asynchronous_siz);
      schrp += asynchronous_siz;
    }
    if (ISPRESENTC(decimal)) {
      memcpy(decimal_ptr, schrp, decimal_siz);
      schrp += decimal_siz;
    }
    if (ISPRESENTC(encoding)) {
      memcpy(encoding_ptr, schrp, encoding_siz);
      schrp += encoding_siz;
    }
    if (ISPRESENTC(sign)) {
      memcpy(sign_ptr, schrp, sign_siz);
      schrp += sign_siz;
    }
    if (ISPRESENTC(stream)) {
      memcpy(stream_ptr, schrp, stream_siz);
      schrp += stream_siz;
    }
    if (ISPRESENTC(round)) {
      memcpy(round_ptr, schrp, round_siz);
      schrp += round_siz;
    }

    return (sint[0]);
  }

  /* i/o processor */

  s = inquire(unit, file_ptr, bitv, iostat, exist, opened, number, named,
              name_ptr, acc_ptr, sequential_ptr, direct_ptr, form_ptr,
              formatted_ptr, unformatted_ptr, recl, nextrec, blank_ptr,
              position_ptr, action_ptr, read_ptr, write0_ptr, readwrite_ptr,
              delim_ptr, pad_ptr, id, pending, pos, size, asynchronous_ptr,
              decimal_ptr, encoding_ptr, sign_ptr, stream_ptr, round_ptr,
              file_siz, name_siz, acc_siz, sequential_siz, direct_siz, form_siz,
              formatted_siz, unformatted_siz, blank_siz, position_siz,
              action_siz, read_siz, write0_siz, readwrite_siz, delim_siz,
              pad_siz, asynchronous_siz, decimal_siz, encoding_siz, sign_siz,
              stream_siz, round_siz);

  sint[0] = s;

  if (*bitv & FIO_BITV_IOSTAT) {
    *sintp++ = *iostat;
  }
  if (ISPRESENTC(acc)) {
    memcpy(schrp, CADR(acc), CLEN(acc));
    schrp += CLEN(acc);
  }
  if (ISPRESENTC(action)) {
    memcpy(schrp, CADR(action), CLEN(action));
    schrp += CLEN(action);
  }
  if (ISPRESENTC(blank)) {
    memcpy(schrp, CADR(blank), CLEN(blank));
    schrp += CLEN(blank);
  }
  if (ISPRESENTC(delim)) {
    memcpy(schrp, CADR(delim), CLEN(delim));
    schrp += CLEN(delim);
  }
  if (ISPRESENTC(direct)) {
    memcpy(schrp, CADR(direct), CLEN(direct));
    schrp += CLEN(direct);
  }
  if (ISPRESENT(exist)) {
    *sintp++ = *exist;
  }
  if (ISPRESENTC(form)) {
    memcpy(schrp, CADR(form), CLEN(form));
    schrp += CLEN(form);
  }
  if (ISPRESENTC(formatted)) {
    memcpy(schrp, CADR(formatted), CLEN(formatted));
    schrp += CLEN(formatted);
  }
  if (ISPRESENTC(name)) {
    memcpy(schrp, CADR(name), CLEN(name));
    schrp += CLEN(name);
  }
  if (ISPRESENT(named)) {
    *sintp++ = *named;
  }
  if (ISPRESENT(nextrec)) {
    *sintp++ = *nextrec;
  }
  if (ISPRESENT(number)) {
    *sintp++ = *number;
  }
  if (ISPRESENT(opened)) {
    *sintp++ = *opened;
  }
  if (ISPRESENTC(pad)) {
    memcpy(schrp, CADR(pad), CLEN(pad));
    schrp += CLEN(pad);
  }
  if (ISPRESENTC(position)) {
    memcpy(schrp, CADR(position), CLEN(position));
    schrp += CLEN(position);
  }
  if (ISPRESENTC(read)) {
    memcpy(schrp, CADR(read), CLEN(read));
    schrp += CLEN(read);
  }
  if (ISPRESENTC(readwrite)) {
    memcpy(schrp, CADR(readwrite), CLEN(readwrite));
    schrp += CLEN(readwrite);
  }
  if (ISPRESENT(recl)) {
    *sintp++ = *recl;
  }
  if (ISPRESENTC(sequential)) {
    memcpy(schrp, CADR(sequential), CLEN(sequential));
    schrp += CLEN(sequential);
  }
  if (ISPRESENTC(unformatted)) {
    memcpy(schrp, CADR(unformatted), CLEN(unformatted));
    schrp += CLEN(unformatted);
  }
  if (ISPRESENTC(write0)) {
    memcpy(schrp, CADR(write0), CLEN(write0));
    schrp += CLEN(write0);
  }
  if (ISPRESENT(id)) {
    *sintp++ = *id;
  }
  if (ISPRESENT(pending)) {
    *sintp++ = *pending;
  }
  if (ISPRESENT(pos)) {
    *sintp++ = *pos;
  }
  if (ISPRESENT(size)) {
    *sintp++ = *size;
  }
  if (ISPRESENTC(asynchronous)) {
    memcpy(schrp, CADR(asynchronous), CLEN(asynchronous));
    schrp += CLEN(asynchronous);
  }
  if (ISPRESENTC(decimal)) {
    memcpy(schrp, CADR(decimal), CLEN(decimal));
    schrp += CLEN(decimal);
  }
  if (ISPRESENTC(encoding)) {
    memcpy(schrp, CADR(encoding), CLEN(encoding));
    schrp += CLEN(encoding);
  }
  if (ISPRESENTC(sign)) {
    memcpy(schrp, CADR(sign), CLEN(sign));
    schrp += CLEN(sign);
  }
  if (ISPRESENTC(stream)) {
    memcpy(schrp, CADR(stream), CLEN(stream));
    schrp += CLEN(stream);
  }
  if (ISPRESENTC(round)) {
    memcpy(schrp, CADR(round), CLEN(round));
    schrp += CLEN(round);
  }

  if (!LOCAL_MODE) {
    DIST_RBCST(ioproc, sint, SINTN, 1, __INT8);
    DIST_RBCST(ioproc, schr, schr_len, 1, __CHAR);
  }

  if (schr)
    __fort_free(schr);

  __fortio_errend03();
  return (sint[0]);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(INQUIRE03_2, inquire03_2)
(__INT_T *unit, DCHAR(file), __INT_T *bitv, __INT_T *iostat, bool *exist,
 bool *opened, __INT8_T *number, bool *named, DCHAR(name), DCHAR(acc),
 DCHAR(sequential), DCHAR(direct), DCHAR(form), DCHAR(formatted),
 DCHAR(unformatted), __INT8_T *recl, __INT8_T *nextrec, DCHAR(blank),
 DCHAR(position), DCHAR(action), DCHAR(read), DCHAR(write0), DCHAR(readwrite),
 DCHAR(delim), DCHAR(pad), __INT_T *id, __INT_T *pending, __INT8_T *pos,
 __INT8_T *size, DCHAR(asynchronous), DCHAR(decimal), DCHAR(encoding),
 DCHAR(sign), DCHAR(stream),
 DCHAR(round) DCLEN(file) DCLEN(name) DCLEN(acc) DCLEN(sequential) DCLEN(direct)
 DCLEN(form) DCLEN(formatted) DCLEN(unformatted) DCLEN(blank)
 DCLEN(position) DCLEN(action) DCLEN(read) DCLEN(write0)
 DCLEN(readwrite) DCLEN(delim) DCLEN(pad) DCLEN(asynchronous)
 DCLEN(decimal) DCLEN(encoding) DCLEN(sign) DCLEN(stream)
 DCLEN(round))
{
  return ENTF90IO(INQUIRE03_2A, inquire03_2a) (unit, CADR(file), bitv, iostat,

                  exist, opened, number, named, CADR(name), CADR(acc),
                  CADR(sequential), CADR(direct), CADR(form), CADR(formatted),
                  CADR(unformatted), recl, nextrec, CADR(blank),
                  CADR(position), CADR(action), CADR(read), CADR(write0),
                  CADR(readwrite), CADR(delim), CADR(pad), id, pending, pos,
                  size, CADR(asynchronous), CADR(decimal), CADR(encoding),
                  CADR(sign), CADR(stream), CADR(round), (__CLEN_T)CLEN(file),
                  (__CLEN_T)CLEN(name), (__CLEN_T)CLEN(acc),
                  (__CLEN_T)CLEN(sequential), (__CLEN_T)CLEN(direct),
                  (__CLEN_T)CLEN(form), (__CLEN_T)CLEN(formatted),
                  (__CLEN_T)CLEN(unformatted), (__CLEN_T)CLEN(blank),
                  (__CLEN_T)CLEN(position), (__CLEN_T)CLEN(action),
                  (__CLEN_T)CLEN(read), (__CLEN_T)CLEN(write0),
                  (__CLEN_T)CLEN(readwrite), (__CLEN_T)CLEN(delim),
                  (__CLEN_T)CLEN(pad), (__CLEN_T)CLEN(asynchronous),
                  (__CLEN_T)CLEN(decimal), (__CLEN_T)CLEN(encoding),
                  (__CLEN_T)CLEN(sign), (__CLEN_T)CLEN(stream),
                  (__CLEN_T)CLEN(round));
}
