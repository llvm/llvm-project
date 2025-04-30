/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief
 * Implements the iargc, cmd_arg_cnt, get_cmd_arg, getarg and get_env_var extentions.
 */

#include <stdlib.h>
#include <string.h>
#include "fioMacros.h"
#include "utils3f.h"

extern int __io_get_argc();
extern char **__io_get_argv();

static void store_int_kind(void *, __INT_T *, int);

__INT_T
ENTRY(IARGC, iargc)() { return __io_get_argc() - 1; }

void
ENTF90(GETARGA, getarga)(__INT_T *n, DCHAR(arg), __INT_T *int_kind DCLEN64(arg))
{
  char *p;
  __INT_T i, len = CLEN(arg);
  __INT_T argc;
  char **argv;

  i = I8(__fort_varying_int)(n, int_kind);
  argc = __io_get_argc();
  argv = __io_get_argv();

  if (i < 0 || i >= argc) {
    i = 0;
  } else {
    p = argv[i];
    /*  copy characters into arg:  */
    for (i = 0; p[i] != 0 && i < len; ++i)
      CADR(arg)[i] = p[i];
  }
  /*  blank fill if necessary:  */
  for (; i < len; ++i)
    CADR(arg)[i] = ' ';
}

/* 32 bit CLEN version */
void
ENTF90(GETARG, getarg)(__INT_T *n, DCHAR(arg), __INT_T *int_kind DCLEN(arg))
{
  ENTF90(GETARGA, getarga)(n, CADR(arg), int_kind, (__CLEN_T)CLEN(arg));
}

__INT4_T
ENTF90(CMD_ARG_CNT, cmd_arg_cnt)()
{
  int n;
  n = __io_get_argc();
  if (n > 0)
    n--;
  return n;
}

__INT8_T
ENTF90(KCMD_ARG_CNT, kcmd_arg_cnt)()
{
  return ENTF90(CMD_ARG_CNT, cmd_arg_cnt)();
}

void
ENTF90(GET_CMD_ARGA, get_cmd_arga)(__INT_T *number, DCHAR(value),
                                 __INT_T *length, __INT_T *status,
                                 __INT_T *int_kind DCLEN64(value))
{
  __CLEN_T len;
  char *p, *q;
  int n;
  __CLEN_T i;
  char **v;
  __CLEN_T arg_len;
  int stat;
  int fail;

  fail = 0;
  i = 0;
  arg_len = 0;
  len = CLEN(value);
  n = I8(__fort_varying_int)(number, int_kind);
  q = (char *)CADR(value);
  if (0 <= n && n < __io_get_argc()) {
    v = __io_get_argv();
    p = v[n];
    /*  compute length of argument:  */
    arg_len = strlen(p);
    if (ISPRESENTC(value)) {
      /*  copy characters into value:  */
      for (i = 0; *p != 0 && i < len; q++, p++, i++) {
        *q = *p;
      }
    }
  } else
    fail = 1;
  if (ISPRESENTC(value)) {
    /*  blank fill if necessary:  */
    for (; i < len; q++, i++)
      *q = ' ';
  }
  if (ISPRESENT(length))
    store_int_kind(length, int_kind, arg_len);
  if (ISPRESENT(status)) {
    stat = 0;
    if (ISPRESENTC(value) && len < arg_len)
      stat = -1;
    else if (fail)
      stat = 1;
    store_int_kind(status, int_kind, stat);
  }
}

/* 32 bit CLEN version */
void
ENTF90(GET_CMD_ARG, get_cmd_arg)(__INT_T *number, DCHAR(value),
                                 __INT_T *length, __INT_T *status,
                                 __INT_T *int_kind DCLEN(value))
{
  ENTF90(GET_CMD_ARGA, get_cmd_arga)(number, CADR(value), length, status,
                                     int_kind, (__CLEN_T)CLEN(value));
}

void
ENTF90(GET_CMDA, get_cmda)(DCHAR(command), __INT_T *length, __INT_T *status,
                         __INT_T *int_kind DCLEN64(command))
{
  __CLEN_T len;
  char *p, *q;
  int argc;
  int n;
  __CLEN_T i;
  char **v;
  __CLEN_T arg_len;
  int stat;
  int fail;

  fail = 0;
  arg_len = 0;
  len = CLEN(command);
  argc = __io_get_argc();
  q = (char *)CADR(command);
  v = __io_get_argv();
  i = 0;
  for (n = 0; n < argc; n++) {
    p = v[n];
    if (ISPRESENTC(command)) {
      /*  copy characters into command:  */
      if (n && i < len) {
        /* insert separator */
        i++;
        *q = ' ';
        arg_len++;
        q++;
      }
      for (; *p != 0 && i < len; q++, p++, i++) {
        *q = *p;
        arg_len++;
      }
    } else {
      /*  just compute length of argument:  */
      if (n) {
        arg_len++; /* separator */
      }
      for (; *p != 0; p++, i++) {
        arg_len++;
      }
    }
  }
  if (ISPRESENTC(command)) {
    /*  blank fill if necessary:  */
    for (; i < len; q++, i++)
      *q = ' ';
  }
  if (ISPRESENT(length))
    store_int_kind(length, int_kind, arg_len);
  if (ISPRESENT(status)) {
    stat = 0;
    if (ISPRESENTC(command) && len < arg_len)
      stat = -1;
    else if (fail)
      stat = 1;
    store_int_kind(status, int_kind, stat);
  }
}

/* 32 bit CLEN version */
void
ENTF90(GET_CMD, get_cmd)(DCHAR(command), __INT_T *length, __INT_T *status,
                         __INT_T *int_kind DCLEN(command))
{
  ENTF90(GET_CMDA, get_cmda)(CADR(command), length, status, int_kind,
                             (__CLEN_T)CLEN(command));
}

void
ENTF90(GET_ENV_VARA, get_env_vara)(DCHAR(name), DCHAR(value), __INT_T *length,
                                 __INT_T *status, __LOG_T *trim_name,
                                 __INT_T *int_kind DCLEN64(name) DCLEN64(value))
{
  __CLEN_T len;
  char *envnm;
  char *p, *q;
  __CLEN_T i = 0;
  __CLEN_T env_len;
  int stat;
  int sigblanks;

  sigblanks = 0;
  if (ISPRESENT(trim_name)) {
    if (I8(__fort_varying_log)(trim_name, int_kind))
      /* trailing blanks in the name are significant
       * --- and currently not supported.
       */
      sigblanks = 1;
  }
  stat = 0;
  env_len = 0;
  len = CLEN(value);
  envnm = __fstr2cstr(CADR(name), CLEN(name));
  p = getenv(envnm);
  __cstr_free(envnm);
  q = (char *)CADR(value);
  if (p) {
    char *pv;
    /*  compute length of the environment variable's value */
    pv = p;
    for (i = 0; *pv != 0; pv++, i++) {
      env_len++;
    }
    if (ISPRESENTC(value)) {
      /*  copy characters into value:  */
      for (i = 0; *p != 0 && i < len; q++, p++, i++) {
        *q = *p;
      }
    }
  } else
    stat = 1;
  if (ISPRESENTC(value)) {
    /*  blank fill if necessary:  */
    for (; i < len; q++, i++)
      *q = ' ';
  }
  if (ISPRESENT(length))
    store_int_kind(length, int_kind, env_len);
  if (ISPRESENT(status)) {
    if (ISPRESENTC(value) && len < env_len)
      stat = -1;
    store_int_kind(status, int_kind, stat);
  }
}

/* 32 bit CLEN version */
void
ENTF90(GET_ENV_VAR, get_env_var)(DCHAR(name), DCHAR(value), __INT_T *length,
                                 __INT_T *status, __LOG_T *trim_name,
                                 __INT_T *int_kind DCLEN(name) DCLEN(value))
{
  ENTF90(GET_ENV_VARA, get_env_vara)(CADR(name), CADR(value), length, status,
                                     trim_name, int_kind, (__CLEN_T)CLEN(name),
                                     (__CLEN_T)CLEN(value));
}

/*
 * helper function to store an int/logical value into a varying int/logical
 */
static void
store_int_kind(void *b, __INT_T *int_kind, int v)
{
  switch (*int_kind) {
  case 1:
    *(__INT1_T *)b = (__INT1_T)v;
    break;
  case 2:
    *(__INT2_T *)b = (__INT2_T)v;
    break;
  case 4:
    *(__INT4_T *)b = (__INT4_T)v;
    break;
  case 8:
    *(__INT8_T *)b = (__INT8_T)v;
    break;
  }
}
