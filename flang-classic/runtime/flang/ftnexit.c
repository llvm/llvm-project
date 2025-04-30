/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* fortran control support routines */

#include "global.h"
#include <fenv.h>

#include "fioMacros.h"
#include "llcrit.h"

void ENTF90(EXIT, exit)(__INT_T *exit_status)
{
  __fort_exit(ISPRESENT(exit_status) ? *exit_status : 0);
}

void ENTCRF90(EXIT, exit)(__INT_T *exit_status)
{
  __fort_exit(ISPRESENT(exit_status) ? *exit_status : 0);
}

static void
_f90io_f2003_stop_with_ieee_warnings(int exc)
{
  if ((exc & FE_INVALID) == FE_INVALID)
    fprintf(__io_stderr(), "Warning: ieee_invalid is signaling\n");
#ifdef FE_DENORM
  if ((exc & FE_DENORM) == FE_DENORM)
    fprintf(__io_stderr(), "Warning: ieee_denorm is signaling\n");
#endif
  if ((exc & FE_DIVBYZERO) == FE_DIVBYZERO)
    fprintf(__io_stderr(), "Warning: ieee_divide_by_zero is signaling\n");
  if ((exc & FE_OVERFLOW) == FE_OVERFLOW)
    fprintf(__io_stderr(), "Warning: ieee_overflow is signaling\n");
  if ((exc & FE_UNDERFLOW) == FE_UNDERFLOW)
    fprintf(__io_stderr(), "Warning: ieee_underflow is signaling\n");
  if ((exc & FE_INEXACT) == FE_INEXACT)
    fprintf(__io_stderr(), "Warning: ieee_inexact is signaling\n");
}

static void
_f90io_stop(int exit_status, char *str, __CLEN_T str_siz, __LOG_T is_quiet, __LOG_T is_errorstop)
{
  int __fenv_fetestexcept(int);
  int anyexc;
  anyexc = __fenv_fetestexcept(FE_ALL_EXCEPT);
  MP_P_STDIO;
  if (str) {
    if (is_quiet == 0)
      _f90io_f2003_stop_with_ieee_warnings(anyexc);
    if (is_errorstop)
      fprintf(__io_stderr(), "ERROR STOP ");

    fprintf(__io_stderr(), "%.*s\n", str_siz, str);
  } else if (__fort_getenv("NO_STOP_MESSAGE") == 0
                 ) {
    if (is_quiet == 0)
      _f90io_f2003_stop_with_ieee_warnings(anyexc);
    if (is_quiet == 0) {
      if (is_errorstop)
        fprintf(__io_stderr(), "ERROR STOP\n");
      else
        fprintf(__io_stderr(), "FORTRAN STOP\n");
    }
  }
  MP_V_STDIO;
  __fort_exit(exit_status);
}

/** \brief
 * ERROR STOP runtime when the stop code is scalar-default-char-expr
 *
 * \param isquiet is the bool value to check the QUIET clause is specified.
 * \param str is the char-expr
 */
void ENTF90(ERRORSTOP08a_char, errorstop08a_char)(__LOG_T *isquiet, DCHAR(str) DCLEN64(str))
{
  if (*isquiet == 0) {
    if (GET_DIST_LCPU != GET_DIST_IOPROC && !LOCAL_MODE)
      __fort_exit(0);
    _f90io_stop(1, CADR(str), CLEN(str), 0, 1);
  } else {
    _f90io_stop(1, NULL, 0, 1, 1);
  }
}

/** \brief
 * ERROR STOP runtime when the stop code is scalar-default-int-expr
 *
 * \param stop_code is the stop code integer value.
 * \param isquiet is the bool value to check the QUIET clause is specified.
 */
void ENTF90(ERRORSTOP08a_int, errorstop08a_int)(__INT_T *stop_code, __LOG_T *isquiet)
{
  char stop_code_str[16];
  sprintf(stop_code_str, "%d", *stop_code);
  if (*isquiet == 0) {
    if (GET_DIST_LCPU != GET_DIST_IOPROC && !LOCAL_MODE)
      __fort_exit(0);
    _f90io_stop(*stop_code, stop_code_str, 16, 0, 1);
  } else {
    _f90io_stop(*stop_code, NULL, 0, 1, 1);
  }
}

void ENTF90(STOP08a, stop08a)(__INT_T *exit_status, DCHAR(str) DCLEN64(str))
{
  char statstr[7];

  if (GET_DIST_LCPU != GET_DIST_IOPROC && !LOCAL_MODE)
    __fort_exit(0);
  if (ISPRESENTC(str))
    _f90io_stop(*exit_status, CADR(str), CLEN(str), 0, 0);
  else if (*exit_status != 0) {
    sprintf(statstr, "%5d", *exit_status);
    _f90io_stop(*exit_status, statstr, 6, 0, 0);
  } else {
    _f90io_stop(*exit_status, NULL, 0, 0, 0);
  }
}

/* 32 bit CLEN version */
void ENTF90(STOP08, stop08)(__INT_T *exit_status, DCHAR(str) DCLEN(str))
{
  ENTF90(STOP08A, stop08a)(exit_status, CADR(str), (__CLEN_T)CLEN(str));
}

void ENTF90(STOPA, stopa)(DCHAR(str) DCLEN64(str))
{
  if (GET_DIST_LCPU != GET_DIST_IOPROC && !LOCAL_MODE)
    __fort_exit(0);
  if (ISPRESENTC(str))
    _f90io_stop(0, CADR(str), CLEN(str), 0, 0);
  else
    _f90io_stop(0, NULL, 0, 0, 0);
}
/* 32 bit CLEN version */
void ENTF90(STOP, stop)(DCHAR(str) DCLEN(str))
{
  ENTF90(STOPA, stopa)(CADR(str), (__CLEN_T)CLEN(str));
}

void ENTCRF90(STOPA, stopa)(DCHAR(str) DCLEN64(str))
{
  if (ISPRESENTC(str))
    _f90io_stop(0, CADR(str), CLEN(str), 0, 0);
  else
    _f90io_stop(0, NULL, 0, 0, 0);
}
/* 32 bit CLEN version */
void ENTCRF90(STOP, stop)(DCHAR(str) DCLEN(str))
{
  ENTCRF90(STOPA, stopa)(CADR(str), (__CLEN_T)CLEN(str));
}

static void
_f90io_pause(char *str, __CLEN_T str_siz)
{
  MP_P_STDIO;
  if (str)
    fprintf(__io_stderr(), "FORTRAN PAUSE: %.*s\n", str_siz, str);
  if (__fort_isatty(__fort_getfd(__io_stdin()))) {
    fprintf(__io_stderr(),
            "FORTRAN PAUSE: enter <return> or <ctrl>d to continue>");
    while (1) {
      char c = fgetc(__io_stdin());
      if (c == '\n')
        break;
      if (__io_feof(__io_stdin())) {
        fprintf(__io_stderr(), "\n");
        break;
      }
    }
  } else /* print message but don't pause: */
    fprintf(__io_stderr(), "FORTRAN PAUSE: continuing...\n");
  MP_V_STDIO;
}

void ENTF90(PAUSEA, pausea)(DCHAR(str) DCLEN64(str))
{
  if (GET_DIST_LCPU == GET_DIST_IOPROC || LOCAL_MODE) {
    if (ISPRESENTC(str))
      _f90io_pause(CADR(str), CLEN(str));
    else
      _f90io_pause(NULL, 0);
  }
  if (!LOCAL_MODE)
    __fort_barrier();
}
/* 32 bit CLEN version */
void ENTF90(PAUSE, pause)(DCHAR(str) DCLEN(str))
{
  ENTF90(PAUSEA, pausea)(CADR(str), (__CLEN_T)CLEN(str));
}

void ENTCRF90(PAUSEA, pausea)(DCHAR(str) DCLEN64(str))
{
  if (ISPRESENTC(str))
    _f90io_pause(CADR(str), CLEN(str));
  else
    _f90io_pause(NULL, 0);
}
/* 32 bit CLEN version */
void ENTCRF90(PAUSE, pause)(DCHAR(str) DCLEN(str))
{
  ENTCRF90(PAUSEA, pausea)(CADR(str), (__CLEN_T)CLEN(str));
}
