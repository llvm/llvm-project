/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	unpacktimeqq3f.c - Implements DFLIB packtimeqq subprogram.  */
#if defined(_WIN64)
#include <windows.h>
#endif
#include <string.h>
#include <stdlib.h>
#include "mpalloc.h"
/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"

#if defined(_WIN64)
extern void __UnpackTime(unsigned int secsSince1970, ULARGE_INTEGER *fileTime);

void ENT3F(UNPACKTIMEQQ, unpacktimeqq)(unsigned int *timedate, int *year,
                                       int *month, int *day, int *hour,
                                       int *minute, int *second)
{
  SYSTEMTIME *sysTime;
  FILETIME *fileTime;
  ULARGE_INTEGER quadTime;

  fileTime = (FILETIME *)_mp_malloc(sizeof(FILETIME));
  sysTime = (SYSTEMTIME *)_mp_malloc(sizeof(SYSTEMTIME));
  __UnpackTime(*timedate, &quadTime);
  fileTime->dwLowDateTime = quadTime.u.LowPart;
  fileTime->dwHighDateTime = quadTime.u.HighPart;
  FileTimeToSystemTime(fileTime, sysTime);

  *year = sysTime->wYear;
  *month = sysTime->wMonth;
  *day = sysTime->wDay;
  *hour = sysTime->wHour;
  *minute = sysTime->wMinute;
  *second = sysTime->wSecond;

  _mp_free(fileTime);
  _mp_free(sysTime);
}
#else
void ENT3F(UNPACKTIMEQQ, unpacktimeqq)(int *timedate, int *year, int *month,
                                       int *day, int *hour, int *minute,
                                       int *second)
{
  fprintf(__io_stderr(), "unpacktimeqq() not implemented on this target\n");
}

#endif
