/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	setfiletimeqq3f.c - Implements DFLIB setfiletimeqq subprogram.  */
#if defined(_WIN64)
#include <windows.h>
#endif
#include <string.h>
#include <stdlib.h>
/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

#define FILE$FIRST -1
#define FILE$LAST -2
#define FILE$ERROR -3
#define FILE$CURTIME -1

#if defined(_WIN64)
extern void __UnpackTime(unsigned int secsSince1970, ULARGE_INTEGER *fileTime);
extern int __GETFILEINFOQQ(DCHAR(ffiles), char *buffer,
                           int *handle DCLEN(ffiles));
int ENT3F(SETFILETIMEQQ, setfiletimeqq)(DCHAR(ffile),
                                        unsigned int *timedate DCLEN(ffile))
{

  HANDLE handle;
  char *fileName = 0;
  int rslt = 0, success;
  ULARGE_INTEGER *fileTime = 0;
  struct FILE$INFO {
    int creation;
    int lastWrite;
    int lastAccess;
    int length;
    int permit;
    char name[256];
  } fileInfo, fileInfo2;
  FILETIME *creation = 0, *lastAccess = 0, *lastWrite = 0;

  fileTime = (ULARGE_INTEGER *)_mp_malloc(sizeof(ULARGE_INTEGER));
  if (*timedate == FILE$CURTIME) {
    SYSTEMTIME *sysTime;
    sysTime = (SYSTEMTIME *)_mp_malloc(sizeof(SYSTEMTIME));
    GetSystemTime(sysTime);
    lastWrite = (FILETIME *)_mp_malloc(sizeof(FILETIME));
    SystemTimeToFileTime(sysTime, lastWrite);
    _mp_free(sysTime);
  } else {
    __UnpackTime(*timedate, fileTime);
    lastWrite = (FILETIME *)_mp_malloc(sizeof(FILETIME));
    lastWrite->dwLowDateTime = fileTime->u.LowPart;
    lastWrite->dwHighDateTime = fileTime->u.HighPart;
  }

  handle = FILE$FIRST;
  __GETFILEINFOQQ(CADR(ffile), (char *)&fileInfo, (int *)&handle, CLEN(ffile));
  while ((int)handle >= 0) {
    __GETFILEINFOQQ(CADR(ffile), (char *)&fileInfo2, (int *)&handle,
                    CLEN(ffile));
  }

  if ((int)handle == FILE$LAST) {
    fileName = __fstr2cstr(CADR(ffile), CLEN(ffile));
    if (!fileName)
      return 0;

    handle = CreateFile(fileName, GENERIC_WRITE, 0, 0, OPEN_EXISTING,
                        fileInfo.permit, 0);

    if (handle == INVALID_HANDLE_VALUE)
      goto rtn;

    __UnpackTime(fileInfo.creation, fileTime);
    creation = (FILETIME *)_mp_malloc(sizeof(FILETIME));
    creation->dwLowDateTime = fileTime->u.LowPart;
    creation->dwHighDateTime = fileTime->u.HighPart;

    __UnpackTime(fileInfo.lastAccess, fileTime);
    lastAccess = (FILETIME *)_mp_malloc(sizeof(FILETIME));
    lastAccess->dwLowDateTime = fileTime->u.LowPart;
    lastAccess->dwHighDateTime = fileTime->u.HighPart;

    success = SetFileTime(handle, creation, lastAccess, lastWrite);
    if (success) {
      rslt = -1;
    }
    CloseHandle(handle);
  }

rtn:
  if (fileName)
    __cstr_free(fileName);
  _mp_free(fileTime);
  _mp_free(lastWrite);
  _mp_free(lastAccess);
  _mp_free(creation);
  return rslt;
}
#else
int ENT3F(SETFILETIMEQQ, setfiletimeqq)(DCHAR(ffile),
                                        unsigned int *timedate DCLEN(ffile))
{
  fprintf(__io_stderr(), "setfiletimeqq() not implemented on this target\n");
  return 0;
}

#endif
