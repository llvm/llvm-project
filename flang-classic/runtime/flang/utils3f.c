/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*	  */
#if defined(_WIN64)
#include <windows.h>
#endif
#include "io3f.h"
#include <string.h>
#include "async.h"
#include "mpalloc.h"

char *
__fstr2cstr(char *from, int from_len)
{
  char *str;
  int len;

  for (len = from_len; len > 0; len--)
    if (from[len - 1] != ' ')
      break;
  str = _mp_malloc(len + 1);
  memcpy(str, from, len);
  str[len] = '\0';
  return str;
}

void
__cstr_free(char *from)
{
  _mp_free(from);
}

void
__fcp_cstr(char *to, int to_len, char *from)
{
  char ch;

  if (from)
    while ((ch = *from++)) {
      if (to_len-- <= 0)
        break;
      *to++ = ch;
    }
  while (to_len-- > 0)
    *to++ = ' ';
}

/* --------------------------------------------------------------- */
/* io-related routines needed to support 3f routines               */
/* --------------------------------------------------------------- */

/* --------------------------------------------------------------- */

bool __isatty3f(int unit)
{
  void *p;
  int fd;

  p = __fio_find_unit(unit);
  if (p == NULL || FIO_FCB_STDUNIT(p)) {
    switch (unit) {
    case 0:
      fd = 2; /* stderr */
      break;
    case 5:
      fd = 0; /* stdin */
      break;
    case 6:
      fd = 1; /* stdout */
      break;
    default:
      return FALSE;
    }
    return __io_isatty(fd);
  }
  return FALSE;
}

/* --------------------------------------------------------------- */

FILE *__getfile3f(int unit)
{
  void *p;

  p = __fio_find_unit(unit);
  if (p != NULL) {
    if (FIO_FCB_ASY_RW(p)) {/* disable any async i/o */
      FIO_FCB_SET_ASY_RW(p, 0);
      if (Fio_asy_disable(FIO_FCB_ASYPTR(p)) == -1) {
        __abort(1, "3F routine found asynchronous I/O error");
      }
    }
    return FIO_FCB_FP(p);
  }
  switch (unit) {
  case 0:
    return __io_stderr();
  case 5:
    return __io_stdin();
  case 6:
    return __io_stdout();
  default:
    return NULL;
  }
}

#if defined(_WIN64)
void
__GetTimeToSecondsSince1970(ULARGE_INTEGER *fileTime, unsigned int *out)
{

  /* From RtlTimeToSecondsSince1970 Function */

  FILETIME *firstFileTime;
  ULARGE_INTEGER *firstLarge;
  SYSTEMTIME *firstTime;
  firstLarge = (ULARGE_INTEGER *)_mp_malloc(sizeof(ULARGE_INTEGER));
  firstTime = (SYSTEMTIME *)_mp_malloc(sizeof(SYSTEMTIME));
  firstTime->wYear = 1970;
  firstTime->wMonth = 1;
  firstTime->wDayOfWeek = 4;
  firstTime->wDay = 1;
  firstTime->wHour = 0;
  firstTime->wMinute = 0;
  firstTime->wSecond = 0;
  firstTime->wMilliseconds = 0;

  firstFileTime = (FILETIME *)_mp_malloc(sizeof(FILETIME));
  SystemTimeToFileTime(firstTime, firstFileTime);
  firstLarge->u.LowPart = firstFileTime->dwLowDateTime;
  firstLarge->u.HighPart = firstFileTime->dwHighDateTime;

  *out =
      (unsigned int)((fileTime->QuadPart - firstLarge->QuadPart) / 10000000LL);

  _mp_free(firstFileTime);
  _mp_free(firstTime);
  _mp_free(firstLarge);
}

void
__UnpackTime(unsigned int secsSince1970, ULARGE_INTEGER *fileTime)
{

  FILETIME *firstFileTime;
  ULARGE_INTEGER *firstLarge;
  SYSTEMTIME *firstTime;

  firstTime = (SYSTEMTIME *)_mp_malloc(sizeof(SYSTEMTIME));
  firstTime->wYear = 1970;
  firstTime->wMonth = 1;
  firstTime->wDayOfWeek = 4;
  firstTime->wDay = 1;
  firstTime->wHour = 0;
  firstTime->wMinute = 0;
  firstTime->wSecond = 0;
  firstTime->wMilliseconds = 0;

  firstFileTime = (FILETIME *)_mp_malloc(sizeof(FILETIME));

  firstLarge = (ULARGE_INTEGER *)_mp_malloc(sizeof(ULARGE_INTEGER));
  SystemTimeToFileTime(firstTime, firstFileTime);
  firstLarge->u.LowPart = firstFileTime->dwLowDateTime;
  firstLarge->u.HighPart = firstFileTime->dwHighDateTime;

  fileTime->QuadPart =
      ((unsigned long long)secsSince1970 * 10000000LL) + firstLarge->QuadPart;

  _mp_free(firstFileTime);
  _mp_free(firstTime);
  _mp_free(firstLarge);
}

#define FILE$FIRST -1
#define FILE$LAST -2
#define FILE$ERROR -3

int
__GETFILEINFOQQ(char *ffiles_arg, char *buffer, int *handle, int ffiles_len)
{
  char *files;
  int rslt = 0, i;
  WIN32_FIND_DATA FindFileData;
  HANDLE hFind;
  ULARGE_INTEGER fileTime;
  char *name;
  struct FILE$INFO {
    unsigned int creation;
    unsigned int lastWrite;
    unsigned int lastAccess;
    int length;
    int permit;
    char name[256];
  } fileInfo;

  files = __fstr2cstr(ffiles_arg, ffiles_len);
  if (!files || !buffer || !handle) {
    __io_errno();
    return 0;
  }
  if (*handle <= 0) {
    hFind = FindFirstFile(files, &FindFileData);
    *handle = (int)hFind;
  } else {
    hFind = (HANDLE)*handle;
    if (FindNextFile(hFind, &FindFileData) == 0) {
      *handle = FILE$LAST;
      FindClose(hFind);
      goto rtn;
    }
  }

  if (hFind == INVALID_HANDLE_VALUE) {
    *handle = FILE$ERROR;
    goto rtn;
  }

  fileTime.u.LowPart = FindFileData.ftCreationTime.dwLowDateTime;
  fileTime.u.HighPart = FindFileData.ftCreationTime.dwHighDateTime;
  __GetTimeToSecondsSince1970(&fileTime, &(fileInfo.creation));

  fileTime.u.LowPart = FindFileData.ftLastWriteTime.dwLowDateTime;
  fileTime.u.HighPart = FindFileData.ftLastWriteTime.dwHighDateTime;
  __GetTimeToSecondsSince1970(&fileTime, &(fileInfo.lastWrite));

  fileTime.u.LowPart = FindFileData.ftLastAccessTime.dwLowDateTime;
  fileTime.u.HighPart = FindFileData.ftLastAccessTime.dwHighDateTime;
  __GetTimeToSecondsSince1970(&fileTime, &(fileInfo.lastAccess));

  fileInfo.length = (int)((FindFileData.nFileSizeHigh * (MAXDWORD + 1)) +
                          FindFileData.nFileSizeLow);
  fileInfo.permit = FindFileData.dwFileAttributes;
  name = FindFileData.cFileName;
  rslt = strlen(name);

  __fcp_cstr(fileInfo.name, 255, name);
  memcpy(buffer, &fileInfo, sizeof(fileInfo));
rtn:
  __cstr_free(files);
  return rslt;
}

#endif
