/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	getfileinfoqq3f.c - Implements DFLIB getfileinfoqq subprogram.  */
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

#if defined(_WIN64)
extern void __GetTimeToSecondsSince1970(ULARGE_INTEGER *fileTime,
                                        unsigned int *out);
int ENT3F(GETFILEINFOQQI8, getfileinfoqqi8)(DCHAR(ffiles), char *buffer,
                                            int *handle DCLEN(ffiles))
{
  char *files;
  int rslt = 0, i;
  WIN32_FIND_DATA FindFileData;
  HANDLE hFind;
  ULARGE_INTEGER fileTime;
  char *name;
  struct FILE$INFO8 {
    int creation;
    int lastWrite;
    int lastAccess;
    long long length;
    int permit;
    char name[256];
  } fileInfo;

  files = __fstr2cstr(CADR(ffiles), CLEN(ffiles));
  if (!files || !buffer || !handle) {
    __io_errno();
    return 0;
  }
  if (*handle <= 0) {
    hFind = FindFirstFile(files, &FindFileData);
    *handle = hFind;
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
#else
int ENT3F(GETFILEINFOQQI8, getfileinfoqqi8)(DCHAR(ffiles), int *buffer,
                                            int *handle DCLEN(ffiles))
{
  fprintf(__io_stderr(),
          "getfileinfoqqi8() not implemented on this target\n");
  return 0;
}

#endif
