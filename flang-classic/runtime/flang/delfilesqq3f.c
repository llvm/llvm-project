/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	delfilesqq3f.c - Implements DFLIB delfilesqq subprogram.  */
#if defined(_WIN64)
#include <windows.h>
#endif
#include <string.h>
#include <stdlib.h>
/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

#if defined(_WIN64)
int ENT3F(DELFILESQQ, delfilesqq)(DCHAR(ffiles) DCLEN(ffiles))
{
  char *files;
  int rslt = 0, i;
  WIN32_FIND_DATA FindFileData;
  HANDLE hFind;

  files = __fstr2cstr(CADR(ffiles), CLEN(ffiles));
  if (!files) {
    __io_errno();
    return 0;
  }
  hFind = FindFirstFile(files, &FindFileData);
  do {
    if (hFind == INVALID_HANDLE_VALUE)
      break;
    if (FindFileData.dwFileAttributes &
        (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_HIDDEN |
         FILE_ATTRIBUTE_SYSTEM | FILE_ATTRIBUTE_READONLY)) {
      continue;
    }
    if (_unlink(FindFileData.cFileName) != -1) {
      ++rslt;
    }

  } while (FindNextFile(hFind, &FindFileData) != 0);
  FindClose(hFind);
  __cstr_free(files);
  return rslt;
}
#else
int ENT3F(DELFILESQQ, delfilesqq)(DCHAR(ffiles) DCLEN(ffiles))
{
  fprintf(__io_stderr(), "delfilesqq() not implemented on this target\n");
  return 0;
}

#endif
