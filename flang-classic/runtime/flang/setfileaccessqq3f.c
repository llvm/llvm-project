/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	setfileaccessqq3f.c - Implements DFLIB setfileaccessqq subprogram.  */
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
int ENT3F(SETFILEACCESSQQ, setfileaccessqq)(DCHAR(ffile),
                                            int *access DCLEN(ffile))
{

  int success;
  char *fileName;

  fileName = __fstr2cstr(CADR(ffile), CLEN(ffile));
  if (!fileName)
    return 0;
  success = SetFileAttributes(fileName, *access);
  __cstr_free(fileName);
  return success ? -1 : 0;
}
#else
int ENT3F(SETFILEACCESSQQ, setfileaccessqq)(DCHAR(ffiles),
                                            int *access DCLEN(ffiles))
{
  fprintf(__io_stderr(),
          "setfileaccessqq() not implemented on this target\n");
  return 0;
}

#endif
