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

#if defined(_WIN64)
extern void __GetTimeToSecondsSince1970(ULARGE_INTEGER *fileTime,
                                        unsigned int *out);
extern int __GETFILEINFOQQ(DCHAR(ffiles), char *buffer,
                           int *handle DCLEN(ffiles));

int ENT3F(GETFILEINFOQQ, getfileinfoqq)(DCHAR(ffiles), char *buffer,
                                        int *handle DCLEN(ffiles))
{
  return __GETFILEINFOQQ(CADR(ffiles), buffer, handle, CLEN(ffiles));
}
#else
int ENT3F(GETFILEINFOQQ, getfileinfoqq)(DCHAR(ffiles), int *buffer,
                                        int *handle DCLEN(ffiles))
{
  fprintf(__io_stderr(), "getfileinfoqq() not implemented on this target\n");
  return 0;
}

#endif
