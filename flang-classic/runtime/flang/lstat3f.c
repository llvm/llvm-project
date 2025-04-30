/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	lstat3f.c - Implements LIB3F lstat subprogram.  */

#if !defined(_WIN64)

/* must include ent3f.h AFTER io3f.h */
#include <sys/stat.h>
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(LSTAT, lstat)(DCHAR(nm), int *statb DCLEN(nm))
{
  struct stat b;
  char *p;
  int i;

  p = __fstr2cstr(CADR(nm), CLEN(nm));
  if ((i = lstat(p, &b)))
    i = __io_errno();
  __cstr_free(p);
  statb[0] = b.st_dev;
  statb[1] = b.st_ino;
  statb[2] = b.st_mode;
  statb[3] = b.st_nlink;
  statb[4] = b.st_uid;
  statb[5] = b.st_gid;
  statb[6] = b.st_rdev;
  statb[7] = b.st_size;
  statb[8] = b.st_atime;
  statb[9] = b.st_mtime;
  statb[10] = b.st_ctime;
  statb[11] = b.st_blksize;
  statb[12] = b.st_blocks;
  return i;
}

#endif /* !_WIN64 */
