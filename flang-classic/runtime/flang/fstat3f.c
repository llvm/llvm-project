/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Implements LIB3F fstat subprogram.  */

/* must include ent3f.h AFTER io3f.h */
#include <sys/stat.h>
#include "io3f.h"
#include "ent3f.h"

int ENT3F(FSTAT, fstat)(int *lu, int *statb)
{
#if defined(_WIN64)
  struct _stat32 b;
  char *p;
  int i;
  void *f;

  f = __fio_find_unit(*lu);
  if (f && !FIO_FCB_STDUNIT(f)) {
    /* need a way to get f->fp's fildes */
    if (i = _stat32(FIO_FCB_NAME(f), &b))
      i = __io_errno();
  } else {
    switch (*lu) {
    case 0:
      i = 2;
      break;
    case 5:
      i = 0;
      break;
    case 6:
      i = 1;
      break;
    default:
      i = -1;
      break;
    }
    if (i = _fstat32(i, &b))
      i = __io_errno();
  }
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
  statb[11] = 0;
  statb[12] = 0;
  return i;
#else
  struct stat b;
  int i;
  void *f;

  f = __fio_find_unit(*lu);
  if (f && !FIO_FCB_STDUNIT(f)) {
    /** need a way to get f->fp's fildes **/
    if ((i = stat(FIO_FCB_NAME(f), &b)))
      i = __io_errno();
  } else {
    switch (*lu) {
    case 0:
      i = 2;
      break;
    case 5:
      i = 0;
      break;
    case 6:
      i = 1;
      break;
    default:
      i = -1;
      break;
    }
    if ((i = fstat(i, &b)))
      i = __io_errno();
  }
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
#if !defined(_WIN64)
  statb[11] = b.st_blksize;
  statb[12] = b.st_blocks;
#else
  statb[11] = 0;
  statb[12] = 0;
#endif
  return i;
#endif
}
