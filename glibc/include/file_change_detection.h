/* Detecting file changes using modification times.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _FILE_CHANGE_DETECTION_H
#define _FILE_CHANGE_DETECTION_H

#include <stdbool.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

/* Items for identifying a particular file version.  Excerpt from
   struct stat64.  */
struct file_change_detection
{
  /* Special values: 0 if file does not exist.  -1 to force mismatch
     with the next comparison.  */
  off64_t size;

  ino64_t ino;
  struct __timespec64 mtime;
  struct __timespec64 ctime;
};

/* Returns true if *LEFT and *RIGHT describe the same version of the
   same file.  */
bool __file_is_unchanged (const struct file_change_detection *left,
                          const struct file_change_detection *right);

/* Extract file change information to *FILE from the stat buffer
   *ST.  */
void __file_change_detection_for_stat (struct file_change_detection *file,
                                       const struct __stat64_t64 *st);

/* Writes file change information for PATH to *FILE.  Returns true on
   success.  For benign errors, *FILE is cleared, and true is
   returned.  For errors indicating resource outages and the like,
   false is returned.  */
bool __file_change_detection_for_path (struct file_change_detection *file,
                                       const char *path);

/* Writes file change information for the stream FP to *FILE.  Returns
   ture on success, false on failure.  If FP is NULL, treat the file
   as non-existing.  */
bool __file_change_detection_for_fp (struct file_change_detection *file,
                                     FILE *fp);

#ifndef _ISOMAC
libc_hidden_proto (__file_is_unchanged)
libc_hidden_proto (__file_change_detection_for_stat)
libc_hidden_proto (__file_change_detection_for_path)
libc_hidden_proto (__file_change_detection_for_fp)
#endif

#endif /* _FILE_CHANGE_DETECTION_H */
