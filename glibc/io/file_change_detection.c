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

#include <file_change_detection.h>

#include <errno.h>
#include <stddef.h>

bool
__file_is_unchanged (const struct file_change_detection *left,
                     const struct file_change_detection *right)
{
  if (left->size < 0 || right->size < 0)
    /* Negative sizes are used as markers and never match.  */
    return false;
  else if (left->size == 0 && right->size == 0)
    /* Both files are empty or do not exist, so they have the same
       content, no matter what the other fields indicate.  */
    return true;
  else
    return left->size == right->size
      && left->ino == right->ino
      && left->mtime.tv_sec == right->mtime.tv_sec
      && left->mtime.tv_nsec == right->mtime.tv_nsec
      && left->ctime.tv_sec == right->ctime.tv_sec
      && left->ctime.tv_nsec == right->ctime.tv_nsec;
}
libc_hidden_def (__file_is_unchanged)

void
__file_change_detection_for_stat (struct file_change_detection *file,
                                  const struct __stat64_t64 *st)
{
  if (S_ISDIR (st->st_mode))
    /* Treat as empty file.  */
    file->size = 0;
  else if (!S_ISREG (st->st_mode))
    /* Non-regular files cannot be cached.  */
    file->size = -1;
  else
    {
      file->size = st->st_size;
      file->ino = st->st_ino;
      file->mtime = (struct __timespec64) { st->st_mtim.tv_sec,
					    st->st_mtim.tv_nsec };
      file->ctime = (struct __timespec64) { st->st_ctim.tv_sec,
					    st->st_ctim.tv_nsec };
    }
}
libc_hidden_def (__file_change_detection_for_stat)

bool
__file_change_detection_for_path (struct file_change_detection *file,
                                  const char *path)
{
  struct __stat64_t64 st;
  if (__stat64_time64 (path, &st) != 0)
    switch (errno)
      {
      case EACCES:
      case EISDIR:
      case ELOOP:
      case ENOENT:
      case ENOTDIR:
      case EPERM:
        /* Ignore errors due to file system contents.  Instead, treat
           the file as empty.  */
        file->size = 0;
        return true;
      default:
        /* Other errors are fatal.  */
        return false;
      }
  else /* stat64 was successfull.  */
    {
      __file_change_detection_for_stat (file, &st);
      return true;
    }
}
libc_hidden_def (__file_change_detection_for_path)

bool
__file_change_detection_for_fp (struct file_change_detection *file,
                                FILE *fp)
{
  if (fp == NULL)
    {
      /* The file does not exist.  */
      file->size = 0;
      return true;
    }
  else
    {
      struct __stat64_t64 st;
      if (__fstat64_time64 (__fileno (fp), &st) != 0)
        /* If we already have a file descriptor, all errors are fatal.  */
        return false;
      else
        {
          __file_change_detection_for_stat (file, &st);
          return true;
        }
    }
}
libc_hidden_def (__file_change_detection_for_fp)
