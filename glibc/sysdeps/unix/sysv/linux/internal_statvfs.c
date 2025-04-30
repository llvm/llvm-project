/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <internal_statvfs.h>
#include <string.h>
#include <time.h>
#include <kernel_stat.h>

/* Special internal-only bit value.  */
# define ST_VALID 0x0020

#if !STATFS_IS_STATFS64
void
__internal_statvfs (struct statvfs *buf, const struct statfs *fsbuf)
{
  /* Now fill in the fields we have information for.  */
  buf->f_bsize = fsbuf->f_bsize;
  /* Linux has the f_frsize size only in later version of the kernel.
     If the value is not filled in use f_bsize.  */
  buf->f_frsize = fsbuf->f_frsize ?: fsbuf->f_bsize;
  buf->f_blocks = fsbuf->f_blocks;
  buf->f_bfree = fsbuf->f_bfree;
  buf->f_bavail = fsbuf->f_bavail;
  buf->f_files = fsbuf->f_files;
  buf->f_ffree = fsbuf->f_ffree;
  if (sizeof (buf->f_fsid) == sizeof (fsbuf->f_fsid))
    /* The shifting uses 'unsigned long long int' even though the target
       field might only have 32 bits.  This is OK since the 'if' branch
       is not used in this case but the compiler would still generate
       warnings.  */
    buf->f_fsid = ((fsbuf->f_fsid.__val[0]
		    & ((1ULL << (8 * sizeof (fsbuf->f_fsid.__val[0]))) - 1))
		   | ((unsigned long long int) fsbuf->f_fsid.__val[1]
		      << (8 * (sizeof (buf->f_fsid)
			       - sizeof (fsbuf->f_fsid.__val[0])))));
  else
    /* We cannot help here.  The statvfs element is not large enough to
       contain both words of the statfs f_fsid field.  */
    buf->f_fsid = fsbuf->f_fsid.__val[0];
#ifdef _STATVFSBUF_F_UNUSED
  buf->__f_unused = 0;
#endif
  buf->f_namemax = fsbuf->f_namelen;
  memset (buf->__f_spare, '\0', sizeof (buf->__f_spare));

  /* What remains to do is to fill the fields f_favail and f_flag.  */

  /* XXX I have no idea how to compute f_favail.  Any idea???  */
  buf->f_favail = buf->f_ffree;

  buf->f_flag = fsbuf->f_flags ^ ST_VALID;
}
#endif

void
__internal_statvfs64 (struct statvfs64 *buf, const struct statfs64 *fsbuf)
{
  /* Now fill in the fields we have information for.  */
  buf->f_bsize = fsbuf->f_bsize;
  /* Linux has the f_frsize size only in later version of the kernel.
     If the value is not filled in use f_bsize.  */
  buf->f_frsize = fsbuf->f_frsize ?: fsbuf->f_bsize;
  buf->f_blocks = fsbuf->f_blocks;
  buf->f_bfree = fsbuf->f_bfree;
  buf->f_bavail = fsbuf->f_bavail;
  buf->f_files = fsbuf->f_files;
  buf->f_ffree = fsbuf->f_ffree;
  if (sizeof (buf->f_fsid) == sizeof (fsbuf->f_fsid))
    /* The shifting uses 'unsigned long long int' even though the target
       field might only have 32 bits.  This is OK since the 'if' branch
       is not used in this case but the compiler would still generate
       warnings.  */
    buf->f_fsid = ((fsbuf->f_fsid.__val[0]
		    & ((1ULL << (8 * sizeof (fsbuf->f_fsid.__val[0]))) - 1))
		   | ((unsigned long long int) fsbuf->f_fsid.__val[1]
		      << (8 * (sizeof (buf->f_fsid)
			       - sizeof (fsbuf->f_fsid.__val[0])))));
  else
    /* We cannot help here.  The statvfs element is not large enough to
       contain both words of the statfs f_fsid field.  */
    buf->f_fsid = fsbuf->f_fsid.__val[0];
#ifdef _STATVFSBUF_F_UNUSED
  buf->__f_unused = 0;
#endif
  buf->f_namemax = fsbuf->f_namelen;
  memset (buf->__f_spare, '\0', sizeof (buf->__f_spare));

  /* What remains to do is to fill the fields f_favail and f_flag.  */

  /* XXX I have no idea how to compute f_favail.  Any idea???  */
  buf->f_favail = buf->f_ffree;

  buf->f_flag = fsbuf->f_flags ^ ST_VALID;
}
