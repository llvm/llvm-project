/* Definition of `struct statvfs', information about a filesystem.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#ifndef _SYS_STATVFS_H
# error "Never include <bits/statvfs.h> directly; use <sys/statvfs.h> instead."
#endif

#include <bits/types.h>

/* GNU Hurd NOTE: The size of this structure (16 ints) is known in
   <hurd/hurd_types.defs>, since it is used in the `file_statfs' RPC.  MiG
   does not cope at all well with the passed C structure not being of the
   expected size.  There are some filler words at the end to allow for
   future expansion.  To increase the size of the structure used in the RPC
   and retain binary compatibility, we would need to assign a new message
   number.  */

struct statvfs
  {
    unsigned long int f_bsize;
    unsigned long int f_frsize;
#ifndef __USE_FILE_OFFSET64
    __fsblkcnt_t f_blocks;
    __fsblkcnt_t f_bfree;
    __fsblkcnt_t f_bavail;
    __fsfilcnt_t f_files;
    __fsfilcnt_t f_ffree;
    __fsfilcnt_t f_favail;
#else
    __fsblkcnt64_t f_blocks;
    __fsblkcnt64_t f_bfree;
    __fsblkcnt64_t f_bavail;
    __fsfilcnt64_t f_files;
    __fsfilcnt64_t f_ffree;
    __fsfilcnt64_t f_favail;
#endif
    __fsid_t f_fsid;
    unsigned long int f_flag;
    unsigned long int f_namemax;
    unsigned int f_spare[6];
  };

#ifdef __USE_LARGEFILE64
struct statvfs64
  {
    unsigned long int f_bsize;
    unsigned long int f_frsize;
    __fsblkcnt64_t f_blocks;
    __fsblkcnt64_t f_bfree;
    __fsblkcnt64_t f_bavail;
    __fsfilcnt64_t f_files;
    __fsfilcnt64_t f_ffree;
    __fsfilcnt64_t f_favail;
    __fsid_t f_fsid;
    unsigned long int f_flag;
    unsigned long int f_namemax;
    unsigned int f_spare[6];
  };
#endif

/* Definitions for the flag in `f_flag'.  */
enum
{
  ST_RDONLY = 1,
#define ST_RDONLY	ST_RDONLY
  ST_NOSUID = 2
#define ST_NOSUID	ST_NOSUID
};
