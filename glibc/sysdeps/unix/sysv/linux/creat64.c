/* Linux default implementation for LFS creat.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <fcntl.h>
#include <sys/types.h>
#include <sysdep-cancel.h>

/* Create FILE with protections MODE.  */
int
__creat64 (const char *file, mode_t mode)
{
#if defined __OFF_T_MATCHES_OFF64_T && defined __NR_creat
  return SYSCALL_CANCEL (creat, file, mode);
#else
  /* We need to pass O_LARGEFILE.  */
  return __open64 (file, O_WRONLY | O_CREAT | O_TRUNC, mode);
#endif
}
weak_alias (__creat64, creat64)

#ifdef __OFF_T_MATCHES_OFF64_T
strong_alias (__creat64, __creat)
weak_alias (__creat64, creat)
#endif
