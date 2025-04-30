/* Test for access to file, relative to open directory.  Linux version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sysdep.h>


int
__faccessat (int fd, const char *file, int mode, int flag)
{
  int ret = INLINE_SYSCALL_CALL (faccessat2, fd, file, mode, flag);
#if __ASSUME_FACCESSAT2
  return ret;
#else
  if (ret == 0 || errno != ENOSYS)
    return ret;

  if (flag & ~(AT_SYMLINK_NOFOLLOW | AT_EACCESS))
    return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);

  if ((flag == 0 || ((flag & ~AT_EACCESS) == 0 && ! __libc_enable_secure)))
    return INLINE_SYSCALL (faccessat, 3, fd, file, mode);

  struct stat64 stats;
  if (__fstatat64 (fd, file, &stats, flag & AT_SYMLINK_NOFOLLOW))
    return -1;

  mode &= (X_OK | W_OK | R_OK);	/* Clear any bogus bits. */
# if R_OK != S_IROTH || W_OK != S_IWOTH || X_OK != S_IXOTH
#  error Oops, portability assumptions incorrect.
# endif

  if (mode == F_OK)
    return 0;			/* The file exists. */

  uid_t uid = (flag & AT_EACCESS) ? __geteuid () : __getuid ();

  /* The super-user can read and write any file, and execute any file
     that anyone can execute. */
  if (uid == 0 && ((mode & X_OK) == 0
		   || (stats.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH))))
    return 0;

  int granted = (uid == stats.st_uid
		 ? (unsigned int) (stats.st_mode & (mode << 6)) >> 6
		 : (stats.st_gid == ((flag & AT_EACCESS)
				     ? __getegid () : __getgid ())
		    || __group_member (stats.st_gid))
		 ? (unsigned int) (stats.st_mode & (mode << 3)) >> 3
		 : (stats.st_mode & mode));

  if (granted == mode)
    return 0;

  return INLINE_SYSCALL_ERROR_RETURN_VALUE (EACCES);
#endif /* !__ASSUME_FACCESSAT2 */
}
weak_alias (__faccessat, faccessat)
