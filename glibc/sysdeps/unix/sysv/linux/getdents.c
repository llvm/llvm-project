/* Get directory entries.  Linux non-LFS version.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#include <dirent.h>

#if !_DIRENT_MATCHES_DIRENT64

# include <unistd.h>
# include <string.h>
# include <errno.h>

# ifndef DIRENT_SET_DP_INO
#  define DIRENT_SET_DP_INO(dp, value) (dp)->d_ino = (value)
# endif

/* Pack the dirent64 struct down into 32-bit offset/inode fields, and
   ensure that no overflow occurs.  */
ssize_t
__getdents (int fd, void *buf0, size_t nbytes)
{
  char *buf = buf0;

  union
  {
    /* For !_DIRENT_MATCHES_DIRENT64 kernel 'linux_dirent64' has the same
       layout of 'struct dirent64'.  */
    struct dirent64 k;
    struct dirent u;
    char b[1];
  } *kbuf = (void *) buf, *outp, *inp;
  size_t kbytes = nbytes;
  off64_t last_offset = -1;
  ssize_t retval;

# define size_diff (offsetof (struct dirent64, d_name) \
		    - offsetof (struct dirent, d_name))
  char kbuftmp[sizeof (struct dirent) + size_diff];
  if (nbytes <= sizeof (struct dirent))
    kbuf = (void*) kbuftmp;

  retval = INLINE_SYSCALL_CALL (getdents64, fd, kbuf, kbytes);
  if (retval == -1)
    return -1;

  /* These two pointers might alias the same memory buffer.
     Standard C requires that we always use the same type for them,
     so we must use the union type.  */
  inp = kbuf;
  outp = (void *) buf;

  while (&inp->b < &kbuf->b + retval)
    {
      const size_t alignment = _Alignof (struct dirent);
      /* Since inp->k.d_reclen is already aligned for the kernel
         structure this may compute a value that is bigger
         than necessary.  */
      size_t old_reclen = inp->k.d_reclen;
      size_t new_reclen = ((old_reclen - size_diff + alignment - 1)
                           & ~(alignment - 1));

      /* Copy the data out of the old structure into temporary space.
         Then copy the name, which may overlap if BUF == KBUF.  */
      const uint64_t d_ino = inp->k.d_ino;
      const int64_t d_off = inp->k.d_off;
      const uint8_t d_type = inp->k.d_type;

      memmove (outp->u.d_name, inp->k.d_name,
               old_reclen - offsetof (struct dirent64, d_name));

      /* Now we have copied the data from INP and access only OUTP.  */

      DIRENT_SET_DP_INO (&outp->u, d_ino);
      outp->u.d_off = d_off;
      if ((sizeof (outp->u.d_ino) != sizeof (inp->k.d_ino)
           && outp->u.d_ino != d_ino)
          || (sizeof (outp->u.d_off) != sizeof (inp->k.d_off)
              && outp->u.d_off != d_off))
        {
          /* Overflow.  If there was at least one entry before this one,
             return them without error, otherwise signal overflow.  */
          if (last_offset != -1)
            {
              __lseek64 (fd, last_offset, SEEK_SET);
              return outp->b - buf;
            }
	  return INLINE_SYSCALL_ERROR_RETURN_VALUE (EOVERFLOW);
        }

      last_offset = d_off;
      outp->u.d_reclen = new_reclen;
      outp->u.d_type = d_type;

      inp = (void *) inp + old_reclen;
      outp = (void *) outp + new_reclen;
    }

  return outp->b - buf;
}

# undef DIRENT_SET_DP_INO

#endif /* _DIRENT_MATCHES_DIRENT64  */
