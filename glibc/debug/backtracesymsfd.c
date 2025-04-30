/* Write formatted list with names for addresses in backtrace to a file.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <execinfo.h>
#include <string.h>
#include <sys/uio.h>

#include <_itoa.h>
#include <ldsodefs.h>

#if __ELF_NATIVE_CLASS == 32
# define WORD_WIDTH 8
#else
/* We assume 64bits.  */
# define WORD_WIDTH 16
#endif


void
__backtrace_symbols_fd (void *const *array, int size, int fd)
{
  struct iovec iov[9];
  int cnt;

  for (cnt = 0; cnt < size; ++cnt)
    {
      char buf[WORD_WIDTH];
      char buf2[WORD_WIDTH];
      Dl_info info;
      struct link_map *map;
      size_t last = 0;

      if (_dl_addr (array[cnt], &info, &map, NULL)
	  && info.dli_fname != NULL && info.dli_fname[0] != '\0')
	{
	  /* Name of the file.  */
	  iov[0].iov_base = (void *) info.dli_fname;
	  iov[0].iov_len = strlen (info.dli_fname);
	  last = 1;

	  if (info.dli_sname != NULL || map->l_addr != 0)
	    {
	      size_t diff;

	      iov[last].iov_base = (void *) "(";
	      iov[last].iov_len = 1;
	      ++last;

	      if (info.dli_sname != NULL)
		{
		  /* We have a symbol name.  */
		  iov[last].iov_base = (void *) info.dli_sname;
		  iov[last].iov_len = strlen (info.dli_sname);
		  ++last;
		}
	      else
		/* We have no symbol, so describe it as relative to the file.
		   The load bias is more useful to the user than the load
		   address.  The use of these addresses is to calculate an
		   address in the ELF file, so its prelinked bias is not
		   something we want to subtract out.  */
		info.dli_saddr = (void *) map->l_addr;

	      if (array[cnt] >= (void *) info.dli_saddr)
		{
		  iov[last].iov_base = (void *) "+0x";
		  diff = array[cnt] - info.dli_saddr;
		}
	      else
		{
		  iov[last].iov_base = (void *) "-0x";
		  diff = info.dli_saddr - array[cnt];
		}
	      iov[last].iov_len = 3;
	      ++last;

	      iov[last].iov_base = _itoa_word ((unsigned long int) diff,
					       &buf2[WORD_WIDTH], 16, 0);
	      iov[last].iov_len = (&buf2[WORD_WIDTH]
				   - (char *) iov[last].iov_base);
	      ++last;

	      iov[last].iov_base = (void *) ")";
	      iov[last].iov_len = 1;
	      ++last;
	    }
	}

      iov[last].iov_base = (void *) "[0x";
      iov[last].iov_len = 3;
      ++last;

      iov[last].iov_base = _itoa_word ((unsigned long int) array[cnt],
				       &buf[WORD_WIDTH], 16, 0);
      iov[last].iov_len = &buf[WORD_WIDTH] - (char *) iov[last].iov_base;
      ++last;

      iov[last].iov_base = (void *) "]\n";
      iov[last].iov_len = 2;
      ++last;

      __writev (fd, iov, last);
    }
}
weak_alias (__backtrace_symbols_fd, backtrace_symbols_fd)
libc_hidden_def (__backtrace_symbols_fd)
