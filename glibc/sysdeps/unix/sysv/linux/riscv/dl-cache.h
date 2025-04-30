/* Support for reading /etc/ld.so.cache files written by Linux ldconfig.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <ldconfig.h>
#include <assert.h>

/* For now we only support the natural XLEN ABI length on all targets, so the
   only bits that need to go into ld.so.cache are the FLEG ABI length.  */
#if defined __riscv_float_abi_double
# define _DL_CACHE_DEFAULT_ID    (FLAG_RISCV_FLOAT_ABI_DOUBLE | FLAG_ELF_LIBC6)
#else
# define _DL_CACHE_DEFAULT_ID    (FLAG_RISCV_FLOAT_ABI_SOFT | FLAG_ELF_LIBC6)
#endif

#define _dl_cache_check_flags(flags)		    			\
  ((flags) == _DL_CACHE_DEFAULT_ID)

/* If given a path to one of our library directories, adds every library
   directory via add_dir (), otherwise just adds the giver directory.  On
   RISC-V, libraries can be found in paths ending in:
     - /lib64/lp64d
     - /lib64/lp64
     - /lib32/ilp32d
     - /lib32/ilp32
     - /lib (only ld.so)
   so this will add all of those paths.

   According to Joseph Myers:
       My reasoning for that would be: generic autoconf-configured (etc.)
       software may only know about using the lib directory, so you want the
       lib directory to be searched regardless of the ABI - but it's also
       useful to be able to e.g. list /usr/local/lib in /etc/ld.so.conf for all
       architectures and have that automatically imply /usr/local/lib64/lp64d
       etc. so that libraries can be found that come from software that does
       use the ABI-specific directories.  */
#define add_system_dir(dir) 						\
  do							    		\
    {									\
      static const char* lib_dirs[] = {					\
	"/lib64/lp64d",							\
	"/lib64/lp64",							\
	"/lib32/ilp32d",						\
	"/lib32/ilp32",							\
	NULL,								\
      };								\
      const size_t lib_len = sizeof ("/lib") - 1;			\
      size_t len = strlen (dir);					\
      char path[len + 10];						\
      const char **ptr;							\
									\
      memcpy (path, dir, len + 1);					\
									\
      for (ptr = lib_dirs; *ptr != NULL; ptr++)				\
	{								\
	  const char *lib_dir = *ptr;					\
	  size_t dir_len = strlen (lib_dir);				\
									\
	  if (len >= dir_len						\
	      && !memcmp (path + len - dir_len, lib_dir, dir_len))	\
	    {								\
	      len -= dir_len - lib_len;					\
	      path[len] = '\0';						\
	      break;							\
	    }								\
	}								\
      add_dir (path);							\
      if (len >= lib_len						\
	  && !memcmp (path + len - lib_len, "/lib", lib_len))		\
	for (ptr = lib_dirs; *ptr != NULL; ptr++)			\
	  {								\
	    const char *lib_dir = *ptr;					\
	    size_t dir_len = strlen (lib_dir);				\
									\
	    assert (dir_len >= lib_len);				\
	    memcpy (path + len, lib_dir + lib_len,			\
		    dir_len - lib_len + 1);				\
	    add_dir (path);						\
	  }								\
    } while (0)


#include_next <dl-cache.h>
