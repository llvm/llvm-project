/* Dynamic linker system dependencies for Linux.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

/* Linux needs some special initialization, but otherwise uses
   the generic dynamic linker system interface code.  */

#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/utsname.h>
#include <ldsodefs.h>
#include <not-cancel.h>

#ifdef SHARED
# define DL_SYSDEP_INIT frob_brk ()

static inline void
frob_brk (void)
{
  __brk (0);			/* Initialize the break.  */
}

# include <elf/dl-sysdep.c>
#endif


int
attribute_hidden
_dl_discover_osversion (void)
{
#if defined NEED_DL_SYSINFO_DSO && defined SHARED
  if (GLRO(dl_sysinfo_map) != NULL)
    {
      /* If the kernel-supplied DSO contains a note indicating the kernel's
	 version, we don't need to call uname or parse any strings.  */

      static const struct
      {
	ElfW(Nhdr) hdr;
	char vendor[8];
      } expected_note = { { sizeof "Linux", sizeof (ElfW(Word)), 0 }, "Linux" };
      const ElfW(Phdr) *const phdr = GLRO(dl_sysinfo_map)->l_phdr;
      const ElfW(Word) phnum = GLRO(dl_sysinfo_map)->l_phnum;
      for (uint_fast16_t i = 0; i < phnum; ++i)
	if (phdr[i].p_type == PT_NOTE)
	  {
	    const ElfW(Addr) start = (phdr[i].p_vaddr
				      + GLRO(dl_sysinfo_map)->l_addr);
	    const ElfW(Nhdr) *note = (const void *) start;
	    while ((ElfW(Addr)) (note + 1) - start < phdr[i].p_memsz)
	      {
		if (!memcmp (note, &expected_note, sizeof expected_note))
		  return *(const ElfW(Word) *) ((const void *) note
						+ sizeof expected_note);
#define ROUND(len) (((len) + sizeof note->n_type - 1) & -sizeof note->n_type)
		note = ((const void *) (note + 1)
			+ ROUND (note->n_namesz) + ROUND (note->n_descsz));
#undef ROUND
	      }
	  }
    }
#endif

  char bufmem[64];
  char *buf = bufmem;
  unsigned int version;
  int parts;
  char *cp;
  struct utsname uts;

  /* Try the uname system call.  */
  if (__uname (&uts))
    {
      /* This was not successful.  Now try reading the /proc filesystem.  */
      int fd = __open64_nocancel ("/proc/sys/kernel/osrelease", O_RDONLY);
      if (fd < 0)
	return -1;
      ssize_t reslen = __read_nocancel (fd, bufmem, sizeof (bufmem));
      __close_nocancel (fd);
      if (reslen <= 0)
	/* This also didn't work.  We give up since we cannot
	   make sure the library can actually work.  */
	return -1;
      buf[MIN (reslen, (ssize_t) sizeof (bufmem) - 1)] = '\0';
    }
  else
    buf = uts.release;

  /* Now convert it into a number.  The string consists of at most
     three parts.  */
  version = 0;
  parts = 0;
  cp = buf;
  while ((*cp >= '0') && (*cp <= '9'))
    {
      unsigned int here = *cp++ - '0';

      while ((*cp >= '0') && (*cp <= '9'))
	{
	  here *= 10;
	  here += *cp++ - '0';
	}

      ++parts;
      version <<= 8;
      version |= here;

      if (*cp++ != '.' || parts == 3)
	/* Another part following?  */
	break;
    }

  if (parts < 3)
    version <<= 8 * (3 - parts);

  return version;
}
