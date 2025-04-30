/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger-Tang <davidm@hpl.hp.com>.

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

/* I/O access is restricted to ISA port space (ports 0..65535).
   Modern devices hopefully are sane enough not to put any performance
   critical registers in i/o space.

   On the first call to ioperm() or iopl(), the entire (E)ISA port
   space is mapped into the virtual address space at address io.base.
   mprotect() calls are then used to enable/disable access to ports.
   Per 4KB page, there are 4 I/O ports.  */

#include <errno.h>
#include <fcntl.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/mman.h>

#define MAX_PORT	0x10000

/*
 * Memory fence w/accept.  This should never be used in code that is
 * not IA-64 specific.
 */
#define __ia64_mf_a()	__asm__ __volatile__ ("mf.a" ::: "memory")

static struct
  {
    unsigned long int base;
    unsigned long int page_mask;
  }
io;

__inline__ unsigned long int
io_offset (unsigned long int port)
{
	return ((port >> 2) << 12) | (port & 0xfff);
}

int
_ioperm (unsigned long int from, unsigned long int num, int turn_on)
{
  unsigned long int base;

  /* this test isn't as silly as it may look like; consider overflows! */
  if (from >= MAX_PORT || from + num > MAX_PORT)
    {
      __set_errno (EINVAL);
      return -1;
    }

  if (turn_on)
    {
      if (!io.base)
	{
	  unsigned long phys_io_base, len;
	  int fd;

	  io.page_mask = ~(__getpagesize() - 1);

	  /* get I/O base physical address from ar.k0 as per PRM: */
	  __asm__ ("mov %0=ar.k0" : "=r"(phys_io_base));

	  /* The O_SYNC flag tells the /dev/mem driver to map the
             memory uncached: */
	  fd = __open ("/dev/mem", O_RDWR | O_SYNC);
	  if (fd < 0)
	    return -1;

	  len = io_offset (MAX_PORT);
	  /* see comment below */
	  base = (unsigned long int) __mmap (0, len, PROT_READ | PROT_WRITE, MAP_SHARED,
						fd, phys_io_base);
	  __close (fd);

	  if ((long) base == -1)
	    return -1;

	  io.base = base;
	}
    }
  else
    {
      if (!io.base)
	return 0;	/* never was turned on... */
    }

  /* We can't do mprotect because that would cause us to lose the
     uncached flag that the /dev/mem driver turned on.  A MAP_UNCACHED
     flag seems so much cleaner...

     See the history of this file for a version that tried mprotect.  */
  return 0;
}

int
_iopl (unsigned int level)
{
  if (level > 3)
    {
      __set_errno (EINVAL);
      return -1;
    }
  if (level)
    {
      int retval = _ioperm (0, MAX_PORT, 1);
      /* Match the documented error returns of the x86 version.  */
      if (retval < 0 && errno == EACCES)
	__set_errno (EPERM);
      return retval;
    }
  return 0;
}

unsigned int
_inb (unsigned long int port)
{
  volatile unsigned char *addr = (void *) io.base + io_offset (port);
  unsigned char ret;

  ret = *addr;
  __ia64_mf_a();
  return ret;
}

unsigned int
_inw (unsigned long int port)
{
  volatile unsigned short *addr = (void *) io.base + io_offset (port);
  unsigned short ret;

  ret = *addr;
  __ia64_mf_a();
  return ret;
}

unsigned int
_inl (unsigned long int port)
{
  volatile unsigned int *addr = (void *) io.base + io_offset (port);
  unsigned int ret;

  ret = *addr;
  __ia64_mf_a();
  return ret;
}

void
_outb (unsigned char val, unsigned long int port)
{
  volatile unsigned char *addr = (void *) io.base + io_offset (port);

  *addr = val;
  __ia64_mf_a();
}

void
_outw (unsigned short val, unsigned long int port)
{
  volatile unsigned short *addr = (void *) io.base + io_offset (port);

  *addr = val;
  __ia64_mf_a();
}

void
_outl (unsigned int val, unsigned long int port)
{
  volatile unsigned int *addr = (void *) io.base + io_offset (port);

  *addr = val;
  __ia64_mf_a();
}

weak_alias (_ioperm, ioperm);
weak_alias (_iopl, iopl);
weak_alias (_inb, inb);
weak_alias (_inw, inw);
weak_alias (_inl, inl);
weak_alias (_outb, outb);
weak_alias (_outw, outw);
weak_alias (_outl, outl);
