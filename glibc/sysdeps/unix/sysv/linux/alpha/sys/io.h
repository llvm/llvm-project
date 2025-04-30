/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#ifndef	_SYS_IO_H

#define	_SYS_IO_H	1
#include <features.h>

__BEGIN_DECLS

/* If TURN_ON is TRUE, request for permission to do direct i/o on the
   port numbers in the range [FROM,FROM+NUM-1].  Otherwise, turn I/O
   permission off for that range.  This call requires root privileges.

   Portability note: not all Linux platforms support this call.  Most
   platforms based on the PC I/O architecture probably will, however.
   E.g., Linux/Alpha for Alpha PCs supports this.  */
extern int ioperm (unsigned long int __from, unsigned long int __num,
		   int __turn_on) __THROW;

/* Set the I/O privilege level to LEVEL.  If LEVEL>3, permission to
   access any I/O port is granted.  This call requires root
   privileges. */
extern int iopl (int __level) __THROW;

/* Return the physical address of the DENSE I/O memory or NULL if none
   is available (e.g. on a jensen).  */
extern unsigned long int _bus_base (void) __THROW __attribute__ ((const));
extern unsigned long int bus_base (void) __THROW __attribute__ ((const));

/* Return the physical address of the SPARSE I/O memory.  */
extern unsigned long _bus_base_sparse (void) __THROW __attribute__ ((const));
extern unsigned long bus_base_sparse (void) __THROW __attribute__ ((const));

/* Return the HAE shift used by the SPARSE I/O memory.  */
extern int _hae_shift (void) __THROW __attribute__ ((const));
extern int hae_shift (void) __THROW __attribute__ ((const));

/* Previous three are deprecated in favour of the following, which
   knows about multiple PCI "hoses".  Provide the PCI bus and dfn
   numbers just as to pciconfig_read/write.  */

enum __pciconfig_iobase_which
{
  IOBASE_HOSE = 0,		/* Return hose index. */
  IOBASE_SPARSE_MEM = 1,	/* Return physical memory addresses.  */
  IOBASE_DENSE_MEM = 2,
  IOBASE_SPARSE_IO = 3,
  IOBASE_DENSE_IO = 4
};

extern long pciconfig_iobase(enum __pciconfig_iobase_which __which,
			     unsigned long int __bus,
			     unsigned long int __dfn)
     __THROW __attribute__ ((const));

/* Access PCI space protected from machine checks.  */
extern int pciconfig_read (unsigned long int __bus,
			   unsigned long int __dfn,
			   unsigned long int __off,
			   unsigned long int __len,
			   unsigned char *__buf) __THROW;

extern int pciconfig_write (unsigned long int __bus,
			    unsigned long int __dfn,
			    unsigned long int __off,
			    unsigned long int __len,
			    unsigned char *__buf) __THROW;

/* Userspace declarations.  */
extern unsigned int inb (unsigned long __port) __THROW;
extern unsigned int inw (unsigned long __port) __THROW;
extern unsigned int inl (unsigned long __port) __THROW;
extern void outb (unsigned char __b, unsigned long __port) __THROW;
extern void outw (unsigned short __w, unsigned long __port) __THROW;
extern void outl (unsigned int __l, unsigned long __port) __THROW;

__END_DECLS

#endif /* _SYS_IO_H */
