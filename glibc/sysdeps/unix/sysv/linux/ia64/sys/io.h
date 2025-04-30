/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
     Contributed by David Mosberger-Tang <davidm@hpl.hp.com>

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
		   int __turn_on);

/* Set the I/O privilege level to LEVEL.  If LEVEL>3, permission to
   access any I/O port is granted.  This call requires root
   privileges. */
extern int iopl (int __level);

extern unsigned int _inb (unsigned long int __port);
extern unsigned int _inb (unsigned long int __port);
extern unsigned int _inw (unsigned long int __port);
extern unsigned int _inl (unsigned long int __port);
extern void _outb (unsigned char __val, unsigned long int __port);
extern void _outw (unsigned short __val, unsigned long int __port);
extern void _outl (unsigned int __val, unsigned long int __port);

#define inb	_inb
#define inw	_inw
#define inl	_inl
#define outb	_outb
#define outw	_outw
#define outl	_outl

/* Access PCI space protected from machine checks.  */
extern int pciconfig_read (unsigned long int __bus, unsigned long int __dfn,
			   unsigned long int __off, unsigned long int __len,
			   unsigned char *__buf);

extern int pciconfig_write (unsigned long int __bus, unsigned long int __dfn,
			    unsigned long int __off, unsigned long int __len,
			    unsigned char *__buf);

__END_DECLS

#endif /* _SYS_IO_H */
