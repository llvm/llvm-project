/* Access to hardware i/o ports.  GNU/x86 version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#ifndef	_SYS_IO_H
#define	_SYS_IO_H	1

#include <features.h>

__BEGIN_DECLS

/* If TURN_ON is TRUE, request for permission to do direct i/o on the
   port numbers in the range [FROM,FROM+NUM-1].  Otherwise, turn I/O
   permission off for that range.  This call requires root privileges.  */
extern int ioperm (unsigned long int __from, unsigned long int __num,
                   int __turn_on) __THROW;

/* Set the I/O privilege level to LEVEL.  If LEVEL>3, permission to
   access any I/O port is granted.  This call requires root
   privileges. */
extern int iopl (int __level) __THROW;

#if defined __GNUC__ && __GNUC__ >= 2

static __inline unsigned char
inb (unsigned short int port)
{
  unsigned char _v;

  __asm__ __volatile__ ("inb %w1,%0":"=a" (_v):"Nd" (port));
  return _v;
}

static __inline unsigned char
inb_p (unsigned short int port)
{
  unsigned char _v;

  __asm__ __volatile__ ("inb %w1,%0\noutb %%al,$0x80":"=a" (_v):"Nd" (port));
  return _v;
}

static __inline unsigned short int
inw (unsigned short int port)
{
  unsigned short _v;

  __asm__ __volatile__ ("inw %w1,%0":"=a" (_v):"Nd" (port));
  return _v;
}

static __inline unsigned short int
inw_p (unsigned short int port)
{
  unsigned short int _v;

  __asm__ __volatile__ ("inw %w1,%0\noutb %%al,$0x80":"=a" (_v):"Nd" (port));
  return _v;
}

static __inline unsigned int
inl (unsigned short int port)
{
  unsigned int _v;

  __asm__ __volatile__ ("inl %w1,%0":"=a" (_v):"Nd" (port));
  return _v;
}

static __inline unsigned int
inl_p (unsigned short int port)
{
  unsigned int _v;
  __asm__ __volatile__ ("inl %w1,%0\noutb %%al,$0x80":"=a" (_v):"Nd" (port));
  return _v;
}

static __inline void
outb (unsigned char value, unsigned short int port)
{
  __asm__ __volatile__ ("outb %b0,%w1": :"a" (value), "Nd" (port));
}

static __inline void
outb_p (unsigned char value, unsigned short int port)
{
  __asm__ __volatile__ ("outb %b0,%w1\noutb %%al,$0x80": :"a" (value),
			"Nd" (port));
}

static __inline void
outw (unsigned short int value, unsigned short int port)
{
  __asm__ __volatile__ ("outw %w0,%w1": :"a" (value), "Nd" (port));

}

static __inline void
outw_p (unsigned short int value, unsigned short int port)
{
  __asm__ __volatile__ ("outw %w0,%w1\noutb %%al,$0x80": :"a" (value),
			"Nd" (port));
}

static __inline void
outl (unsigned int value, unsigned short int port)
{
  __asm__ __volatile__ ("outl %0,%w1": :"a" (value), "Nd" (port));
}

static __inline void
outl_p (unsigned int value, unsigned short int port)
{
  __asm__ __volatile__ ("outl %0,%w1\noutb %%al,$0x80": :"a" (value),
			"Nd" (port));
}

static __inline void
insb (unsigned short int port, void *addr, unsigned long int count)
{
  __asm__ __volatile__ ("cld ; rep ; insb":"=D" (addr),
			"=c" (count):"d" (port), "0" (addr), "1" (count));
}

static __inline void
insw (unsigned short int port, void *addr, unsigned long int count)
{
  __asm__ __volatile__ ("cld ; rep ; insw":"=D" (addr),
			"=c" (count):"d" (port), "0" (addr), "1" (count));
}

static __inline void
insl (unsigned short int port, void *addr, unsigned long int count)
{
  __asm__ __volatile__ ("cld ; rep ; insl":"=D" (addr),
			"=c" (count):"d" (port), "0" (addr), "1" (count));
}

static __inline void
outsb (unsigned short int port, const void *addr, unsigned long int count)
{
  __asm__ __volatile__ ("cld ; rep ; outsb":"=S" (addr),
			"=c" (count):"d" (port), "0" (addr), "1" (count));
}

static __inline void
outsw (unsigned short int port, const void *addr, unsigned long int count)
{
  __asm__ __volatile__ ("cld ; rep ; outsw":"=S" (addr),
			"=c" (count):"d" (port), "0" (addr), "1" (count));
}

static __inline void
outsl (unsigned short int port, const void *addr, unsigned long int count)
{
  __asm__ __volatile__ ("cld ; rep ; outsl":"=S" (addr),
			"=c" (count):"d" (port), "0" (addr), "1" (count));
}

#endif	/* GNU C */

__END_DECLS
#endif /* _SYS_IO_H */
