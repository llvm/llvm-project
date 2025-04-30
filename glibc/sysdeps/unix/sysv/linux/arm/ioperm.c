/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Phil Blundell, based on the Alpha version by
   David Mosberger.

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

#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_4, GLIBC_2_30)
# include <errno.h>

int
ioperm (unsigned long int from, unsigned long int num, int turn_on)
{
  __set_errno (ENOSYS);
  return -1;
}
compat_symbol (libc, ioperm, ioperm, GLIBC_2_4);

int
iopl (unsigned int level)
{
  __set_errno (ENOSYS);
  return -1;
}
compat_symbol (libc, iopl, iopl, GLIBC_2_4);


/* The remaining functions do not have any way to indicate failure.
   However, it is only valid to call them after calling ioperm/iopl,
   which will have indicated failure.  */

void
outb (unsigned char b, unsigned long int port)
{
}
compat_symbol (libc, outb, outb, GLIBC_2_4);

void
outw (unsigned short b, unsigned long int port)
{
}
compat_symbol (libc, outw, outw, GLIBC_2_4);

void
outl (unsigned int b, unsigned long int port)
{
}
compat_symbol (libc, outl, outl, GLIBC_2_4);

unsigned int
inb (unsigned long int port)
{
  return 0;
}
compat_symbol (libc, inb, inb, GLIBC_2_4);


unsigned int
inw (unsigned long int port)
{
  return 0;
}
compat_symbol (libc, inw, inw, GLIBC_2_4);


unsigned int
inl (unsigned long int port)
{
  return 0;
}
compat_symbol (libc, inl, inl, GLIBC_2_4);

#endif /* SHLIB_COMAT */
