/* Compute hash alue for given string according to ELF standard.
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

#ifndef _DL_HASH_H
#define _DL_HASH_H	1


/* This is the hashing function specified by the ELF ABI.  It is highly
   optimized for the PII processors.  Though it will run on i586 it
   would be much slower than the generic C implementation.  So don't
   use it.  */
static unsigned int
__attribute__ ((unused))
_dl_elf_hash (const char *name)
{
  unsigned int result;
  unsigned int temp0;
  unsigned int temp1;

  __asm__ __volatile__
    ("movzbl (%1),%2\n\t"
     "testl %2, %2\n\t"
     "jz 1f\n\t"
     "movl %2, %0\n\t"
     "movzbl 1(%1), %2\n\t"
     "jecxz 1f\n\t"
     "shll $4, %0\n\t"
     "addl %2, %0\n\t"
     "movzbl 2(%1), %2\n\t"
     "jecxz 1f\n\t"
     "shll $4, %0\n\t"
     "addl %2, %0\n\t"
     "movzbl 3(%1), %2\n\t"
     "jecxz 1f\n\t"
     "shll $4, %0\n\t"
     "addl %2, %0\n\t"
     "movzbl 4(%1), %2\n\t"
     "jecxz 1f\n\t"
     "shll $4, %0\n\t"
     "addl $5, %1\n\t"
     "addl %2, %0\n\t"
     "movzbl (%1), %2\n\t"
     "jecxz 1f\n"
     "2:\t"
     "shll $4, %0\n\t"
     "movl $0xf0000000, %3\n\t"
     "incl %1\n\t"
     "addl %2, %0\n\t"
     "andl %0, %3\n\t"
     "andl $0x0fffffff, %0\n\t"
     "shrl $24, %3\n\t"
     "movzbl (%1), %2\n\t"
     "xorl %3, %0\n\t"
     "testl %2, %2\n\t"
     "jnz 2b\n"
     "1:\t"
     : "=&r" (result), "=r" (name), "=&c" (temp0), "=&r" (temp1)
     : "0" (0), "1" ((const unsigned char *) name));

  return result;
}

#endif /* dl-hash.h */
