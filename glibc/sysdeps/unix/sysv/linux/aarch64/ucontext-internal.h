/* Copyright (C) 2009-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#define SP_ALIGN_SIZE       15

#define SP_ALIGN_MASK	   ~15

/* Size of an X regiser in bytes. */
#define SZREG                8

/* Size of a V register in bytes. */
#define SZVREG              16

/* Number of integer parameter passing registers. */
#define NUMXREGARGS          8

/* Number of FP parameter passing registers. */
#define NUMDREGARGS          8

/* Size of named integer argument in bytes when passed on the
   stack.  */
#define SIZEOF_NAMED_INT     4

/* Size of an anonymous integer argument in bytes when passed on the
   stack.  */
#define SIZEOF_ANONYMOUS_INT 8

#define oX21 (oX0 + 21*8)
#define oFP  (oX0 + 29*8)
#define oLR  (oX0 + 30*8)
