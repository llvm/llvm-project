/* Bug 20419: test for stack overflow in elf/dl-load.c open_verify()
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

/* This creates a .so with 8MiB PT_NOTE segment.
   On a typical Linux system with 8MiB "ulimit -s", that was enough
   to trigger stack overflow in open_verify.  */

#define NOTE_SIZE 8*1024*1024

.pushsection .note.big,"a"
	.balign 4
	.long 5 		/* n_namesz.  Length of "GLIBC".  */
	.long NOTE_SIZE		/* n_descsz.  */
	.long 0			/* n_type.  */
	.ascii "GLIBC\0\0\0"	/* Name and alignment to four bytes.  */
	.fill NOTE_SIZE, 1, 0
.popsection
