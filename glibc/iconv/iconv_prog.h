/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.

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

#ifndef _ICONV_PROG_H
#define _ICONV_PROG_H	1

#include <stdio.h>

#include <charmap.h>
#include <record-status.h>

/* If nonzero omit invalid character from output.  */
extern int omit_invalid;

/* Perform the conversion using a charmap or two.  */
extern int charmap_conversion (const char *from_code,
			       struct charmap_t *from_charmap,
			       const char *to_code,
			       struct charmap_t *to_charmap,
			       int argc, int remaining, char *argv[],
			       const char *output_file);


#endif	/* iconv_prog.h */
