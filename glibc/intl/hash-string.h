/* Description of GNU message catalog format: string hashing function.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 2.1 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* @@ end of prolog @@ */

/* We assume to have `unsigned long int' value with at least 32 bits.  */
#define HASHWORDBITS 32


#ifndef _LIBC
# ifdef IN_LIBINTL
#  define __hash_string libintl_hash_string
# else
#  define __hash_string hash_string
# endif
#endif

/* Defines the so called `hashpjw' function by P.J. Weinberger
   [see Aho/Sethi/Ullman, COMPILERS: Principles, Techniques and Tools,
   1986, 1987 Bell Telephone Laboratories, Inc.]  */
extern unsigned long int __hash_string (const char *str_param)
     attribute_hidden;
