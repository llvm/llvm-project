/* Functions with hidden attribute internal to ld.so, which are declared
   in include/unistd.h.  Generic version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

extern __typeof (__access) __access attribute_hidden;
extern __typeof (__brk) __brk attribute_hidden;
extern __typeof (__close) __close attribute_hidden;
extern __typeof (__getcwd) __getcwd attribute_hidden;
extern __typeof (__getpid) __getpid attribute_hidden;
extern __typeof (__libc_read) __libc_read attribute_hidden;
extern __typeof (__libc_write) __libc_write attribute_hidden;
extern __typeof (__lseek) __lseek attribute_hidden;
extern __typeof (__profil) __profil attribute_hidden;
extern __typeof (__read) __read attribute_hidden;
extern __typeof (__sbrk) __sbrk attribute_hidden;
