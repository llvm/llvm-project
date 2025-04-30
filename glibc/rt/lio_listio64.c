/* Enqueue and list of read or write requests.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <bits/wordsize.h>
#if __WORDSIZE != 64
# define AIOCB aiocb64
# define LIO_LISTIO lio_listio64
# define LIO_LISTIO_OLD __lio_listio64_21
# define LIO_LISTIO_NEW __lio_listio64_24
# define LIO_OPCODE_BASE 128

# include <rt/lio_listio-common.c>
#endif
