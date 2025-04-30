/* Copyright (C) 2009-2021 Free Software Foundation, Inc.

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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _LIBC_VDSO_H
#define _LIBC_VDSO_H

/* Adjust the return IFUNC value from a vDSO symbol accordingly required
   by the ELFv1 ABI.  It is used by the architecture to create an ODP
   entry since the kernel vDSO does not provide it.  */
#define VDSO_IFUNC_RET(__value) (__value)

#endif
