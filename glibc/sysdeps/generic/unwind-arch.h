/* Dynamic loading of the libgcc unwinder.  Generic version of parameters.
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

#ifndef _ARCH_UNWIND_LINK_H
#define _ARCH_UNWIND_LINK_H

/* The _Unwind_GetIP function is supported.  */
#define UNWIND_LINK_GETIP 1

/* The __frame_state_for function is needed and re-exported from glibc.  */
#define UNWIND_LINK_FRAME_STATE_FOR 0

/* No adjustment of the is needed.  */
#define UNWIND_LINK_FRAME_ADJUSTMENT 0

/* There are no extra fields in struct unwind_link in the generic version.  */
#define UNWIND_LINK_EXTRA_FIELDS
#define UNWIND_LINK_EXTRA_INIT

#endif /* _ARCH_UNWIND_LINK_H */
