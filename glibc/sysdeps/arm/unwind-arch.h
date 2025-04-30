/* Dynamic loading of the libgcc unwinder.  Arm customization.
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

/* On arm, _Unwind_GetIP is a macro.  */
#define UNWIND_LINK_GETIP 0

#define UNWIND_LINK_FRAME_STATE_FOR 0
#define UNWIND_LINK_FRAME_ADJUSTMENT 0
#define UNWIND_LINK_EXTRA_FIELDS \
  __typeof (_Unwind_VRS_Get) *ptr__Unwind_VRS_Get;
#define UNWIND_LINK_EXTRA_INIT                                \
  local.ptr__Unwind_VRS_Get                                   \
    = __libc_dlsym (local_libgcc_handle, "_Unwind_VRS_Get");  \
  assert (local.ptr__Unwind_VRS_Get != NULL);                 \
  PTR_MANGLE (local.ptr__Unwind_VRS_Get);

/* This is used by the _Unwind_Resume assembler implementation to
   obtain the address to jump to.  */
void *__unwind_link_get_resume (void) attribute_hidden;

#endif /* _ARCH_UNWIND_LINK_H */
