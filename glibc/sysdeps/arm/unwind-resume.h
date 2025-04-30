/* Definitions for unwind-resume.c.  ARM (EABI) version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

/* The EABI personality routine has a different signature than the
   canonical one.  These macros tell sysdeps/gnu/unwind*.c how to
   define __gcc_personality_v0.  */
#define PERSONALITY_PROTO                       \
  (_Unwind_State state,                         \
   struct _Unwind_Exception *ue_header,         \
   struct _Unwind_Context *context)
#define PERSONALITY_ARGS                        \
  (state, ue_header, context)

/* It's vitally important that _Unwind_Resume not have a stack frame; the
   ARM unwinder relies on register state at entrance.  So we write this in
   assembly (see arm-unwind-resume.S).  This macro tells the generic code
   not to provide the generic C definition.  */
#define HAVE_ARCH_UNWIND_RESUME                 1
