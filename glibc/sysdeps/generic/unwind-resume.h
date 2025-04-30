/* Definitions for unwind-resume.c.  Generic version.
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

/* These describe the arguments to unwinder personality functions,
   specifically __gcc_personality_v0.  A machine-specific sysdeps
   file might define them differently.  */
#define PERSONALITY_PROTO                       \
  (int version, _Unwind_Action actions,         \
   _Unwind_Exception_Class exception_class,     \
   struct _Unwind_Exception *ue_header,         \
   struct _Unwind_Context *context)
#define PERSONALITY_ARGS                                        \
  (version, actions, exception_class, ue_header, context)

/* This is defined nonzero by a machine-specific sysdeps file if
   _Unwind_Resume is provided separately and thus the generic C
   version should not be defined.  */
#define HAVE_ARCH_UNWIND_RESUME		0
