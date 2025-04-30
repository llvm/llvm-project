/* Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <memcopy.h>

extern __typeof (_wordcopy_fwd_aligned) _wordcopy_fwd_aligned_ppc;
extern __typeof (_wordcopy_fwd_dest_aligned) _wordcopy_fwd_dest_aligned_ppc;
extern __typeof (_wordcopy_bwd_aligned) _wordcopy_bwd_aligned_ppc;
extern __typeof (_wordcopy_bwd_dest_aligned) _wordcopy_bwd_dest_aligned_ppc;

#define _wordcopy_fwd_aligned       _wordcopy_fwd_aligned_ppc
#define _wordcopy_fwd_dest_aligned  _wordcopy_fwd_dest_aligned_ppc
#define _wordcopy_bwd_aligned       _wordcopy_bwd_aligned_ppc
#define _wordcopy_bwd_dest_aligned  _wordcopy_bwd_dest_aligned_ppc

extern __typeof (memmove) __memmove_ppc attribute_hidden;
#define MEMMOVE __memmove_ppc

extern __typeof (memcpy) __memcpy_ppc attribute_hidden;
#ifdef SHARED
# define memcpy __memcpy_ppc
#endif

#if IS_IN (libc) && defined(SHARED)
# undef libc_hidden_builtin_def
# define libc_hidden_builtin_def(name)
#endif

#include <string/memmove.c>
