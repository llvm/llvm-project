/* Real definitions for extern inline functions in argp-fmtstream.h
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Miles Bader <miles@gnu.ai.mit.edu>.

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

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#define ARGP_FS_EI
#undef __OPTIMIZE__
#define __OPTIMIZE__ 1
#include <argp-fmtstream.h>

#if 0
/* Not exported.  */
/* Add weak aliases.  */
#if _LIBC - 0 && !defined (ARGP_FMTSTREAM_USE_LINEWRAP) && defined (weak_alias)

weak_alias (__argp_fmtstream_putc, argp_fmtstream_putc)
weak_alias (__argp_fmtstream_puts, argp_fmtstream_puts)
weak_alias (__argp_fmtstream_write, argp_fmtstream_write)
weak_alias (__argp_fmtstream_set_lmargin, argp_fmtstream_set_lmargin)
weak_alias (__argp_fmtstream_set_rmargin, argp_fmtstream_set_rmargin)
weak_alias (__argp_fmtstream_set_wmargin, argp_fmtstream_set_wmargin)
weak_alias (__argp_fmtstream_point, argp_fmtstream_point)

#endif
#endif
