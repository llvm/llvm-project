/* _Float128 aliasing macro support for ifunc generation on PPC.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef _FLOAT128_IFUNC_REDIRECT_MACROS_PPC64LE
#define _FLOAT128_IFUNC_REDIRECT_MACROS_PPC64LE 1

/* Define the redirection macros used throughout most of the IFUNC headers.
   The variant is inferred via compiler options.

   F128_REDIR_PFX_R(function, destination_prefix, reentrant_suffix)
     Redirect function, optionally suffixed by reentrant_suffix, to a function
     named destination_prefix ## function ## variant ## reentrant_suffix.

   F128_SFX_APPEND(sym)
     Append the the multiarch variant specific suffix to the sym. sym is not
     expanded.  This is sym ## variant.

   F128_REDIR_R(func, reentrant_suffix)
     Redirect func to a function named function ## variant ## reentrant_suffix

   F128_REDIR(function)
     Convience wrapper for F128_REDIR_R where function does not require
     a suffix argument.

*/
#ifndef _ARCH_PWR9
#define F128_REDIR_PFX_R(func, pfx, r) \
  extern __typeof(func ## r) func ## r __asm( #pfx #func "_power8" #r );
#define F128_SFX_APPEND(x) x ## _power8
#else
#define F128_REDIR_PFX_R(func, pfx, r) \
  extern __typeof(func ## r) func ## r __asm( #pfx #func "_power9" #r );
#define F128_SFX_APPEND(x) x ## _power9
#endif
#define F128_REDIR_R(func, r) F128_REDIR_PFX_R (func, , r)
#define F128_REDIR(func) F128_REDIR_R (func, )

#endif /*_FLOAT128_IFUNC_REDIRECT_MACROS_PPC64LE */
