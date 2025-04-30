/* Direct access for nss_files functions for NSS module loading.
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

#include <nss_module.h>
#include <nss_files.h>

void
__nss_files_functions (nss_module_functions_untyped pointers)
{
  void **fptr = pointers;

  /* Functions which are not implemented.  */
#define _nss_files_getcanonname_r NULL
#define _nss_files_gethostbyaddr2_r NULL
#define _nss_files_getpublickey NULL
#define _nss_files_getsecretkey NULL
#define _nss_files_netname2user NULL

#undef DEFINE_NSS_FUNCTION
#define DEFINE_NSS_FUNCTION(x) *fptr++ = _nss_files_##x;
#include "function.def"
}
