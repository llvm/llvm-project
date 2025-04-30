/* Define INIT_ARCH so that midr is initialized before use by IFUNCs.
   This file is part of the GNU C Library.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

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

#include <ldsodefs.h>
#include <sys/auxv.h>

/* Make glibc MTE-safe on a system that supports MTE in case user code
   enables tag checks independently of the mte_status of glibc.  There
   is currently no ABI contract for enabling tag checks in user code,
   but this can be useful for debugging with MTE.  */
#define MTE_ENABLED() (GLRO(dl_hwcap2) & HWCAP2_MTE)

#define INIT_ARCH()							      \
  uint64_t __attribute__((unused)) midr =				      \
    GLRO(dl_aarch64_cpu_features).midr_el1;				      \
  unsigned __attribute__((unused)) zva_size =				      \
    GLRO(dl_aarch64_cpu_features).zva_size;				      \
  bool __attribute__((unused)) bti =					      \
    HAVE_AARCH64_BTI && GLRO(dl_aarch64_cpu_features).bti;		      \
  bool __attribute__((unused)) mte =					      \
    MTE_ENABLED ();							      \
  bool __attribute__((unused)) sve =					      \
    GLRO(dl_aarch64_cpu_features).sve;
