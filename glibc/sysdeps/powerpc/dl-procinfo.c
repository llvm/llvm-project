/* Data for processor capability information.  PowerPC version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

/* This information must be kept in sync with the _DL_HWCAP_COUNT
   definition in procinfo.h.

   If anything should be added here check whether the size of each string
   is still ok with the given array size.

   All the #ifdefs in the definitions are quite irritating but
   necessary if we want to avoid duplicating the information.  There
   are three different modes:

   - PROCINFO_DECL is defined.  This means we are only interested in
     declarations.

   - PROCINFO_DECL is not defined:

     + if SHARED is defined the file is included in an array
       initializer.  The .element = { ... } syntax is needed.

     + if SHARED is not defined a normal array initialization is
       needed.
  */

#ifndef PROCINFO_CLASS
# define PROCINFO_CLASS
#endif

#if !IS_IN (ldconfig)
# if !defined PROCINFO_DECL && defined SHARED
  ._dl_powerpc_cpu_features
# else
PROCINFO_CLASS struct cpu_features _dl_powerpc_cpu_features
# endif
# ifndef PROCINFO_DECL
= { }
# endif
# if !defined SHARED || defined PROCINFO_DECL
;
# else
,
# endif
#endif

#if !defined PROCINFO_DECL && defined SHARED
  ._dl_powerpc_cap_flags
#else
PROCINFO_CLASS const char _dl_powerpc_cap_flags[64][15]
#endif
#ifndef PROCINFO_DECL
= {
    "ppcle", "true_le", "", "",
    "", "", "archpmu", "vsx",
    "arch_2_06", "power6x", "dfp", "pa6t",
    "arch_2_05", "ic_snoop", "smt", "booke",
    "cellbe", "power5+", "power5", "power4",
    "notb", "efpdouble", "efpsingle", "spe",
    "ucache", "4xxmac", "mmu", "fpu",
    "altivec", "ppc601", "ppc64", "ppc32",
    "", "", "", "",
    "", "", "", "",
    "", "", "", "",
    "", "", "", "",
    "", "mma", "arch_3_1", "htm-no-suspend",
    "scv", "darn", "ieee128", "arch_3_00",
    "htm-nosc", "vcrypto", "tar", "isel",
    "ebb", "dscr", "htm", "arch_2_07",
  }
#endif
#if !defined SHARED || defined PROCINFO_DECL
;
#else
,
#endif

#if !IS_IN (ldconfig)
# if !defined PROCINFO_DECL && defined SHARED
     ._dl_cache_line_size
# else
PROCINFO_CLASS int _dl_cache_line_size
# endif
# ifndef PROCINFO_DECL
     = 0
# endif
# if !defined SHARED || defined PROCINFO_DECL
;
# else
,
# endif
#endif


#undef PROCINFO_DECL
#undef PROCINFO_CLASS
