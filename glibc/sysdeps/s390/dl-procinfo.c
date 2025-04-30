/* Data for s390 version of processor capability information.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Martin Schwidefsky <schwidefsky@de.ibm.com>, 2006.

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

/* This information must be kept in sync with the _DL_HWCAP_COUNT and
   _DL_PLATFORM_COUNT definitions in procinfo.h.

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

#if !defined PROCINFO_DECL && defined SHARED
  ._dl_s390_cap_flags
#else
PROCINFO_CLASS const char _dl_s390_cap_flags[21][9]
#endif
#ifndef PROCINFO_DECL
= {
     "esan3", "zarch", "stfle", "msa", "ldisp", "eimm", "dfp", "edat", "etf3eh",
     "highgprs", "te", "vx", "vxd", "vxe", "gs", "vxe2", "vxp", "sort", "dflt",
     "vxp2", "nnpa"
  }
#endif
#if !defined SHARED || defined PROCINFO_DECL
;
#else
,
#endif

#if !defined PROCINFO_DECL && defined SHARED
  ._dl_s390_platforms
#else
PROCINFO_CLASS const char _dl_s390_platforms[10][7]
#endif
#ifndef PROCINFO_DECL
= {
    "g5", "z900", "z990", "z9-109", "z10", "z196", "zEC12", "z13", "z14", "z15"
  }
#endif
#if !defined SHARED || defined PROCINFO_DECL
;
#else
,
#endif

#undef PROCINFO_DECL
#undef PROCINFO_CLASS
