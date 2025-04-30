/* Data for Linux/sparc version of processor capability information.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2002.

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

/* This information must be kept in sync with the _DL_HWCAP_COUNT
   definition in procinfo.h.

   If anything should be added here check whether the size of each string
   is still ok with the given array size.

   All the #ifdefs in the definitions ar equite irritating but
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
#define PROCINFO_CLASS
#endif

#if !defined PROCINFO_DECL && defined SHARED
  ._dl_sparc_cap_flags
#else
PROCINFO_CLASS const char _dl_sparc_cap_flags[28][11]
#endif
#ifndef PROCINFO_DECL
  = { "flush", "stbar", "swap", "muldiv", "v9", "ultra3", "v9v", "v9v2",
      "mul32", "div32", "fsmuld", "v8plus", "popc", "vis", "vis2",
      "ASIBlkInit", "fmaf", "vis3", "hpc", "random", "trans", "fjfmau",
      "ima", "cspare", "pause", "cbcond", "crypto", "adp" }
#endif
#if !defined SHARED || defined PROCINFO_DECL
;
#else
,
#endif

#undef PROCINFO_DECL
#undef PROCINFO_CLASS
