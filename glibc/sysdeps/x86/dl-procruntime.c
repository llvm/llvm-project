/* Data for processor runtime information.  x86 version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

/* This information must be kept in sync with the _DL_HWCAP_COUNT,
   HWCAP_PLATFORMS_START and HWCAP_PLATFORMS_COUNT definitions in
   dl-hwcap.h.

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
  ._dl_x86_feature_1
# else
PROCINFO_CLASS unsigned int _dl_x86_feature_1
# endif
# ifndef PROCINFO_DECL
= 0
# endif
# if !defined SHARED || defined PROCINFO_DECL
;
# else
,
# endif

# if !defined PROCINFO_DECL && defined SHARED
  ._dl_x86_feature_control
# else
PROCINFO_CLASS struct dl_x86_feature_control _dl_x86_feature_control
# endif
# ifndef PROCINFO_DECL
= {
    .ibt = DEFAULT_DL_X86_CET_CONTROL,
    .shstk = DEFAULT_DL_X86_CET_CONTROL,
  }
# endif
# if !defined SHARED || defined PROCINFO_DECL
;
# else
,
# endif
#endif
