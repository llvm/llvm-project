/* Data for i386 version of processor capability information.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.

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

/* If anything should be added here check whether the size of each string
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

#include <sysdeps/x86/dl-procinfo.c>

#if !defined PROCINFO_DECL && defined SHARED
  ._dl_x86_cap_flags
#else
PROCINFO_CLASS const char _dl_x86_cap_flags[32][8]
#endif
#ifndef PROCINFO_DECL
= {
    "fpu", "vme", "de", "pse", "tsc", "msr", "pae", "mce",
    "cx8", "apic", "10", "sep", "mtrr", "pge", "mca", "cmov",
    "pat", "pse36", "pn", "clflush", "20", "dts", "acpi", "mmx",
    "fxsr", "sse", "sse2", "ss", "ht", "tm", "ia64", "pbe"
  }
#endif
#if !defined SHARED || defined PROCINFO_DECL
;
#else
,
#endif

#undef PROCINFO_DECL
#undef PROCINFO_CLASS
