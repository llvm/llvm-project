/* Resolve function pointers to VDSO functions.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */


#ifndef _LIBC_POWERPC_VDSO_H
#define _LIBC_POWERPC_VDSO_H

#include <sysdep.h>
#include <sysdep-vdso.h>

#if (defined(__PPC64__) || defined(__powerpc64__)) && _CALL_ELF != 2
# include <dl-machine.h>
/* The correct solution is for _dl_vdso_vsym to return the address of the OPD
   for the kernel VDSO function.  That address would then be stored in the
   __vdso_* variables and returned as the result of the IFUNC resolver function.
   Yet, the kernel does not contain any OPD entries for the VDSO functions
   (incomplete implementation).  However, PLT relocations for IFUNCs still expect
   the address of an OPD to be returned from the IFUNC resolver function (since
   PLT entries on PPC64 are just copies of OPDs).  The solution for now is to
   create an artificial static OPD for each VDSO function returned by a resolver
   function.  The TOC value is set to a non-zero value to avoid triggering lazy
   symbol resolution via .glink0/.plt0 for a zero TOC (requires thread-safe PLT
   sequences) when the dynamic linker isn't prepared for it e.g. RTLD_NOW.  None
   of the kernel VDSO routines use the TOC or AUX values so any non-zero value
   will work.  Note that function pointer comparisons will not use this artificial
   static OPD since those are resolved via ADDR64 relocations and will point at
   the non-IFUNC default OPD for the symbol.  Lastly, because the IFUNC relocations
   are processed immediately at startup the resolver functions and this code need
   not be thread-safe, but if the caller writes to a PLT slot it must do so in a
   thread-safe manner with all the required barriers.  */
# define VDSO_IFUNC_RET(value)                           \
  ({                                                     \
    static Elf64_FuncDesc vdso_opd = { .fd_toc = ~0x0 }; \
    vdso_opd.fd_func = (Elf64_Addr)value;                \
    &vdso_opd;                                           \
  })

#else
# define VDSO_IFUNC_RET(value)  ((void *) (value))
#endif

#endif /* _LIBC_VDSO_H */
