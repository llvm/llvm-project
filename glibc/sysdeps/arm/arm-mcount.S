/* Implementation of profiling support.  ARM EABI version.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
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

/* Don't call mcount when calling mcount...  */
#undef PROF

#include <sysdep.h>

#undef mcount

#ifdef __thumb2__
	.thumb
#endif
	.syntax unified


/* Use an assembly stub with a special ABI.  The calling lr has been
   pushed to the stack (which will be misaligned).  We should preserve
   all registers except ip and pop a word off the stack.

   NOTE: This assumes mcount_internal does not clobber any non-core
   (coprocessor) registers.  Currently this is true, but may require
   additional attention in the future.

   The calling sequence looks something like:
func:
   push {lr}
   bl __gnu_mcount_nc
   <function body>
*/

ENTRY(__gnu_mcount_nc)
	push {r0, r1, r2, r3, lr}
	cfi_adjust_cfa_offset (20)
	cfi_rel_offset (r0, 0)
	cfi_rel_offset (r1, 4)
	cfi_rel_offset (r2, 8)
	cfi_rel_offset (r3, 12)
	cfi_rel_offset (lr, 16)
	bic r1, lr, #1
	ldr r0, [sp, #20]
	bl __mcount_internal
	pop {r0, r1, r2, r3, ip, lr}
	cfi_adjust_cfa_offset (-24)
	cfi_restore (r0)
	cfi_restore (r1)
	cfi_restore (r2)
	cfi_restore (r3)
	cfi_register (lr, ip)
	bx ip
END(__gnu_mcount_nc)


#include <gcc-compat.h>
#include <shlib-compat.h>

/* The new __gnu_mcount_nc entry point was introduced in 4.4, so the
   static library needs the old one only to support older compilers.
   Even in a configuration that only cares about newer compilers, the
   shared library might need it only for strict ABI compatibility.  */

#if GCC_COMPAT (4, 3) || SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_19)

/* Provide old mcount for backwards compatibility.  This requires
   code be compiled with APCS frame pointers.  */

ENTRY(__mcount_arm_compat)
	push	{r0, r1, r2, r3, fp, lr}
	cfi_adjust_cfa_offset (24)
	cfi_rel_offset (r0, 0)
	cfi_rel_offset (r1, 4)
	cfi_rel_offset (r2, 8)
	cfi_rel_offset (r3, 12)
	cfi_rel_offset (fp, 16)
	cfi_rel_offset (lr, 20)
	movs r0, fp
	ittt ne
	ldrne r0, [r0, #-4]
	movsne r1, lr
	blne __mcount_internal
# if defined (__ARM_ARCH_4T__) && defined (__THUMB_INTERWORK__)
	pop	{r0, r1, r2, r3, fp, lr}
	cfi_adjust_cfa_offset (-24)
	cfi_restore (r0)
	cfi_restore (r1)
	cfi_restore (r2)
	cfi_restore (r3)
	cfi_restore (fp)
	cfi_restore (lr)
	bx lr
# else
	pop	{r0, r1, r2, r3, fp, pc}
# endif
END(__mcount_arm_compat)

#endif

#if GCC_COMPAT (4, 3)

strong_alias (__mcount_arm_compat, _mcount)

/* The canonical name for the function is `_mcount' in both C and asm,
   but some old asm code might assume it's `mcount'.  */
weak_alias (_mcount, mcount)

#elif SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_19)

compat_symbol (libc, __mcount_arm_compat, _mcount, GLIBC_2_0)

strong_alias (__mcount_arm_compat, __mcount_arm_compat_1)
compat_symbol (libc, __mcount_arm_compat_1, mcount, GLIBC_2_0)

#endif
