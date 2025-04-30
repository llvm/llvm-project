/* Machine-specific calling sequence for `mcount' profiling function.  ia64
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
     Contributed by David Mosberger <davidm@hpl.hp.com>
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

/* Assembly stub to invoke _mcount().  Compiler generated code calls
   this stub before executing a function's prologue and without saving
   any registers.  It is therefore necessary to preserve the input
   registers as they may contain function arguments.  To work
   correctly with frame-less functions, it is also necessary to
   preserve the return pointer (b0 aka rp).

   State upon entering _mcount:

    r8  address of return value structure (used only when called
	function returns a large structure)
    r15 static link (used only for nested functions)
    in0 ar.pfs to restore before returning to the function that
	called _mcount
    in1 gp value to restore before returning to the function that
	called _mcount
    in2 return address in the function that invoked the caller
	of _mcount (frompc)
    in3	address of the global-offset table entry that holds the
	profile count dword allocated by the compiler; to get
	the address of this dword, use "ld8 in2=[in2]; this
	dword can be used in any way by _mcount (including
	not at all, as is the case with the current implementation)
    b0	address to return to after _mcount is done
*/

#include <sysdep.h>

#undef ret

LEAF(_mcount)
	.prologue ASM_UNW_PRLG_RP|ASM_UNW_PRLG_PFS, ASM_UNW_PRLG_GRSAVE(4)
	alloc loc1 = ar.pfs, 4, 4, 3, 0
	mov loc0 = rp
	.body
	mov loc2 = r8	// gcc uses r8 to pass pointer to return structure
	;;
	mov loc3 = r15	// gcc uses r15 to pass the static link to nested functions
	mov out0 = in2
	mov out1 = rp
	br.call.sptk.few rp = __mcount
	;;
.here:
{
	.mii
	mov gp = in1
	mov r2 = ip
	mov ar.pfs = loc1
}
	;;
	adds r2 = _mcount_ret_helper - .here, r2
	mov b7 = loc0
	mov rp = in2
	;;
	mov r3 = in0
	mov r8 = loc2
	mov r15 = loc3
	mov b6 = r2
	br.ret.sptk.few b6
END(_mcount)

LOCAL_LEAF(_mcount_ret_helper)
	.prologue
	.altrp b7
	.save ar.pfs, r3
	.body
	alloc r2 = ar.pfs, 0, 0, 8, 0
	mov ar.pfs = r3
	br b7
END(_mcount_ret_helper)

weak_alias (_mcount, mcount)
