** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Subprograms with many arguments

	parameter(n = 4)
	integer result(n), expect(n)
	data expect/11,12,13,14/

	call t0(result, 1,2,3,4,5,6,7,8,9,10,11,12,13,14)
	call check (result, expect, n)
	end
	subroutine t0(a,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14)
	integer a(*)
	call t1		! to force memory arg ptr to memory
	a(1) = i11
	a(2) = i12
	a(3) = i13
	a(4) = i14
	end
	subroutine t1
	end
