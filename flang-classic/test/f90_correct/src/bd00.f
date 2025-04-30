** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   INTRINSIC statements.

C  Tests:
C   (1) generic in INTRINSIC stmt doesn't remove generic property.
C   (2) intrinsic in INTRINSIC stmt passed as argument.

	intrinsic iabs
	integer rslts(3), expect(3)
	intrinsic sngl, dprod, or, dfloat
	intrinsic int

        data expect / 2, 7, 7 /

	rslts(1) = int(2.9D0)

	call sub(iabs, ires)
	rslts(2) = ires

	call sub2(dprod, ires)
	rslts(3) = ires
	
	call check(rslts, expect, 3)
	end
CCCCC
	subroutine sub(ifunc, iout)
	iout = ifunc(-7)
	end
CCCCC
	subroutine sub2(dfunc, iout)
	double precision dfunc
	iout = dfunc(2.35E0, 3.0E0)
	end
