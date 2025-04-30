** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   EXTERNAL statements.

c  Items tested include:
c  (1) blockdata name in EXTERNAL statement.
c  (2) redefinition of an intrinsic using EXTERNAL statement.
c  (3) name of function which is to be passed as a dummy
c      argument in EXTERNAL stmt.
c  (4) name of dummy function in EXTERNAL statement.
c  (5) use of a dummy function as a subroutine (CALLed) and
c      as a function.

	EXTERNAL blockdat, ifunc
	integer rslts(4), expect(4)
	common rslts, expect
	external iabs
	external sub3

	rslts(1) = iabs(3)
	call sub(ifunc, ifunc, sub3)

	call check(rslts, expect, 4)
	end
c-----------------------------------------c
	blockdata blockdat
	integer rslts(4), expect(4)
	common rslts, expect

	data expect / 4, 19, 99, -1 /
	end
c-----------------------------------------c
	integer function iabs(j)
	iabs = j + 1
	return
	end
c-----------------------------------------c
	integer function ifunc(i)
	ifunc = i - 1
	return
	end
c-----------------------------------------c
	subroutine sub(if, jf, kf)
	integer rslts(4), expect(4)
	common rslts, expect
	external if, jf, kf

	rslts(2) = if(20)
	call sub2(jf)
	call kf(-1)
	end
c-----------------------------------------c
	subroutine sub2(jf)
c  -- external stmt should not be required to call jf:
	integer rslts(4), expect(4)
	common rslts, expect

	rslts(3) = jf(100)
	return
	end
c-----------------------------------------c
	subroutine sub3(jf)
	integer rslts(4), expect(4)
	common rslts, expect

	rslts(4) = jf
	return
	end
