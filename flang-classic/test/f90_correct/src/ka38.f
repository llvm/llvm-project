** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*--- Induction variables and countable loops.
*    Tests of countable loops whose induction variables are potentially
*    reached by defs which store constants.

	program p
	parameter (N=7)
	integer i, j, result(N), expect(N)
	common i, j, result, expect
	integer safe_local, unsafe_local

	data j /1/
	data expect / 10, 10, 10, 10, 10, 10, 10 /

*    /* test 1 - only one def reaches do, but does not dominate do */
	if (j .ne. 0) then
	    i = 5
	    call seti()
	else
	    i = 6
	endif
	do while (i .le. 10)
	    result(1) = result(1) + 1
	    i = i + 1
	enddo

*    /* test 2 - a single def, but is killed by call */
	i = 10
	call seti()
	do while (i .le. 10)
	    result(2) = result(2) + 1
	    i = i + 1
	enddo

*    /* test 3 - is okay to use const as initial value */
	do i = 1, 10 , 1
	    result(3) = result(3) + 1
	enddo

*    /* test 4 - single def, but does not dominate do */
	call seti()
	if (j .eq. 0) then
	    i = 10
	endif
	do while (i .le. 10)
	    result(4) = result(4) + 1
	    i = i + 1
	enddo

*    /* test 5 - safe induction variable but multiple defs reaching do */
	if (j .ne. 0) then
	    safe_local = 1
	else
	    safe_local = 10
	endif
	do while (safe_local .le. 10)
	    result(5) = result(5) + 1
	    safe_local = safe_local + 1
	enddo

*    /* test 6 - single safe def unaffected by call */
	safe_local = 1
	call seti()
	do while(safe_local .le. 10)
	    result(6) = result(6) + 1
	    safe_local = safe_local + 1
	enddo

*    /* test 7 - unsafe local since its address is taken */
	unsafe_local = 10
	call func(unsafe_local)
	do while (unsafe_local .le. 10)
	    unsafe_local = unsafe_local + 1
	    result(7) = result(7) + 1
	enddo

	call check(result, expect, N)
	end

	subroutine seti
	common i
	i = 1
	end

	subroutine func(p)
	integer p
	p = 1
	end
