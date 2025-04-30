** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Non-deletable induction variables.
*   Optimizer tests of induction variables that should not
*   be deleted from a loop

	subroutine prin(p, result)
	integer p, result
	result = p
	end

	program ka36
	parameter(N=2)
	integer result(N), expect(N)
	integer i, a(0:9), j

	do j = 0, 2, 1         ! this j has an explict use later on
	    a(j) = 0
	enddo

	do i = 0, 1, 1         ! this i has an implicit use due to addrtkn
	    a(i) = 0
	enddo
	call prin(i, result)
	result(2) = j
        call check (result, expect, N)
	data expect /2, 3/
	end
