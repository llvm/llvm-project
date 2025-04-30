** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   VMS old-style PARAMETER statements.

C   - parens must be omitted,
C   - type of the parameter must not be declared,
C   - constant determines the type, not the identifier.

	program bp20
	implicit character*26 (a-i)

	parameter i3 = 3, x25 = 2.5
	parameter in25 = -2.5, x6 = 6
	parameter true = .true.
	parameter f = .not. true
	parameter a = 'b', b = 'ABCDEFGHIJKLMNOP' //
     +			'QRSTUVWXYZ'
	parameter j = 1D-9   ! watch exponent wrt Newton's method
	parameter x = 'FFFFFFFF'x

	parameter XX = 10
	integer rslts(XX), expect(XX)

	rslts(1) = i3			! 3
	rslts(2) = 2.01 * x25		! 5
	rslts(3) = 2 * in25		! -5
	rslts(4) = (x6/4)*4		! 4

	if (true) rslts(5) = 2		! 2
	if (.not. f) rslts(6) = 3	! 3

	rslts(7) = ICHAR(a)		! 98
	c = b			! (c is implicitly declared)
	rslts(8) = ichar(c(25:25)) - ichar('Z')		! -1

	rslts(9) = nint(j * (1 / j ))	! 1.0  -->  1
	if (x) rslts(10) = x		! -1

	call check(rslts, expect, 10)
	data expect / 3, 5, -5, 4, 2, 3, 98, -1, 1, -1 /
	end
