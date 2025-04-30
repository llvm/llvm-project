** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Exponentiation (** operator) to double complex powers.

	program p
	implicit double complex (c)

	parameter (N=6)
	integer rslts(N), expect(N)

	integer ctoi
	ctoi(c) = real(c) + dimag(c)

	data x2 / 2.0/
	data c2, c23, c11, c12/ (2, 0), (2, 3), (1, 1), (1, 2) /

	rslts(1) = ctoi( (2.1d0, 0.0) ** (2.0d0, 0.0) )
	rslts(2) = ctoi( (2.1, 0.0d0) ** c2)
	rslts(3) = ctoi( c23 ** x2 + .001 )
	rslts(4) = c23 ** c11 + 10
	rslts(5) = ctoi( 10 ** c12 )
	rslts(6) = ctoi(x2 ** (-1.0d0, 2.0) * 10)

c --- check results:

	call check(rslts, expect, N)

	data expect /
     +                4, 4, 7, 9, -11, 5      /

	end
