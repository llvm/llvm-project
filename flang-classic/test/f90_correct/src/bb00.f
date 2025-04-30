** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   IMPLICIT statements.

C  Items tested include:
c   (1)  Both ranges and individual letters in IMPLICIT stmt.
c   (2)  PARAMETER statement which precedes IMPLICIT statements.
c        (IMPLICIT stmt which follows must not change type
c         of parameter).
c   (3)  Single and multiple types specified in IMPLICIT stmt.
c   (4)  Type declaration which overrides IMPLICIT type and 
c        character length.
c   (5)  Definition of character lengths using IMPLICIT.
c   (6)  Constant expressions and parameter references used to
c        define character length.
c   (7)  Upper case letters used for character range.
c   (8)  Length specification on REAL.
c   (9)  IMPLICIT specification for special characters (_, $).
c   (10) IMPLICIT statement which causes type of enclosing
c        function and its dummy arguments to be changed.

C    NONSTANDARD:
C      Identifiers beginning with _ and $. (non-VMS)

	implicit integer(a), real(i, b), logical(c-d, j),
     +           integer(k)
	parameter(i = 2.6, aa = 5, n = 15)

	implicit character (e)
	implicit character*2 (f), character*(aa) (g),
     +           character*(3 + 7 - 1)(H, l-M)
	implicit real*8 (n), complex(_, $), integer(x, r)

	integer expect(n)
	dimension rslts(n)
	character*300 fcharvar

	rslts(1) = aa
	rslts(2) = i + i

	data c, ci, d / .true., .false., .true. /
	rslts(3) = OR(0, c .and..not.ci.and.d)
	k = 5
	rslts(4) = k/2*2

	rslts(5) = len(e)
	rslts(6) = len(enough)
	rslts(7) = len(ff)
	rslts(8) = len(g)
	rslts(9) = len(hvar_)
	rslts(10) = len(m) + len(fcharvar)

	data nn /12345678909.1D0 /
	rslts(11) = nn - 12345678905D0

	_x = (1.0, 2.0)
	$ = (2.0, 3.0)
	rslts(12) = aimag(_x + $)

	rslts(13) = xfunc(10, 2.6, 300)
	xvar = -7
	rslts(14) = iabs(xvar)
	rslts(15) = afunc(2, 20)

	call check(rslts, expect, n)

	data expect / 5, 5, -1, 4,
     +                1, 1, 2, 5, 9, 309,
     +                4, 5, 315, 7, 22    /

	end

	function xfunc(y, z, x)
c  the following implicit stmt should set the type of xfunc, x,
c  and y to INTEGER.
	implicit integer(x-y)
	j = z + z
	xfunc = y + x + j
	end

	integer function afunc(i, j)
c  the following IMPLICIT stmt should not have any net effect:
	implicit real(a-h, j-z)
	integer j
	afunc = i + j
	return
	end
