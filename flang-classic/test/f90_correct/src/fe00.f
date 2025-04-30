** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Intrinsic names passed as arguments.
*   (all intrinsics which are allowed to be passed are tested once).

	parameter(n = 55)
	implicit real*8 (d)
	integer rslts(n), expect(n)

	intrinsic aint, anint, abs, sqrt, exp, alog, alog10, sin,
     +            cos, tan, asin, acos, atan, sinh, cosh, tanh,
     +            dint, dnint, dabs, dsqrt, dexp, dlog, dlog10,
     +            dsin, dcos, dtan, dasin, dacos, datan, dsinh,
     +            dcosh, dtanh, nint, idnint, dprod, len, index,
     +            aimag
	integer ctoi
	intrinsic conjg, csqrt, cexp, clog, csin, ccos
	intrinsic sign, dsign, dim, ddim, atan2, datan2
	intrinsic cabs, iabs, mod, isign, idim

c --- tests 1 - 16: intrinsics which take 1 REAL arg and return REAL:

	rslts(1) = f1(aint, 2.9)
	rslts(2) = f1(anint, 2.9)
	rslts(3) = f1(abs, -6.0)
	rslts(4) = f1(sqrt, 16.1)
	rslts(5) = f1(exp, 0.0)
	rslts(6) = f1(alog, exp(5.1))
	rslts(7) = f1(alog10, 1000.1)
	rslts(8) = f1(sin, 1.0) * 100
	rslts(9) = f1(cos, 2.0) * 100
	rslts(10) = f1(tan, 3.0) * 100
	rslts(11) = f1(asin, 0.5) * 100
	rslts(12) = f1(acos, -.5) * 100
	rslts(13) = f1(atan, 5.0) * 100
	rslts(14) = f1(sinh, .5) * 100
	rslts(15) = f1(cosh, 1.0) * 100
	rslts(16) = f1(tanh, 2.0) * 100

c --- tests 17 - 32: intrinsics which take 1 d.p. arg and return d.p.:

	rslts(17) = df1(dint, 2.9d0)
	rslts(18) = df1(dnint, 2.9d0)
	rslts(19) = df1(dabs, -6.0d0)
	rslts(20) = df1(dsqrt, 16.1d0)
	rslts(21) = df1(dexp, 0.0d0)
	rslts(22) = df1(dlog, dexp(5.1d0))
	rslts(23) = df1(dlog10, 1000.1d0)
	rslts(24) = df1(dsin, 1.0d0) * 100
	rslts(25) = df1(dcos, 2.0d0) * 100
	rslts(26) = df1(dtan, 3.0d0) * 100
	rslts(27) = df1(dasin, 0.5d0) * 100
	rslts(28) = df1(dacos, -.5d0) * 100
	rslts(29) = df1(datan, 5.0d0) * 100
	rslts(30) = df1(dsinh, .5d0) * 100
	rslts(31) = df1(dcosh, 1d0) * 100
	rslts(32) = df1(dtanh, 2d0) * 100

c  --- tests 33 - 38: miscellaneous intrinsics:

	call sub1(nint, 2.9, idnint, -2.9d0, rslts)

	call sub2(rslts, dprod, len, index, aimag)

c  --- tests 39 - 44: complex intrinsics:

	rslts(39) = ctoi((1, 2), conjg)
	rslts(40) = ctoi((2, -3), csqrt)
	rslts(41) = ctoi((-1, 1), cexp)
	rslts(42) = ctoi((3, 4), clog)
	rslts(43) = ctoi((3, 4), csin)
	rslts(44) = ctoi((3, 4), ccos)

c  --- tests 45 - 50: 2 argument real and d.p. intrinsics:

	call sub3(sign, dsign, rslts(45))
	call sub3(dim, ddim, rslts(47))
	call sub3(atan2, datan2, rslts(49))

c  --- tests 51 - 55:  cabs, iabs, mod, isign, idim.

	call sub4(rslts, cabs, iabs, mod, isign, idim)

c  --- check results:

	call check(rslts, expect, n)

	data expect /2, 3, 6, 4, 1, 5, 3, 84, -41, -14, 52, 209,
     +                              137, 52, 154, 96,
     +               2, 3, 6, 4, 1, 5, 3, 84, -41, -14, 52, 209,
     +                              137, 52, 154, 96,
     +               3, -3, 10000, 3, 3, 10,
     +               980, 1665, 201, 1618, 3583, -27073,
     +               -23, -23, 33, 33, 19, 19,
     +               5, 4, 3, 3, 0                          /
	end

c ccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	real function f1(f, a)
	f1 = f(a)
	end


	function df1(f, d)
	implicit real*8 (d-f)
	external f
	df1 = f(d)
	return
	end


	subroutine sub1(i, x, j, d, a)
	integer a(*), i
	real*8 d

	a(33) = i(x)
	a(34) = j(d)
	end


	subroutine sub2(a, dp, le, in, ai)
	integer a(1:*)
	external dp, ai
	double precision dp
        interface
           integer function le (string)
           character*(*) :: string
           end function le
           integer function in (string, substring)
           character*(*) :: string, substring
           end function in
        end interface

	a(35) = dp(100., 100.)
	a(36) = le('abc')
	a(37) = in('abcd', 'c')
	a(38) = ai( (2.0, 10.1) )

	end


	integer function ctoi(c, f)
	complex c, f, t
	external f
	
	t = f(c)
	ctoi = real(t) * 1000 + aimag(t) * 10
	end


	subroutine sub3(rf, df, a)
	integer a(*)
	real rf
	real*8 df
	external rf, df
	a(1) = rf(2.31, -1.0) * 10
	a(2) = df(2.31d0, -1.0d0) * 10
	end


	subroutine sub4(a, cabs, iabs, mod, isign, idim)
	external cabs, iabs, mod, isign, idim
	real cabs
	integer mod, iabs, isign, idim, a(*)

	a(51) = cabs((3.1, 4.0))
	a(52) = iabs(-4)
	a(53) = mod(7, 4)
	a(54) = isign(3, 0)
	a(55) = idim(3, 4)
	end
