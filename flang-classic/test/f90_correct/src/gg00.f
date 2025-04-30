** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Exponentiation operator (**) - exponentiation of integer,
*   real, d.p., or complex values to an integer power, 
*   including constant folding.

	implicit double precision (d), complex(c)
	parameter(n = 34)
	integer rslts(n), expect(n), iexpect(12)
	real rrslts(13:18), rexpect(13:18)
	double precision drslts(4), dexpect(4)
	complex crslts(4), cexpect(4)

	equivalence (expect, iexpect), (expect(13), rexpect),
     +              (expect(19), dexpect), (expect(27), cexpect)
	equivalence (rslts(13), rrslts(13)), (rslts(19), drslts),
     +              (rslts(27), crslts)

	parameter(ip1 = 138 ** 1, ip2 = -4 ** 2, ip3 = 5**0,
     +            ip4 = -5**3, ip5 = 1 ** 20, ip6 = 1**(-3),
     +            ip7 = 2 ** (-1), ip8=5**(-2), ip9=2**3**2  )

	parameter(xp1 = 2.0 ** 2, xp2 = 4.0 ** (-1), xp3 = -3.0 ** 3,
     +            xp4 = 2.3 ** 1, xp5 = 2.0 ** (-2) )

	parameter(dp1 = 2.0D0 ** 3, dp2 = 2.3d45**0, dp3=(-2.0d0)**(-2))

	parameter(cp1 = (2.0, 0.0) ** 3, cp2 = (2.3, 2.3) ** 1,
     +            cp3 = (2.3, 2.3) ** 0, cp4 = (1.0, 1.0) ** 2  )

c ----------- tests 1 - 12:    integer ** integer

	data iexpect / ip1, ip2, ip2, ip3, ip4, ip5,
     +                 ip6, ip7, ip8, ip9, ip9, 125  /

	data i138, i4, i2, i3, i5, i20, in3, i1, in2, in1 /
     +        138,  4,  2,  3,  5,  20,  -3,  1,  -2, -1  /

	rslts(1) = i138 ** i1
	rslts(2) = - (i4 ** i2)
	rslts(3) = - i4 ** i2
	rslts(4) = i5 ** (i2 / i3)
	rslts(5) = - i5 ** (i2 + i1)
	rslts(6) = i1 ** i20
	rslts(7) = i1 ** (in3)
	rslts(8) = (i1 * i2) ** (-i1)
	rslts(9) = 5 ** (in2)
	rslts(10) = (-in2) ** if(3) ** 2
	rslts(11) = i2 ** (i3 ** i2)
	rslts(12) = if(5) ** 3

c ------------ tests 13 - 18:   real ** integer

	data rexpect / xp1, xp2, xp3, xp4, xp5, 9.0 /

	data x3, x23, x2 / 3.0, 2.3, 2.0 /

	rrslts(13) = 4.0
	rrslts(14) = 4.0 ** in1
	rrslts(15) = -x3**3
	rrslts(16) = x23 ** (i2 - 1)
	rrslts(17) = x2 ** (- i2)
	rrslts(18) = xf( -x3 ) ** 2

c ------------- tests 19 - 26:   double ** integer

	data dexpect / dp1, dp2, dp3, 4.0D0  /

	data d2 / 2.0d0 /

	drslts(1) = d2 ** i3
	drslts(2) = 2.3D45 ** (i2 - i1 - i1)
	drslts(3) = (-d2) ** (-i2)
	drslts(4) = d2 ** 2

c -------------- tests 27 - 34:  complex ** integer

	data cexpect / cp1, cp2, cp3, cp4 /

	data         c2,        c23,       cn1_1  /
     +       (2.0, 0.0), (2.3, 2.3), (-1.0, 1.0)  /

        crslts(1) = c2 ** i3
        crslts(2) = (2.3, 2.3) ** (i2 - i1)
        crslts(3) = c23 ** 0
        crslts(4) = (c2 + cn1_1) ** 2

c --------------- check results:

	call check(rslts, expect, n)
	end


	integer function if(i)
	common /comif/ ii
	data ii /0/
	if (ii .gt. 1)  stop "'if' called too often"
        ii = ii + 1
	if =  i
	end

	real function xf(x)
	common /comxf/ ii
	data ii /0/
	if (ii .gt. 1)  stop "'xf' called too often"
        ii = ii + 1
	xf = x
	end
