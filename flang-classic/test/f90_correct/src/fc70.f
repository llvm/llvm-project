** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


*   ABS,MIN,MAX constant folding(mostly) in the expander.

	parameter(n = 83)
	integer rslts(n), expect(n)
	real  rrslts(n), rexpect(n)
	equivalence (rslts, rrslts), (expect, rexpect)

        integer  i, j, k, l
        real     r, s, t, u
        double precision  d, e, f, g
        
	data (expect(i), i = 1, 5) /2, 34, 0, 34, 2 /
        i = ABS( 2)
        rslts(1) = i
        j = ABS( -34)
        rslts(2) = j
        k = ABS(0)
        rslts(3) = k
        l = ABS(j)
        rslts(4) = l
        l = ABS(i)
        rslts(5) = l

	data (rexpect(i), i = 6, 9) /2.45, 34.6, 0.0, 34.6/
        r = ABS(2.45)
        rrslts(6) = r
        s = ABS(-34.6)
        rrslts(7) = s
        t = ABS(0.0)
        rrslts(8) = t
        u = ABS(s)
        rrslts(9) = u

	data (rexpect(i), i = 10, 13) /2.45, 34.6, 0.0, 34.6/
        d = ABS(2.45D0)
        rrslts(10) = d
        e = ABS(-34.6D0)
        rrslts(11) = e
        f = ABS(0.0D0)
        rrslts(12) = f
        g = ABS(e)
        rrslts(13) = g

        data (expect(i), i = 14, 24) /23,23,-23,-23,23,34, 23, 
     - 23, 0 ,34, 34/
        i = MAX( 2, 23)
        rslts(14) = i
        i = MAX( 23, 2)
        rslts(15) = i
        j = MAX( -34, -23)
        rslts(16) = j
        j = MAX( -23, -34)
        rslts(17) = j
        j = MAX( 23, -34)
        rslts(18) = j
        j = MAX( -23, 34)
        rslts(19) = j
        k = MAX(0, 23)
        rslts(20) = k
        k = MAX(23, 0)
        rslts(21) = k
        k = MAX(0,0)
        rslts(22) = k
        l = MAX(j,k)
        rslts(23) = l
        l = MAX(k,l)
        rslts(24) = l

	data (rexpect(i), i = 25, 36) /23.2,23.2,-23.4,-23.4, 34.6,
     - 23.4, 0.0, 34.5, 0.0, 23.4, 23.4, 23.4 /
        r = MAX(2.45,23.2)
        rrslts(25) = r
        r = MAX(23.2,2.45)
        rrslts(26) = r
        s = MAX(-34.6,-23.4)
        rrslts(27) = s
        s = MAX(-23.4,-34.6)
        rrslts(28) = s
        s = MAX(-23.4,34.6)
        rrslts(29) = s
        s = MAX(23.4,-34.6)
        rrslts(30) = s
        t = MAX(0.0, -34.5)
        rrslts(31) = t
        t = MAX(0.0, 34.5)
        rrslts(32) = t
        t = MAX(0.0, 0.0)
        rrslts(33) = t
        u = MAX(s,t)
        rrslts(34) = u
        u = MAX(t,s)
        rrslts(35) = u
        u = MAX(s,s)
        rrslts(36) = u

        data (rexpect(i),i=37,48)/34.3D0,34.3D0,-23.6D0,-34.6D0,
     - 34.6D0,57.9D0,34.8D0,34.8D0,0.0D0,57.9D0, 57.9D0, 57.9D0/
        d = MAX(2.45D0,34.3D0)
        rrslts(37) = d
        d = MAX(34.3D0, 2.45D0)
        rrslts(38) = d
        e = MAX(-34.6D0,-23.6D0)
        rrslts(39) = e
        e = MAX(-34.6D0, -57.9D0)
        rrslts(40) = e
        e = MAX(34.6D0, -57.9D0)
        rrslts(41) = e
        e = MAX(-34.6D0, 57.9D0)
        rrslts(42) = e
        f = MAX(0.0D0,34.8D0)
        rrslts(43) = f
        f = MAX(34.8D0,0.0D0)
        rrslts(44) = f
        f = MAX(0.0D0,0.0D0)
        rrslts(45) = f
        g = MAX(e,f)
        rrslts(46) = g
        g = MAX(g,e)
        rrslts(47) = g
        g = MAX(g,g)
        rrslts(48) = g
        

	data (expect(i),i= 49,59)/2,2,-34,-34,-34,-23,0,0,0,-23,-23/
        i = MIN( 2, 23)
        rslts(49) = i
        i = MIN( 23, 2)
        rslts(50) = i
        j = MIN( -34, -23)
        rslts(51) = j
        j = MIN( -23, -34)
        rslts(52) = j
        j = MIN( 23, -34)
        rslts(53) = j
        j = MIN( -23, 34)
        rslts(54) = j
        k = MIN(0, 23)
        rslts(55) = k
        k = MIN(23, 0)
        rslts(56) = k
        k = MIN(0,0)
        rslts(57) = k
        l = MIN(j,k)
        rslts(58) = l
        l = MIN(k,l)
        rslts(59) = l

	data (rexpect(i), i = 60, 71) /2.45,2.45,-34.6,-34.6,-23.4,-34.6,
     - -34.5, 0.0,0.0,-34.6,-34.6,-34.6/
        r = MIN(2.45,23.2)
        rrslts(60) = r
        r = MIN(23.2,2.45)
        rrslts(61) = r
        s = MIN(-34.6,-23.4)
        rrslts(62) = s
        s = MIN(-23.4,-34.6)
        rrslts(63) = s
        s = MIN(-23.4,34.6)
        rrslts(64) = s
        s = MIN(23.4,-34.6)
        rrslts(65) = s
        t = MIN(0.0, -34.5)
        rrslts(66) = t
        t = MIN(0.0, 34.5)
        rrslts(67) = t
        t = MIN(0.0, 0.0)
        rrslts(68) = t
        u = MIN(s,t)
        rrslts(69) = u
        u = MIN(t,s)
        rrslts(70) = u
        u = MIN(s,s)
        rrslts(71) = u

	data(rexpect(i),i = 72, 83)/2.45D0,2.45D0,-34.6D0,-57.9D0,
     - -57.9D0,-34.6D0,0.0D0,0.0D0,0.0D0,-34.6D0,-34.6D0,-34.6D0/
        d = MIN(2.45D0,34.3D0)
        rrslts(72) = d
        d = MIN(34.3D0, 2.45D0)
        rrslts(73) = d
        e = MIN(-34.6D0,-23.6D0)
        rrslts(74) = e
        e = MIN(-34.6D0, -57.9D0)
        rrslts(75) = e
        e = MIN(34.6D0, -57.9D0)
        rrslts(76) = e
        e = MIN(-34.6D0, 57.9D0)
        rrslts(77) = e
        f = MIN(0.0D0,34.8D0)
        rrslts(78) = f
        f = MIN(34.8D0,0.0D0)
        rrslts(79) = f
        f = MIN(0.0D0,0.0D0)
        rrslts(80) = f
        g = MIN(e,f)
        rrslts(81) = g
        g = MIN(g,e)
        rrslts(82) = g
        g = MIN(g,g)
        rrslts(83) = g

        
	call check(rslts, expect, n)

        end
