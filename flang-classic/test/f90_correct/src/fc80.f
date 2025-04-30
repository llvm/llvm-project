** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception



*   DIM constant folding in the expander.

	parameter(n = 35)
	integer rslts(n), expect(n)
	real  rrslts(n), rexpect(n)
	equivalence (rslts, rrslts), (expect, rexpect)

        integer  i, j, k, l
        real     r, s, t, u
        double precision  d, e, f, g
        
	data (expect(i), i = 1, 11) /0,21,0,11,57,0,0,23,0,0,0/
        i = DIM( 2, 23)
        rslts(1) = i
        i = DIM( 23, 2)
        rslts(2) = i
        j = DIM( -34, -23)
        rslts(3) = j
        j = DIM( -23, -34)
        rslts(4) = j
        j = DIM( 23, -34)
        rslts(5) = j
        j = DIM( -23, 34)
        rslts(6) = j
        k = DIM(0, 23)
        rslts(7) = k
        k = DIM(23, 0)
        rslts(8) = k
        k = DIM(0,0)
        rslts(9) = k
        l = DIM(j,k)
        rslts(10) = l
        l = DIM(k,l)
        rslts(11) = l

	data (rexpect(i), i = 12, 23) /0.0,21.2,0.0,11.6,0.0,58.0,
     -  34.5,0.0,0.0,58.0,0.0,0.0/
        r = DIM(2.45,23.2)
        rrslts(12) = r
        r = DIM(23.45,2.25)
        rrslts(13) = r
        s = DIM(-34.6,-23.4)
        rrslts(14) = s
        s = DIM(-23.4,-35.0)
        rrslts(15) = s
        s = DIM(-23.4,34.6)
        rrslts(16) = s
        s = DIM(23.4,-34.6)
        rrslts(17) = s
        t = DIM(0.0, -34.5)
        rrslts(18) = t
        t = DIM(0.0, 34.5)
        rrslts(19) = t
        t = DIM(0.0, 0.0)
        rrslts(20) = t
        u = DIM(s,t)
        rrslts(21) = u
        u = DIM(t,s)
        rrslts(22) = u
        u = DIM(s,s)
        rrslts(23) = u

	data (rexpect(i), i = 24, 35) /0.0,32.0,0.0,23.3,92.5,0.0,
     -  0.0,34.8,0.0,0.0,0.0,0.0/
        d = DIM(2.45D0,34.3D0)
        rrslts(24) = d
        d = DIM(34.45D0, 2.45D0)
        rrslts(25) = d
        e = DIM(-34.6D0,-23.6D0)
        rrslts(26) = e
        e = DIM(-34.6D0, -57.9D0)
        rrslts(27) = e
        e = DIM(34.6D0, -57.9D0)
        rrslts(28) = e
        e = DIM(-34.6D0, 57.9D0)
        rrslts(29) = e
        f = DIM(0.0D0,34.8D0)
        rrslts(30) = f
        f = DIM(34.8D0,0.0D0)
        rrslts(31) = f
        f = DIM(0.0D0,0.0D0)
        rrslts(32) = f
        g = DIM(e,f)
        rrslts(33) = g
        g = DIM(g,e)
        rrslts(34) = g
        g = DIM(g,g)
        rrslts(35) = g

        call check(rslts,expect,n)
        end
