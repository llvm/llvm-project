** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

* Many invariant array references

	program ka52
	call test
	call chck
	end
	subroutine test
	common /a/ a(10)
	common /b/ b(10)
	common /c/ c(10)
	common /d/ d(10)
	common /e/ e(10)
	common /f/ f(10)
	common /g/ g(10)
	common /h/ h(10)
	common /xj/ xj(10)
	common /xk/ xk(10)
	common /xl/ xl(10)
	common /xm/ xm(10)
	common /xn/ xn(10)
	common /o/ o(10)
	common /p/ p(10)
	common /q/ q(10)
	common /r/ r(10)
	common /s/ s(10)
	common /t/ t(10)
	common /u/ u(10)
	common /kk/kk

        do 70 i = 1,10
        a(kk) = a(kk) + 1.0
        b(kk) = b(kk) + 1.0
        c(kk) = c(kk) + 1.0
        d(kk) = d(kk) + 1.0
        e(kk) = e(kk) + 1.0
        f(kk) = f(kk) + 1.0
        g(kk) = g(kk) + 1.0
        h(kk) = h(kk) + 1.0
        xj(kk) = xj(kk) + 1.0
        xk(kk) = xk(kk) + 1.0
        xl(kk) = xl(kk) + 1.0
        xm(kk) = xm(kk) + 1.0
        xn(kk) = xn(kk) + 1.0
        o(kk) = o(kk) + 1.0
        p(kk) = p(kk) + 1.0
        q(kk) = q(kk) + 1.0
        r(kk) = r(kk) + 1.0
        s(kk) = s(kk) + 1.0
        t(kk) = t(kk) + 1.0
        u(kk) = u(kk) + 1.0
70      continue
        end
	subroutine chck
	common /a/ a(10)
	common /b/ b(10)
	common /c/ c(10)
	common /d/ d(10)
	common /e/ e(10)
	common /f/ f(10)
	common /g/ g(10)
	common /h/ h(10)
	common /xj/ xj(10)
	common /xk/ xk(10)
	common /xl/ xl(10)
	common /xm/ xm(10)
	common /xn/ xn(10)
	common /o/ o(10)
	common /p/ p(10)
	common /q/ q(10)
	common /r/ r(10)
	common /s/ s(10)
	common /t/ t(10)
	common /u/ u(10)
	common /res/expect(20),result(20), icnt
	icnt = 1
        call fillres(a)
        call fillres(b)
        call fillres(c)
        call fillres(d)
        call fillres(e)
        call fillres(f)
        call fillres(g)
        call fillres(h)
        call fillres(xj)
        call fillres(xk)
        call fillres(xl)
        call fillres(xm)
        call fillres(xn)
        call fillres(o)
        call fillres(p)
        call fillres(q)
        call fillres(r)
        call fillres(s)
        call fillres(t)
        call fillres(u)
	call check (result, expect, 20)
	end
	subroutine fillres(a)
	dimension a(10)
	common /res/expect(20), result(20), icnt
	result(icnt) = a(2)
	icnt = icnt + 1
	end
	block data
	common /kk/kk
	data kk/2/

	common /res/expect(20), result(20), icnt
	data expect/
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0,
     +  10.0
     +  /
	end
