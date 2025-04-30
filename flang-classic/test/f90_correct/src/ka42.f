** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*--- Large loop with many induction pointers. Can cause block breaking
*--- and register usage problems.
	program ka42
	common /res/iexp(80), ires(80)
	common /com/ a(10), b(10), c(10), d(10), e(10)
	common /com/ f(10), g(10), h(10), p(10), q(10)
	common /com/ r(10), s(10), t(10), u(10), v(10)
	common /com/ w(10), x(10), y(10), z(10), za(10)
	common /com/ ab(10), bb(10), cb(10), db(10), eb(10)
	common /com/ fb(10), gb(10), hb(10), pb(10), qb(10)
	common /com/ rb(10), sb(10), tb(10), ub(10), vb(10)
	common /com/ wb(10), xb(10), yb(10), zb(10), zab(10)
	common /com/ ac(10), bc(10), cc(10), dc(10), ec(10)
	common /com/ fc(10), gc(10), hc(10), pc(10), qc(10)
	common /com/ rc(10), sc(10), tc(10), uc(10), vc(10)
	common /com/ wc(10), xc(10), yc(10), zc(10), zac(10)
	common /com/ ad(10), bd(10), cd(10), dd(10), ed(10)
	common /com/ fd(10), gd(10), hd(10), pd(10), qd(10)
	common /com/ rd(10), sd(10), td(10), ud(10), vd(10)
	common /com/ wd(10), xd(10), yd(10), zd(10), zad(10)

	do 10 i = 2,10
	    a(i) = a(i-1) + b(i)
	    c(i) = d(i) + a(i) * e(i)
	    f(i) = f(i-1) + g(i)
	    h(i) = p(i) + f(i) * q(i)
	    r(i) = r(i-1) + s(i)
	    t(i) = u(i) + r(i) * v(i)
	    w(i) = w(i-1) + x(i)
	    y(i) = z(i) + w(i) * za(i)
	    ab(i) = ab(i-1) + bb(i)
	    cb(i) = db(i) + ab(i) * eb(i)
	    fb(i) = fb(i-1) + gb(i)
	    hb(i) = pb(i) + fb(i) * qb(i)
	    rb(i) = rb(i-1) + sb(i)
	    tb(i) = ub(i) + rb(i) * vb(i)
	    wb(i) = wb(i-1) + xb(i)
	    yb(i) = zb(i) + wb(i) * zab(i)
	    ac(i) = ac(i-1) + bc(i)
	    cc(i) = dc(i) + ac(i) * ec(i)
	    fc(i) = fc(i-1) + gc(i)
	    hc(i) = pc(i) + fc(i) * qc(i)
	    rc(i) = rc(i-1) + sc(i)
	    tc(i) = uc(i) + rc(i) * vc(i)
	    wc(i) = wc(i-1) + xc(i)
	    yc(i) = zc(i) + wc(i) * zac(i)
	    ad(i) = ad(i-1) + bd(i)
	    cd(i) = dd(i) + ad(i) * ed(i)
	    fd(i) = fd(i-1) + gd(i)
	    hd(i) = pd(i) + fd(i) * qd(i)
	    rd(i) = rd(i-1) + sd(i)
	    td(i) = ud(i) + rd(i) * vd(i)
	    wd(i) = wd(i-1) + xd(i)
	    yd(i) = zd(i) + wd(i) * zad(i)
10	continue

	call dosums

	call check (ires, iexp, 80)
	end
	subroutine dosums
	common /com/ a(10), b(10), c(10), d(10), e(10)
	common /com/ f(10), g(10), h(10), p(10), q(10)
	common /com/ r(10), s(10), t(10), u(10), v(10)
	common /com/ w(10), x(10), y(10), z(10), za(10)
	common /com/ ab(10), bb(10), cb(10), db(10), eb(10)
	common /com/ fb(10), gb(10), hb(10), pb(10), qb(10)
	common /com/ rb(10), sb(10), tb(10), ub(10), vb(10)
	common /com/ wb(10), xb(10), yb(10), zb(10), zab(10)
	common /com/ ac(10), bc(10), cc(10), dc(10), ec(10)
	common /com/ fc(10), gc(10), hc(10), pc(10), qc(10)
	common /com/ rc(10), sc(10), tc(10), uc(10), vc(10)
	common /com/ wc(10), xc(10), yc(10), zc(10), zac(10)
	common /com/ ad(10), bd(10), cd(10), dd(10), ed(10)
	common /com/ fd(10), gd(10), hd(10), pd(10), qd(10)
	common /com/ rd(10), sd(10), td(10), ud(10), vd(10)
	common /com/ wd(10), xd(10), yd(10), zd(10), zad(10)
	call sum (a)		! test 1
	call sum (b)		! test 2
	call sum (c)		! test 3
	call sum (d)		! test 4
	call sum (e)		! test 5
	call sum (f)		! test 6
	call sum (g)		! test 7
	call sum (h)		! test 8
	call sum (p)		! test 9
	call sum (q)		! test 10
	call sum (r)		! test 11
	call sum (s)		! test 12
	call sum (t)		! test 13
	call sum (u)		! test 14
	call sum (v)		! test 15
	call sum (w)		! test 16
	call sum (x)		! test 17
	call sum (y)		! test 18
	call sum (z)		! test 19
	call sum (za)		! test 20
	call sum (ab)		! test 21
	call sum (bb)		! test 22
	call sum (cb)		! test 23
	call sum (db)		! test 24
	call sum (eb)		! test 25
	call sum (fb)		! test 26
	call sum (gb)		! test 27
	call sum (hb)		! test 28
	call sum (pb)		! test 29
	call sum (qb)		! test 30
	call sum (rb)		! test 31
	call sum (sb)		! test 32
	call sum (tb)		! test 33
	call sum (ub)		! test 34
	call sum (vb)		! test 35
	call sum (wb)		! test 36
	call sum (xb)		! test 37
	call sum (yb)		! test 38
	call sum (zb)		! test 39
	call sum (ac)		! test 40
	call sum (bc)		! test 41
	call sum (cc)		! test 42
	call sum (dc)		! test 43
	call sum (ec)		! test 44
	call sum (fc)		! test 45
	call sum (gc)		! test 46
	call sum (hc)		! test 47
	call sum (pc)		! test 48
	call sum (qc)		! test 49
	call sum (rc)		! test 50
	call sum (sc)		! test 51
	call sum (tc)		! test 52
	call sum (uc)		! test 53
	call sum (vc)		! test 54
	call sum (wc)		! test 55
	call sum (xc)		! test 56
	call sum (yc)		! test 57
	call sum (zc)		! test 58
	call sum (ad)		! test 59
	call sum (bd)		! test 60
	call sum (cd)		! test 61
	call sum (dd)		! test 62
	call sum (ed)		! test 63
	call sum (fd)		! test 64
	call sum (gd)		! test 65
	call sum (hd)		! test 66
	call sum (pd)		! test 67
	call sum (qd)		! test 68
	call sum (rd)		! test 69
	call sum (sd)		! test 70
	call sum (td)		! test 71
	call sum (ud)		! test 72
	call sum (vd)		! test 73
	call sum (wd)		! test 74
	call sum (xd)		! test 75
	call sum (yd)		! test 76
	call sum (zd)		! test 77
	call sum (zab)		! test 78
	call sum (zac)		! test 79
	call sum (zad)		! test 80
	end
	subroutine sum (a)
	real a(10)
	common /res/iexp(80), ires(80)
	integer icnt
	save icnt
	data icnt/0/
	fsum = 0.0
	do 10 i = 1, 10
10	fsum = fsum + a(i)
	icnt = icnt + 1
	ires(icnt) = fsum
	end
	blockdata
	common /res/iexp(80), ires(80)
	common /com/ a(10), b(10), c(10), d(10), e(10)
	common /com/ f(10), g(10), h(10), p(10), q(10)
	common /com/ r(10), s(10), t(10), u(10), v(10)
	common /com/ w(10), x(10), y(10), z(10), za(10)
	common /com/ ab(10), bb(10), cb(10), db(10), eb(10)
	common /com/ fb(10), gb(10), hb(10), pb(10), qb(10)
	common /com/ rb(10), sb(10), tb(10), ub(10), vb(10)
	common /com/ wb(10), xb(10), yb(10), zb(10), zab(10)
	common /com/ ac(10), bc(10), cc(10), dc(10), ec(10)
	common /com/ fc(10), gc(10), hc(10), pc(10), qc(10)
	common /com/ rc(10), sc(10), tc(10), uc(10), vc(10)
	common /com/ wc(10), xc(10), yc(10), zc(10), zac(10)
	common /com/ ad(10), bd(10), cd(10), dd(10), ed(10)
	common /com/ fd(10), gd(10), hd(10), pd(10), qd(10)
	common /com/ rd(10), sd(10), td(10), ud(10), vd(10)
	common /com/ wd(10), xd(10), yd(10), zd(10), zad(10)
	data a /1,2,3,4,5,6,7,8,9,10/
	data b /1,2,3,4,5,6,7,8,9,10/
	data c /1,2,3,4,5,6,7,8,9,10/
	data d /1,2,3,4,5,6,7,8,9,10/
	data e /1,2,3,4,5,6,7,8,9,10/
	data f /1,2,3,4,5,6,7,8,9,10/
	data g /1,2,3,4,5,6,7,8,9,10/
	data h /1,2,3,4,5,6,7,8,9,10/
	data p /1,2,3,4,5,6,7,8,9,10/
	data q /1,2,3,4,5,6,7,8,9,10/
	data r /1,2,3,4,5,6,7,8,9,10/
	data s /1,2,3,4,5,6,7,8,9,10/
	data t /1,2,3,4,5,6,7,8,9,10/
	data u /1,2,3,4,5,6,7,8,9,10/
	data v /1,2,3,4,5,6,7,8,9,10/
	data w /1,2,3,4,5,6,7,8,9,10/
	data x /1,2,3,4,5,6,7,8,9,10/
	data y /1,2,3,4,5,6,7,8,9,10/
	data z /1,2,3,4,5,6,7,8,9,10/
	data za/1,2,3,4,5,6,7,8,9,10/
	data ab/1,2,3,4,5,6,7,8,9,10/
	data bb/1,2,3,4,5,6,7,8,9,10/
	data cb/1,2,3,4,5,6,7,8,9,10/
	data db/1,2,3,4,5,6,7,8,9,10/
	data eb/1,2,3,4,5,6,7,8,9,10/
	data fb/1,2,3,4,5,6,7,8,9,10/
	data gb/1,2,3,4,5,6,7,8,9,10/
	data hb/1,2,3,4,5,6,7,8,9,10/
	data pb/1,2,3,4,5,6,7,8,9,10/
	data qb/1,2,3,4,5,6,7,8,9,10/
	data rb/1,2,3,4,5,6,7,8,9,10/
	data sb/1,2,3,4,5,6,7,8,9,10/
	data tb/1,2,3,4,5,6,7,8,9,10/
	data ub/1,2,3,4,5,6,7,8,9,10/
	data vb/1,2,3,4,5,6,7,8,9,10/
	data wb/1,2,3,4,5,6,7,8,9,10/
	data xb/1,2,3,4,5,6,7,8,9,10/
	data yb/1,2,3,4,5,6,7,8,9,10/
	data zb/1,2,3,4,5,6,7,8,9,10/
	data ac/1,2,3,4,5,6,7,8,9,10/
	data bc/1,2,3,4,5,6,7,8,9,10/
	data cc/1,2,3,4,5,6,7,8,9,10/
	data dc/1,2,3,4,5,6,7,8,9,10/
	data ec/1,2,3,4,5,6,7,8,9,10/
	data fc/1,2,3,4,5,6,7,8,9,10/
	data gc/1,2,3,4,5,6,7,8,9,10/
	data hc/1,2,3,4,5,6,7,8,9,10/
	data pc/1,2,3,4,5,6,7,8,9,10/
	data qc/1,2,3,4,5,6,7,8,9,10/
	data rc/1,2,3,4,5,6,7,8,9,10/
	data sc/1,2,3,4,5,6,7,8,9,10/
	data tc/1,2,3,4,5,6,7,8,9,10/
	data uc/1,2,3,4,5,6,7,8,9,10/
	data vc/1,2,3,4,5,6,7,8,9,10/
	data wc/1,2,3,4,5,6,7,8,9,10/
	data xc/1,2,3,4,5,6,7,8,9,10/
	data yc/1,2,3,4,5,6,7,8,9,10/
	data zc/1,2,3,4,5,6,7,8,9,10/
	data ad/1,2,3,4,5,6,7,8,9,10/
	data bd/1,2,3,4,5,6,7,8,9,10/
	data cd/1,2,3,4,5,6,7,8,9,10/
	data dd/1,2,3,4,5,6,7,8,9,10/
	data ed/1,2,3,4,5,6,7,8,9,10/
	data fd/1,2,3,4,5,6,7,8,9,10/
	data gd/1,2,3,4,5,6,7,8,9,10/
	data hd/1,2,3,4,5,6,7,8,9,10/
	data pd/1,2,3,4,5,6,7,8,9,10/
	data qd/1,2,3,4,5,6,7,8,9,10/
	data rd/1,2,3,4,5,6,7,8,9,10/
	data sd/1,2,3,4,5,6,7,8,9,10/
	data td/1,2,3,4,5,6,7,8,9,10/
	data ud/1,2,3,4,5,6,7,8,9,10/
	data vd/1,2,3,4,5,6,7,8,9,10/
	data wd/1,2,3,4,5,6,7,8,9,10/
	data xd/1,2,3,4,5,6,7,8,9,10/
	data yd/1,2,3,4,5,6,7,8,9,10/
	data zd/1,2,3,4,5,6,7,8,9,10/
	data zab/10*7/
	data zac/10*8/
	data zad/10*9/
	data iexp/
     + 220, 55, 1759, 55, 55, 220, 55, 1759, 55, 55,
     + 220, 55, 1759, 55, 55, 220, 55, 1759, 55, 55,
     + 220, 55, 1759, 55, 55, 220, 55, 1759, 55, 55,
     + 220, 55, 1759, 55, 55, 220, 55, 1588, 55, 220,
     + 55, 1759, 55, 55, 220, 55, 1759, 55, 55, 220,
     + 55, 1759, 55, 55, 220, 55, 1807, 55, 220, 55,
     + 1759, 55, 55, 220, 55, 1759, 55, 55, 220, 55,
     + 1759, 55, 55, 220, 55, 2026, 55, 70, 80, 90
     + /
	end
