** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Use of integer*2 variables in as many contexts as possible.

	integer * 2 n, nn
	parameter (n = 18, nn = 0)
	integer rslts(n), expect(n)
	integer*2 a(nn:3), k, i1, in2, f, i, i2, i7
	integer*2 sf, aa, b, sf2
	character*4 c
	integer*2  f2

	sf(aa, b) = aa + b
	sf2(aa) = aa

	data i1 / 1/, in2 / -2 /, i0 / 0 /, i7 / 7 /, i2 / 2 /
	data c / 'xyzw' /

	rslts(1) = i1
	rslts(2) = in2
	rslts(3) = 4 + in2
	rslts(4) = i1 * (in2 + 5)

	a(0) = 7
	a(3) = 77
	rslts(5) = f(a, i0)
	rslts(6) = a(1)

	rslts(7) = 10
	do 10 i = i7, i7, i1
10		rslts(i7) = rslts(i7) + i

	rslts(8) = sf(in2, -7)
	rslts(9) = sf2(999)
	rslts(10) = sf2(in2) + i1

	rslts(11) = ichar( c(i2:i2) )
	rslts(12) = abs(in2)
	rslts(13) = 2.51 * real(i2)
	rslts(14) = f2(in2)
	
	rslts(15) = 0
	assign 20 to ilab
	goto ilab
	rslts(15) = 10
20	rslts(15) = rslts(15) + 2

	rslts(16) = 0
	goto (30, 40, 50) i2
30	rslts(16) = 10
40	rslts(16) = rslts(16) + 1
50	rslts(16) = rslts(16) + 3

	rslts(17) = 0
	if (i2) 60, 70, 80
60	rslts(17) = rslts(17) + 1
70	rslts(17) = rslts(17) + 1
80	rslts(17) = rslts(17) + 1

	a(1) = 12.9
	rslts(18) = a(i1)

c ******* check results:

	call check(rslts, expect, int(n))

	data expect / 1, -2, 2, 3,        
     +                7, 75, 17,
     +                -9, 999, -1,
     +                121, 2, 5, -4,
     +                2, 4, 1, 12    /
	end

	

	function f(a, b)
	implicit integer*2 (a-f)
	dimension a(b:3)

	f = a(0) + b
	a(1) = a(3) - 2
	return

	entry f2(b)
	f2 = b + b
	end


	subroutine abcdefg
c    use of integer*2 in read and write statements:

        implicit integer*2 (i, j, k)
    	dimension kk(5)

        read (i, *) kk(1), j
        write (3, 'ab') i, kk(j)
        read(6, 100) (kk(i), i = 1, 4)
        write(6, 100) (kk(i), i = j, k, j)
100      format(2x)

        endfile(j)
        rewind(k)

 	end
