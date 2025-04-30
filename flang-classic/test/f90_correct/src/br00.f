** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   VMS STRUCTURE/RECORD

C   Structure assignment tests:
C   test1 makes calls to the run-time support routines
C        c$bcopy, c$hcopy, and c$wcopy.
C   test2 assigns structures by loads and stores.
	program p
	parameter(N=36)

	integer result(N), expect(N)
	common result

	structure /b/
	    character*1 x(10)
	endstructure
	structure /h/
	    integer*2 x(10)
	endstructure
	structure /w/
	    integer*4 x(10)
	endstructure

	record /b/ bb, byte
	record /h/ hh, hw
	record /w/ ww, word

	data expect /
     +   48, 49, 50, 51, 52, 53, 54, 55, 56, 57,             ! bb
     +   -1, 1, 2, 3, 4, 5, 6, 7, 8, 9,                      ! hh
     +   -1, 1, 2, 3, 4, 5, 6, 7, 8, 9,                      ! ww
     +   1, 2,                                               ! s1
     +   3, 4, 5, 1 /                                        ! s2(1)

	call initb(bb.x)
	call inithw(hh.x)
	call initww(ww.x)

c --- test1

	byte = bb
	hw = hh
	word = ww
	call fill(result, byte.x, hw.x, word.x)
	 
	call test2

	call check(result, expect, N)

	end

	subroutine fill(p, b, h, w)
	integer p(30)
	character*1 b(10)
	integer*2 h(10)
	integer*4 w(10)
	do i = 1, 10
	    p(i) = ichar(b(i))
	enddo
	do i = 1, 10
	    p(i+10) = h(i)
	enddo
	do i = 1, 10
	    p(i+20) = w(i)
	enddo
	end

	subroutine test2
	structure /s1/
	   integer*2 a
	   integer*2 b
	endstructure
	structure /s2/
	   integer a
	   integer b
	   integer c
	endstructure
	record /s1/ s1, a1
	record /s2/ s2(2), a2

	call inits1(s1.a)
	call inits2(s2(1).a)
	a1 = s1
	a2 = s2(indx())
	call r31_35(a1, a2)
	end

	subroutine r31_35(a1, a2)
	integer result(36)
	common result
	structure /s1/
	   integer*2 a
	   integer*2 b
	endstructure
	structure /s2/
	   integer a
	   integer b
	   integer c
	endstructure
	record/s1/a1
	record/s2/a2

	result(31) = a1.a
	result(32) = a1.b
	result(33) = a2.a
	result(34) = a2.b
	result(35) = a2.c
	return
	end

	integer function indx()
	integer result
	common result(36)
	result(36) = result(36) + 1    ! function is called once
	indx = 2
	return
	end

c  the following routines are necessary since it's too awkward to init
c  records

	subroutine initb(bb)
	character*10 bb
	bb = '0123456789'
	return
	end
	subroutine inithw(hh)
	integer*2 hh(0:9)
	hh(0) = -1
	do i=1, 9
	    hh(i) = i
	enddo
	end
	subroutine initww(ww)
	integer ww(0:9)
	ww(0) = -1
	do i=1, 9
	    ww(i) = i
	enddo
	end
	subroutine inits1(s1)
	integer*2 s1(2)
	s1(1) = 1
	s1(2) = 2
	end
	subroutine inits2(s2)
	integer s2(6)
	s2(1) = 11
	s2(2) = 12
	s2(3) = 13
	s2(4) = 3
	s2(5) = 4
	s2(6) = 5
	end
