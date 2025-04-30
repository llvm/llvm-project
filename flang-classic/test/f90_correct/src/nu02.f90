!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!       Modify test bs04.f90 to use NEWUNIT

	program p
	 implicit none
	 character*50 a(10),b(10)
	 integer ::i,errors=0,expect=0
	 integer*1 i1(10),j1
	 integer*2 i2(10),j2
	 integer*4 i4(10),j4
	 integer*8 i8(10),j8
	 integer*1 u1(4)
	 integer*2 u2(4)
	 integer*4 u3(4)
	 integer*8 u4(4)

	 a(1) = 'abc abc abc abc abc'
	 a(2) = 'abc def abc abc abc'
	 a(3) = 'abc def ghi abc abc'
	 a(4) = 'abc def ghi jkl abc'
	 a(5) = 'abc def ghi jkl mno'
	 a(6) = 'abc def ghi        '
	 a(7) = 'abc def            '
	 a(8) = 'abc                '
	 a(9) = '0000               '
	 a(10)= '-------------------'

	 open(newunit=u1(2),file='test.txt',access='DIRECT',&
		form='UNFORMATTED',recl=50)
	 do i = 1,10
	  write(u1(2),rec=i) a(i)
	 enddo
	 close(unit=u1(2))

	 i1 = 66
	 i1(8) = 50
	 i1(6) = 11
	 open(newunit=u2(2),file='test.txt',access='DIRECT',&
		action='READ',form='UNFORMATTED',recl=i1(8))
	 do j1 = 1,10
	  i1(3) = j1
	  read(u2(2),rec=i1(3)) b(i1(3))
	  if( b(i1(3)) .ne. a(i1(3)) ) then
	   print *,'i1 error ',i1(3),' is ',b(i1(3))
	   errors = errors + 1
	  endif
	 enddo
	 close(unit=u2(2))

	 i2 = 6666
	 i2(4) = 50
	 i2(9) = 12
	 open(newunit=u3(3),file='test.txt',access='DIRECT',&
		action='READ',form='UNFORMATTED',recl=i2(4))
	 do j2 = 1,10
	  i2(2) = j2
	  read(u3(3),rec=i2(2)) b(i2(2))
	  if( b(i2(2)) .ne. a(i2(2)) ) then
	   print *,'i2 error ',i2(2),' is ',b(i2(2))
	   errors = errors + 1
	  endif
	 enddo
	 close(unit=u3(3))

	 i4 = 6666666
	 i4(1) = 50
	 i4(3) = 14
	 open(newunit=u4(4),file='test.txt',access='DIRECT',&
		action='READ',form='UNFORMATTED',recl=i4(1))
	 do j4 = 1,10
	  i4(5) = j4
	  read(u4(4),rec=i4(5)) b(i4(5))
	  if( b(i4(5)) .ne. a(i4(5)) ) then
	   print *,'i4 error ',i4(5),' is ',b(i4(5))
	   errors = errors + 1
	  endif
	 enddo
	 close(unit=u4(4))

	 i8 = 6666666666
	 i8(9) = 50
	 i8(8) = 18
	 open(newunit=u1(1),file='test.txt',access='DIRECT',&
		action='READ',form='UNFORMATTED',recl=i8(9))
	 do j8 = 1,10
	  i8(7) = j8
	  read(u1(1),rec=i8(7)) b(i8(7))
	  if( b(i8(7)) .ne. a(i8(7)) ) then
	   print *,'i8 error ',i8(7),' is ',b(i8(7))
	   errors = errors + 1
	  endif
	 enddo
	 close(unit=u1(1))

	 !print *,errors,' errors'
	 call check(errors,expect,1)
	end
