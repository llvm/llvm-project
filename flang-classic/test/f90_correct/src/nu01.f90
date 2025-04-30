!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!       Modify bs03.f90 to tests NEWUNIT       
	program p
	 implicit none
	 character*50 a(10),b(10)
	 integer ::i,errors=0,expect=0
	 integer*1 i1,r1
	 integer*2 i2,r2
	 integer*4 i4,r4
	 integer*8 i8,r8
	 integer*1 u1
	 integer*2 u2
	 integer*4 u3
	 integer*8 u4

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

	 open(newunit=u1,file='test.txt',access='DIRECT',&
		form='UNFORMATTED',recl=50)
	 do i = 1,10
	  write(u1,rec=i) a(i)
	 enddo
	 close(unit=u1)

	 r1 = 50
	 open(newunit=u2,file='test.txt',access='DIRECT',&
		action='READ',form='UNFORMATTED',recl=r1)
	 do i1 = 1,10
	  read(u2,rec=i1) b(i1)
	  if( b(i1) .ne. a(i1) ) then
	   print *,'i1 error ',i1,' is ',b(i1)
	   errors = errors + 1
	  endif
	 enddo
	 close(unit=u2)

	 r2 = 50
	 open(newunit=u3,file='test.txt',access='DIRECT',&
		action='READ',form='UNFORMATTED',recl=r2)
	 do i2 = 1,10
	  read(u3,rec=i2) b(i2)
	  if( b(i2) .ne. a(i2) ) then
	   print *,'i2 error ',i2,' is ',b(i2)
	   errors = errors + 1
	  endif
	 enddo
	 close(unit=u3)

	 r4 = 50
	 open(newunit=u4,file='test.txt',access='DIRECT',&
		action='READ',form='UNFORMATTED',recl=r4)
	 do i4 = 1,10
	  read(u4,rec=i4) b(i4)
	  if( b(i4) .ne. a(i4) ) then
	   print *,'i4 error ',i4,' is ',b(i4)
	   errors = errors + 1
	  endif
	 enddo
	 close(unit=u4)

	 r8 = 50
	 open(newunit=u1,file='test.txt',access='DIRECT',&
		action='READ',form='UNFORMATTED',recl=r8)
	 do i8 = 1,10
	  read(u1,rec=i8) b(i8)
	  if( b(i8) .ne. a(i8) ) then
	   print *,'i8 error ',i8,' is ',b(i8)
	   errors = errors + 1
	  endif
	 enddo
	 close(unit=u1)

	 !print *,errors,' errors'
	 call check(errors,expect,1)
	end
