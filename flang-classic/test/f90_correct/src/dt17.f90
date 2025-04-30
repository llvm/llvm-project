!*** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!*** See https://llvm.org/LICENSE.txt for license information.
!*** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! CSHIFT, EOSHIFT, TRANSPOSE, SPREAD with arrays of derived type
! where the derived type has an array member

	module mmm
	! set 'd' to .true. to get intermediate output
	logical,parameter::d=.false.
	type dt
	 integer i,j(3)
	end type dt

	contains
	 subroutine check1(s,a,ia,ja,r)
	! this seems to fail with type(dt)::a(:)
	  type(dt)::a(4)
	  integer:: ia(4),ja(3,4),i
	  character*(*) s
	  integer r
	  r = 1
	  do i = lbound(a,1),ubound(a,1)
	   if(a(i)%i .ne. ia(i) )then
	    write(*,'(a,a,i1,a,i4,a,i4)') s,'a(',i,')%i=',a(i)%i,' != ',ia(i)
	    r = 0
	   endif
	   do j = lbound(ja,1),ubound(ja,1)
	    if(a(i)%j(j) .ne. ja(j,i) )then
	     write(*,'(a,a,i1,a,i1,a,i4,a,i4)') s,'a(',i,')%j(',j,')=',a(i)%j(j),' != ',ja(j,i)
	     r = 0
	    endif
	   enddo
	  enddo
	 end subroutine
	 subroutine check2(s,a,ia,ja,r)
	  type(dt)::a(4,4)
	  integer:: ia(4,4),ja(3,4,4),i,j,k
	  character*(*) s
	  integer r
	  r = 1
	  do i = lbound(a,1),ubound(a,1)
	   do j = lbound(a,2),ubound(a,2)
	    if(a(i,j)%i .ne. ia(i,j) )then
	     write(*,'(a,a,i1,a,i1,a,i4,a,i4)') s,'a(',i,',',j,')%i=',a(i,j)%i,' != ',ia(i)
	     r = 0
	    endif
	    do k = lbound(ja,1),ubound(ja,1)
	     if(a(i,j)%j(k) .ne. ja(k,i,j) )then
	      write(*,'(a,a,i1,a,i1,a,i1,a,i4,a,i4)') s,'a(',i,',',j,')%j(',k,')=',a(i,j)%j(k),' != ',ja(k,i,j)
	      r = 0
	     endif
	    enddo
	   enddo
	  enddo
	 end subroutine
	end module
	use mmm
	! test that CSHIFT, EOSHIFT, SPREAD, TRANSPOSE, MERGE
	! work with derived type arguments

	! for CSHIFT, EOSHIFT
	type(dt):: a(4),b(4),e
	integer ia(4),ja(3,4),ib(4),jb(3,4),ie,je(3)

	! for TRANSPOSE, SPREAD
	type(dt):: aa(4,4),bb(4,4)
	integer iaa(4,4),jaa(3,4,4),ibb(4,4),jbb(3,4,4)
	integer result(6), expect(6)
	data expect/1,1,1,1,1,1/

	do i = lbound(b,1),ubound(b,1)
	  b(i)%i = i*10
	  b(i)%j(1) = i*10+1
	  b(i)%j(2) = i*10+2
	  b(i)%j(3) = i*10+3
	  ib(i) =  i*10
	  jb(1,i) =  i*10+1
	  jb(2,i) =  i*10+2
	  jb(3,i) =  i*10+3
	enddo
	e%i = 990
	e%j(1) = 991
	e%j(2) = 992
	e%j(3) = 993
	ie = 990
	je(1) = 991
	je(2) = 992
	je(3) = 993

	a = cshift(b,2)
	ia = cshift(ib,2)
	ja = cshift(jb,2,2)
	if(d)write(*,100) 'CSHIFT(b,2)',(a(i),i=1,4)
	call check1('CSHIFT-1: ',a,ia,ja,result(1))
 	a = cshift(b,-1)
 	ia = cshift(ib,-1)
 	ja = cshift(jb,-1,2)
 	if(d)write(*,100) 'CSHIFT(b,-1)',(a(i),i=1,4)
 	call check1('CSHIFT-2: ',a,ia,ja,result(2))
 
 	a = eoshift(b,2,e)
 	ia = eoshift(ib,2,ie)
 	ja = eoshift(jb,2,je,2)
 	if(d)write(*,100) 'EOSHIFT(b,2,e)',(a(i),i=1,4)
 	call check1('EOSHIFT: ',a,ia,ja,result(3))
 
 	aa = spread(b,1,4)
 	iaa = spread(ib,1,4)
 	jaa = spread(jb,2,4)
 	if(d)write(*,200) 'SPREAD(b,1,4)',((aa(i,j),j=1,4),i=1,4)
 	call check2('SPREAD-1: ',aa,iaa,jaa,result(4))
 
 	aa = spread(b,2,4)
 	iaa = spread(ib,2,4)
 	jaa = spread(jb,3,4)
 	if(d)write(*,200) 'SPREAD(b,2,4)',((aa(i,j),j=1,4),i=1,4)
 	call check2('SPREAD-2: ',aa,iaa,jaa,result(5))
 
	do i = lbound(bb,1),ubound(bb,1)
	 do j = lbound(bb,2),ubound(bb,2)
	  bb(i,j)%i = i*100 + j*10 
	  bb(i,j)%j(1) = i*100 + j*10 + 1
	  bb(i,j)%j(2) = i*100 + j*10 + 2
	  bb(i,j)%j(3) = i*100 + j*10 + 3
	  ibb(i,j) = i*100 + j*10
	  jbb(1,i,j) = i*100 + j*10 + 1
	  jbb(2,i,j) = i*100 + j*10 + 2
	  jbb(3,i,j) = i*100 + j*10 + 3
	 enddo
	enddo

	aa = transpose(bb)
	iaa = transpose(ibb)
	forall(i=1:3,j=1:4,k=1:4) jaa(i,j,k) = jbb(i,k,j)
	if(d)write(*,200) 'TRANSPOSE(bb)',((aa(i,j),j=1,4),i=1,4)
	call check2('TRANSPOSE: ',aa,iaa,jaa,result(6))
	call check(result,expect,6)
100	format(a,':',1(/1x,4(4i4,2x)))
200	format(a,':',4(/1x,4(4i4,2x)))
	end
