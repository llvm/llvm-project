! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test EOSHIFT and CSHIFT
! one dimension
	program p
	 implicit none
	 integer, parameter :: N=5
	 integer, dimension(N) :: src
	 integer, dimension(N) :: eopc,eopv,cpc,cpv
	 integer, dimension(N) :: eomc,eomv,cmc,cmv
	 integer, dimension(N) :: eopcres,eopvres,cpcres,cpvres
	 integer, dimension(N) :: eomcres,eomvres,cmcres,cmvres
        data eopcres/   2,   3,   4,   5,   0/
        data eopvres/   2,   3,   4,   5,   0/
        data eomcres/   0,   1,   2,   3,   4/
        data eomvres/   0,   1,   2,   3,   4/
        data  cpcres/   2,   3,   4,   5,   1/
        data  cpvres/   2,   3,   4,   5,   1/
        data  cmcres/   5,   1,   2,   3,   4/
        data  cmvres/   5,   1,   2,   3,   4/
	logical,parameter :: doprint = .false.

	 integer :: i,j,k
	 eopc = -10	! fill in with garbage
	 eopv = -10
	 eomc = -10
	 eomv = -10
	 cpc = -10
	 cpv = -10
	 cmc = -10
	 cmv = -10
	 call sub(j,k)	! j is positive, k is negative
	 src=(/(i,i=1,N)/)	! initialize
	 eopc=eoshift(src, 1)	! eoshift, positive, constant
	 eopv=eoshift(src, j)	! eoshift, positive, variable
	 eomc=eoshift(src,-1)	! eoshift, negative, constant
	 eomv=eoshift(src, k)	! eoshift, negative, variable
	  cpc= cshift(src, 1)	!  cshift, positive, constant
	  cpv= cshift(src, j)	!  cshift, positive, variable
	  cmc= cshift(src,-1)	!  cshift, negative, constant
	  cmv= cshift(src, k)	!  cshift, negative, variable
	 if( doprint )then
	  print 10, 'eopc',eopc
	  print 10, 'eopv',eopv
	  print 10, 'eomc',eomc
	  print 10, 'eomv',eomv
	  print 10, ' cpc', cpc
	  print 10, ' cpv', cpv
	  print 10, ' cmc', cmc
	  print 10, ' cmv', cmv
10	  format( '        data ',a,'res/',4(i4,','),i4,'/')
	 else
	  call check(eopc,eopcres,N)
	  call check(eopv,eopvres,N)
	  call check(eomc,eomcres,N)
	  call check(eomv,eomvres,N)
	  call check(cpc,cpcres,N)
	  call check(cpv,cpvres,N)
	  call check(cmc,cmcres,N)
	  call check(cmv,cmvres,N)
	 endif
	end
	subroutine sub(j,k)
	 j = 1
	 k = -1
	end
