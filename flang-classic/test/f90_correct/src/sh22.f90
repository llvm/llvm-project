! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test EOSHIFT and CSHIFT
! one dimension out of two
	program p
	 implicit none
	 integer, parameter :: N=5, M=6
	 integer, dimension(N,M) :: src
	 integer, dimension(N,M) :: eopc,eopv,cpc,cpv
	 integer, dimension(N,M) :: eomc,eomv,cmc,cmv
	 integer, dimension(N,M) :: eopcres,eopvres,cpcres,cpvres
	 integer, dimension(N,M) :: eomcres,eomvres,cmcres,cmvres
	 integer :: i,j,k
        data eopcres/&
        &   3,   4,   5,   0,   0,&
        &   8,   9,  10,   0,   0,&
        &  13,  14,  15,   0,   0,&
        &  18,  19,  20,   0,   0,&
        &  23,  24,  25,   0,   0,&
        &  28,  29,  30,   0,   0/
        data eopvres/&
        &   3,   4,   5,   0,   0,&
        &   8,   9,  10,   0,   0,&
        &  13,  14,  15,   0,   0,&
        &  18,  19,  20,   0,   0,&
        &  23,  24,  25,   0,   0,&
        &  28,  29,  30,   0,   0/
        data eomcres/&
        &   0,   0,   1,   2,   3,&
        &   0,   0,   6,   7,   8,&
        &   0,   0,  11,  12,  13,&
        &   0,   0,  16,  17,  18,&
        &   0,   0,  21,  22,  23,&
        &   0,   0,  26,  27,  28/
        data eomvres/&
        &   0,   0,   1,   2,   3,&
        &   0,   0,   6,   7,   8,&
        &   0,   0,  11,  12,  13,&
        &   0,   0,  16,  17,  18,&
        &   0,   0,  21,  22,  23,&
        &   0,   0,  26,  27,  28/
        data  cpcres/&
        &   3,   4,   5,   1,   2,&
        &   8,   9,  10,   6,   7,&
        &  13,  14,  15,  11,  12,&
        &  18,  19,  20,  16,  17,&
        &  23,  24,  25,  21,  22,&
        &  28,  29,  30,  26,  27/
        data  cpvres/&
        &   3,   4,   5,   1,   2,&
        &   8,   9,  10,   6,   7,&
        &  13,  14,  15,  11,  12,&
        &  18,  19,  20,  16,  17,&
        &  23,  24,  25,  21,  22,&
        &  28,  29,  30,  26,  27/
        data  cmcres/&
        &   4,   5,   1,   2,   3,&
        &   9,  10,   6,   7,   8,&
        &  14,  15,  11,  12,  13,&
        &  19,  20,  16,  17,  18,&
        &  24,  25,  21,  22,  23,&
        &  29,  30,  26,  27,  28/
        data  cmvres/&
        &   4,   5,   1,   2,   3,&
        &   9,  10,   6,   7,   8,&
        &  14,  15,  11,  12,  13,&
        &  19,  20,  16,  17,  18,&
        &  24,  25,  21,  22,  23,&
        &  29,  30,  26,  27,  28/
	 logical,parameter :: doprint = .false.
	 eopc = -10	! fill in with garbage
	 eopv = -10
	 eomc = -10
	 eomv = -10
	 cpc = -10
	 cpv = -10
	 cmc = -10
	 cmv = -10
	 call sub(j,k)	! j is positive, k is negative
	 src=reshape((/(i,i=1,N*M)/),(/N,M/))	! initialize
	 eopc=eoshift(src, 2, dim=1)	! eoshift, positive, constant
	 eopv=eoshift(src, j, dim=1)	! eoshift, positive, variable
	 eomc=eoshift(src,-2, dim=1)	! eoshift, negative, constant
	 eomv=eoshift(src, k, dim=1)	! eoshift, negative, variable
	  cpc= cshift(src, 2, dim=1)	!  cshift, positive, constant
	  cpv= cshift(src, j, dim=1)	!  cshift, positive, variable
	  cmc= cshift(src,-2, dim=1)	!  cshift, negative, constant
	  cmv= cshift(src, k, dim=1)	!  cshift, negative, variable
	 if( doprint )then
	  print 10, ' src',src
	  print 10, 'eopc',eopc
	  print 10, 'eopv',eopv
	  print 10, 'eomc',eomc
	  print 10, 'eomv',eomv
	  print 10, ' cpc', cpc
	  print 10, ' cpv', cpv
	  print 10, ' cmc', cmc
	  print 10, ' cmv', cmv
10	  format('        data ',a,'res/&'/'        &',5(5(i4,','),'&'/'        &'),&
		& 4(i4,','),i4,'/')
	 else
	  call check(eopc,eopcres,N*M)
	  call check(eopv,eopvres,N*M)
	  call check(eomc,eomcres,N*M)
	  call check(eomv,eomvres,N*M)
	  call check(cpc,cpcres,N*M)
	  call check(cpv,cpvres,N*M)
	  call check(cmc,cmcres,N*M)
	  call check(cmv,cmvres,N*M)
	 endif
	end
	subroutine sub(j,k)
	 j = 2
	 k = -2
	end
