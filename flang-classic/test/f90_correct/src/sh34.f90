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
        &   0,   0,   0,   0,   0,&
        &   2,   3,   4,   5,   0,&
        &   7,   8,   9,  10,   0,&
        &  12,  13,  14,  15,   0,&
        &  17,  18,  19,  20,   0,&
        &  22,  23,  24,  25,   0/
        data eopvres/&
        &   0,   0,   0,   0,   0,&
        &   2,   3,   4,   5,   0,&
        &   7,   8,   9,  10,   0,&
        &  12,  13,  14,  15,   0,&
        &  17,  18,  19,  20,   0,&
        &  22,  23,  24,  25,   0/
        data eomcres/&
        &   0,   0,   0,   0,   0,&
        &   0,   1,   2,   3,   4,&
        &   0,   6,   7,   8,   9,&
        &   0,  11,  12,  13,  14,&
        &   0,  16,  17,  18,  19,&
        &   0,  21,  22,  23,  24/
        data eomvres/&
        &   0,   0,   0,   0,   0,&
        &   0,   1,   2,   3,   4,&
        &   0,   6,   7,   8,   9,&
        &   0,  11,  12,  13,  14,&
        &   0,  16,  17,  18,  19,&
        &   0,  21,  22,  23,  24/
        data  cpcres/&
        &   0,   0,   0,   0,   0,&
        &   2,   3,   4,   5,   1,&
        &   7,   8,   9,  10,   6,&
        &  12,  13,  14,  15,  11,&
        &  17,  18,  19,  20,  16,&
        &  22,  23,  24,  25,  21/
        data  cpvres/&
        &   0,   0,   0,   0,   0,&
        &   2,   3,   4,   5,   1,&
        &   7,   8,   9,  10,   6,&
        &  12,  13,  14,  15,  11,&
        &  17,  18,  19,  20,  16,&
        &  22,  23,  24,  25,  21/
        data  cmcres/&
        &   0,   0,   0,   0,   0,&
        &   5,   1,   2,   3,   4,&
        &  10,   6,   7,   8,   9,&
        &  15,  11,  12,  13,  14,&
        &  20,  16,  17,  18,  19,&
        &  25,  21,  22,  23,  24/
        data  cmvres/&
        &   0,   0,   0,   0,   0,&
        &   5,   1,   2,   3,   4,&
        &  10,   6,   7,   8,   9,&
        &  15,  11,  12,  13,  14,&
        &  20,  16,  17,  18,  19,&
        &  25,  21,  22,  23,  24/
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
	 eopc=eoshift(eoshift(src,k,dim=2), 1, dim=1)	! eoshift, positive, constant
	 eopv=eoshift(eoshift(src,k,dim=2), j, dim=1)	! eoshift, positive, variable
	 eomc=eoshift(eoshift(src,k,dim=2),-1, dim=1)	! eoshift, negative, constant
	 eomv=eoshift(eoshift(src,k,dim=2), k, dim=1)	! eoshift, negative, variable
	  cpc= cshift(eoshift(src,k,dim=2), 1, dim=1)	!  cshift, positive, constant
	  cpv= cshift(eoshift(src,k,dim=2), j, dim=1)	!  cshift, positive, variable
	  cmc= cshift(eoshift(src,k,dim=2),-1, dim=1)	!  cshift, negative, constant
	  cmv= cshift(eoshift(src,k,dim=2), k, dim=1)	!  cshift, negative, variable
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
	 j = 1
	 k = -1
	end
