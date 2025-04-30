! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This has failed from time to time:
!  assumed-shape, assumed-length character array
!
 program main
  interface
   subroutine chfunc( aslc )
   character*(*) aslc(:)
   end subroutine
  end interface
  integer r,e
 
  character*5 c(4)
 
  c(1) = 'abcde'
  c(2) = 'fghij'
  c(3) = 'klmno'
  c(4) = 'pqrst'
 
  call chfunc( c )     ! call subroutine
  r = 1
  e = 1
  call check(r,e,1)
 end program
 
 subroutine chfunc( aslc )
  character*(*) aslc(:)
  do i = 1,ubound(aslc,1)
   print *,aslc(i)     ! error appears when compiling this ref to aslc
  enddo
 end subroutine

