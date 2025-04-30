! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test array parameter in contained subprogram
! was failing with -Mipa
!
program p
 interface
  subroutine t1(n)
    integer :: n(*)
  end subroutine
 end interface

  integer :: rr(8)
  integer :: ex(8)
  data ex/97,98,99,100,97,98,99,100/
  call t1( rr )
  call check(rr,ex,8)
  !print *,rr
end

subroutine t1(n)
  integer, intent(inout) :: n(*)
  call t2( n(1), 0, 1 )
  call t2( n(2), 1, 1 )
  call t2( n(3), 2, 1 )
  call t2( n(4), 3, 1 )
  call t2( n(5), 0, 7 )
  call t2( n(6), 1, 7 )
  call t2( n(7), 2, 7 )
  call t2( n(8), 3, 7 )
contains
  subroutine t2( r, n, m )
    integer :: r, n, m
    character(len=8),parameter,dimension(0:3):: hm = &
      & (/'aaaaaaaa','bbbbbbbb','cccccccc','dddddddd'/)
    r = iachar( hm(n)(m:m) )
  end subroutine
end subroutine

