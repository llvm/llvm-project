! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!  array constructors of character and assumed-length character
!
module stopwatch

 interface watch
  module procedure watch_a, watch_s
 end interface

 integer n
 integer result(100)
contains

 subroutine create()
  call watch(clock=(/"cpu ","user","sys ","wall"/))
 end subroutine

 subroutine watch_a(clock)
  character(len=*), intent(in), dimension(:) :: clock
  !print *,'in watch_a, character len is ', len(clock(1))
  n = n + 1
  result(n) = len(clock(1))
  do i = 1,ubound(clock,1)
   !print *, i, '->', clock(i)
   n = n + 1
   result(n) = iachar(clock(i)(1:1))
  enddo
 end subroutine

 subroutine watch_s(clock)
  character(len=*), optional, intent(in) :: clock
  if (present(clock)) then
   call watch_a((/clock/))
  else
   call watch_a((/"none"/))
  end if
 end subroutine
end module

 use stopwatch
 integer expect(15)
 data expect/4,99,117,115,119,4,110,10,117,4,100,98,97,102,-1/

 n = 0
 call create
 call watch_s()
 call watch_s('ugly clock')
 call watch_a( (/ 'dont','be','a','fool' /) )
 !print *,n
 !print *,result(1:n)
 n = n + 1
 result(n) = -1
 call check(result,expect,15)
end
