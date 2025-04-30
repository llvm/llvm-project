!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! array-valued internal function whose result specification
! depends on the value of a host dummy and the host is a module routine
!
module f21260
contains
  subroutine host(xxx,ub)
  integer :: xxx, ub
  integer :: m
  m = xxx
  call internal()
  contains
    subroutine internal()
      call zzz(xxx)
! for the invocation of foo(), a temporary is allocated for the array result.
! Its upper bound references a host dummy argument.  At the point of call,
! there are two symbol table entries for host (an ST_PROC & ST_ENTRY); there
! are also two instances of the host dummy, xxx.  
      ub = ubound(foo(), 1)
    end subroutine
    function foo()
      integer, dimension(xxx) :: foo
      foo(xxx) = 3
    end function
  end subroutine
end module
  use f21260
  common /result/ires(2)
  integer :: expect(2) = (/3,3/)
  call host(3, iub)
!  print *, 'expect 3 ',iub
  ires(2) = iub
!  print *, ires
  call check(ires, expect, 2)
end
subroutine zzz(iii)
  common /result/ires(2)
!  print *, 'zzz ', iii
  ires(1) = iii
end
