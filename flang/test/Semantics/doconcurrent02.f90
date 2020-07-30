! when the loops are not DO CONCURRENT

! RUN: not %f18 -funparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK-NOT: image control statement not allowed in DO CONCURRENT
! CHECK-NOT: RETURN not allowed in DO CONCURRENT
! CHECK-NOT: call to impure procedure in DO CONCURRENT not allowed
! CHECK-NOT: IEEE_GET_FLAG not allowed in DO CONCURRENT
! CHECK-NOT: ADVANCE specifier not allowed in DO CONCURRENT
! CHECK-NOT: SYNC ALL
! CHECK-NOT: SYNC IMAGES

module ieee_exceptions
  interface
     subroutine ieee_get_flag(i, j)
       integer :: i, j
     end subroutine ieee_get_flag
  end interface
end module ieee_exceptions

subroutine do_concurrent_test1(i,n)
  implicit none
  integer :: i, n
  do 10 i = 1,n
     SYNC ALL
     SYNC IMAGES (*)
     return
10 continue
end subroutine do_concurrent_test1

subroutine do_concurrent_test2(i,j,n,flag)
  use ieee_exceptions
  implicit none
  integer :: i, j, n, flag, flag2
  do i = 1,n
    change team (j)
      call ieee_get_flag(flag, flag2)
    end team
    write(*,'(a35)',advance='no')
  end do
end subroutine do_concurrent_test2
