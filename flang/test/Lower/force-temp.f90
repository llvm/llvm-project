! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
! Ensure that we still create copy_in/copy_out for non-contiguous input,
! despite having IGNORE_TKR.
! 
module test
  implicit none(type, external)
contains
  subroutine pass_ignore_tkr(buf, n)
    implicit none
    !DIR$ IGNORE_TKR buf
    real, intent(inout) :: buf(n)
    integer, intent(in) :: n
  end subroutine
  
  subroutine s1()

!CHECK-LABEL: func.func @_QMtestPs1()
!CHECK: hlfir.copy_in
!CHECK: fir.call @_QMtestPpass_ignore_tkr
!CHECK: hlfir.copy_out

  integer :: x(5)
  x = [1,2,3,4,5]
  ! Non-contiguous input
  call pass_ignore_tkr(x(1::2), size(x(1::2)))

  end subroutine s1

  subroutine s2()

!CHECK-LABEL: func.func @_QMtestPs2()
!CHECK-NOT: hlfir.copy_in
!CHECK: fir.call @_QMtestPpass_ignore_tkr
!CHECK-NOT: hlfir.copy_out

  integer :: x(5)
  x = [1,2,3,4,5]
  ! Contiguous input
  call pass_ignore_tkr(x(1:3), size(x(1:3)))

  end subroutine s2
end module test


end program
