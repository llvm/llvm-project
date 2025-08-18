! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
! Ensure that copy-in/copy-out happens with specific ignore_tkr settings
module test
  interface
    subroutine pass_ignore_tkr(buf)
      implicit none
      !DIR$ IGNORE_TKR buf
      real :: buf
    end subroutine
    subroutine pass_ignore_tkr_c(buf)
      implicit none
      !DIR$ IGNORE_TKR (tkrc) buf
      real :: buf
    end subroutine
  end interface
contains
  subroutine s1(buf)
!CHECK-LABEL: func.func @_QMtestPs1
!CHECK: hlfir.copy_in
!CHECK: fir.call @_QPpass_ignore_tkr
!CHECK: hlfir.copy_out
    real, intent(inout) :: buf(:)
    ! Create temp here
    call pass_ignore_tkr(buf)
  end subroutine
  subroutine s2(buf)
!CHECK-LABEL: func.func @_QMtestPs2
!CHECK-NOT: hlfir.copy_in
!CHECK: fir.call @_QPpass_ignore_tkr_c
!CHECK-NOT: hlfir.copy_out
    real, intent(inout) :: buf(:)
    ! Don't create temp here
    call pass_ignore_tkr_c(buf)
  end subroutine
end module
