! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
! Ensure that copy-in/copy-out happens with specific ignore_tkr settings
module test
  interface
    subroutine pass_ignore_tkr(buf)
      implicit none
      !DIR$ IGNORE_TKR buf
      real :: buf
    end subroutine
    subroutine pass_ignore_tkr_2(buf)
      implicit none
      !DIR$ IGNORE_TKR(tkrdm) buf
      type(*) :: buf
    end subroutine
    subroutine pass_ignore_tkr_c(buf)
      implicit none
      !DIR$ IGNORE_TKR (tkrc) buf
      real :: buf
    end subroutine
    subroutine pass_ignore_tkr_c_2(buf)
      implicit none
      !DIR$ IGNORE_TKR (tkrcdm) buf
      type(*) :: buf
    end subroutine
    subroutine pass_intent_out(buf)
      implicit none
      integer, intent(out) :: buf(5)
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
  subroutine s3(buf)
!CHECK-LABEL: func.func @_QMtestPs3
!CHECK: hlfir.copy_in
!CHECK: fir.call @_QPpass_ignore_tkr_2
!CHECK: hlfir.copy_out
    real, intent(inout) :: buf(:)
    ! Create temp here
    call pass_ignore_tkr_2(buf)
  end subroutine
  subroutine s4(buf)
!CHECK-LABEL: func.func @_QMtestPs4
!CHECK-NOT: hlfir.copy_in
!CHECK: fir.call @_QPpass_ignore_tkr_c_2
!CHECK-NOT: hlfir.copy_out
    real, intent(inout) :: buf(:)
    ! Don't create temp here
    call pass_ignore_tkr_c_2(buf)
  end subroutine
  subroutine s5()
  ! TODO: pass_intent_out() has intent(out) dummy argument, so as such it
  ! should have copy-out, but not copy-in. Unfortunately, at the moment flang
  ! can only do copy-in/copy-out together. When this is fixed, this test should
  ! change from 'CHECK' for hlfir.copy_in to 'CHECK-NOT' for hlfir.copy_in
!CHECK-LABEL: func.func @_QMtestPs5
!CHECK: hlfir.copy_in
!CHECK: fir.call @_QPpass_intent_out
!CHECK: hlfir.copy_out
    implicit none
    integer, target :: x(10)
    integer, pointer :: p(:)
    p => x(::2) ! pointer to non-contiguous array section
    call pass_intent_out(p)
  end subroutine
end module
