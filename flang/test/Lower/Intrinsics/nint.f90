! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: nint_test1
subroutine nint_test1(i, a)
    integer :: i
    real :: a
    i = nint(a)
    ! CHECK: fir.call @llvm.lround.i32.f32
  end subroutine
  ! CHECK-LABEL: nint_test2
  subroutine nint_test2(i, a)
    integer(8) :: i
    real(8) :: a
    i = nint(a, 8)
    ! CHECK: fir.call @llvm.lround.i64.f64
  end subroutine
  