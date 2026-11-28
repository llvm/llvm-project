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
    ! An INTEGER(8) result must not use lround: it returns a C long, which is
    ! 32 bits wide on an ILP32 target.
    ! CHECK: fir.call @llvm.llround.i64.f64
  end subroutine
  ! CHECK-LABEL: nint_test3
  subroutine nint_test3(i, a)
    integer(8) :: i
    real(4) :: a
    i = nint(a, 8)
    ! CHECK: fir.call @llvm.llround.i64.f32
  end subroutine
