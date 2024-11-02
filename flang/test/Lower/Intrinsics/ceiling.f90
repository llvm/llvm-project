! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: ceiling_test1
subroutine ceiling_test1(i, a)
    integer :: i
    real :: a
    i = ceiling(a)
    ! CHECK: %[[f:.*]] = math.ceil %{{.*}} : f32
    ! CHECK: fir.convert %[[f]] : (f32) -> i32
  end subroutine
  ! CHECK-LABEL: ceiling_test2
  subroutine ceiling_test2(i, a)
    integer(8) :: i
    real :: a
    i = ceiling(a, 8)
    ! CHECK: %[[f:.*]] = math.ceil %{{.*}} : f32
    ! CHECK: fir.convert %[[f]] : (f32) -> i64
  end subroutine
  
  
