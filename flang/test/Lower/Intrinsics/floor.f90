! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: floor_test1
subroutine floor_test1(i, a)
    integer :: i
    real :: a
    i = floor(a)
    ! CHECK: %[[f:.*]] = math.floor %{{.*}} : f32
    ! CHECK: fir.convert %[[f]] : (f32) -> i32
  end subroutine
  ! CHECK-LABEL: floor_test2
  subroutine floor_test2(i, a)
    integer(8) :: i
    real :: a
    i = floor(a, 8)
    ! CHECK: %[[f:.*]] = math.floor %{{.*}} : f32
    ! CHECK: fir.convert %[[f]] : (f32) -> i64
  end subroutine
  
