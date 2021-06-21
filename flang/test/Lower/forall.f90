! Test forall lowering
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPfoo(
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.array<200xf32>>,
! CHECK-SAME: %[[mask:.*]]: !fir.ref<!fir.array<200x!fir.logical<4>>>)
subroutine foo(x, mask)
  logical :: mask(200)
  real :: x(200)
  ! CHECK: %[[ivar:.*]] = fir.alloca i32 {uniq_name = "i"}
  ! CHECK: %[[c1:.*]] = fir.convert %c1{{.*}} : (i32) -> index
  ! CHECK: %[[c200:.*]] = fir.convert %c100{{.*}} : (i32) -> index
  ! CHECK: fir.do_loop %[[i:.*]] = %[[c1]] to %[[c200]] step %c1{{.*}} unordered {
  ! CHECK:   %[[icast:.*]] = fir.convert %[[i]] : (index) -> i32
  ! CHECK:   fir.store %[[icast]] to %[[ivar]] : !fir.ref<i32>
  ! CHECK:   %[[iload:.*]] = fir.load %[[ivar]] : !fir.ref<i32>
  ! CHECK:   %[[icast2:.*]] = fir.convert %[[iload]] : (i32) -> i64
  ! CHECK:   %[[offset:.*]] = subi %[[icast2]], %c1{{.*}} : i64
  ! CHECK:   %[[coor:.*]] = fir.coordinate_of %[[mask]], %[[offset]] : (!fir.ref<!fir.array<200x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
  ! CHECK:   %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.logical<4>>
  ! CHECK:   %[[maskval:.*]] = fir.convert %[[load]] : (!fir.logical<4>) -> i1
  ! CHECK:   fir.if %[[maskval]] {
  ! CHECK:     %[[cst:.*]] = constant 1.000000e+00 : f32
  ! CHECK:     %[[iload2:.*]] = fir.load %[[ivar]] : !fir.ref<i32>
  ! CHECK:     %[[icast2:.*]] = fir.convert %[[iload2]] : (i32) -> i64
  ! CHECK:     %[[offset2:.*]] = subi %[[icast2]], %c1{{.*}} : i64
  ! CHECK:     %[[xcoor:.*]] = fir.coordinate_of %[[x]], %[[offset2]] : (!fir.ref<!fir.array<200xf32>>, i64) -> !fir.ref<f32>
  ! CHECK:     fir.store %[[cst]] to %[[xcoor]] : !fir.ref<f32>
  ! CHECK:   }
  ! CHECK: }
  forall (i=1:100,mask(i)) x(i) = 1.
end subroutine

