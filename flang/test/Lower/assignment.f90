! RUN: bbc %s -o "-" -emit-fir | FileCheck %s

subroutine sub1(a)
  integer :: a
  a = 1
end

! CHECK-LABEL: func @_QPsub1(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32>
! CHECK:         %[[C1:.*]] = arith.constant 1 : i32
! CHECK:         fir.store %[[C1]] to %[[ARG0]] : !fir.ref<i32>

subroutine sub2(a, b)
  integer(4) :: a
  integer(8) :: b
  a = b
end

! CHECK-LABEL: func @_QPsub2(
! CHECK:         %[[A:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}
! CHECK:         %[[B:.*]]: !fir.ref<i64> {fir.bindc_name = "b"}
! CHECK:         %[[B_VAL:.*]] = fir.load %arg1 : !fir.ref<i64>
! CHECK:         %[[B_CONV:.*]] = fir.convert %[[B_VAL]] : (i64) -> i32
! CHECK:         fir.store %[[B_CONV]] to %[[A]] : !fir.ref<i32>
