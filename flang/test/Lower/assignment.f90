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

integer function negi(a)
  integer :: a
  negi = -a
end 

! CHECK-LABEL: func @_QPnegi(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}) -> i32 {
! CHECK:         %[[FCTRES:.*]] = fir.alloca i32 {bindc_name = "negi", uniq_name = "_QFnegiEnegi"}
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
! CHECK:         %[[C0:.*]] = arith.constant 0 : i32
! CHECK:         %[[NEG:.*]] = arith.subi %[[C0]], %[[A_VAL]] : i32
! CHECK:         fir.store %[[NEG]] to %[[FCTRES]] : !fir.ref<i32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<i32>
! CHECK:         return %[[RET]] : i32

real function negr(a)
  real :: a
  negr = -a
end 

! CHECK-LABEL: func @_QPnegr(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<f32> {fir.bindc_name = "a"}) -> f32 {
! CHECK:         %[[FCTRES:.*]] = fir.alloca f32 {bindc_name = "negr", uniq_name = "_QFnegrEnegr"}
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<f32>
! CHECK:         %[[NEG:.*]] = arith.negf %[[A_VAL]] : f32
! CHECK:         fir.store %[[NEG]] to %[[FCTRES]] : !fir.ref<f32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<f32>
! CHECK:         return %[[RET]] : f32

complex function negc(a)
  complex :: a
  negc = -a
end 

! CHECK-LABEL: func @_QPnegc(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.complex<4>> {fir.bindc_name = "a"}) -> !fir.complex<4> {
! CHECK:         %[[FCTRES:.*]] = fir.alloca !fir.complex<4> {bindc_name = "negc", uniq_name = "_QFnegcEnegc"}
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[NEG:.*]] = fir.negc %[[A_VAL]] : !fir.complex<4>
! CHECK:         fir.store %[[NEG]] to %[[FCTRES]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<!fir.complex<4>>
! CHECK:         return %[[RET]] : !fir.complex<4>
