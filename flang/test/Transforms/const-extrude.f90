! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --const-extruder-opt | FileCheck %s

subroutine sub1(x,y)
  implicit none
  integer x, y
  
  call sub2(0.0d0, 1.0d0, x, y, 1)
end subroutine sub1

!CHECK-LABEL: func.func @_QPsub1
!CHECK-SAME: [[ARG0:%.*]]: !fir.ref<i32> {{{.*}}},
!CHECK-SAME: [[ARG1:%.*]]: !fir.ref<i32> {{{.*}}}) {
!CHECK: [[X:%.*]] = fir.declare [[ARG0]] {{.*}}
!CHECK: [[Y:%.*]] = fir.declare [[ARG1]] {{.*}}
!CHECK: [[CONST_R0:%.*]] = fir.address_of([[EXTR_0:@.*]]) : !fir.ref<f64>
!CHECK: [[CONST_R1:%.*]] = fir.address_of([[EXTR_1:@.*]]) : !fir.ref<f64>
!CHECK: [[CONST_I:%.*]] = fir.address_of([[EXTR_2:@.*]]) : !fir.ref<i32>
!CHECK: fir.call @_QPsub2([[CONST_R0]], [[CONST_R1]], [[X]], [[Y]], [[CONST_I]])
!CHECK: return

!CHECK: fir.global internal [[EXTR_0]] constant : f64 {
!CHECK: %{{.*}} = arith.constant 0.000000e+00 : f64
!CHECK: fir.has_value %{{.*}} : f64
!CHECK: }
!CHECK: fir.global internal [[EXTR_1]] constant : f64 {
!CHECK: %{{.*}} = arith.constant 1.000000e+00 : f64
!CHECK: fir.has_value %{{.*}} : f64
!CHECK: }
!CHECK: fir.global internal [[EXTR_2]] constant : i32 {
!CHECK: %{{.*}} = arith.constant 1 : i32
!CHECK: fir.has_value %{{.*}} : i32
!CHECK: }
