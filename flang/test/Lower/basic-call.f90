! RUN: bbc %s -o "-" -emit-fir | FileCheck %s

subroutine sub1()
end
! CHECK-LABEL: func @_QPsub1()

subroutine sub2()
  call sub1()
end

! CHECK-LABEL: func @_QPsub2()
! CHECK:         fir.call @_QPsub1() : () -> ()

subroutine sub3(a, b)
  integer :: a
  real :: b
end

! CHECK-LABEL: func @_QPsub3(
! CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:    %{{.*}}: !fir.ref<f32> {fir.bindc_name = "b"})

subroutine sub4()
  call sub3(2, 3.0)
end

! CHECK-LABEL: func @_QPsub4() {
! CHECK-DAG: %[[REAL_VALUE:.*]] = fir.alloca f32 {adapt.valuebyref}
! CHECK-DAG: %[[INT_VALUE:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK:     %[[C2:.*]] = arith.constant 2 : i32
! CHECK:     fir.store %[[C2]] to %[[INT_VALUE]] : !fir.ref<i32>
! CHECK:     %[[C3:.*]] = arith.constant 3.000000e+00 : f32
! CHECK:     fir.store %[[C3]] to %[[REAL_VALUE]] : !fir.ref<f32>
! CHECK:     fir.call @_QPsub3(%[[INT_VALUE]], %[[REAL_VALUE]]) : (!fir.ref<i32>, !fir.ref<f32>) -> ()

subroutine call_fct1()
  real :: a, b, c
  c = fct1(a, b)
end

! CHECK-LABEL: func @_QPcall_fct1()
! CHECK:         %[[A:.*]] = fir.alloca f32 {bindc_name = "a", uniq_name = "_QFcall_fct1Ea"}
! CHECK:         %[[B:.*]] = fir.alloca f32 {bindc_name = "b", uniq_name = "_QFcall_fct1Eb"}
! CHECK:         %[[C:.*]] = fir.alloca f32 {bindc_name = "c", uniq_name = "_QFcall_fct1Ec"}
! CHECK:         %[[RES:.*]] = fir.call @_QPfct1(%[[A]], %[[B]]) : (!fir.ref<f32>, !fir.ref<f32>) -> f32
! CHECK:         fir.store %[[RES]] to %[[C]] : !fir.ref<f32>
! CHECK:         return

! CHECK: func private @_QPfct1(!fir.ref<f32>, !fir.ref<f32>) -> f32
