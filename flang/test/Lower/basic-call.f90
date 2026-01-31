! RUN: bbc %s -o "-" -emit-hlfir | FileCheck %s

subroutine sub1()
end
! CHECK-LABEL: func @_QPsub1()

subroutine sub2()
  call sub1()
end

! CHECK-LABEL: func @_QPsub2()
! CHECK:         fir.call @_QPsub1() {{.*}}: () -> ()

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
! CHECK-DAG: %[[INT_VALUE:.*]]:3 = hlfir.associate %{{.*}} {adapt.valuebyref}
! CHECK-DAG: %[[REAL_VALUE:.*]]:3 = hlfir.associate %{{.*}} {adapt.valuebyref}
! CHECK:     fir.call @_QPsub3(%[[INT_VALUE]]#0, %[[REAL_VALUE]]#0) {{.*}}: (!fir.ref<i32>, !fir.ref<f32>) -> ()

subroutine call_fct1()
  real :: a, b, c
  c = fct1(a, b)
end

! CHECK-LABEL: func @_QPcall_fct1()
! CHECK:         %[[A:.*]] = fir.alloca f32 {bindc_name = "a", uniq_name = "_QFcall_fct1Ea"}
! CHECK:         %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] {{.*}}
! CHECK:         %[[B:.*]] = fir.alloca f32 {bindc_name = "b", uniq_name = "_QFcall_fct1Eb"}
! CHECK:         %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] {{.*}}
! CHECK:         %[[C:.*]] = fir.alloca f32 {bindc_name = "c", uniq_name = "_QFcall_fct1Ec"}
! CHECK:         %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] {{.*}}
! CHECK:         %[[RES:.*]] = fir.call @_QPfct1(%[[A_DECL]]#0, %[[B_DECL]]#0) {{.*}}: (!fir.ref<f32>, !fir.ref<f32>) -> f32
! CHECK:         hlfir.assign %[[RES]] to %[[C_DECL]]#0 : f32, !fir.ref<f32>
! CHECK:         return

! CHECK: func private @_QPfct1(!fir.ref<f32>, !fir.ref<f32>) -> f32
