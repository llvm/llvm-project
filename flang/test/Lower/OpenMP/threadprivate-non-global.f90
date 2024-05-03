! This test checks lowering of OpenMP Threadprivate Directive.
! Test for non-character non-SAVEd non-initialized scalars with or without
! allocatable or pointer attribute in main program.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

program test
  integer :: x
  real :: y
  logical :: z
  complex :: w
  integer, pointer :: a
  real, allocatable :: b

!CHECK-DAG:  %[[A:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %[[OMP_A:.*]] = omp.threadprivate %[[A]] : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %[[OMP_A_DECL:.*]]:2 = hlfir.declare %[[OMP_A]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK-DAG:  %[[B:.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:  %[[OMP_B:.*]] = omp.threadprivate %[[B]] : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:  %[[OMP_B_DECL:.*]]:2 = hlfir.declare %[[OMP_B]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEb"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
!CHECK-DAG:  %[[W:.*]] = fir.address_of(@_QFEw) : !fir.ref<!fir.complex<4>>
!CHECK-DAG:  %[[OMP_W:.*]] = omp.threadprivate %[[W]] : !fir.ref<!fir.complex<4>> -> !fir.ref<!fir.complex<4>>
!CHECK-DAG:  %[[OMP_W_DECL:.*]]:2 = hlfir.declare %[[OMP_W]] {uniq_name = "_QFEw"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
!CHECK-DAG:  %[[X:.*]] = fir.address_of(@_QFEx) : !fir.ref<i32>
!CHECK-DAG:  %[[OMP_X:.*]] = omp.threadprivate %[[X]] : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:  %[[OMP_X_DECL:.*]]:2 = hlfir.declare %[[OMP_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:  %[[Y:.*]] = fir.address_of(@_QFEy) : !fir.ref<f32>
!CHECK-DAG:  %[[OMP_Y:.*]] = omp.threadprivate %[[Y]] : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:  %[[OMP_Y_DECL:.*]]:2 = hlfir.declare %[[OMP_Y]] {uniq_name = "_QFEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK-DAG:  %[[Z:.*]] = fir.address_of(@_QFEz) : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %[[OMP_Z:.*]] = omp.threadprivate %[[Z]] : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %[[OMP_Z_DECL:.*]]:2 = hlfir.declare %[[OMP_Z]] {uniq_name = "_QFEz"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  !$omp threadprivate(x, y, z, w, a, b)

  call sub(a, b)

!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_X_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_Y_DECL]]#0 : !fir.ref<f32>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_Z_DECL]]#0 : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_W_DECL]]#0 : !fir.ref<!fir.complex<4>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_A_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_B_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  print *, x, y, z, w, a, b

  !$omp parallel
!CHECK-DAG:  %[[X_PVT:.*]] = omp.threadprivate %[[X]] : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:  %[[X_PVT_DECL:.*]]:2 = hlfir.declare %[[X_PVT]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:  %[[Y_PVT:.*]] = omp.threadprivate %[[Y]] : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:  %[[Y_PVT_DECL:.*]]:2 = hlfir.declare %[[Y_PVT]] {uniq_name = "_QFEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK-DAG:  %[[Z_PVT:.*]] = omp.threadprivate %[[Z]] : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %[[Z_PVT_DECL:.*]]:2 = hlfir.declare %[[Z_PVT]] {uniq_name = "_QFEz"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK-DAG:  %[[W_PVT:.*]] = omp.threadprivate %[[W]] : !fir.ref<!fir.complex<4>> -> !fir.ref<!fir.complex<4>>
!CHECK-DAG:  %[[W_PVT_DECL:.*]]:2 = hlfir.declare %[[W_PVT]] {uniq_name = "_QFEw"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
!CHECK-DAG:  %[[A_PVT:.*]] = omp.threadprivate %[[A]] : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %[[A_PVT_DECL:.*]]:2 = hlfir.declare %[[A_PVT]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK-DAG:  %[[B_PVT:.*]] = omp.threadprivate %[[B]] : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:  %[[B_PVT_DECL:.*]]:2 = hlfir.declare %[[B_PVT]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEb"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
!CHECK-DAG:  %{{.*}} = fir.load %[[X_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load %[[Y_PVT_DECL]]#0 : !fir.ref<f32>
!CHECK-DAG:  %{{.*}} = fir.load %[[Z_PVT_DECL]]#0 : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %{{.*}} = fir.load %[[W_PVT_DECL]]#0 : !fir.ref<!fir.complex<4>>
!CHECK-DAG:  %{{.*}} = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[B_PVT_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  print *, x, y, z, w, a, b
  !$omp end parallel

!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_X_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_Y_DECL]]#0 : !fir.ref<f32>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_Z_DECL]]#0 : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_W_DECL]]#0 : !fir.ref<!fir.complex<4>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_A_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_B_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  print *, x, y, z, w, a, b

!CHECK:  return

!CHECK-DAG: fir.global internal @_QFEa : !fir.box<!fir.ptr<i32>> {
!CHECK-DAG:   [[Z0:%.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK-DAG:   [[E0:%.*]] = fir.embox [[Z0]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK-DAG:   fir.has_value [[E0]] : !fir.box<!fir.ptr<i32>>
!CHECK-DAG: }
!CHECK-DAG: fir.global internal @_QFEb : !fir.box<!fir.heap<f32>> {
!CHECK-DAG:   [[Z1:%.*]] = fir.zero_bits !fir.heap<f32>
!CHECK-DAG:   [[E1:%.*]] = fir.embox [[Z1]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
!CHECK-DAG:   fir.has_value [[E1]] : !fir.box<!fir.heap<f32>>
!CHECK-DAG: }
!CHECK-DAG: fir.global internal @_QFEw : !fir.complex<4> {
!CHECK-DAG:   [[Z2:%.*]] = fir.undefined !fir.complex<4>
!CHECK-DAG:   fir.has_value [[Z2]] : !fir.complex<4>
!CHECK-DAG: }
!CHECK-DAG: fir.global internal @_QFEx : i32 {
!CHECK-DAG:   [[Z3:%.*]] = fir.undefined i32
!CHECK-DAG:   fir.has_value [[Z3]] : i32
!CHECK-DAG: }
!CHECK-DAG: fir.global internal @_QFEy : f32 {
!CHECK-DAG:   [[Z4:%.*]] = fir.undefined f32
!CHECK-DAG:   fir.has_value [[Z4]] : f32
!CHECK-DAG: }
!CHECK-DAG: fir.global internal @_QFEz : !fir.logical<4> {
!CHECK-DAG:   [[Z5:%.*]] = fir.undefined !fir.logical<4>
!CHECK-DAG:   fir.has_value [[Z5]] : !fir.logical<4>
!CHECK-DAG: }
end
