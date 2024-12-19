! This test checks lowering of OpenMP Threadprivate Directive.
! Test for real, logical, complex, and derived type.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module test
  type my_type
    integer :: t_i
    real :: t_arr(5)
  end type my_type
  real :: x
  complex :: y
  logical :: z
  type(my_type) :: t

  !$omp threadprivate(x, y, z, t)

!CHECK-DAG: fir.global @_QMtestEt : !fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}> {
!CHECK-DAG: fir.global @_QMtestEx : f32 {
!CHECK-DAG: fir.global @_QMtestEy : complex<f32> {
!CHECK-DAG: fir.global @_QMtestEz : !fir.logical<4> {

contains
!CHECK-LABEL: func.func @_QMtestPsub
  subroutine sub()
!CHECK-DAG:  %[[T:.*]] = fir.address_of(@_QMtestEt) : !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>
!CHECK-DAG:  %[[T_DECL:.*]]:2 = hlfir.declare %[[T]] {uniq_name = "_QMtestEt"} : (!fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>) -> (!fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>, !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>)
!CHECK-DAG:  %[[OMP_T:.*]] = omp.threadprivate %[[T_DECL]]#1 : !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>> -> !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>
!CHECK-DAG:  %[[OMP_T_DECL:.*]]:2 = hlfir.declare %[[OMP_T]] {uniq_name = "_QMtestEt"} : (!fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>) -> (!fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>, !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>)
!CHECK-DAG:  %[[X:.*]] = fir.address_of(@_QMtestEx) : !fir.ref<f32>
!CHECK-DAG:  %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QMtestEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK-DAG:  %[[OMP_X:.*]] = omp.threadprivate %[[X_DECL]]#1 : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:  %[[OMP_X_DECL:.*]]:2 = hlfir.declare %[[OMP_X]] {uniq_name = "_QMtestEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK-DAG:  %[[Y:.*]] = fir.address_of(@_QMtestEy) : !fir.ref<complex<f32>>
!CHECK-DAG:  %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QMtestEy"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
!CHECK-DAG:  %[[OMP_Y:.*]] = omp.threadprivate %[[Y_DECL]]#1 : !fir.ref<complex<f32>> -> !fir.ref<complex<f32>>
!CHECK-DAG:  %[[OMP_Y_DECL:.*]]:2 = hlfir.declare %[[OMP_Y]] {uniq_name = "_QMtestEy"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
!CHECK-DAG:  %[[Z:.*]] = fir.address_of(@_QMtestEz) : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QMtestEz"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK-DAG:  %[[OMP_Z:.*]] = omp.threadprivate %[[Z_DECL]]#1 : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %[[OMP_Z_DECL:.*]]:2 = hlfir.declare %[[OMP_Z]] {uniq_name = "_QMtestEz"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_X_DECL]]#0 : !fir.ref<f32>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_Y_DECL]]#0 : !fir.ref<complex<f32>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_Z_DECL]]#0 : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %{{.*}} = hlfir.designate %[[OMP_T_DECL]]#0{"t_i"}   : (!fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>) -> !fir.ref<i32>
    print *, x, y, z, t%t_i

    !$omp parallel
!CHECK-DAG:  %[[T_PVT:.*]] = omp.threadprivate %[[T_DECL]]#1 : !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>> -> !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>
!CHECK-DAG:  %[[T_PVT_DECL:.*]]:2 = hlfir.declare %[[T_PVT]] {uniq_name = "_QMtestEt"} : (!fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>) -> (!fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>, !fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>)
!CHECK-DAG:  %[[X_PVT:.*]] = omp.threadprivate %[[X_DECL]]#1 : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:  %[[X_PVT_DECL:.*]]:2 = hlfir.declare %[[X_PVT]] {uniq_name = "_QMtestEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK-DAG:  %[[Y_PVT:.*]] = omp.threadprivate %[[Y_DECL]]#1 : !fir.ref<complex<f32>> -> !fir.ref<complex<f32>>
!CHECK-DAG:  %[[Y_PVT_DECL:.*]]:2 = hlfir.declare %[[Y_PVT]] {uniq_name = "_QMtestEy"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
!CHECK-DAG:  %[[Z_PVT:.*]] = omp.threadprivate %[[Z_DECL]]#1 : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %[[Z_PVT_DECL:.*]]:2 = hlfir.declare %[[Z_PVT]] {uniq_name = "_QMtestEz"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK-DAG:  %{{.*}} = fir.load %[[X_PVT_DECL]]#0 : !fir.ref<f32>
!CHECK-DAG:  %{{.*}} = fir.load %[[Y_PVT_DECL]]#0 : !fir.ref<complex<f32>>
!CHECK-DAG:  %{{.*}} = fir.load %[[Z_PVT_DECL]]#0 : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %{{.*}} = hlfir.designate %[[T_PVT_DECL]]#0{"t_i"}   : (!fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>) -> !fir.ref<i32>
print *, x, y, z, t%t_i
    !$omp end parallel
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_X_DECL]]#0 : !fir.ref<f32>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_Y_DECL]]#0 : !fir.ref<complex<f32>>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_Z_DECL]]#0 : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %{{.*}} = hlfir.designate %[[OMP_T_DECL]]#0{"t_i"}   : (!fir.ref<!fir.type<_QMtestTmy_type{t_i:i32,t_arr:!fir.array<5xf32>}>>) -> !fir.ref<i32>
    print *, x, y, z, t%t_i

  end
end
