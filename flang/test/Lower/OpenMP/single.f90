!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s

!===============================================================================
! Single construct
!===============================================================================

!CHECK-LABEL: func @_QPomp_single
!CHECK-SAME: (%[[X:.*]]: !fir.ref<i32> {fir.bindc_name = "x"})
subroutine omp_single(x)
  integer, intent(inout) :: x
  !CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFomp_singleEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  !CHECK: omp.parallel
  !$omp parallel
  !CHECK: omp.single
  !$omp single
    !CHECK: %[[xval:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
    !CHECK: %[[res:.*]] = arith.addi %[[xval]], %{{.*}} : i32
    !CHECK: hlfir.assign %[[res]] to %[[X_DECL]]#0 : i32, !fir.ref<i32>
    x = x + 12
  !CHECK: omp.terminator
  !$omp end single
  !CHECK: omp.terminator
  !$omp end parallel
end subroutine omp_single

!===============================================================================
! Single construct with nowait
!===============================================================================

!CHECK-LABEL: func @_QPomp_single_nowait
!CHECK-SAME: (%[[X:.*]]: !fir.ref<i32> {fir.bindc_name = "x"})
subroutine omp_single_nowait(x)
  integer, intent(inout) :: x
  !CHECK:   %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFomp_single_nowaitEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  !CHECK: omp.parallel
  !$omp parallel
  !CHECK: omp.single nowait
  !$omp single
    !CHECK: %[[xval:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
    !CHECK: %[[res:.*]] = arith.addi %[[xval]], %{{.*}} : i32
    !CHECK: hlfir.assign %[[res]] to %[[X_DECL]]#0 : i32, !fir.ref<i32>
    x = x + 12
  !CHECK: omp.terminator
  !$omp end single nowait
  !CHECK: omp.terminator
  !$omp end parallel
end subroutine omp_single_nowait

!===============================================================================
! Single construct with allocate
!===============================================================================

!CHECK-LABEL: func @_QPsingle_allocate
subroutine single_allocate()
  use omp_lib
  integer :: x
  !CHECK: omp.parallel {
  !$omp parallel
  !CHECK: omp.single allocate(%{{.+}} : i32 -> %{{.+}} : !fir.ref<i32>) {
  !$omp single allocate(omp_high_bw_mem_alloc: x) private(x)
  !CHECK: arith.addi
  x = x + 12
  !CHECK: omp.terminator
  !$omp end single
  !CHECK: omp.terminator
  !$omp end parallel
end subroutine single_allocate

!===============================================================================
! Single construct with private/firstprivate
!===============================================================================

! CHECK-LABEL: func.func @_QPsingle_privatization(
! CHECK-SAME:                                     %[[X:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}, 
! CHECK-SAME:                                     %[[Y:.*]]: !fir.ref<f64> {fir.bindc_name = "y"}) {
! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFsingle_privatizationEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFsingle_privatizationEy"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK:           omp.single   {
! CHECK:             %[[X_PVT:.*]] = fir.alloca f32 {bindc_name = "x", pinned, uniq_name = "_QFsingle_privatizationEx"}
! CHECK:             %[[X_PVT_DECL:.*]]:2 = hlfir.declare %[[X_PVT]] {uniq_name = "_QFsingle_privatizationEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:             %[[Y_PVT:.*]] = fir.alloca f64 {bindc_name = "y", pinned, uniq_name = "_QFsingle_privatizationEy"}
! CHECK:             %[[Y_PVT_DECL:.*]]:2 = hlfir.declare %[[Y_PVT]] {uniq_name = "_QFsingle_privatizationEy"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK:             %[[Y_LOAD:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<f64>
! CHECK:             hlfir.assign %[[Y_LOAD]] to %[[Y_PVT_DECL]]#0 temporary_lhs : f64, !fir.ref<f64>
! CHECK:             fir.call @_QPbar(%[[X_PVT_DECL]]#1, %[[Y_PVT_DECL]]#1) fastmath<contract> : (!fir.ref<f32>, !fir.ref<f64>) -> ()
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine single_privatization(x, y)
  real :: x
  real(8) :: y

  !$omp single private(x) firstprivate(y)
  call bar(x, y)
  !$omp end single
end subroutine

! CHECK-LABEL: func.func @_QPsingle_privatization2(
! CHECK-SAME:                                      %[[X:.*]]: !fir.ref<f32> {fir.bindc_name = "x"},
! CHECK-SAME:                                      %[[Y:.*]]: !fir.ref<f64> {fir.bindc_name = "y"}) {
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFsingle_privatization2Ex"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:         %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFsingle_privatization2Ey"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK:         omp.parallel   {
! CHECK:           omp.single   {
! CHECK:             %[[X_PVT:.*]] = fir.alloca f32 {bindc_name = "x", pinned, uniq_name = "_QFsingle_privatization2Ex"}
! CHECK:             %[[X_PVT_DECL:.*]]:2 = hlfir.declare %[[X_PVT]] {uniq_name = "_QFsingle_privatization2Ex"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:             %[[Y_PVT:.*]] = fir.alloca f64 {bindc_name = "y", pinned, uniq_name = "_QFsingle_privatization2Ey"}
! CHECK:             %[[Y_PVT_DECL:.*]]:2 = hlfir.declare %[[Y_PVT]] {uniq_name = "_QFsingle_privatization2Ey"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK:             %[[Y_LOAD:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<f64>
! CHECK:             hlfir.assign %[[Y_LOAD]] to %[[Y_PVT_DECL]]#0 temporary_lhs : f64, !fir.ref<f64>
! CHECK:             fir.call @_QPbar(%[[X_PVT_DECL]]#1, %[[Y_PVT_DECL]]#1) fastmath<contract> : (!fir.ref<f32>, !fir.ref<f64>) -> ()
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine single_privatization2(x, y)
  real :: x
  real(8) :: y

  !$omp parallel
  !$omp single private(x) firstprivate(y)
  call bar(x, y)
  !$omp end single
  !$omp end parallel
end subroutine
