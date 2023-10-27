!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s
!RUN: bbc -emit-fir -fopenmp %s -o - | FileCheck %s

!===============================================================================
! Single construct
!===============================================================================

!CHECK-LABEL: func @_QPomp_single
!CHECK-SAME: (%[[x:.*]]: !fir.ref<i32> {fir.bindc_name = "x"})
subroutine omp_single(x)
  integer, intent(inout) :: x
  !CHECK: omp.parallel
  !$omp parallel
  !CHECK: omp.single
  !$omp single
    !CHECK: %[[xval:.*]] = fir.load %[[x]] : !fir.ref<i32>
    !CHECK: %[[res:.*]] = arith.addi %[[xval]], %{{.*}} : i32
    !CHECK: fir.store %[[res]] to %[[x]] : !fir.ref<i32>
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
!CHECK-SAME: (%[[x:.*]]: !fir.ref<i32> {fir.bindc_name = "x"})
subroutine omp_single_nowait(x)
  integer, intent(inout) :: x
  !CHECK: omp.parallel
  !$omp parallel
  !CHECK: omp.single nowait
  !$omp single
    !CHECK: %[[xval:.*]] = fir.load %[[x]] : !fir.ref<i32>
    !CHECK: %[[res:.*]] = arith.addi %[[xval]], %{{.*}} : i32
    !CHECK: fir.store %[[res]] to %[[x]] : !fir.ref<i32>
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
! CHECK-SAME:                                     %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "x"},
! CHECK-SAME:                                     %[[VAL_1:.*]]: !fir.ref<f64> {fir.bindc_name = "y"}) {
! CHECK:         omp.single   {
! CHECK:           %[[VAL_2:.*]] = fir.alloca f32 {bindc_name = "x", pinned, uniq_name = "_QFsingle_privatizationEx"}
! CHECK:           %[[VAL_3:.*]] = fir.alloca f64 {bindc_name = "y", pinned, uniq_name = "_QFsingle_privatizationEy"}
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<f64>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_3]] : !fir.ref<f64>
! CHECK:           fir.call @_QPbar(%[[VAL_2]], %[[VAL_3]]) {{.*}}: (!fir.ref<f32>, !fir.ref<f64>) -> ()
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
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "x"},
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.ref<f64> {fir.bindc_name = "y"}) {
! CHECK:         omp.parallel   {
! CHECK:           omp.single   {
! CHECK:             %[[VAL_2:.*]] = fir.alloca f32 {bindc_name = "x", pinned, uniq_name = "_QFsingle_privatization2Ex"}
! CHECK:             %[[VAL_3:.*]] = fir.alloca f64 {bindc_name = "y", pinned, uniq_name = "_QFsingle_privatization2Ey"}
! CHECK:             %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<f64>
! CHECK:             fir.store %[[VAL_4]] to %[[VAL_3]] : !fir.ref<f64>
! CHECK:             fir.call @_QPbar(%[[VAL_2]], %[[VAL_3]]) {{.*}}: (!fir.ref<f32>, !fir.ref<f64>) -> ()
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
