! This test checks the lowering of OpenMP scope construct inside a target region.

! RUN: bbc -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

! Module-level declarations emitted before any function body.

! CHECK: omp.private {type = private} @_QFtarget_scope_allocateEi_private_i32 : i32

! CHECK-LABEL: omp.private {type = firstprivate} @_QFtarget_scope_firstprivateEi_firstprivate_i32 : i32 copy {
! CHECK:         fir.load %{{.*}} : !fir.ref<i32>
! CHECK:         hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! CHECK:         omp.yield

! CHECK-LABEL: omp.declare_reduction @add_reduction_i32 : i32 init {
! CHECK:          arith.constant 0 : i32
! CHECK:          omp.yield
! CHECK-LABEL: } combiner {
! CHECK:          arith.addi
! CHECK:          omp.yield

! CHECK: omp.private {type = private} @_QFtarget_scope_privateEi_private_i32 : i32

! CHECK-LABEL: func @_QPtarget_scope_basic
subroutine target_scope_basic()
  integer :: x
  x = 10

  !$omp target
    ! CHECK: omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32)
    ! CHECK: omp.target map_entries(%{{.*}} -> %[[XARG:.*]] : !fir.ref<i32>) {
    ! CHECK:   hlfir.declare %[[XARG]] {uniq_name = "_QFtarget_scope_basicEx"}
    ! CHECK:   omp.scope {
    !$omp scope
    x = x + 1
    ! CHECK:   omp.terminator
    ! CHECK:   }
    !$omp end scope
  !$omp end target
end subroutine

! CHECK-LABEL: func @_QPtarget_scope_nowait
subroutine target_scope_nowait()
  integer :: x
  x = 10

  !$omp target
    ! CHECK: omp.target map_entries(%{{.*}} -> %[[XARG:.*]] : !fir.ref<i32>) {
    ! CHECK:   hlfir.declare %[[XARG]] {uniq_name = "_QFtarget_scope_nowaitEx"}
    ! CHECK:   omp.scope nowait {
    !$omp scope
    x = x + 1
    ! CHECK:   omp.terminator
    ! CHECK:   }
    !$omp end scope nowait
  !$omp end target
end subroutine

! CHECK-LABEL: func @_QPtarget_scope_private
subroutine target_scope_private()
  integer :: i
  i = 0

  !$omp target
    ! CHECK: omp.target map_entries(%{{.*}} -> %[[IARG:.*]] : !fir.ref<i32>) {
    ! CHECK:   hlfir.declare %[[IARG]] {uniq_name = "_QFtarget_scope_privateEi"}
    ! CHECK:   omp.scope private(@_QFtarget_scope_privateEi_private_i32 %{{.*}}#0 -> %[[PRIV:.*]] : !fir.ref<i32>) {
    ! CHECK:     %[[PDECL:.*]]:2 = hlfir.declare %[[PRIV]] {uniq_name = "_QFtarget_scope_privateEi"}
    !$omp scope private(i)
    ! CHECK:     hlfir.assign %{{.*}} to %[[PDECL]]#0 : i32, !fir.ref<i32>
    i = 42
    ! CHECK:   omp.terminator
    ! CHECK:   }
    !$omp end scope
  !$omp end target
end subroutine

! CHECK-LABEL: func @_QPtarget_scope_reduction
subroutine target_scope_reduction()
  integer :: sum
  sum = 0

  !$omp target
    ! CHECK: omp.target map_entries(%{{.*}} -> %[[SARG:.*]] : !fir.ref<i32>) {
    ! CHECK:   hlfir.declare %[[SARG]] {uniq_name = "_QFtarget_scope_reductionEsum"}
    ! CHECK:   omp.scope reduction(@add_reduction_i32 %{{.*}}#0 -> %[[REDUC:.*]] : !fir.ref<i32>) {
    ! CHECK:     %[[RDECL:.*]]:2 = hlfir.declare %[[REDUC]] {uniq_name = "_QFtarget_scope_reductionEsum"}
    !$omp scope reduction(+:sum)
    ! CHECK:     fir.load %[[RDECL]]#0 : !fir.ref<i32>
    ! CHECK:     hlfir.assign %{{.*}} to %[[RDECL]]#0 : i32, !fir.ref<i32>
    sum = sum + 1
    ! CHECK:   omp.terminator
    ! CHECK:   }
    !$omp end scope
  !$omp end target
end subroutine

! CHECK-LABEL: func @_QPtarget_scope_firstprivate
subroutine target_scope_firstprivate()
  integer :: i
  i = 42

  !$omp target
    ! CHECK: omp.target map_entries(%{{.*}} -> %[[IARG:.*]] : !fir.ref<i32>) {
    ! CHECK:   hlfir.declare %[[IARG]] {uniq_name = "_QFtarget_scope_firstprivateEi"}
    ! CHECK:   omp.scope private(@_QFtarget_scope_firstprivateEi_firstprivate_i32 %{{.*}}#0 -> %[[FP:.*]] : !fir.ref<i32>) {
    ! CHECK:     %[[FPDECL:.*]]:2 = hlfir.declare %[[FP]] {uniq_name = "_QFtarget_scope_firstprivateEi"}
    !$omp scope firstprivate(i)
    ! CHECK:     fir.load %[[FPDECL]]#0 : !fir.ref<i32>
    ! CHECK:     hlfir.assign %{{.*}} to %[[FPDECL]]#0 : i32, !fir.ref<i32>
    i = i + 1
    ! CHECK:   omp.terminator
    ! CHECK:   }
    !$omp end scope
  !$omp end target
end subroutine

! CHECK-LABEL: func @_QPtarget_scope_allocate
subroutine target_scope_allocate()
  integer :: i
  i = 0

  !$omp target
    ! CHECK: omp.target map_entries(%{{.*}} -> %[[IARG:.*]] : !fir.ref<i32>) {
    ! CHECK:   hlfir.declare %[[IARG]] {uniq_name = "_QFtarget_scope_allocateEi"}
    ! CHECK:   omp.scope allocate(%{{.*}} : i32 -> %{{.*}}#0 : !fir.ref<i32>) private(@_QFtarget_scope_allocateEi_private_i32 %{{.*}}#0 -> %[[APRIV:.*]] : !fir.ref<i32>) {
    ! CHECK:     %[[ADECL:.*]]:2 = hlfir.declare %[[APRIV]] {uniq_name = "_QFtarget_scope_allocateEi"}
    !$omp scope private(i) allocate(i)
    ! CHECK:     hlfir.assign %{{.*}} to %[[ADECL]]#0 : i32, !fir.ref<i32>
    i = 1
    ! CHECK:   omp.terminator
    ! CHECK:   }
    !$omp end scope
  !$omp end target
end subroutine
