! This test checks the lowering of OpenMP scope construct.

! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

! Module-level declarations emitted before any function body.

! CHECK-LABEL: omp.private {type = firstprivate} @_QFomp_scope_firstprivateEi_firstprivate_i32 : i32 copy {
! CHECK:         fir.load %{{.*}} : !fir.ref<i32>
! CHECK:         hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! CHECK:         omp.yield

! CHECK-LABEL: omp.declare_reduction @add_reduction_i32 : i32 init {
! CHECK:          arith.constant 0 : i32
! CHECK:          omp.yield
! CHECK-LABEL: } combiner {
! CHECK:          arith.addi
! CHECK:          omp.yield

! CHECK: omp.private {type = private} @_QFomp_scope_privateEi_private_i32 : i32

! CHECK-LABEL: func @_QPomp_scope_basic
subroutine omp_scope_basic()
  integer :: x
  ! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFomp_scope_basicEx"}
  x = 10

  ! CHECK: omp.scope {
  !$omp scope
  print *, "In scope", x
  ! CHECK: omp.terminator
  ! CHECK: }
  !$omp end scope
end subroutine

! CHECK-LABEL: func @_QPomp_scope_nowait
subroutine omp_scope_nowait()
  integer :: x
  ! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFomp_scope_nowaitEx"}
  x = 10

  ! CHECK: omp.scope nowait {
  !$omp scope
  print *, "In scope", x
  ! CHECK: omp.terminator
  ! CHECK: }
  !$omp end scope nowait
end subroutine

! CHECK-LABEL: func @_QPomp_scope_private
subroutine omp_scope_private()
  integer :: i
  ! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFomp_scope_privateEi"}
  i = 10

  ! CHECK: omp.scope private(@_QFomp_scope_privateEi_private_i32 %{{.*}}#0 -> %[[PRIV:.*]] : !fir.ref<i32>) {
  ! CHECK: %[[PDECL:.*]]:2 = hlfir.declare %[[PRIV]] {uniq_name = "_QFomp_scope_privateEi"}
  !$omp scope private(i)
  ! CHECK: fir.load %[[PDECL]]#0 : !fir.ref<i32>
  print *, "omp scope", i
  ! CHECK: omp.terminator
  ! CHECK: }
  !$omp end scope
end subroutine

! CHECK-LABEL: func @_QPomp_scope_reduction
subroutine omp_scope_reduction()
  integer :: sum
  ! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFomp_scope_reductionEsum"}
  sum = 0

  ! CHECK: omp.scope reduction(@add_reduction_i32 %{{.*}}#0 -> %[[REDUC:.*]] : !fir.ref<i32>) {
  ! CHECK: %[[RDECL:.*]]:2 = hlfir.declare %[[REDUC]] {uniq_name = "_QFomp_scope_reductionEsum"}
  ! CHECK: fir.load %[[RDECL]]#0 : !fir.ref<i32>
  ! CHECK: arith.addi %{{.*}}, %{{.*}} : i32
  ! CHECK: hlfir.assign %{{.*}} to %[[RDECL]]#0 : i32, !fir.ref<i32>
  !$omp scope reduction(+:sum)
  sum = sum + 1
  ! CHECK: omp.terminator
  ! CHECK: }
  !$omp end scope
end subroutine

! CHECK-LABEL: func @_QPomp_scope_firstprivate
subroutine omp_scope_firstprivate()
  integer :: i
  ! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFomp_scope_firstprivateEi"}
  i = 42

  ! CHECK: omp.scope private(@_QFomp_scope_firstprivateEi_firstprivate_i32 %{{.*}}#0 -> %[[FP:.*]] : !fir.ref<i32>) {
  ! CHECK: %[[FPDECL:.*]]:2 = hlfir.declare %[[FP]] {uniq_name = "_QFomp_scope_firstprivateEi"}
  !$omp scope firstprivate(i)
  ! CHECK: fir.load %[[FPDECL]]#0 : !fir.ref<i32>
  print *, "omp scope", i
  ! CHECK: omp.terminator
  ! CHECK: }
  !$omp end scope
end subroutine

! CHECK-LABEL: func @_QPomp_scope_allocate
subroutine omp_scope_allocate()
  integer :: i
  ! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFomp_scope_allocateEi"}
  i = 0

  ! CHECK: omp.scope allocate(%{{.*}} : i32 -> %{{.*}}#0 : !fir.ref<i32>) private(@_QFomp_scope_allocateEi_private_i32 %{{.*}}#0 -> %[[APRIV:.*]] : !fir.ref<i32>) {
  ! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[APRIV]] {uniq_name = "_QFomp_scope_allocateEi"}
  !$omp scope private(i) allocate(i)
  ! CHECK: hlfir.assign %{{.*}} to %[[ADECL]]#0 : i32, !fir.ref<i32>
  i = 1
  ! CHECK: omp.terminator
  ! CHECK: }
  !$omp end scope
end subroutine
