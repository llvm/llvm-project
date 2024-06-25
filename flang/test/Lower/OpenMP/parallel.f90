! REQUIRES: openmp_runtime

!RUN: %flang_fc1 -emit-hlfir %openmp_flags %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPparallel_simple
subroutine parallel_simple()
   !CHECK: omp.parallel
!$omp parallel
   !CHECK: fir.call
   call f1()
!$omp end parallel
end subroutine parallel_simple

!===============================================================================
! `if` clause
!===============================================================================

!CHECK-LABEL: func @_QPparallel_if
subroutine parallel_if(alpha, beta, gamma)
   integer, intent(in) :: alpha
   logical, intent(in) :: beta
   logical(1) :: logical1
   logical(2) :: logical2
   logical(4) :: logical4
   logical(8) :: logical8

   !CHECK: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(alpha .le. 0)
   !CHECK: fir.call
   call f1()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(.false.)
   !CHECK: fir.call
   call f2()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(alpha .ge. 0)
   !CHECK: fir.call
   call f3()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(.true.)
   !CHECK: fir.call
   call f4()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(beta)
   !CHECK: fir.call
   call f1()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(logical1)
   !CHECK: fir.call
   call f1()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(logical2)
   !CHECK: fir.call
   call f1()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(logical4)
   !CHECK: fir.call
   call f1()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(logical8)
   !CHECK: fir.call
   call f1()
   !CHECK: omp.terminator
   !$omp end parallel

end subroutine parallel_if

!===============================================================================
! `num_threads` clause
!===============================================================================

!CHECK-LABEL: func @_QPparallel_numthreads
subroutine parallel_numthreads(num_threads)
   integer, intent(inout) :: num_threads

   !CHECK: omp.parallel num_threads(%{{.*}}: i32) {
   !$omp parallel num_threads(16)
   !CHECK: fir.call
   call f1()
   !CHECK: omp.terminator
   !$omp end parallel

   num_threads = 4

   !CHECK: omp.parallel num_threads(%{{.*}} : i32) {
   !$omp parallel num_threads(num_threads)
   !CHECK: fir.call
   call f2()
   !CHECK: omp.terminator
   !$omp end parallel

end subroutine parallel_numthreads

!===============================================================================
! `proc_bind` clause
!===============================================================================

!CHECK-LABEL: func @_QPparallel_proc_bind
subroutine parallel_proc_bind()

   !CHECK: omp.parallel proc_bind(master) {
   !$omp parallel proc_bind(master)
   !CHECK: fir.call
   call f1()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel proc_bind(close) {
   !$omp parallel proc_bind(close)
   !CHECK: fir.call
   call f2()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel proc_bind(spread) {
   !$omp parallel proc_bind(spread)
   !CHECK: fir.call
   call f3()
   !CHECK: omp.terminator
   !$omp end parallel

end subroutine parallel_proc_bind

!===============================================================================
! `allocate` clause
!===============================================================================

!CHECK-LABEL: func @_QPparallel_allocate
subroutine parallel_allocate()
   use omp_lib
   integer :: x
   !CHECK: omp.parallel allocate(
   !CHECK: %{{.+}} : i64 -> %{{.+}} : !fir.ref<i32>
   !CHECK: ) {
   !$omp parallel allocate(omp_high_bw_mem_alloc: x) private(x)
   !CHECK: arith.addi
   x = x + 12
   !CHECK: omp.terminator
   !$omp end parallel
end subroutine parallel_allocate

!===============================================================================
! multiple clauses
!===============================================================================

!CHECK-LABEL: func @_QPparallel_multiple_clauses
subroutine parallel_multiple_clauses(alpha, num_threads)
   use omp_lib
   integer, intent(inout) :: alpha
   integer, intent(in) :: num_threads

   !CHECK: omp.parallel if({{.*}} : i1) proc_bind(master) {
   !$omp parallel if(alpha .le. 0) proc_bind(master)
   !CHECK: fir.call
   call f1()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel num_threads({{.*}} : i32) proc_bind(close) {
   !$omp parallel proc_bind(close) num_threads(num_threads)
   !CHECK: fir.call
   call f2()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel if({{.*}} : i1) num_threads({{.*}} : i32) {
   !$omp parallel num_threads(num_threads) if(alpha .le. 0)
   !CHECK: fir.call
   call f3()
   !CHECK: omp.terminator
   !$omp end parallel

   !CHECK: omp.parallel if({{.*}} : i1) num_threads({{.*}} : i32) allocate(
   !CHECK: %{{.+}} : i64 -> %{{.+}} : !fir.ref<i32>
   !CHECK: ) {
   !$omp parallel num_threads(num_threads) if(alpha .le. 0) allocate(omp_high_bw_mem_alloc: alpha) private(alpha)
   !CHECK: fir.call
   call f3()
   !CHECK: arith.addi
   alpha = alpha + 12
   !CHECK: omp.terminator
   !$omp end parallel

end subroutine parallel_multiple_clauses
