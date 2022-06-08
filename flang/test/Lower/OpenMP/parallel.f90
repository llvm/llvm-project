!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefixes="FIRDialect,OMPDialect"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefixes="LLVMDialect,OMPDialect"

!FIRDialect-LABEL: func @_QPparallel_simple
subroutine parallel_simple()
   !OMPDialect: omp.parallel
!$omp parallel
   !FIRDialect: fir.call
   call f1()
!$omp end parallel
end subroutine parallel_simple

!===============================================================================
! `if` clause
!===============================================================================

!FIRDialect-LABEL: func @_QPparallel_if
subroutine parallel_if(alpha, beta, gamma)
   integer, intent(in) :: alpha
   logical, intent(in) :: beta
   logical(1) :: logical1
   logical(2) :: logical2
   logical(4) :: logical4
   logical(8) :: logical8

   !OMPDialect: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(alpha .le. 0)
   !FIRDialect: fir.call
   call f1()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(.false.)
   !FIRDialect: fir.call
   call f2()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(alpha .ge. 0)
   !FIRDialect: fir.call
   call f3()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(.true.)
   !FIRDialect: fir.call
   call f4()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(beta)
   !FIRDialect: fir.call
   call f1()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(logical1)
   !FIRDialect: fir.call
   call f1()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(logical2)
   !FIRDialect: fir.call
   call f1()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(logical4)
   !FIRDialect: fir.call
   call f1()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel if(%{{.*}} : i1) {
   !$omp parallel if(logical8)
   !FIRDialect: fir.call
   call f1()
   !OMPDialect: omp.terminator
   !$omp end parallel

end subroutine parallel_if

!===============================================================================
! `num_threads` clause
!===============================================================================

!FIRDialect-LABEL: func @_QPparallel_numthreads
subroutine parallel_numthreads(num_threads)
   integer, intent(inout) :: num_threads

   !OMPDialect: omp.parallel num_threads(%{{.*}}: i32) {
   !$omp parallel num_threads(16)
   !FIRDialect: fir.call
   call f1()
   !OMPDialect: omp.terminator
   !$omp end parallel

   num_threads = 4

   !OMPDialect: omp.parallel num_threads(%{{.*}} : i32) {
   !$omp parallel num_threads(num_threads)
   !FIRDialect: fir.call
   call f2()
   !OMPDialect: omp.terminator
   !$omp end parallel

end subroutine parallel_numthreads

!===============================================================================
! `proc_bind` clause
!===============================================================================

!FIRDialect-LABEL: func @_QPparallel_proc_bind
subroutine parallel_proc_bind()

   !OMPDialect: omp.parallel proc_bind(master) {
   !$omp parallel proc_bind(master)
   !FIRDialect: fir.call
   call f1()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel proc_bind(close) {
   !$omp parallel proc_bind(close)
   !FIRDialect: fir.call
   call f2()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel proc_bind(spread) {
   !$omp parallel proc_bind(spread)
   !FIRDialect: fir.call
   call f3()
   !OMPDialect: omp.terminator
   !$omp end parallel

end subroutine parallel_proc_bind

!===============================================================================
! `allocate` clause
!===============================================================================

!FIRDialect-LABEL: func @_QPparallel_allocate
subroutine parallel_allocate()
   use omp_lib
   integer :: x
   !OMPDialect: omp.parallel allocate(
   !FIRDialect: %{{.+}} : i32 -> %{{.+}} : !fir.ref<i32>
   !LLVMDialect: %{{.+}} : i32 -> %{{.+}} : !llvm.ptr<i32>
   !OMPDialect: ) {
   !$omp parallel allocate(omp_high_bw_mem_alloc: x) private(x)
   !FIRDialect: arith.addi
   x = x + 12
   !OMPDialect: omp.terminator
   !$omp end parallel
end subroutine parallel_allocate

!===============================================================================
! multiple clauses
!===============================================================================

!FIRDialect-LABEL: func @_QPparallel_multiple_clauses
subroutine parallel_multiple_clauses(alpha, num_threads)
   use omp_lib
   integer, intent(inout) :: alpha
   integer, intent(in) :: num_threads

   !OMPDialect: omp.parallel if({{.*}} : i1) proc_bind(master) {
   !$omp parallel if(alpha .le. 0) proc_bind(master)
   !FIRDialect: fir.call
   call f1()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel num_threads({{.*}} : i32) proc_bind(close) {
   !$omp parallel proc_bind(close) num_threads(num_threads)
   !FIRDialect: fir.call
   call f2()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel if({{.*}} : i1) num_threads({{.*}} : i32) {
   !$omp parallel num_threads(num_threads) if(alpha .le. 0)
   !FIRDialect: fir.call
   call f3()
   !OMPDialect: omp.terminator
   !$omp end parallel

   !OMPDialect: omp.parallel if({{.*}} : i1) num_threads({{.*}} : i32) allocate(
   !FIRDialect: %{{.+}} : i32 -> %{{.+}} : !fir.ref<i32>
   !LLVMDialect: %{{.+}} : i32 -> %{{.+}} : !llvm.ptr<i32>
   !OMPDialect: ) {
   !$omp parallel num_threads(num_threads) if(alpha .le. 0) allocate(omp_high_bw_mem_alloc: alpha) private(alpha)
   !FIRDialect: fir.call
   call f3()
   !FIRDialect: arith.addi
   alpha = alpha + 12
   !OMPDialect: omp.terminator
   !$omp end parallel

end subroutine parallel_multiple_clauses
