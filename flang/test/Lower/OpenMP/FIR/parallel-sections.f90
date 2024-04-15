! REQUIRES: openmp_runtime

!RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s --check-prefixes="FIRDialect,OMPDialect"
!RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | fir-opt --cfg-conversion-on-func-opt | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefixes="OMPDialect,LLVMDialect"

!===============================================================================
! Parallel sections construct
!===============================================================================

!FIRDialect: func @_QPomp_parallel_sections
subroutine omp_parallel_sections(x, y)
  integer, intent(inout) :: x, y
  !OMPDialect: omp.parallel {
  !OMPDialect: omp.sections {
  !$omp parallel sections
    !OMPDialect: omp.section {
    !$omp section
      !FIRDialect: fir.load
      !FIRDialect: arith.addi
      !FIRDialect: fir.store
      x = x + 12
      !OMPDialect: omp.terminator
    !OMPDialect: omp.section {
    !$omp section
      !FIRDialect: fir.load
      !FIRDialect: arith.subi
      !FIRDialect: fir.store
      y = y - 5
      !OMPDialect: omp.terminator
  !OMPDialect: omp.terminator
  !OMPDialect: omp.terminator
  !$omp end parallel sections
end subroutine omp_parallel_sections

!===============================================================================
! Parallel sections construct with allocate clause
!===============================================================================

!FIRDialect: func @_QPomp_parallel_sections
subroutine omp_parallel_sections_allocate(x, y)
  use omp_lib
  integer, intent(inout) :: x, y
  !FIRDialect: %[[allocator_1:.*]] = arith.constant 4 : i64
  !FIRDialect: %[[allocator_2:.*]] = arith.constant 4 : i64
  !LLVMDialect: %[[allocator_1:.*]] = llvm.mlir.constant(4 : i64) : i64
  !LLVMDialect: %[[allocator_2:.*]] = llvm.mlir.constant(4 : i64) : i64
  !OMPDialect: omp.parallel allocate(
  !FIRDialect: %[[allocator_2]] : i64 -> %{{.*}} : !fir.ref<i32>) {
  !LLVMDialect: %[[allocator_2]] : i64 -> %{{.*}} : !llvm.ptr) {
  !OMPDialect: omp.sections allocate(
  !FIRDialect: %[[allocator_1]] : i64 -> %{{.*}} : !fir.ref<i32>) {
  !LLVMDialect: %[[allocator_1]] : i64 -> %{{.*}} : !llvm.ptr) {
  !$omp parallel sections allocate(omp_high_bw_mem_alloc: x)
    !OMPDialect: omp.section {
    !$omp section
      x = x + 12
      !OMPDialect: omp.terminator
    !OMPDialect: omp.section {
    !$omp section
      y = y + 5
      !OMPDialect: omp.terminator
  !OMPDialect: omp.terminator
  !OMPDialect: omp.terminator
  !$omp end parallel sections
end subroutine omp_parallel_sections_allocate
