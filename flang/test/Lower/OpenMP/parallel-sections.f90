!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefixes="FIRDialect,OMPDialect"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --cfg-conversion | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefixes="OMPDialect,LLVMDialect"

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
  !FIRDialect: %[[allocator_1:.*]] = arith.constant 1 : i32
  !FIRDialect: %[[allocator_2:.*]] = arith.constant 1 : i32
  !LLVMDialect: %[[allocator_1:.*]] = llvm.mlir.constant(1 : i32) : i32
  !LLVMDialect: %[[allocator_2:.*]] = llvm.mlir.constant(1 : i32) : i32
  !OMPDialect: omp.parallel allocate(
  !FIRDialect: %[[allocator_2]] : i32 -> %{{.*}} : !fir.ref<i32>) {
  !LLVMDialect: %[[allocator_2]] : i32 -> %{{.*}} : !llvm.ptr<i32>) {
  !OMPDialect: omp.sections allocate(
  !FIRDialect: %[[allocator_1]] : i32 -> %{{.*}} : !fir.ref<i32>) {
  !LLVMDialect: %[[allocator_1]] : i32 -> %{{.*}} : !llvm.ptr<i32>) {
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
