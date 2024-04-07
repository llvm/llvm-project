! REQUIRES: openmp_runtime

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!===============================================================================
! Parallel sections construct
!===============================================================================

!CHECK: func @_QPomp_parallel_sections
subroutine omp_parallel_sections(x, y)
  integer, intent(inout) :: x, y
  !CHECK: omp.parallel {
  !CHECK: omp.sections {
  !$omp parallel sections
    !CHECK: omp.section {
    !$omp section
      !CHECK: fir.load
      !CHECK: arith.addi
      !CHECK: hlfir.assign
      x = x + 12
      !CHECK: omp.terminator
    !CHECK: omp.section {
    !$omp section
      !CHECK: fir.load
      !CHECK: arith.subi
      !CHECK: hlfir.assign
      y = y - 5
      !CHECK: omp.terminator
  !CHECK: omp.terminator
  !CHECK: omp.terminator
  !$omp end parallel sections
end subroutine omp_parallel_sections

!===============================================================================
! Parallel sections construct with allocate clause
!===============================================================================

!CHECK: func @_QPomp_parallel_sections
subroutine omp_parallel_sections_allocate(x, y)
  use omp_lib
  integer, intent(inout) :: x, y
  !CHECK: %[[allocator_1:.*]] = arith.constant 4 : i64
  !CHECK: %[[allocator_2:.*]] = arith.constant 4 : i64
  !CHECK: omp.parallel allocate(
  !CHECK: %[[allocator_2]] : i64 -> %{{.*}} : !fir.ref<i32>) {
  !CHECK: omp.sections allocate(
  !CHECK: %[[allocator_1]] : i64 -> %{{.*}} : !fir.ref<i32>) {
  !$omp parallel sections allocate(omp_high_bw_mem_alloc: x)
    !CHECK: omp.section {
    !$omp section
      x = x + 12
      !CHECK: omp.terminator
    !CHECK: omp.section {
    !$omp section
      y = y + 5
      !CHECK: omp.terminator
  !CHECK: omp.terminator
  !CHECK: omp.terminator
  !$omp end parallel sections
end subroutine omp_parallel_sections_allocate
