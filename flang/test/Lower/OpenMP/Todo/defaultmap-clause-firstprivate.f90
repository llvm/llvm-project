!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

subroutine f00
    implicit none
    ! NOTE: This is implemented for scalars as it is the default behaviour, so we utilise
    ! a different data type.
    integer, allocatable :: i
    !CHECK: not yet implemented: Firstprivate is currently unsupported defaultmap behaviour
    !$omp target defaultmap(firstprivate)
      i = 10
    !$omp end target
  end
