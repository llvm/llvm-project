! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/no-parallelism.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/no-openmp-first.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/no-openmp-last.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/no-openmp-routines.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/absent-first.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/absent-last.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/contains.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/holds.f90 2>&1 | FileCheck %s

! CHECK: not yet implemented: assumption clauses on the assumes directive

!--- no-parallelism.f90
subroutine p
  !$omp assumes no_parallelism
end subroutine p

!--- no-openmp-first.f90
subroutine p
  !$omp assumes no_openmp no_parallelism
end subroutine p

!--- no-openmp-last.f90
subroutine p
  !$omp assumes no_parallelism no_openmp
end subroutine p

!--- absent-first.f90
subroutine p
  !$omp assumes absent(parallel) no_parallelism
end subroutine p

!--- absent-last.f90
subroutine p
  !$omp assumes no_parallelism absent(parallel)
end subroutine p

!--- no-openmp-routines.f90
subroutine p
  !$omp assumes no_openmp absent(parallel) no_openmp_routines
end subroutine p

!--- contains.f90
subroutine p
  !$omp assumes no_openmp absent(parallel) contains(task)
end subroutine p

!--- holds.f90
subroutine p
  !$omp assumes no_openmp absent(parallel) holds(.true.)
end subroutine p
