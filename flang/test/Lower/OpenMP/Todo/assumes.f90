! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/no-openmp-routines.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/absent.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/contains.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/holds.f90 2>&1 | FileCheck %s

! CHECK: not yet implemented: assumption clauses on the assumes directive

!--- no-openmp-routines.f90
subroutine p
  !$omp assumes no_openmp no_parallelism no_openmp_routines
end subroutine p

!--- absent.f90
subroutine p
  !$omp assumes no_openmp no_parallelism absent(parallel)
end subroutine p

!--- contains.f90
subroutine p
  !$omp assumes no_openmp no_parallelism contains(barrier)
end subroutine p

!--- holds.f90
subroutine p
  !$omp assumes no_openmp no_parallelism holds(.true.)
end subroutine p
