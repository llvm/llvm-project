! A TARGET selected by a METADIRECTIVE follows the same TEAMS nesting rule as a
! directly written TARGET. Every runtime candidate must conform.

! RUN: not %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -cpp -o - %s 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -cpp -DRUNTIME -o - %s 2>&1 | FileCheck %s

! CHECK: error: {{.*}}TARGET construct with nested TEAMS region contains statements or directives outside of the TEAMS construct

subroutine selected_target_teams(flag, x)
  logical :: flag
  integer :: x
#ifdef RUNTIME
  !$omp metadirective when(user={condition(flag)}: target) otherwise(nothing)
#else
  !$omp metadirective when(implementation={vendor(llvm)}: target)
#endif
  block
#ifndef RUNTIME
    x = 1
#endif
    !$omp teams
    !$omp end teams
  end block
end subroutine
