! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

program test
    integer :: tmp
    !$omp target teams ompx_bare num_teams(42) thread_limit(43)
    tmp = 1
    !$omp end target teams
end program

! CHECK: omp.target ompx_bare
