! RUN: %flang_fc1 -emit-fir -O0 -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -O3 -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! Allocatable descriptors crossing the workdistribute target fission are
! reloaded from their maps in the kernel, not cached by value.

! CHECK-LABEL: func @_QPsaxpy_allocatable
! CHECK:         omp.target_data
! CHECK-NOT:     !fir.box<{{.*}}name("__flang_workdistribute
! CHECK:         omp.target
! CHECK:         omp.teams
! CHECK:         omp.parallel
! CHECK:         omp.distribute
! CHECK:         omp.wsloop
! CHECK:         omp.loop_nest
subroutine saxpy_allocatable(a, x, y)
  real :: a
  real, allocatable :: x(:), y(:)

  !$omp target teams workdistribute
  y = a * x + y
  !$omp end target teams workdistribute
end subroutine saxpy_allocatable
