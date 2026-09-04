!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

!CHECK-LABEL: @_QPomp_taskwait_depend
subroutine omp_taskwait_depend
  integer :: x
  !CHECK: omp.taskwait depend(taskdependin -> %{{.+}} : !fir.ref<i32>)
  !$omp taskwait depend(in: x)
end subroutine omp_taskwait_depend
