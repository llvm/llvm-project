!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: @_QPomp_taskwait
subroutine omp_taskwait
  !CHECK: omp.taskwait
  !$omp taskwait
  !CHECK: fir.call @_QPfoo() {{.*}}: () -> ()
  call foo()
  !CHECK: omp.taskwait
  !$omp taskwait
end subroutine omp_taskwait
