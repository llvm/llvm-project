!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: @_QPomp_taskyield
subroutine omp_taskyield
  !CHECK: omp.taskyield
  !$omp taskyield
  !CHECK: fir.call @_QPfoo() {{.*}}: () -> ()
  call foo()
  !CHECK: omp.taskyield
  !$omp taskyield
end subroutine omp_taskyield
