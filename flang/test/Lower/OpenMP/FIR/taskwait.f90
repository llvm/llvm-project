!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefixes="FIRDialect,OMPDialect"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefixes="OMPDialect"

!FIRDialect-LABEL: @_QPomp_taskwait
subroutine omp_taskwait
  !OMPDialect: omp.taskwait
  !$omp taskwait
  !FIRDialect: fir.call @_QPfoo() {{.*}}: () -> ()
  call foo()
  !OMPDialect: omp.taskwait
  !$omp taskwait
end subroutine omp_taskwait
