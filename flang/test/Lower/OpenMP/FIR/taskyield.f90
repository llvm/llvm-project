!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefixes="FIRDialect,OMPDialect"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefixes="OMPDialect"

!FIRDialect-LABEL: @_QPomp_taskyield
subroutine omp_taskyield
  !OMPDialect: omp.taskyield
  !$omp taskyield
  !FIRDialect: fir.call @_QPfoo() {{.*}}: () -> ()
  call foo()
  !OMPDialect: omp.taskyield
  !$omp taskyield
end subroutine omp_taskyield
