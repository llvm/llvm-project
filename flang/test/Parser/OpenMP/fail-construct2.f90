! RUN: not %flang_fc1 -fsyntax-only -fopenmp %s 2>&1 | FileCheck %s

! CHECK: error: Invalid OpenMP directive
!$omp dummy
end
