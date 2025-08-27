! RUN: not %flang_fc1 -fsyntax-only -fopenmp %s 2>&1 | FileCheck %s

!$omp  parallel
! CHECK: error: Expected OpenMP end directive
end
