! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s

! CHECK: !$OMP PARALLEL
! CHECK: !$OMP END PARALLEL
c$omp0parallel
c$omp0endparallel
      end
