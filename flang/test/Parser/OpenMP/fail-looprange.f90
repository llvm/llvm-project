! RUN: not %flang_fc1 -fsyntax-only -fopenmp %s 2>&1 | FileCheck %s

! CHECK: error:
!$omp fuse looprange

! CHECK: error:
!$omp fuse looprange(1)

! CHECK: error:
!$omp fuse looprange(1,2,3)
end
