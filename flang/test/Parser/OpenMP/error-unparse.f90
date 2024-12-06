! RUN: %flang_fc1  -fopenmp-version=51 -fopenmp -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
program main
  !CHECK: !$OMP ERROR AT(COMPILATION) SEVERITY(WARNING) MESSAGE("some message here")
  !$omp error at(compilation) severity(warning) message("some message here")
end program main
