! RUN: %flang_fc1  -fopenmp-version=51 -fopenmp -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
program main
  character(*), parameter :: message = "This is an error"
  !CHECK: !$OMP ERROR AT(COMPILATION) SEVERITY(WARNING) MESSAGE("some message here")
  !$omp error at(compilation) severity(warning) message("some message here")
  !CHECK: !$OMP ERROR AT(COMPILATION) SEVERITY(FATAL) MESSAGE(message)
  !$omp error at(compilation) severity(fatal) message(message)
  !CHECK: !$OMP ERROR AT(EXECUTION) SEVERITY(FATAL) MESSAGE(message)
  !$omp error at(EXECUTION) severity(fatal) message(message)
end program main
