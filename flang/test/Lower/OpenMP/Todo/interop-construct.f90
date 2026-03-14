! REQUIRES: openmp_runtime
! RUN: %not_todo_cmd %flang_fc1 -emit-llvm %openmp_flags -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMPInteropConstruct
program interop_test
  use omp_lib
  integer(omp_interop_kind) :: obj
  !$omp interop init(targetsync,target: obj)
end program interop_test
