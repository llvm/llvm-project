! RUN: mkdir -p %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=51 -module-dir %t %S/assumption.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -I %t -module-dir %t -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK5X
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -I %t -module-dir %t -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK60

submodule (assumes_module) assumes_implementation
contains
  ! CHECK-LABEL: func.func @_QMassumes_modulePseparate_work()
  ! CHECK-NOT: "omp_no_openmp"
  ! CHECK: return
  module procedure separate_work
  end procedure separate_work
end submodule assumes_implementation

submodule (use_parent) use_child_reload
  use scoped_assumes, only:
  implicit none
contains
  ! CHECK-LABEL: func.func @_QMuse_parentPimported_work()
  ! CHECK5X-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_parallelism"]]
  ! CHECK60-NOT: "llvm.assume"
  module procedure imported_work
  end procedure imported_work
end submodule use_child_reload

submodule (parallelism_module) parallelism_implementation
contains
  ! CHECK-LABEL: func.func @_QMparallelism_modulePseparate_parallel_work()
  ! CHECK-NOT: "llvm.assume"
  ! CHECK: return
  module procedure separate_parallel_work
  end procedure separate_parallel_work

  ! CHECK-LABEL: func.func @{{.*}}Plocal_parallel_work()
  ! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
  subroutine local_parallel_work()
    implicit none
    !$omp assumes no_parallelism
  end subroutine local_parallel_work
end submodule parallelism_implementation

! CHECK-LABEL: func.func @_QPuses_reloaded_parallelism()
! CHECK5X-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
! CHECK60-NOT: "llvm.assume"
subroutine uses_reloaded_parallelism()
  use reexport_parallelism, only: module_work
  implicit none
  ! CHECK: fir.call @_QMparallelism_modulePmodule_work()
  call module_work()
end subroutine uses_reloaded_parallelism
