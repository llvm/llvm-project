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
  ! CHECK5X-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
  ! CHECK60-NOT: "omp_no_openmp"
  module procedure imported_work
  end procedure imported_work
end submodule use_child_reload
