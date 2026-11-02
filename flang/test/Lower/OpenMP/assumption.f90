! RUN: mkdir -p %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -module-dir %t -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK5X,SCOPE,SCOPE5X
! RUN: FileCheck %s --check-prefix=MODFILE --input-file=%t/assumes_module.mod
! RUN: FileCheck %s --check-prefix=USE-MODFILE --input-file=%t/use_parent-use_child.mod
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -module-dir %t -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK60,SCOPE,SCOPE60
! RUN: %flang_fc1 -emit-hlfir -fopenmp-simd -fopenmp-version=51 -module-dir %t -o - %s | FileCheck %s --check-prefix=SIMD
! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=51 -module-dir %t %s | FileCheck %s --check-prefix=SYMBOLS

! SIMD: module
! SIMD-NOT: "omp_no_openmp"

! SYMBOLS-DAG: assume_block (Subroutine): Subprogram
! SYMBOLS-DAG: assumes_module (OmpAssumesNoOpenmp): Module
! SYMBOLS-DAG: assumes_no_openmp (Subroutine, OmpAssumesNoOpenmp): Subprogram
! SYMBOLS-DAG: assumes_no_openmp_parallel (Subroutine, OmpAssumesNoOpenmp): Subprogram
! SYMBOLS-DAG: assumes_no_openmp_host (Subroutine, OmpAssumesNoOpenmp): Subprogram
! SYMBOLS-DAG: uses_assumes_module (Subroutine, OmpAssumesNoOpenmp): Subprogram

! Lowering of the OpenMP ASSUME construct. The HOLDS clause asserts a scalar
! logical expression, which is lowered to an llvm.assume intrinsic. The block is
! always lowered.

! CHECK-LABEL: func.func @_QPassume_block
! CHECK-NOT: "omp_no_openmp"
subroutine assume_block(x)
  integer :: x
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare
  !$omp assume holds(x > 0)
  block
    ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0
    ! CHECK: %[[C0:.*]] = arith.constant 0 : i32
    ! CHECK: %[[CMP:.*]] = arith.cmpi sgt, %[[LOAD]], %[[C0]] : i32
    ! CHECK: llvm.intr.assume %[[CMP]] : i1
    ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
    ! CHECK: hlfir.assign %[[C2]] to %[[DECL]]#0
    x = 2
  end block
end subroutine assume_block

module assumes_module
  implicit none
  !$omp assumes no_openmp
  interface
    module subroutine separate_work()
    end subroutine separate_work
  end interface
contains
  ! CHECK-LABEL: func.func @_QMassumes_modulePset_value
  ! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
  subroutine set_value(x)
    integer :: x
    ! CHECK: %[[VALUE:.*]] = arith.constant 7 : i32
    ! CHECK: hlfir.assign %[[VALUE]] to %{{.*}} : i32, !fir.ref<i32>
    ! CHECK: return
    x = 7
  end subroutine set_value
end module assumes_module

! CHECK-LABEL: func.func @_QPassumes_no_openmp
! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
subroutine assumes_no_openmp(x)
  use assumes_module, only: set_value
  implicit none
  !$omp assumes no_openmp
  integer :: x
  ! CHECK: fir.call @_QMassumes_modulePset_value(%{{.*}}) {{.*}} : (!fir.ref<i32>) -> ()
  ! CHECK: fir.call @_QPexternal_helper(%{{.*}}) {{.*}} : (!fir.ref<i32>) -> ()
  ! CHECK: return
  call set_value(x)
  call external_helper(x)
end subroutine assumes_no_openmp

! Compile-only: executing this region would violate the assumption.
! CHECK-LABEL: func.func @_QPassumes_no_openmp_parallel
! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
subroutine assumes_no_openmp_parallel
  implicit none
  !$omp assumes no_openmp
  ! CHECK: omp.parallel {
  !$omp parallel
    ! CHECK: omp.barrier
    !$omp barrier
    ! CHECK: omp.terminator
    ! CHECK-NEXT: }
  !$omp end parallel
  ! CHECK: return
end subroutine assumes_no_openmp_parallel

! CHECK-LABEL: func.func @_QPassumes_no_openmp_host
! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
subroutine assumes_no_openmp_host(x)
  implicit none
  !$omp assumes no_openmp
  integer :: x
  call internal_work(x)
contains
  ! CHECK-LABEL: func.func {{.*}}@_QFassumes_no_openmp_hostPinternal_work
  ! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
  subroutine internal_work(value)
    integer :: value
    value = 9
  end subroutine internal_work
end subroutine assumes_no_openmp_host

! CHECK-LABEL: func.func @_QPuses_assumes_module
! CHECK5X-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
! CHECK60-NOT: "omp_no_openmp"
subroutine uses_assumes_module(x)
  use assumes_module, only: set_value
  implicit none
  integer :: x
  ! CHECK: fir.call @_QMassumes_modulePset_value(%{{.*}}) {{.*}} : (!fir.ref<i32>) -> ()
  ! CHECK: return
  call set_value(x)
end subroutine uses_assumes_module

! CHECK-LABEL: func.func @_QPassumes_no_openmp_local
! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
subroutine assumes_no_openmp_local
  use assumes_module, only: set_value
  implicit none
  !$omp assumes no_openmp
  integer :: x
  ! CHECK: fir.call @_QMassumes_modulePset_value(%{{.*}}) {{.*}} : (!fir.ref<i32>) -> ()
  ! CHECK: return
  call set_value(x)
end subroutine assumes_no_openmp_local

! MODFILE: module assumes_module
! MODFILE: !$omp assumes no_openmp

module scoped_assumes
  implicit none
  !$omp assumes no_openmp
  interface
    module subroutine independent_work() bind(C, name="independent_work")
    end subroutine independent_work
  end interface
contains
  ! SCOPE-LABEL: func.func @_QMscoped_assumesPmodule_work()
  ! SCOPE-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
  subroutine module_work()
  end subroutine module_work
end module scoped_assumes

submodule (scoped_assumes) scoped_implementation
contains
  ! SCOPE-LABEL: func.func @independent_work()
  ! SCOPE-NOT: "omp_no_openmp"
  module procedure independent_work
    ! SCOPE: omp.parallel {
    !$omp parallel
      ! SCOPE: omp.barrier
      !$omp barrier
    !$omp end parallel
  end procedure independent_work

  ! SCOPE-LABEL: func.func @{{.*}}Plocal_work()
  ! SCOPE-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
  subroutine local_work()
    implicit none
    !$omp assumes no_openmp
  end subroutine local_work
end submodule scoped_implementation

module use_parent
  interface
    module subroutine imported_work()
    end subroutine imported_work
  end interface
end module use_parent

! USE-MODFILE: !$omp assumes no_openmp
submodule (use_parent) use_child
  use scoped_assumes, only:
  implicit none
contains
  ! SCOPE-LABEL: func.func @_QMuse_parentPimported_work()
  ! SCOPE5X-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
  ! SCOPE60-NOT: "omp_no_openmp"
  module procedure imported_work
  end procedure imported_work
end submodule use_child
