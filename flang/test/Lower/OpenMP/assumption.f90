! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -module-dir %t -o - %t/bare.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK5X,SCOPE,SCOPE5X
! RUN: FileCheck %s --check-prefix=MODFILE --input-file=%t/assumes_module.mod
! RUN: FileCheck %s --check-prefix=USE-MODFILE --input-file=%t/use_parent-use_child.mod
! RUN: FileCheck %s --check-prefix=PARALLELISM-MODFILE --input-file=%t/parallelism_module.mod
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -module-dir %t -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK60,SCOPE,SCOPE60,CONDITIONAL
! RUN: %flang_fc1 -emit-hlfir -fopenmp-simd -fopenmp-version=51 -module-dir %t -o - %t/bare.f90 | FileCheck %s --check-prefix=SIMD
! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=51 -module-dir %t %t/bare.f90 | FileCheck %s --check-prefix=SYMBOLS
! RUN: FileCheck %s --check-prefix=TRUE-MODFILE --input-file=%t/conditional_true.mod
! RUN: FileCheck %s --check-prefix=FALSE-MODFILE --input-file=%t/conditional_false.mod
! RUN: %flang_fc1 -emit-hlfir -fopenmp-simd -fopenmp-version=60 -module-dir %t -o - %s | FileCheck %s --check-prefix=SIMD

! SIMD: module
! SIMD-NOT: "llvm.assume"

! SYMBOLS-DAG: assume_block (Subroutine): Subprogram
! SYMBOLS-DAG: assumes_module (OmpAssumesNoOpenmp): Module
! SYMBOLS-DAG: assumes_no_openmp (Subroutine, OmpAssumesNoOpenmp): Subprogram
! SYMBOLS-DAG: assumes_no_openmp_parallel (Subroutine, OmpAssumesNoOpenmp): Subprogram
! SYMBOLS-DAG: assumes_no_openmp_host (Subroutine, OmpAssumesNoOpenmp): Subprogram
! SYMBOLS-DAG: uses_assumes_module (Subroutine, OmpAssumesNoOpenmp): Subprogram
! SYMBOLS-DAG: assumes_no_parallelism (Subroutine, OmpAssumesNoParallelism): Subprogram
! SYMBOLS-DAG: parallelism_module (OmpAssumesNoParallelism): Module
! SYMBOLS-DAG: reexport_parallelism (OmpAssumesNoParallelism): Module
! SYMBOLS-DAG: uses_parallelism_module (Subroutine, OmpAssumesNoParallelism): Subprogram
! SYMBOLS-DAG: assumes_mixed (Subroutine, OmpAssumesNoOpenmp, OmpAssumesNoParallelism): Subprogram
! SYMBOLS-DAG: assumes_mixed_reverse (Subroutine, OmpAssumesNoOpenmp, OmpAssumesNoParallelism): Subprogram

!--- bare.f90
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
  !$omp assumes no_openmp no_parallelism
  interface
    module subroutine independent_work() bind(C, name="independent_work")
    end subroutine independent_work
  end interface
contains
  ! SCOPE-LABEL: func.func @_QMscoped_assumesPmodule_work()
  ! SCOPE-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_parallelism"]]
  subroutine module_work()
  end subroutine module_work
end module scoped_assumes

submodule (scoped_assumes) scoped_implementation
contains
  ! SCOPE-LABEL: func.func @independent_work()
  ! SCOPE-NOT: "llvm.assume"
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
! USE-MODFILE-NEXT: !$omp assumes no_parallelism
submodule (use_parent) use_child
  use scoped_assumes, only:
  implicit none
contains
  ! SCOPE-LABEL: func.func @_QMuse_parentPimported_work()
  ! SCOPE5X-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_parallelism"]]
  ! SCOPE60-NOT: "llvm.assume"
  module procedure imported_work
  end procedure imported_work
end submodule use_child

! CHECK-LABEL: func.func @_QPassumes_no_parallelism()
! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
subroutine assumes_no_parallelism()
  implicit none
  !$omp assumes no_parallelism
  ! CHECK: omp.barrier
  !$omp barrier
end subroutine assumes_no_parallelism

! PARALLELISM-MODFILE: module parallelism_module
! PARALLELISM-MODFILE: !$omp assumes no_parallelism
module parallelism_module
  implicit none
  !$omp assumes no_parallelism
  interface
    module subroutine separate_parallel_work()
    end subroutine separate_parallel_work
  end interface
contains
  ! CHECK-LABEL: func.func @_QMparallelism_modulePmodule_work()
  ! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
  subroutine module_work()
  end subroutine module_work
end module parallelism_module

module reexport_parallelism
  use parallelism_module, only: module_work
  implicit none
end module reexport_parallelism

! CHECK-LABEL: func.func @_QPuses_parallelism_module()
! CHECK5X-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
! CHECK60-NOT: "llvm.assume"
subroutine uses_parallelism_module()
  use parallelism_module, only:
  use reexport_parallelism, only: module_work
  implicit none
  ! CHECK: fir.call @_QMparallelism_modulePmodule_work()
  call module_work()
end subroutine uses_parallelism_module

! CHECK-LABEL: func.func @_QPassumes_mixed_import()
! CHECK5X-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_parallelism"]]
! CHECK60-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
subroutine assumes_mixed_import()
  use parallelism_module, only:
  implicit none
  !$omp assumes no_openmp
end subroutine assumes_mixed_import

! CHECK-LABEL: func.func @_QPassumes_mixed()
! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_parallelism"]]
subroutine assumes_mixed()
  implicit none
  !$omp assumes no_openmp no_parallelism
end subroutine assumes_mixed

! CHECK-LABEL: func.func @_QPassumes_mixed_reverse()
! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_parallelism"]]
subroutine assumes_mixed_reverse()
  implicit none
  !$omp assumes no_parallelism no_openmp
end subroutine assumes_mixed_reverse

! CHECK-LABEL: func.func @_QPassumes_parallelism_host()
! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
subroutine assumes_parallelism_host()
  implicit none
  !$omp assumes no_parallelism
  !$omp assumes no_parallelism
  call internal_work()
  call mixed_work()
contains
  ! CHECK-LABEL: func.func {{.*}}@_QFassumes_parallelism_hostPinternal_work()
  ! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
  subroutine internal_work()
  end subroutine internal_work

  ! CHECK-LABEL: func.func {{.*}}@_QFassumes_parallelism_hostPmixed_work()
  ! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_parallelism"]]
  subroutine mixed_work()
    implicit none
    !$omp assumes no_openmp
  end subroutine mixed_work
end subroutine assumes_parallelism_host

! CHECK-LABEL: func.func @_QPwithout_parallelism_assumption()
! CHECK-NOT: "llvm.assume"
! CHECK: return
subroutine without_parallelism_assumption()
end subroutine without_parallelism_assumption

! CHECK-LABEL: func.func @_QQmain()
! CHECK-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
program assumes_parallelism_program
  implicit none
  !$omp assumes no_parallelism
end program assumes_parallelism_program

!--- conditional.f90
! CONDITIONAL-LABEL: func.func @_QPconditional_literal_true()
! CONDITIONAL-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
subroutine conditional_literal_true()
  !$omp assumes no_parallelism(.true.)
  ! CONDITIONAL: omp.barrier
  !$omp barrier
end subroutine

! CONDITIONAL-LABEL: func.func @_QPconditional_literal_false()
! CONDITIONAL-NOT: "llvm.assume"
subroutine conditional_literal_false()
  !$omp assumes no_parallelism(.false.)
  ! CONDITIONAL: omp.parallel
  !$omp parallel
  ! CONDITIONAL: omp.barrier
  !$omp barrier
  !$omp end parallel
end subroutine

! CONDITIONAL-LABEL: func.func @_QPconditional_folded()
! CONDITIONAL-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
subroutine conditional_folded()
  !$omp assumes no_parallelism(enabled .and. .not. disabled)
  logical(kind=1), parameter :: enabled = .true., disabled = .false.
end subroutine

! CONDITIONAL-LABEL: func.func @_QPconditional_mixed_false()
! CONDITIONAL-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
subroutine conditional_mixed_false()
  !$omp assumes no_openmp no_parallelism(.false.)
end subroutine

! CONDITIONAL-LABEL: func.func @_QPconditional_mixed_true()
! CONDITIONAL-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_parallelism"]]
subroutine conditional_mixed_true()
  !$omp assumes no_parallelism(.true.) no_openmp
end subroutine

! CONDITIONAL-LABEL: func.func @_QPconditional_true_then_false()
! CONDITIONAL-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
subroutine conditional_true_then_false()
  !$omp assumes no_parallelism(.true.)
  !$omp assumes no_parallelism(.false.)
end subroutine

! CONDITIONAL-LABEL: func.func @_QPconditional_false_then_true()
! CONDITIONAL-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
subroutine conditional_false_then_true()
  !$omp assumes no_parallelism(.false.)
  !$omp assumes no_parallelism(.true.)
end subroutine

! CONDITIONAL-LABEL: func.func @_QPconditional_default_then_false()
! CONDITIONAL-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
subroutine conditional_default_then_false()
  !$omp assumes no_parallelism
  !$omp assumes no_parallelism(.false.)
end subroutine

module conditional_true
  logical :: enabled
  !$omp assumes no_parallelism(enabled)
  interface
    subroutine interface_condition()
      !$omp assumes no_parallelism(.false.)
    end subroutine
  end interface
  parameter (enabled = .true.)
contains
  ! CONDITIONAL-LABEL: func.func @_QMconditional_truePwork()
  ! CONDITIONAL-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
  subroutine work()
    !$omp assumes no_parallelism(.false.)
  end subroutine
end module

! TRUE-MODFILE: module conditional_true
! TRUE-MODFILE: !$omp assumes no_parallelism{{ *$}}

module conditional_false
  !$omp assumes no_parallelism(enabled)
  logical, parameter :: enabled = .false.
contains
  ! CONDITIONAL-LABEL: func.func @_QMconditional_falsePwork()
  ! CONDITIONAL-NOT: "llvm.assume"
  ! CONDITIONAL: return
  subroutine work()
  end subroutine
end module

! FALSE-MODFILE: module conditional_false
! FALSE-MODFILE-NOT: !$omp assumes
! FALSE-MODFILE: end

module conditional_host_shadowing
  logical, parameter :: enabled = .true.
contains
  ! CONDITIONAL-LABEL: func.func @_QMconditional_host_shadowingPlocal_condition()
  ! CONDITIONAL-NOT: "llvm.assume"
  ! CONDITIONAL: return
  subroutine local_condition()
    !$omp assumes no_parallelism(enabled)
    logical, parameter :: enabled = .false.
  end subroutine

  ! CONDITIONAL-LABEL: func.func @_QMconditional_host_shadowingPhost_condition()
  ! CONDITIONAL-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallelism"]]
  subroutine host_condition()
    !$omp assumes no_parallelism(enabled)
  end subroutine
end module
