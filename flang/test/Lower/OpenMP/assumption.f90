! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %t/source.f90 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/source.f90 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=51 -o - %t/source.f90 | FileCheck %s --check-prefix=LLVM
! RUN: %flang_fc1 -emit-llvm -fopenmp-simd -fopenmp-version=51 -o - %t/source.f90 | FileCheck %s --check-prefix=SIMD
! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=51 %t/source.f90 | FileCheck %s --check-prefix=SYMBOLS
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=51 -module-dir %t %t/source.f90
! RUN: FileCheck %s --check-prefix=MODFILE --input-file=%t/assumes_module.mod
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -I %t -module-dir %t -o - %t/submodule.f90 | FileCheck %s --check-prefix=SUBMODULE
! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=51 -I %t -module-dir %t -o - %t/submodule.f90 | FileCheck %s --check-prefix=SUBMODULE-LLVM

!--- source.f90
! SIMD: target triple =
! SIMD-NOT: "llvm.assume"="omp_no_openmp"

! SYMBOLS-DAG: assume_block (Subroutine): Subprogram
! SYMBOLS-DAG: assumes_module (OmpAssumesNoOpenmp): Module
! SYMBOLS-DAG: assumes_no_openmp (Subroutine, OmpAssumesNoOpenmp): Subprogram
! SYMBOLS-DAG: assumes_no_openmp_parallel (Subroutine, OmpAssumesNoOpenmp): Subprogram
! SYMBOLS-DAG: assumes_no_openmp_host (Subroutine, OmpAssumesNoOpenmp): Subprogram
! SYMBOLS-DAG: uses_assumes_module (Subroutine): Subprogram

! Lowering of the OpenMP ASSUME construct. The HOLDS clause asserts a scalar
! logical expression, which is lowered to an llvm.assume intrinsic. The block is
! always lowered.

! CHECK-LABEL: func.func @_QPassume_block
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

!--- submodule.f90
submodule (assumes_module) assumes_implementation
contains
  module procedure separate_work
  end procedure separate_work
end submodule assumes_implementation

subroutine uses_assumes_module_reload()
  use assumes_module, only: separate_work
  implicit none
  call separate_work()
end subroutine uses_assumes_module_reload

! MODFILE: module assumes_module
! MODFILE: !$omp assumes no_openmp

! SUBMODULE-LABEL: func.func @_QMassumes_modulePseparate_work()
! SUBMODULE-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp"]]
! SUBMODULE-LABEL: func.func @_QPuses_assumes_module_reload() {

! LLVM-LABEL: define void @assume_block_(
! LLVM-SAME: {{.*}}) {
! LLVM-LABEL: define void @_QMassumes_modulePset_value(
! LLVM-SAME: {{.*}}) #[[NO_OPENMP:[0-9]+]] {
! LLVM-LABEL: define void @assumes_no_openmp_(
! LLVM-SAME: {{.*}}) #[[NO_OPENMP]] {
! LLVM-LABEL: define void @assumes_no_openmp_parallel_(
! LLVM-SAME: ) #[[NO_OPENMP]] {
! LLVM: call void {{.*}}@__kmpc_fork_call
! LLVM-LABEL: define void @assumes_no_openmp_host_(
! LLVM-SAME: {{.*}}) #[[NO_OPENMP]] {
! LLVM-LABEL: define internal void @_QFassumes_no_openmp_hostPinternal_work(
! LLVM-SAME: {{.*}}) #[[NO_OPENMP]] {
! LLVM-LABEL: define void @uses_assumes_module_(
! LLVM-SAME: {{.*}}) {
! LLVM-LABEL: define void @assumes_no_openmp_local_(
! LLVM-SAME: ) #[[NO_OPENMP]] {
! LLVM-LABEL: declare void @external_helper_(ptr)
! LLVM: attributes #[[NO_OPENMP]] = { "llvm.assume"="omp_no_openmp" }

! SUBMODULE-LLVM-LABEL: define void @_QMassumes_modulePseparate_work()
! SUBMODULE-LLVM-SAME: #[[ATTR:[0-9]+]] {
! SUBMODULE-LLVM-LABEL: define void @uses_assumes_module_reload_() {
! SUBMODULE-LLVM: attributes #[[ATTR]] = { "llvm.assume"="omp_no_openmp" }
