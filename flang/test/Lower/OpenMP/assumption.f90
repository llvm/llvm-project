! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -module-dir %t -o - %t/source.f90 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -module-dir %t -o - %t/source.f90 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp-simd -fopenmp-version=51 -module-dir %t -o - %t/source.f90 | FileCheck %s --check-prefix=SIMD
! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=51 -module-dir %t %t/source.f90 | FileCheck %s --check-prefix=SYMBOLS
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=51 -module-dir %t %t/source.f90
! RUN: FileCheck %s --check-prefix=MODFILE --input-file=%t/assumes_module.mod
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -I %t -module-dir %t -o - %t/submodule.f90 | FileCheck %s --check-prefix=SUBMODULE
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -I %t -module-dir %t -o - %t/submodule.f90 | FileCheck %s --check-prefix=SUBMODULE
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -module-dir %t -o - %t/versioned-scope.f90 | FileCheck %s --check-prefix=SCOPE
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -module-dir %t -o - %t/versioned-scope.f90 | FileCheck %s --check-prefix=SCOPE
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -module-dir %t -o - %t/absent.f90 | FileCheck %s --check-prefix=ABSENT
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -module-dir %t -o - %t/absent.f90 | FileCheck %s --check-prefix=ABSENT
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -module-dir %t -o - %t/absent.f90 | FileCheck %s --check-prefix=ABSENT
! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=52 -module-dir %t -o - %t/absent.f90 | FileCheck %s --check-prefix=LLVM
! RUN: %flang_fc1 -emit-llvm -fopenmp-simd -fopenmp-version=52 -module-dir %t -o - %t/absent.f90 | FileCheck %s --check-prefix=ABSENT-SIMD
! RUN: FileCheck %s --check-prefix=ABSENT-MODFILE --input-file=%t/absent_module.mod
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -I %t -module-dir %t -o - %t/absent-submodule.f90 | FileCheck %s --check-prefix=RELOAD
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -I %t -module-dir %t -o - %t/absent-submodule.f90 | FileCheck %s --check-prefix=RELOAD
! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=52 -I %t -module-dir %t -o - %t/absent-submodule.f90 | FileCheck %s --check-prefix=RELOAD-LLVM

!--- source.f90
! SIMD: module
! SIMD-NOT: "omp_no_openmp"

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
! CHECK-NOT: llvm.passthrough
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
! CHECK-NOT: "omp_no_openmp"
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

! MODFILE: module assumes_module
! MODFILE: !$omp assumes no_openmp

! SUBMODULE-LABEL: func.func @_QMassumes_modulePseparate_work()
! SUBMODULE-NOT: "omp_no_openmp"
! SUBMODULE: return

!--- versioned-scope.f90
module scoped_assumes
  implicit none
  !$omp assumes no_openmp absent(parallel, barrier)
  interface
    module subroutine independent_work() bind(C, name="independent_work")
    end subroutine independent_work
  end interface
contains
  ! SCOPE-LABEL: func.func @_QMscoped_assumesPmodule_work()
  ! SCOPE-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_barrier,omp_no_parallel"]]
  subroutine module_work()
  end subroutine module_work
end module scoped_assumes

submodule (scoped_assumes) scoped_implementation
contains
  ! SCOPE-LABEL: func.func @independent_work()
  ! SCOPE-NOT: llvm.passthrough
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

!--- absent.f90
! ABSENT-LABEL: func.func @_QPassumes_absent
! ABSENT-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_barrier,omp_no_parallel"]]
subroutine assumes_absent(x)
  implicit none
  !$omp assumes absent(parallel, barrier)
  integer :: x
  ! ABSENT: hlfir.assign
  x = 7
end subroutine assumes_absent

module absent_module
  implicit none
  !$omp assumes no_openmp absent(parallel, barrier)
  interface
    module subroutine separate_absent_work() bind(C, name="separate_absent_work")
    end subroutine separate_absent_work
  end interface
contains
  ! ABSENT-LABEL: func.func @_QMabsent_modulePmodule_work()
  ! ABSENT-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_barrier,omp_no_parallel"]]
  subroutine module_work()
  end subroutine module_work

  ! ABSENT-LABEL: func.func @_QMabsent_modulePlocal_work()
  ! ABSENT-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_barrier,omp_no_critical,omp_no_parallel"]]
  subroutine local_work()
    !$omp assumes absent(critical)
  end subroutine local_work
end module absent_module

! ABSENT-LABEL: func.func @_QPassumes_host()
! ABSENT-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_parallel"]]
subroutine assumes_host()
  !$omp assumes absent(parallel)
  call internal_work()
contains
  ! ABSENT-LABEL: func.func {{.*}}@_QFassumes_hostPinternal_work()
  ! ABSENT-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_openmp,omp_no_barrier,omp_no_parallel"]]
  subroutine internal_work()
    !$omp assumes absent(parallel, barrier) no_openmp
    !$omp assumes absent(barrier)
  end subroutine internal_work
end subroutine assumes_host

! ABSENT-LABEL: func.func @_QPassumes_multiword()
! ABSENT-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_target_enter_data"]]
subroutine assumes_multiword()
  !$omp assumes absent(target enter data)
end subroutine assumes_multiword

! ABSENT-LABEL: func.func @_QPuses_absent_module()
! ABSENT-NOT: llvm.passthrough
subroutine uses_absent_module()
  use absent_module, only: module_work
  implicit none
  ! ABSENT: fir.call @_QMabsent_modulePmodule_work()
  call module_work()
end subroutine uses_absent_module

! ABSENT-LABEL: func.func @_QPassumes_keeps_parallel()
! ABSENT-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_target"]]
subroutine assumes_keeps_parallel()
  !$omp assumes absent(target)
  ! ABSENT: omp.parallel {
  !$omp parallel
    ! ABSENT: omp.barrier
    !$omp barrier
  !$omp end parallel
end subroutine assumes_keeps_parallel

! ABSENT-SIMD: define
! ABSENT-SIMD-NOT: "omp_no_

! ABSENT-MODFILE: module absent_module
! ABSENT-MODFILE: !$omp assumes no_openmp
! ABSENT-MODFILE: !$omp assumes absent(barrier, parallel)

! LLVM-LABEL: define {{.*}}@assumes_absent_(
! LLVM-SAME: #[[PAIR:[0-9]+]]
! LLVM-LABEL: define {{.*}}@_QMabsent_modulePmodule_work(
! LLVM-SAME: #[[BOTH:[0-9]+]]
! LLVM-LABEL: define {{.*}}@_QMabsent_modulePlocal_work(
! LLVM-SAME: #[[MERGED:[0-9]+]]
! LLVM-LABEL: define {{.*}}@assumes_host_(
! LLVM-SAME: #[[PARALLEL:[0-9]+]]
! LLVM-LABEL: define {{.*}}@_QFassumes_hostPinternal_work(
! LLVM-SAME: #[[BOTH]]
! LLVM-LABEL: define {{.*}}@assumes_multiword_(
! LLVM-SAME: #[[MULTIWORD:[0-9]+]]
! LLVM-LABEL: define {{.*}}@uses_absent_module_() {
! LLVM-LABEL: define {{.*}}@assumes_keeps_parallel_(
! LLVM-SAME: #[[TARGET:[0-9]+]]
! LLVM: call {{.*}}@__kmpc_fork_call(
! LLVM-DAG: attributes #[[PAIR]] = {{.*}}"llvm.assume"="omp_no_barrier,omp_no_parallel"
! LLVM-DAG: attributes #[[BOTH]] = {{.*}}"llvm.assume"="omp_no_openmp,omp_no_barrier,omp_no_parallel"
! LLVM-DAG: attributes #[[MERGED]] = {{.*}}"llvm.assume"="omp_no_openmp,omp_no_barrier,omp_no_critical,omp_no_parallel"
! LLVM-DAG: attributes #[[PARALLEL]] = {{.*}}"llvm.assume"="omp_no_parallel"
! LLVM-DAG: attributes #[[MULTIWORD]] = {{.*}}"llvm.assume"="omp_no_target_enter_data"
! LLVM-DAG: attributes #[[TARGET]] = {{.*}}"llvm.assume"="omp_no_target"

!--- absent-submodule.f90
submodule (absent_module) absent_implementation
contains
  ! RELOAD-LABEL: func.func @separate_absent_work()
  ! RELOAD-NOT: llvm.passthrough
  ! RELOAD: return
  module procedure separate_absent_work
  end procedure separate_absent_work

  ! RELOAD-LABEL: func.func @{{.*}}Plocal_work()
  ! RELOAD-SAME: llvm.passthrough = {{\[\[}}"llvm.assume", "omp_no_critical"]]
  subroutine local_work()
    !$omp assumes absent(critical)
  end subroutine local_work
end submodule absent_implementation

! RELOAD-LABEL: func.func @_QPuses_absent_module_reload()
! RELOAD-NOT: llvm.passthrough
subroutine uses_absent_module_reload()
  use absent_module, only: separate_absent_work
  ! RELOAD: fir.call @separate_absent_work()
  call separate_absent_work()
end subroutine uses_absent_module_reload

! RELOAD-LLVM-LABEL: define {{.*}}@separate_absent_work() {
! RELOAD-LLVM-LABEL: define {{.*}}@_QMabsent_moduleSabsent_implementationPlocal_work()
! RELOAD-LLVM-SAME: #[[LOCAL:[0-9]+]]
! RELOAD-LLVM-LABEL: define {{.*}}@uses_absent_module_reload_() {
! RELOAD-LLVM: attributes #[[LOCAL]] = {{.*}}"llvm.assume"="omp_no_critical"
