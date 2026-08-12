! This test checks lowering of an OpenMP allocate Directive that appears in a
! module or submodule declaration scope.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module mymod
  implicit none
  integer :: x
  real :: y
  !$omp allocate(x, y)
end module mymod

submodule (mymod) mysub
  implicit none
  integer :: z
  !$omp allocate(z)
end submodule mysub

! CHECK: fir.global @_QMmymodEx : i32
! CHECK: fir.global @_QMmymodEy : f32

! CHECK: llvm.func internal @_QMmymodEx_omp_allocate_ctor() {
! CHECK:   %[[X:.*]] = fir.address_of(@_QMmymodEx) : !fir.ref<i32>
! CHECK:   omp.allocate_dir(%[[X]] : !fir.ref<i32>)
! CHECK:   llvm.return
! CHECK: }

! CHECK: llvm.mlir.global_ctors ctors = [@_QMmymodEx_omp_allocate_ctor, @_QMmymodEy_omp_allocate_ctor, @_QMmymodSmysubEz_omp_allocate_ctor]

! CHECK: llvm.func internal @_QMmymodEy_omp_allocate_ctor() {
! CHECK:   %[[Y:.*]] = fir.address_of(@_QMmymodEy) : !fir.ref<f32>
! CHECK:   omp.allocate_dir(%[[Y]] : !fir.ref<f32>)
! CHECK:   llvm.return
! CHECK: }

! CHECK: fir.global @_QMmymodSmysubEz : i32

! CHECK: llvm.func internal @_QMmymodSmysubEz_omp_allocate_ctor() {
! CHECK:   %[[Z:.*]] = fir.address_of(@_QMmymodSmysubEz) : !fir.ref<i32>
! CHECK:   omp.allocate_dir(%[[Z]] : !fir.ref<i32>)
! CHECK:   llvm.return
! CHECK: }
