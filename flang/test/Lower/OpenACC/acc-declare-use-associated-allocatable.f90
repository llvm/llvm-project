! Test separate compilation of !$acc declare create on a module allocatable:
! - the defining unit exports linkable declare-action recipe definitions
! - the using unit declares the module global and external recipe symbols so
!   acc-declare-action-conversion can insert fir.call to the recipes.

! RUN: split-file %s %t
! RUN: bbc -fopenacc -emit-hlfir %t/mod.f90 -o %t/mod.mlir --module=%t
! RUN: bbc -fopenacc -emit-hlfir %t/use.f90 -o %t/use.mlir -I %t
! RUN: FileCheck %s --check-prefix=MOD < %t/mod.mlir
! RUN: FileCheck %s --check-prefix=USE < %t/use.mlir
! RUN: fir-opt %t/use.mlir --acc-declare-action-conversion -o - | FileCheck %s --check-prefix=CONV

//--- mod.f90
module acc_declare_alloc_mod
  integer, allocatable :: data(:)
  !$acc declare create(data)
end module

//--- use.f90
subroutine use_mod()
  use acc_declare_alloc_mod
  implicit none
  allocate(data(100))
end subroutine

! MOD: func.func @_QMacc_declare_alloc_modEdata_acc_declare_post_alloc() attributes {acc.declare_action} {
! MOD: acc.declare_enter
! MOD: func.func @_QMacc_declare_alloc_modEdata_acc_declare_post_dealloc() attributes {acc.declare_action} {

! USE: func.func @_QPuse_mod() {
! USE: acc.declare_action = #acc.declare_action<postAlloc = @_QMacc_declare_alloc_modEdata_acc_declare_post_alloc>
! USE: fir.global @_QMacc_declare_alloc_modEdata {acc.declare = #acc.declare<dataClause = acc_create>
! USE: func.func private @_QMacc_declare_alloc_modEdata_acc_declare_post_alloc()
! USE: func.func private @_QMacc_declare_alloc_modEdata_acc_declare_post_dealloc()
! USE-NOT: acc.declare_enter

! CONV: fir.call @_QMacc_declare_alloc_modEdata_acc_declare_post_alloc()
