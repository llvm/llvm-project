! Test that use associated module variables are only instantiated
! in a subprogram when they are referenced in it.

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module test_use_instantiation
  integer :: var1, var2, var3
  integer :: a, b, c
  equivalence (a, b)
  integer :: n = 10, l = 3
  integer, target :: pointee_target
  real, target :: tgt(10)
  integer, pointer :: p => null()
  procedure(real), pointer :: proc_ptr
  namelist /nml/ var3
end module

subroutine foo()
  use test_use_instantiation
  call bar(var1)
end subroutine
! CHECK-LABEL: func.func @_QPfoo(
! CHECK-NOT: fir.address_of
! CHECK: %[[ADDR1:.*]] = fir.address_of(@_QMtest_use_instantiationEvar1) : !fir.ref<i32>
! CHECK: hlfir.declare %[[ADDR1]]
! CHECK-NOT: fir.address_of
! CHECK: return

subroutine foo_equiv()
  use test_use_instantiation
  b = 1
end subroutine
! CHECK-LABEL: func.func @_QPfoo_equiv(
! CHECK-NOT: hlfir.declare
! CHECK: %[[ADDR_A:.*]] = fir.address_of(@_QMtest_use_instantiationEa) : !fir.ref<!fir.array<4xi8>>
! CHECK: hlfir.declare %{{.*}} storage(%[[ADDR_A]][0]) {uniq_name = "_QMtest_use_instantiationEb"}
! CHECK-NOT: hlfir.declare
! CHECK: return

subroutine foo_spec_expr()
  use test_use_instantiation
  real :: x(n)
  call bar2(x)
end subroutine
! CHECK-LABEL: func.func @_QPfoo_spec_expr(
! CHECK-NOT: fir.address_of
! CHECK: %[[ADDR_N:.*]] = fir.address_of(@_QMtest_use_instantiationEn) : !fir.ref<i32>
! CHECK: hlfir.declare %[[ADDR_N]]
! CHECK-NOT: fir.address_of
! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFfoo_spec_exprEx"}
! CHECK-NOT: fir.address_of
! CHECK: return

subroutine foo_char_len()
  use test_use_instantiation
  character(l) :: ch
  call bar3(ch)
end subroutine
! CHECK-LABEL: func.func @_QPfoo_char_len(
! CHECK-NOT: fir.address_of
! CHECK: %[[ADDR_L:.*]] = fir.address_of(@_QMtest_use_instantiationEl) : !fir.ref<i32>
! CHECK: hlfir.declare %[[ADDR_L]]
! CHECK-NOT: fir.address_of
! CHECK: hlfir.declare %{{.*}} typeparams %{{.*}} {uniq_name = "_QFfoo_char_lenEch"}
! CHECK-NOT: fir.address_of
! CHECK: return

subroutine foo_equiv_spec_expr()
  use test_use_instantiation
  real :: x(a)
  call bar2(x)
end subroutine
! CHECK-LABEL: func.func @_QPfoo_equiv_spec_expr(
! CHECK-NOT: fir.address_of
! CHECK: %[[ADDR_A2:.*]] = fir.address_of(@_QMtest_use_instantiationEa) : !fir.ref<!fir.array<4xi8>>
! CHECK: hlfir.declare %{{.*}} storage(%[[ADDR_A2]][0]) {uniq_name = "_QMtest_use_instantiationEa"}
! CHECK-NOT: fir.address_of
! CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFfoo_equiv_spec_exprEx"}
! CHECK-NOT: fir.address_of
! CHECK: return

subroutine foo_init_target()
  use test_use_instantiation
  real, pointer :: ptr(:) => tgt
  call bar4(ptr)
end subroutine
! CHECK-LABEL: func.func @_QPfoo_init_target(
! CHECK-NOT: fir.address_of
! CHECK: %[[ADDR_TGT:.*]] = fir.address_of(@_QMtest_use_instantiationEtgt) : !fir.ref<!fir.array<10xf32>>
! CHECK: hlfir.declare %[[ADDR_TGT]]
! CHECK-NOT: fir.address_of(@_QMtest_use_instantiationE
! CHECK: hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFfoo_init_targetEptr"}
! CHECK-NOT: fir.address_of(@_QMtest_use_instantiationE
! CHECK: return

subroutine foo_init_target_scalar()
  use test_use_instantiation
  integer, pointer :: iptr => pointee_target
  call bar5(iptr)
end subroutine
! CHECK-LABEL: func.func @_QPfoo_init_target_scalar(
! CHECK-NOT: fir.address_of
! CHECK: %[[ADDR_PT:.*]] = fir.address_of(@_QMtest_use_instantiationEpointee_target) : !fir.ref<i32>
! CHECK: hlfir.declare %[[ADDR_PT]]
! CHECK-NOT: fir.address_of(@_QMtest_use_instantiationE
! CHECK: hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFfoo_init_target_scalarEiptr"}
! CHECK-NOT: fir.address_of(@_QMtest_use_instantiationE
! CHECK: return

subroutine foo_namelist()
  use test_use_instantiation
  read(*, nml)
end subroutine
! CHECK-LABEL: func.func @_QPfoo_namelist(
! CHECK-NOT: fir.address_of
! CHECK: %[[ADDR3:.*]] = fir.address_of(@_QMtest_use_instantiationEvar3) : !fir.ref<i32>
! CHECK: hlfir.declare %[[ADDR3]]
! CHECK-NOT: fir.address_of(@_QMtest_use_instantiationE
! CHECK: return

subroutine foo_internal()
  use test_use_instantiation
  call internal()
contains
  subroutine internal()
    var2 = 42
  end subroutine
end subroutine
! CHECK-LABEL: func.func @_QPfoo_internal(
! CHECK-NOT: fir.address_of
! CHECK: fir.address_of(@_QMtest_use_instantiationEvar2) : !fir.ref<i32>
! CHECK-NOT: fir.address_of
! CHECK: return

! CHECK-LABEL: func.func private @_QFfoo_internalPinternal(
! CHECK-NOT: fir.address_of
! CHECK: %[[ADDR2:.*]] = fir.address_of(@_QMtest_use_instantiationEvar2) : !fir.ref<i32>
! CHECK: hlfir.declare %[[ADDR2]]
! CHECK-NOT: fir.address_of
! CHECK: return

subroutine foo_unused()
  use test_use_instantiation
end subroutine
! CHECK-LABEL: func.func @_QPfoo_unused(
! CHECK-NOT: fir.address_of
! CHECK-NOT: hlfir.declare
! CHECK: return
