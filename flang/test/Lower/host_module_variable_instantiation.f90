! Test that host module variables are not instantiated inside
! module procedure when they are not needed.

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module test_host_module_instantiation
  integer :: var1, var2
contains
subroutine foo()
  call bar(var1)
end subroutine
end module

! CHECK-LABEL:  func.func @_QMtest_host_module_instantiationPfoo(
! CHECK-NOT: fir.address_of
! CHECK: %[[ADDR1:.*]] = fir.address_of(@_QMtest_host_module_instantiationEvar1) : !fir.ref<i32>
! CHECK: hlfir.declare %[[ADDR1]]
! CHECK-NOT: fir.address_of
! CHECK: return
