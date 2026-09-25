! Test that host module variables are not instantiated inside
! module procedure when they are not needed.

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module test_host_module_instantiation
  integer :: var1, var2
  integer :: a, b, c
  equivalence (a, b)
contains
subroutine foo()
  call bar(var1)
end subroutine

subroutine foo_equiv
  b = 1
end subroutine
end module

! CHECK-LABEL:  func.func @_QMtest_host_module_instantiationPfoo(
! CHECK-NOT: fir.address_of
! CHECK: %[[ADDR1:.*]] = fir.address_of(@_QMtest_host_module_instantiationEvar1) : !fir.ref<i32>
! CHECK: hlfir.declare %[[ADDR1]]
! CHECK-NOT: fir.address_of
! CHECK: return


! CHECK-LABEL:   func.func @_QMtest_host_module_instantiationPfoo_equiv(
! CHECK-NOT:       hlfir.declare
! CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMtest_host_module_instantiationEa) : !fir.ref<!fir.array<4xi8>>
! CHECK-NEXT:      %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK-NEXT:      %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[ADDRESS_OF_0]], %[[CONSTANT_0]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! CHECK-NEXT:      %[[CONVERT_0:.*]] = fir.convert %[[COORDINATE_OF_0]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! CHECK-NEXT:      %[[DECLARE_0:.*]]:2 = hlfir.declare %[[CONVERT_0]] storage(%[[ADDRESS_OF_0]][0]) {uniq_name = "_QMtest_host_module_instantiationEb"} : (!fir.ptr<i32>, !fir.ref<!fir.array<4xi8>>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! CHECK-NOT:       hlfir.declare
! CHECK:           return
