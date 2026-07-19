! RUN: %flang_fc1 -emit-hlfir -fopenacc %s -o - | FileCheck %s

! Test that module symbols used inside the use_device clause
! in procedures from that same module are correctly instantiated.

module test_use_device
  real(4) :: a(100)
  integer :: i, j
  interface
  subroutine something(a)
   real(4) :: a(100)
  end subroutine
  module subroutine test_submodule()
  end subroutine
  end interface
contains
  subroutine test_simple
    !$acc host_data use_device(a)
    call something(a)
    !$acc end host_data
  end subroutine
  subroutine test_designator
    !$acc host_data use_device(a(i:j))
    !$acc end host_data
  end subroutine
  subroutine test_shadowed
    real :: a(100)
    !$acc declare copy(a)
    !$acc host_data use_device(a)
    call something(a)
    !$acc end host_data
  end subroutine
end module

!  CHECK-LABEL:   func.func @_QMtest_use_devicePtest_simple() {
!  CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
!  CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMtest_use_deviceEa) : !fir.ref<!fir.array<100xf32>>
!  CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
!  CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
!  CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ADDRESS_OF_0]](%[[SHAPE_0]]) {uniq_name = "_QMtest_use_deviceEa"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
!  CHECK:           %[[USE_DEVICE_0:.*]] = acc.use_device varPtr(%[[DECLARE_0]]#0 : !fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>> {name = "a"}
!  CHECK:           acc.host_data dataOperands(%[[USE_DEVICE_0]] : !fir.ref<!fir.array<100xf32>>) {
!  CHECK:             %[[DECLARE_1:.*]]:2 = hlfir.declare %[[USE_DEVICE_0]](%[[SHAPE_0]]) {uniq_name = "_QMtest_use_deviceEa"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
!  CHECK:             fir.call @_QPsomething(%[[DECLARE_1]]#0)
!  CHECK:             acc.terminator
!  CHECK:           }
!  CHECK:           return
!  CHECK:         }

!  CHECK-LABEL:   func.func @_QMtest_use_devicePtest_designator() {
!  CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMtest_use_deviceEa) : !fir.ref<!fir.array<100xf32>>
!  CHECK:           %[[ADDRESS_OF_1:.*]] = fir.address_of(@_QMtest_use_deviceEi) : !fir.ref<i32>
!  CHECK:           %[[ADDRESS_OF_2:.*]] = fir.address_of(@_QMtest_use_deviceEj) : !fir.ref<i32>
!  CHECK:           acc.use_device

!  CHECK-LABEL:   func.func @_QMtest_use_devicePtest_shadowed() {
!  CHECK-NOT:      fir.address_of(@_QMtest_use_deviceEa)
!  CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "a", uniq_name = "_QMtest_use_deviceFtest_shadowedEa"}
!  CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](
!  CHECK-NOT:      fir.address_of(@_QMtest_use_deviceEa)
!  CHECK:           %[[USE_DEVICE_0:.*]] = acc.use_device varPtr(%[[DECLARE_0]]#0

submodule (test_use_device) some_submodule
contains
  module subroutine test_submodule
    !$acc host_data use_device(a)
    call something(a)
    !$acc end host_data
  end subroutine
end submodule

!  CHECK-LABEL:   func.func @_QMtest_use_devicePtest_submodule() {
!  CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMtest_use_deviceEa) : !fir.ref<!fir.array<100xf32>>
!  CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ADDRESS_OF_0]]
!  CHECK:           acc.use_device varPtr(%[[DECLARE_0]]#0
