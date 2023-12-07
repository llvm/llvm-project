! RUN: bbc -emit-fir -o - %s | FileCheck %s

! Test that allocatable or pointer item from namelist are retrieved correctly
! if they are part of a common block as well.

program nml_common
  integer :: i
  real, pointer :: p(:)
  namelist /t/i,p
  common /c/i,p
  
  allocate(p(2))
  call print_t()
contains
  subroutine print_t()
    write(*,t)
  end subroutine
end

! CHECK-LABEL: fir.global linkonce @_QFNt.list constant : !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>> {
! CHECK: %[[CB_ADDR:.*]] = fir.address_of(@_QCc) : !fir.ref<!fir.array<56xi8>>
! CHECK: %[[CB_CAST:.*]] = fir.convert %[[CB_ADDR]] : (!fir.ref<!fir.array<56xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK: %[[OFFSET:.*]] = arith.constant 8 : index
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[CB_CAST]], %[[OFFSET]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK: %[[CAST_BOX:.*]] = fir.convert %[[COORD]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK: %[[CAST_BOX_NONE:.*]] = fir.convert %[[CAST_BOX]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[RES:.*]] = fir.insert_value %{{.*}}, %[[CAST_BOX_NONE]], [1 : index, 1 : index] : (!fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>, !fir.ref<!fir.box<none>>) -> !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
! CHECK: fir.has_value %[[RES]] : !fir.array<2xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
