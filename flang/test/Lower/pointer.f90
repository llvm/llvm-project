! RUN: bbc -emit-fir %s -o - | FileCheck %s

! TODO: Descriptor (fir.box) will most likely be used for pointers
! (at least for the character case below). This code is hitting a
! hard todo until pointers are handled correctly.
! XFAIL: true

! CHECK-LABEL: func @_QPpointertests
subroutine pointerTests
  ! CHECK: fir.global internal @_QFpointertestsEptr1 : !fir.ptr<i32>
  integer, pointer :: ptr1 => NULL()
  ! CHECK: %[[c0:.*]] = arith.constant 0 : index
  ! CHECK: [[reg1:%[0-9]+]] = fir.convert %[[c0:.*]] : (index) -> !fir.ref<none>
  ! CHECK: [[reg2:%[0-9]+]] = fir.convert [[reg1]] : (!fir.ref<none>) -> !fir.ptr<i32>
  ! CHECK: fir.has_value [[reg2]] : !fir.ptr<i32>

  ! CHECK: fir.global internal @_QFpointertestsEptr2 : !fir.ptr<f32>
  real, pointer :: ptr2 => NULL()
  ! CHECK: %[[c0:.*]] = arith.constant 0 : index
  ! CHECK: [[reg1:%[0-9]+]] = fir.convert %[[c0:.*]] : (index) -> !fir.ref<none>
  ! CHECK: [[reg2:%[0-9]+]] = fir.convert [[reg1]] : (!fir.ref<none>) -> !fir.ptr<f32>
  ! CHECK: fir.has_value [[reg2]] : !fir.ptr<f32>

  ! CHECK: fir.global internal @_QFpointertestsEptr3 : !fir.ptr<complex<f32>>
  complex, pointer :: ptr3 => NULL()
  ! CHECK: %[[c0:.*]] = arith.constant 0 : index
  ! CHECK: [[reg1:%[0-9]+]] = fir.convert %[[c0:.*]] : (index) -> !fir.ref<none>
  ! CHECK: [[reg2:%[0-9]+]] = fir.convert [[reg1]] : (!fir.ref<none>) -> !fir.ptr<complex<f32>>
  ! CHECK: fir.has_value [[reg2]] : !fir.ptr<complex<f32>>

  ! CHECK: fir.global internal @_QFpointertestsEptr4 : !fir.ptr<!fir.char<1,?>>
  character(:), pointer :: ptr4 => NULL()
  ! CHECK: %[[c0:.*]] = arith.constant 0 : index
  ! CHECK: [[reg1:%[0-9]+]] = fir.convert %[[c0:.*]] : (index) -> !fir.ref<none>
  ! CHECK: [[reg2:%[0-9]+]] = fir.convert [[reg1]] : (!fir.ref<none>) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK: fir.has_value [[reg2]] : !fir.ptr<!fir.char<1,?>>

  ! CHECK: fir.global internal @_QFpointertestsEptr5 : !fir.ptr<!fir.logical<4>>
  logical, pointer :: ptr5 => NULL()
  ! CHECK: %[[c0:.*]] = arith.constant 0 : index
  ! CHECK: [[reg1:%[0-9]+]] = fir.convert %[[c0:.*]] : (index) -> !fir.ref<none>
  ! CHECK: [[reg2:%[0-9]+]] = fir.convert [[reg1]] : (!fir.ref<none>) -> !fir.ptr<!fir.logical<4>>
  ! CHECK: fir.has_value [[reg2]] : !fir.ptr<!fir.logical<4>>

end subroutine pointerTests
