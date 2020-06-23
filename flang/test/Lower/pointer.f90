! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPpointertests
subroutine pointerTests
  ! CHECK: fir.global @_QFpointertestsEptr1 : !fir.ptr<i32>
  integer, pointer :: ptr1 => NULL()
  ! CHECK: fir.global @_QFpointertestsEptr2 : !fir.ptr<f32>
  real, pointer :: ptr2 => NULL()
  ! CHECK: fir.global @_QFpointertestsEptr3 : !fir.ptr<!fir.complex<4>>
  complex, pointer :: ptr3 => NULL()
  ! CHECK: fir.global @_QFpointertestsEptr4 : !fir.ptr<!fir.char<1>
  character, pointer :: ptr4 => NULL()
  ! CHECK: fir.global @_QFpointertestsEptr5 : !fir.ptr<!fir.logical<4>>
  logical, pointer :: ptr5 => NULL()
end subroutine pointerTests

