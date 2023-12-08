! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test propagation of Fortran attributes to FIR.


! CHECK-LABEL: func @_QPfoo1(
! CHECK-SAME: %arg0: !fir.ref<f32> {fir.bindc_name = "x", fir.optional},
! CHECK-SAME: %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y", fir.optional},
! CHECK-SAME: %arg2: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "i", fir.optional},
! CHECK-SAME: %arg3: !fir.boxchar<1> {fir.bindc_name = "c", fir.optional}
subroutine foo1(x, y, i, c)
  real, optional :: x, y(:)
  integer, allocatable, optional :: i(:)
  character, optional :: c
end subroutine

! CHECK-LABEL: func @_QPfoo2(
! CHECK-SAME: %arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous},
! CHECK-SAME: %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "i", fir.contiguous}
subroutine foo2(x, i)
  real, contiguous :: x(:)
  integer, pointer, contiguous :: i(:)
end subroutine

! CHECK-LABEL: func @_QPfoo3
! CHECK-SAME: %arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous, fir.optional}
subroutine foo3(x)
  real, optional, contiguous :: x(:)
end subroutine
