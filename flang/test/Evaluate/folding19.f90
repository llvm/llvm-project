! RUN: not %flang_fc1 %s 2>&1 | FileCheck %s
! Check errors found in folding
! TODO: test others emitted from flang/lib/Evaluate
module m
 contains
  subroutine s1(a,b)
    real :: a(*), b(:)
    !CHECK: error: DIM=1 dimension is out of range for rank-1 assumed-size array
    integer :: ub1(ubound(a,1))
    !CHECK-NOT: error: DIM=1 dimension is out of range for rank-1 assumed-size array
    integer :: lb1(lbound(a,1))
    !CHECK: error: DIM=0 dimension is out of range for rank-1 array
    integer :: ub2(ubound(a,0))
    !CHECK: error: DIM=2 dimension is out of range for rank-1 array
    integer :: ub3(ubound(a,2))
    !CHECK: error: DIM=0 dimension is out of range for rank-1 array
    integer :: lb2(lbound(b,0))
    !CHECK: error: DIM=2 dimension is out of range for rank-1 array
    integer :: lb3(lbound(b,2))
  end subroutine
  subroutine s2
    integer, parameter :: array(2,3) = reshape([(j, j=1, 6)], shape(array))
    integer :: x(2, 3)
    !CHECK: error: Invalid 'dim=' argument (0) in CSHIFT
    x = cshift(array, [1, 2], dim=0)
    !CHECK: error: Invalid 'shift=' argument in CSHIFT: extent on dimension 1 is 2 but must be 3
    x = cshift(array, [1, 2], dim=1)
  end subroutine
  subroutine s3
    integer, parameter :: array(2,3) = reshape([(j, j=1, 6)], shape(array))
    integer :: x(2, 3)
    !CHECK: error: Invalid 'dim=' argument (0) in EOSHIFT
    x = eoshift(array, [1, 2], dim=0)
    !CHECK: error: Invalid 'shift=' argument in EOSHIFT: extent on dimension 1 is 2 but must be 3
    x = eoshift(array, [1, 2], dim=1)
    !CHECK: error: Invalid 'boundary=' argument in EOSHIFT: extent on dimension 1 is 3 but must be 2
    x = eoshift(array, 1, [0, 0, 0], 2)
  end subroutine
  subroutine s4
    integer, parameter :: array(2,3) = reshape([(j, j=1, 6)], shape(array))
    logical, parameter :: mask(*,*) = reshape([(.true., j=1,3),(.false., j=1,3)], shape(array))
    integer :: x(3)
    !CHECK: error: Invalid 'vector=' argument in PACK: the 'mask=' argument has 3 true elements, but the vector has only 2 elements
    x = pack(array, mask, [0,0])
  end subroutine
  subroutine s5
    logical, parameter :: mask(2,3) = reshape([.false., .true., .true., .false., .false., .true.], shape(mask))
    integer, parameter :: field(3,2) = reshape([(-j,j=1,6)], shape(field))
    integer :: x(2,3)
    !CHECK: error: Invalid 'vector=' argument in UNPACK: the 'mask=' argument has 3 true elements, but the vector has only 2 elements
    x = unpack([1,2], mask, 0)
  end subroutine
end module
