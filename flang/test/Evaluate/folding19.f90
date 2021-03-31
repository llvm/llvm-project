! RUN: not %f18 %s 2>&1 | FileCheck %s
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
end module

