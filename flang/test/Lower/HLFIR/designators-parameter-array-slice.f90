! Test non-contiguous slice of parameter array.
! RUN: bbc -emit-hlfir --polymorphic-type -o - %s | FileCheck %s
subroutine test2(i)
  integer, parameter :: a(*,*) = reshape( [ 1,2,3,4 ], [ 2,2 ])
  integer :: x(2)
  x = a(i,:)
end subroutine test2
! Check that the result type of the designate operation
! is a box (as opposed to !fir.ref<!fir.array<>>) that is able
! to represent non-contiguous array section:
! CHECK: hlfir.designate {{.*}} shape {{.*}} : (!fir.ref<!fir.array<2x2xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
