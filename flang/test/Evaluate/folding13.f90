! RUN: %S/test_folding.sh %s %t %f18
! Test folding of array constructors with constant implied DO bounds;
! their indices are constant expressions and can be used as such.
module m1
  integer, parameter :: kinds(*) = [1, 2, 4, 8]
  integer(kind=8), parameter :: clipping(*) = [integer(kind=8) :: &
    (int(z'100010101', kind=kinds(j)), j=1,4)]
  integer(kind=8), parameter :: expected(*) = [ &
    int(z'01',8), int(z'0101',8), int(z'00010101',8), int(z'100010101',8)]
  logical, parameter :: test_clipping = all(clipping == expected)
end module
