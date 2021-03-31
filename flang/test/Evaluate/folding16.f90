! RUN: %S/test_folding.sh %s %t %f18
! Ensure that lower bounds are accounted for in intrinsic folding;
! this is a regression test for a bug in which they were not
module m
  real, parameter :: a(-1:-1) = 1.
  real, parameter :: b(-1:-1) = log(a)
  integer, parameter :: c(-1:1) = [33, 22, 11]
  integer, parameter :: d(1:3) = [33, 22, 11]
  integer, parameter :: e(-2:0) = ([33, 22, 11])
  logical, parameter :: test_1 = lbound((a),1)==-1 .and. lbound(b,1)==-1 .and. &
                               lbound(log(a),1)==1 .and. all(b==0)
  logical, parameter :: test_2 = all(c .eq. d)
  logical, parameter :: test_3 = all(c .eq. e)
  logical, parameter :: test_4 = all(d .eq. e)
end
