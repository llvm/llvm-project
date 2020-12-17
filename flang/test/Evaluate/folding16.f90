! RUN: %S/test_folding.sh %s %t %f18
! Ensure that lower bounds are accounted for in intrinsic folding;
! this is a regression test for a bug in which they were not
real, parameter :: a(-1:-1) = 1.
real, parameter :: b(-1:-1) = log(a)
logical, parameter :: test = lbound(a,1)==-1 .and. lbound(b,1)==-1 .and. &
                             lbound(log(a),1)==1 .and. all(b==0)
end
