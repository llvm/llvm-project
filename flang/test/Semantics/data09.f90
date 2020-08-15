! RUN: %f18 -fparse-only -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! CHECK: init:[INTEGER(4)::1065353216_4,1073741824_4,1077936128_4,1082130432_4]
! Verify that the closure of EQUIVALENCE'd symbols with any DATA
! initialization produces a combined initializer.
real :: a(2), b(2), c(2)
equivalence(a(2),b(1)),(b(2),c(1))
data a(1)/1./,b(1)/2./,c/3.,4./
common /block/ a
end
