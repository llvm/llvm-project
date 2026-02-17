!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
!CHECK: REAL :: avoidkahannan = (1._4/0.)
real :: avoidKahanNaN = sum([1./0., 0.]) ! Inf, not NaN
!CHECK: REAL :: expectnan1 = (0._4/0.)
real :: expectNaN1 = sum([1./0., -1./0.])
!CHECK: REAL :: expectnan2 = (0._4/0.)
real :: expectNaN2 = sum([-1./0., 1./0.])
end
