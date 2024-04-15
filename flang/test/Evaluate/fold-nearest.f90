! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of NEAREST() and its relatives
module m1
  real, parameter :: minSubnormal = 1.e-45
  logical, parameter :: test_1 = nearest(0., 1.) == minSubnormal
  logical, parameter :: test_2 = nearest(minSubnormal, -1.) == 0
  logical, parameter :: test_3 = nearest(1., 1.) == 1.0000001
  logical, parameter :: test_4 = nearest(1.0000001, -1.) == 1
  !WARN: warning: NEAREST intrinsic folding overflow
  real, parameter :: inf = nearest(huge(1.), 1.)
  !WARN: warning: NEAREST intrinsic folding: bad argument
  logical, parameter :: test_5 = nearest(inf, 1.) == inf
  !WARN: warning: NEAREST intrinsic folding: bad argument
  logical, parameter :: test_6 = nearest(-inf, -1.) == -inf
  logical, parameter :: test_7 = nearest(1.9999999, 1.) == 2.
  logical, parameter :: test_8 = nearest(2., -1.) == 1.9999999
#if __x86_64__
  logical, parameter :: test_9 = nearest(1.9999999999999999999_10, 1.) == 2._10
#endif
  logical, parameter :: test_10 = nearest(-1., 1.) == -.99999994
  logical, parameter :: test_11 = nearest(-1., -2.) == -1.0000001
  real, parameter :: negZero = sign(0., -1.)
  logical, parameter :: test_12 = nearest(negZero, 1.) == minSubnormal
  logical, parameter :: test_13 = nearest(negZero, -1.) == -minSubnormal
  !WARN: warning: NEAREST: S argument is zero
  logical, parameter :: test_14 = nearest(0., negZero) == -minSubnormal
  !WARN: warning: NEAREST: S argument is zero
  logical, parameter :: test_15 = nearest(negZero, 0.) == minSubnormal
  logical, parameter :: test_16 = nearest(tiny(1.),-1.) == 1.1754942E-38
  logical, parameter :: test_17 = nearest(tiny(1.),1.) == 1.1754945E-38
end module

module m2
  use ieee_arithmetic, only: ieee_next_after
  real, parameter :: minSubnormal = 1.e-45
  logical, parameter :: test_0 = ieee_next_after(0., 0.) == 0.
  logical, parameter :: test_1 = ieee_next_after(0., 1.) == minSubnormal
  logical, parameter :: test_2 = ieee_next_after(minSubnormal, -1.) == 0
  logical, parameter :: test_3 = ieee_next_after(1., 2.) == 1.0000001
  logical, parameter :: test_4 = ieee_next_after(1.0000001, -1.) == 1
  !WARN: warning: division by zero
  real, parameter :: inf = 1. / 0.
  logical, parameter :: test_5 = ieee_next_after(inf, inf) == inf
  logical, parameter :: test_6 = ieee_next_after(inf, -inf) == inf
  logical, parameter :: test_7 = ieee_next_after(-inf, inf) == -inf
  logical, parameter :: test_8 = ieee_next_after(-inf, -1.) == -inf
  logical, parameter :: test_9 = ieee_next_after(1.9999999, 3.) == 2.
  logical, parameter :: test_10 = ieee_next_after(2., 1.) == 1.9999999
#if __x86_64__
  logical, parameter :: test_11 = ieee_next_after(1.9999999999999999999_10, 3.) == 2._10
#endif
  logical, parameter :: test_12 = ieee_next_after(1., 1.) == 1.
  !WARN: warning: invalid argument on division
  real, parameter :: nan = 0. / 0.
  !WARN: warning: IEEE_NEXT_AFTER intrinsic folding: bad argument
  real, parameter :: x13 = ieee_next_after(nan, nan)
  logical, parameter :: test_13 = .not. (x13 == x13)
  !WARN: warning: IEEE_NEXT_AFTER intrinsic folding: bad argument
  real, parameter :: x14 = ieee_next_after(nan, 0.)
  logical, parameter :: test_14 = .not. (x14 == x14)
end module

module m3
  use ieee_arithmetic, only: ieee_next_up, ieee_next_down
  real(kind(0.d0)), parameter :: minSubnormal = 5.d-324
  logical, parameter :: test_1 = ieee_next_up(0.d0) == minSubnormal
  logical, parameter :: test_2 = ieee_next_down(0.d0) == -minSubnormal
  logical, parameter :: test_3 = ieee_next_up(1.d0) == 1.0000000000000002d0
  logical, parameter :: test_4 = ieee_next_down(1.0000000000000002d0) == 1.d0
  !WARN: warning: division by zero
  real(kind(0.d0)), parameter :: inf = 1.d0 / 0.d0
  !WARN: warning: IEEE_NEXT_UP intrinsic folding overflow
  logical, parameter :: test_5 = ieee_next_up(huge(0.d0)) == inf
  !WARN: warning: IEEE_NEXT_DOWN intrinsic folding overflow
  logical, parameter :: test_6 = ieee_next_down(-huge(0.d0)) == -inf
  !WARN: warning: IEEE_NEXT_UP intrinsic folding: bad argument
  logical, parameter :: test_7 = ieee_next_up(inf) == inf
  !WARN: warning: IEEE_NEXT_DOWN intrinsic folding: bad argument
  logical, parameter :: test_8 = ieee_next_down(inf) == inf
  !WARN: warning: IEEE_NEXT_UP intrinsic folding: bad argument
  logical, parameter :: test_9 = ieee_next_up(-inf) == -inf
  !WARN: warning: IEEE_NEXT_DOWN intrinsic folding: bad argument
  logical, parameter :: test_10 = ieee_next_down(-inf) == -inf
  logical, parameter :: test_11 = ieee_next_up(1.9999999999999997d0) == 2.d0
  logical, parameter :: test_12 = ieee_next_down(2.d0) == 1.9999999999999997d0
  !WARN: warning: invalid argument on division
  real(kind(0.d0)), parameter :: nan = 0.d0 / 0.d0
  !WARN: warning: IEEE_NEXT_UP intrinsic folding: bad argument
  real(kind(0.d0)), parameter :: x13 = ieee_next_up(nan)
  logical, parameter :: test_13 = .not. (x13 == x13)
  !WARN: warning: IEEE_NEXT_DOWN intrinsic folding: bad argument
  real(kind(0.d0)), parameter :: x14 = ieee_next_down(nan)
  logical, parameter :: test_14 = .not. (x14 == x14)
end module
