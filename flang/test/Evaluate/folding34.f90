! RUN: %python %S/test_folding.py %s %flang_fc1
! Constant folding of COMPLEX(4)**INTEGER must match flang-rt's cpowi/cpowk
! (complex-powi.cpp): both accumulate in COMPLEX(8) and round once.
module m
  complex(4), parameter :: base = (1.234567_4, 1.234567_4)
  complex(4), parameter :: p_i4 = base ** 7_4
  complex(4), parameter :: p_i8 = base ** 7_8
  ! Sanity check, not proof of the fix: both exponent widths take the same
  ! code path here (keyed on the COMPLEX(4) result type, not the exponent's
  ! kind), so this holds with or without the fix below. The value checks
  ! that follow are what actually discriminate.
  logical, parameter :: test_cpowi_cpowk_agree = p_i4 == p_i8
  ! Known-correct value (double-precision reference, MPFR-verified
  ! elsewhere): (1.234567,1.234567)**7 = (34.96976852, -34.96976852).
  complex(4), parameter :: expected = (34.96976852_4, -34.96976852_4)
  logical, parameter :: test_cpowi_i4_value = p_i4 == expected
  logical, parameter :: test_cpowk_i8_value = p_i8 == expected
  ! Negative exponent: the folder divides per set bit while the runtime
  ! inverts once at the end, so this case needs checking on both sides;
  ! for this input they agree, and Complex.cpowiRoundsToSingleOnce pins
  ! the runtime half.
  complex(4), parameter :: p_neg = (0.5_4, 0.6_4) ** (-10_4)
  logical, parameter :: test_cpowi_neg_value = &
      p_neg == (-9.3229351_4, -7.29848194_4)
  ! Same negative-exponent case, INTEGER(8) exponent -- both widths take the
  ! same code path (see test_cpowi_cpowk_agree above), so this must match.
  complex(4), parameter :: p_neg_i8 = (0.5_4, 0.6_4) ** (-10_8)
  logical, parameter :: test_cpowk_neg_value = &
      p_neg_i8 == (-9.3229351_4, -7.29848194_4)
  ! An intermediate that overflows in single but not in double: the result
  ! is representable (subnormal), so folding must not produce Inf/NaN -- a
  ! plain underflow on the final narrowing is correct and expected here.
  complex(4), parameter :: big   = (1.0e20_4, 0.0_4)
  !WARN: warning: underflow on power with INTEGER exponent [-Wfolding-exception]
  complex(4), parameter :: bigm2 = big ** (-2)
  ! bigm2%im == 0.0_4 alone would also pass a wrong fold to (0.0, 0.0), so
  ! pin the exact real-part bit pattern too, not just rule out NaN.
  real(4), parameter :: bigm2_re_expected = &
      transfer(int(z'000116C2', 4), 0.0_4)
  logical, parameter :: test_no_bogus_nan = &
      bigm2%im == 0.0_4 .and. bigm2%re == bigm2_re_expected
end module m
