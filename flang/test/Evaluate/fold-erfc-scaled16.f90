! REQUIRES: flang-fold-real16-quadmath
! RUN: %python %S/test_folding.py %s %flang_fc1
! Folding of ERFC_SCALED for REAL(16) on hosts where the compiler folds
! REAL(16) through libquadmath.
!
! The asymptotic-series results (x >= 16) involve no math-library calls and
! are compared exactly; the expected literals are the exact decimal
! expansions of the algorithm's results, which sit within one ulp of a
! 300-bit mpmath oracle. The remaining points go through expq/erfcq, whose
! last-ulp rounding may vary across libquadmath versions, so they are
! compared to a few-ulp relative tolerance (binary128 epsilon is 1.93e-34);
! their reference literals are the correctly rounded true values.
module m
  integer, parameter :: qp = selected_real_kind(33)
  real(qp), parameter :: tol = 5.0e-33_qp

  ! Series branch: exact.
  logical, parameter :: test_p20 = erfc_scaled(20._qp) == &
    2.817434874105131931864915453447075566e-2_qp
  logical, parameter :: test_p1e6 = erfc_scaled(1000000._qp) == &
    5.641895835474741921563059965594862664e-7_qp

  ! Direct branch.
  real(qp), parameter :: ref_half = 0.615690344192925874870793422683741924_qp
  logical, parameter :: test_phalf = &
    abs(erfc_scaled(0.5_qp) - ref_half) <= tol * ref_half

  ! Negative arguments with representable binary128 results. The previous
  ! implementation returned HUGE(x) for x < -26.628; the true values stay
  ! representable down to x ~ -106.567.
  real(qp), parameter :: ref_n27 = &
    7.97457052408519312709372209466870032e316_qp
  logical, parameter :: test_n27 = &
    abs(erfc_scaled(-27._qp) - ref_n27) <= tol * ref_n27
  real(qp), parameter :: ref_n1063 = &
    4.94361012472232021713994535196543716e4907_qp
  logical, parameter :: test_n1063 = &
    abs(erfc_scaled(-106.3_qp) - ref_n1063) <= tol * ref_n1063

  ! Beyond the last representable result, +infinity.
  !WARN: warning: overflow on evaluation of intrinsic function or operation [-Wfolding-exception]
  logical, parameter :: test_n107 = erfc_scaled(-107._qp) > huge(0._qp)
end
