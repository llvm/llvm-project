! RUN: %python %S/test_folding.py %s %flang_fc1
! Folding of ERFC_SCALED for kinds 4 and 8.
!
! On the asymptotic-series branch (x >= 16 in general, x >= 9 for kind 4) the
! computation involves no math-library calls, so the folded values are
! host-independent and compared exactly; the expected literals are the exact
! decimal expansions of the algorithm's results, which measure within 2 ulps
! of a 300-bit mpmath oracle. Direct-branch and negative results pass through
! host exp/erfc, whose last-ulp rounding may vary across hosts, and are
! compared to a small relative tolerance instead.
module m
  ! Series branch: exact.
  logical, parameter :: test_p9_4 = &
    erfc_scaled(9._4) == 0.06230773031711578369140625_4
  logical, parameter :: test_p95_4 = &
    erfc_scaled(9.5_4) == 0.0590646751224994659423828125_4
  logical, parameter :: test_p12_4 = &
    erfc_scaled(12._4) == 0.0468542166054248809814453125_4
  logical, parameter :: test_p20_4 = &
    erfc_scaled(20._4) == 0.02817435003817081451416015625_4
  logical, parameter :: test_p20_8 = erfc_scaled(20._8) == &
    0.028174348741051312428052000313982716761529445648193359375_8

  ! Direct branch: tolerance (about four ulps).
  real(4), parameter :: tol_4 = 5.0e-7_4
  real(8), parameter :: tol_8 = 1.0e-15_8
  real(4), parameter :: ref_p85_4 = 0.065925121307373046875_4
  logical, parameter :: test_p85_4 = &
    abs(erfc_scaled(8.5_4) - ref_p85_4) <= tol_4 * ref_p85_4

  ! Negative arguments with representable results.
  real(4), parameter :: ref_n9_4 = 3.01219472345922556007081388223234048e35_4
  logical, parameter :: test_n9_4 = &
    abs(erfc_scaled(-9._4) - ref_n9_4) <= tol_4 * ref_n9_4
  ! At kind 8 the last finite results lie just below |x| = 26.6287357; the
  ! previous implementation returned HUGE(x) from x < -26.628 on.
  real(8), parameter :: ref_sliver_8 = 1.7378490118139657e308_8
  logical, parameter :: test_sliver_8 = &
    abs(erfc_scaled(-26.6281_8) - ref_sliver_8) <= tol_8 * ref_sliver_8

  ! Negative arguments whose true value exceeds the format's range fold to
  ! +infinity.
  !WARN: warning: overflow on evaluation of intrinsic function or operation [-Wfolding-exception]
  logical, parameter :: test_n12_4 = erfc_scaled(-12._4) > huge(0._4)
  !WARN: warning: overflow on evaluation of intrinsic function or operation [-Wfolding-exception]
  logical, parameter :: test_n27_4 = erfc_scaled(-27._4) > huge(0._4)
  !WARN: warning: overflow on evaluation of intrinsic function or operation [-Wfolding-exception]
  logical, parameter :: test_n27_8 = erfc_scaled(-27._8) > huge(0._8)
end
