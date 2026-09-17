! Accuracy of ERFC_SCALED at REAL(16), against values computed independently.
!
! REQUIRES: flang-supports-f128-math
! RUN: %flang %s -o %t && %t | FileCheck %s
!
! The reference values below are from mpmath at 50 decimal places, not from
! another Fortran compiler. ERFC_SCALED is not a libm entry point: every
! implementation is somebody's approximation, so two of them agreeing would
! show common ancestry rather than correctness. Only an independent ground
! truth distinguishes "both right" from "both wrong the same way".
!
! THE GRID IS DELIBERATELY NOT REGULAR, AND MUST STAY THAT WAY. A step of 0.5
! -- or any of the tidy values 0.125, 0.5, 1.0, 2.0, 5.0, 13.75 -- gives an
! exactly representable square, and exp(x*x) amplifies the rounding of that
! squaring by a factor of x*x. On such points the error is zero by
! construction and an uncompensated implementation measures 0.4 epsilons; off
! them it measures 123. Tidying this grid would blind the test to the defect
! the implementation exists to avoid.
!
! Both sides of the branch threshold are here, against the same references, so
! that the switch at x = 16 cannot hide a discontinuity.
!
! The tolerance is 2 epsilons rather than 1. Both independently chosen grids
! measured under 1 during development, but erfc at binary128 comes from an
! external library whose future versions may move the last place.

program erfc_scaled_real16
  implicit none
  integer, parameter :: qp = selected_real_kind(33)
  integer, parameter :: n = 11
  real(qp), parameter :: delta = 1.0e-10_qp
  ! Not tidy on purpose: see above.
  real(qp), parameter :: x(n) = [ &
      0.1_qp, 1.7_qp, 15.9_qp, &
      16.0_qp - delta, 16.0_qp + delta, &
      20.0_qp, 64.0_qp, 1000.0_qp, &
      -1.7_qp, -13.75_qp, 0.0_qp]
  ! exp(x*x)*erfc(x), mpmath, 50 dps, rounded to binary128.
  real(qp), parameter :: want(n) = [ &
      0.89645697996912664193188374864404227_qp, &
      0.291663297075343466454843377681505751_qp, &
      0.0354138554979901787367777955827486539_qp, &
      0.0351933778251499452361469135833690837_qp, &
      0.0351933778247117298966017592289243255_qp, &
      0.0281743487410513193186491545344707584_qp, &
      0.00881438653054441357566924341796977317_qp, &
      0.000564189301453387654199745028061695727_qp, &
      35.6949559060252863357458060477745282_qp, &
      2.56939266805496674962846043540050838e+82_qp, &
      1.0_qp]
  real(qp) :: got, rel, worst
  integer :: i

  worst = 0.0_qp
  do i = 1, n
    got = erfc_scaled(x(i))
    if (want(i) == 0.0_qp) then
      rel = abs(got)
    else
      rel = abs(got - want(i))/abs(want(i))
    end if
    worst = max(worst, rel)
  end do

  ! CHECK: worst deviation in epsilons:
  ! CHECK-SAME: within tolerance
  write(*,'(A,ES12.5,A)') 'worst deviation in epsilons: ', &
      worst/epsilon(1.0_qp), &
      merge(' within tolerance', ' TOO LARGE       ', &
            worst <= 2.0_qp*epsilon(1.0_qp))
end program erfc_scaled_real16
