! RUN: %S/test_folding.sh %s %t %f18
! Test folding of isnan() extension
module m1
  logical, parameter :: results(*) = isnan([ &
    0., &
    -0., &
!WARN: division by zero
    1./0., &
!WARN: invalid argument on division
    0./0., &
    real(z'7ff80001',kind=4), &
    real(z'fff80001',kind=4), &
    real(z'7ffc0000',kind=4), &
    real(z'7ffe0000',kind=4) ])
  logical, parameter :: expected(*) = [ &
    .false., .false., .false., .true., .true., .true., .true., .true. ]
  logical, parameter :: test_isnan = all(results .eqv. expected)
end module
