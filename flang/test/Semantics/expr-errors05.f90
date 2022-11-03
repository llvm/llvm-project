! RUN: %python %S/test_errors.py %s %flang_fc1
! The components of a complex constructor (extension) must be scalar
!ERROR: Real part of complex constructor must be scalar
complex, parameter :: z1(*) = ([1.,2.], 3.)
!ERROR: Imaginary part of complex constructor must be scalar
complex, parameter :: z2(*) = (4., [5.,6.])
end
