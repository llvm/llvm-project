! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
!PORTABILITY: Real part of complex constructor is not scalar
complex, parameter :: z1(*) = ([1.,2.], 3.)
!PORTABILITY: Imaginary part of complex constructor is not scalar
complex, parameter :: z2(*) = (4., [5.,6.])
real, parameter :: aa(*) = [7.,8.]
!PORTABILITY: Real part of complex literal constant is not scalar
complex, parameter :: z3(*) = (aa, 9.)
!PORTABILITY: Imaginary part of complex literal constant is not scalar
complex, parameter :: z4(*) = (10., aa)
!We need a nonzero exit status to make test_errors.py look at messages :-(
!WARNING: division by zero
real, parameter :: xxx = 1./0.
end
