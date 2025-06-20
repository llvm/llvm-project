! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror -pedantic
!PORTABILITY: nonstandard usage: generalized COMPLEX constructor [-Wno-complex-constructor]
!PORTABILITY: Real part of complex constructor is not scalar [-Wno-complex-constructor]
complex, parameter :: z1(*) = ([1.,2.], 3.)
!PORTABILITY: nonstandard usage: generalized COMPLEX constructor [-Wno-complex-constructor]
!PORTABILITY: Imaginary part of complex constructor is not scalar [-Wno-complex-constructor]
complex, parameter :: z2(*) = (4., [5.,6.])
real, parameter :: aa(*) = [7.,8.]
!PORTABILITY: Real part of complex literal constant is not scalar [-Wno-complex-constructor]
complex, parameter :: z3(*) = (aa, 9.)
!PORTABILITY: Imaginary part of complex literal constant is not scalar [-Wno-complex-constructor]
complex, parameter :: z4(*) = (10., aa)
!We need a nonzero exit status to make test_errors.py look at messages :-(
!WARNING: division by zero [-Wno-folding-exception]
real, parameter :: xxx = 1./0.
end
