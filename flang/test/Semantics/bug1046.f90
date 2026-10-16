! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
!WARNING: INTEGER(4) 0**0 is not defined [-Wfolding-exception]
print *, 0**0
!WARNING: REAL/COMPLEX 0**0 is not defined [-Wfolding-exception]
print *, 0**0.
!WARNING: invalid argument on power with INTEGER exponent [-Wfolding-exception]
print *, 0.0**0
!WARNING: REAL/COMPLEX 0**0 is not defined [-Wfolding-exception]
print *, 0.0**0.
!WARNING: invalid argument on power with INTEGER exponent [-Wfolding-exception]
print *, (0.0, 0.0)**0
!WARNING: REAL/COMPLEX 0**0 is not defined [-Wfolding-exception]
print *, (0.0, 0.0)**0.
print *, (0.0, 0.0)**2.5
end
