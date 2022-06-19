! RUN: %python %S/test_errors.py %s %flang_fc1
! Fortran syntax considers signed int literals in complex literals
! to be a distinct production, not an application of unary +/- to
! an unsigned int literal, so they're used here to test overflow
! on signed int literal constants.  The literals are tested here
! as part of expressions that name resolution must analyze.

complex, parameter :: okj1 = 127_1, okz1 = (+127_1, -128_1)
!ERROR: Integer literal is too large for INTEGER(KIND=1)
complex, parameter :: badj1 = 128_1
!ERROR: Integer literal is too large for INTEGER(KIND=1)
complex, parameter :: badz1 = (+128_1, 0)
complex, parameter :: okj1a = 128_2
complex, parameter :: okz1a = (+128_2, 0)

complex, parameter :: okj2 = 32767_2, okz2 = (+32767_2, -32768_2)
!ERROR: Integer literal is too large for INTEGER(KIND=2)
complex, parameter :: badj2 = 32768_2
!ERROR: Integer literal is too large for INTEGER(KIND=2)
complex, parameter :: badz2 = (+32768_2, 0)
complex, parameter :: okj2a = 32768_4
complex, parameter :: okz2a = (+32768_4, 0)

complex, parameter :: okj4 = 2147483647_4, okz4 = (+2147483647_4, -2147483648_4)
!ERROR: Integer literal is too large for INTEGER(KIND=4)
complex, parameter :: badj4 = 2147483648_4
!ERROR: Integer literal is too large for INTEGER(KIND=4)
complex, parameter :: badz4 = (+2147483648_4, 0)
complex, parameter :: okj4a = 2147483648_8
complex, parameter :: okz4a = (+2147483648_8, 0)

complex, parameter :: okj4d = 2147483647, okz4d = (+2147483647, -2147483648)
!WARNING: Integer literal is too large for default INTEGER(KIND=4); assuming INTEGER(KIND=8)
complex, parameter :: badj4dext = 2147483648
!WARNING: Integer literal is too large for default INTEGER(KIND=4); assuming INTEGER(KIND=8)
complex, parameter :: badz4dext = (+2147483648, 0)

complex, parameter :: okj8 = 9223372036854775807_8, okz8 = (+9223372036854775807_8, -9223372036854775808_8)
!ERROR: Integer literal is too large for INTEGER(KIND=8)
complex, parameter :: badj8 = 9223372036854775808_8
!ERROR: Integer literal is too large for INTEGER(KIND=8)
complex, parameter :: badz8 = (+9223372036854775808_8, 0)
complex, parameter :: okj8a = 9223372036854775808_16
complex, parameter :: okz8a = (+9223372036854775808_16, 0)

complex, parameter :: okj16 = 170141183460469231731687303715884105727_16
complex, parameter :: okz16 = (+170141183460469231731687303715884105727_16, -170141183460469231731687303715884105728_16)
!ERROR: Integer literal is too large for INTEGER(KIND=16)
complex, parameter :: badj16 = 170141183460469231731687303715884105728_16
!ERROR: Integer literal is too large for INTEGER(KIND=16)
complex, parameter :: badz16 = (+170141183460469231731687303715884105728_16, 0)

end
