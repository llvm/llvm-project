! The REAL kind constants of ISO_FORTRAN_ENV reflect the compiler options of
! the compilation that uses the module, not those used to build the module.
! REQUIRES: x86-registered-target, flang-supports-f128-math
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ALL
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fdebug-unparse -fdisable-real-3 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NO3
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fdebug-unparse -fdisable-real-10 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NO10
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fdebug-unparse -fdisable-real-16 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NO16
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fdebug-unparse  -fdisable-real-3 -fdisable-real-10 -fdisable-real-16 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NONE

! CHECK-NOT: error
use iso_fortran_env
use ieee_arithmetic
use iso_c_binding

! ALL: rb16 = 3_4
! NO3: rb16 = -2_4
! NO10: rb16 = 3_4
! NO16: rb16 = 3_4
! NONE: rb16 = -2_4
integer, parameter :: rb16 = bfloat16

! ALL: r80 = 10_4
! NO3: r80 = 10_4
! NO10: r80 = -2_4
! NO16: r80 = 10_4
! NONE: r80 = -1_4
integer, parameter :: r80 = real80

! ALL: r128 = 16_4
! NO3: r128 = 16_4
! NO10: r128 = 16_4
! NO16: r128 = -1_4
! NONE: r128 = -1_4
integer, parameter :: r128 = real128

! ALL: r64x2 = -2_4
! NO3: r64x2 = -2_4
! NO10: r64x2 = -2_4
! NO16: r64x2 = -1_4
! NONE: r64x2 = -1_4
integer, parameter :: r64x2 = real64x2

! ALL: kinds(*) = [INTEGER(4)::2_4,3_4,4_4,8_4,10_4,16_4]
! NO3: kinds(*) = [INTEGER(4)::2_4,4_4,8_4,10_4,16_4]
! NO10: kinds(*) = [INTEGER(4)::2_4,3_4,4_4,8_4,16_4]
! NO16: kinds(*) = [INTEGER(4)::2_4,3_4,4_4,8_4,10_4]
! NONE: kinds(*) = [INTEGER(4)::2_4,4_4,8_4]
integer, parameter :: kinds(*) = real_kinds

! ALL: cld = 10_4
! NO3: cld = 10_4
! NO10: cld = -1_4
! NO16: cld = 10_4
! NONE: cld = -3_4
integer, parameter :: cld = c_long_double

! ALL: cf128 = 16_4
! NO3: cf128 = 16_4
! NO10: cf128 = 16_4
! NO16: cf128 = -1_4
! NONE: cf128 = -3_4
integer, parameter :: cf128 = c_float128

! CHECK: r32 = 4_4
integer, parameter :: r32 = real32

! CHECK: PRINT *, ieee_arithmetic$ieee_arithmetic$ieee_is_finite_a4(1._4)
print *, ieee_is_finite(1.0)
end
