! The INTEGER, UNSIGNED, and LOGICAL kind constants of ISO_FORTRAN_ENV and the
! 128-bit kinds of ISO_C_BINDING reflect the compiler options of the
! compilation that uses the module, not those used to build the module.
! REQUIRES: x86-registered-target
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ALL
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fdebug-unparse -fdisable-integer-2 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NO2
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fdebug-unparse -fdisable-integer-16 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NO16
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fdebug-unparse  -fdisable-integer-2 -fdisable-integer-16 %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOBOTH

! CHECK-NOT: error
use iso_fortran_env
use iso_c_binding

! ALL: i16 = 2_4
! NO2: i16 = -2_4
! NO16: i16 = 2_4
! NOBOTH: i16 = -2_4
integer, parameter :: i16 = int16

! ALL: i128 = 16_4
! NO2: i128 = 16_4
! NO16: i128 = -1_4
! NOBOTH: i128 = -1_4
integer, parameter :: i128 = int128

! ALL: u16 = 2_4
! NO2: u16 = -2_4
! NO16: u16 = 2_4
! NOBOTH: u16 = -2_4
integer, parameter :: u16 = uint16

! ALL: u128 = 16_4
! NO2: u128 = 16_4
! NO16: u128 = -1_4
! NOBOTH: u128 = -1_4
integer, parameter :: u128 = uint128

! ALL: ikinds(*) = [INTEGER(4)::1_4,2_4,4_4,8_4,16_4]
! NO2: ikinds(*) = [INTEGER(4)::1_4,4_4,8_4,16_4]
! NO16: ikinds(*) = [INTEGER(4)::1_4,2_4,4_4,8_4]
! NOBOTH: ikinds(*) = [INTEGER(4)::1_4,4_4,8_4]
integer, parameter :: ikinds(*) = integer_kinds

! ALL: lkinds(*) = [INTEGER(4)::1_4,2_4,4_4,8_4]
! NO2: lkinds(*) = [INTEGER(4)::1_4,4_4,8_4]
! NO16: lkinds(*) = [INTEGER(4)::1_4,2_4,4_4,8_4]
! NOBOTH: lkinds(*) = [INTEGER(4)::1_4,4_4,8_4]
integer, parameter :: lkinds(*) = logical_kinds

! ALL: ci128 = 16_4
! NO2: ci128 = 16_4
! NO16: ci128 = -1_4
! NOBOTH: ci128 = -1_4
integer, parameter :: ci128 = c_int128_t

! ALL: cu128 = 16_4
! NO2: cu128 = 16_4
! NO16: cu128 = -1_4
! NOBOTH: cu128 = -1_4
integer, parameter :: cu128 = c_uint128_t

! CHECK: cimax = 8_4
integer, parameter :: cimax = c_intmax_t

! CHECK: cumax = 8_4
integer, parameter :: cumax = c_uintmax_t

! CHECK: i32 = 4_4
integer, parameter :: i32 = int32
end
