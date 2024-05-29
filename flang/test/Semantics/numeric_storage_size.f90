! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=CHECK
! RUN: %flang_fc1 -fdebug-unparse -fdefault-integer-8 %s 2>&1 | FileCheck %s --check-prefix=CHECK-I8
! RUN: %flang_fc1 -fdebug-unparse %s -fdefault-real-8 2>&1 | FileCheck %s --check-prefix=CHECK-R8
! RUN: %flang_fc1 -fdebug-unparse %s -fdefault-integer-8 -fdefault-real-8  2>&1 | FileCheck %s --check-prefix=CHECK-I8-R8

use iso_fortran_env

!CHECK-NOT: warning
!CHECK: nss = 32_4
!CHECK-I8: warning: NUMERIC_STORAGE_SIZE from ISO_FORTRAN_ENV is not well-defined when default INTEGER and REAL are not consistent due to compiler options
!CHECK-I8: nss = 32_4
!CHECK-R8: warning: NUMERIC_STORAGE_SIZE from ISO_FORTRAN_ENV is not well-defined when default INTEGER and REAL are not consistent due to compiler options
!CHECK-R8: nss = 32_4
!CHECK-I8-R8: nss = 64_4
integer, parameter :: nss = numeric_storage_size

!CHECK: iss = 32_4
!CHECK-I8: iss = 64_8
!CHECK-R8: iss = 32_4
!CHECK-I8-R8: iss = 64_8
integer, parameter :: iss = storage_size(1)

!CHECK: rss = 32_4
!CHECK-I8: rss = 32_8
!CHECK-R8: rss = 64_4
!CHECK-I8-R8: rss = 64_8
integer, parameter :: rss = storage_size(1.)

!CHECK: zss = 64_4
!CHECK-I8: zss = 64_8
!CHECK-R8: zss = 128_4
!CHECK-I8-R8: zss = 128_8
integer, parameter :: zss = storage_size((1.,0.))

!CHECK: lss = 32_4
!CHECK-I8: lss = 64_8
!CHECK-R8: lss = 32_4
!CHECK-I8-R8: lss = 64_8
integer, parameter :: lss = storage_size(.true.)
end
