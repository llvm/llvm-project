! Ensure argument -flarge-sizes works as expected.
! TODO: Add checks when actual codegen is possible.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: rm -rf %t/dir-flang  && mkdir -p %t/dir-flang && %flang -fsyntax-only -module-dir %t/dir-flang %s  2>&1
! RUN: cat %t/dir-flang/m.mod | FileCheck %s --check-prefix=NOLARGE
! RUN: rm -rf %t/dir-flang  && mkdir -p %t/dir-flang && %flang -fsyntax-only -flarge-sizes -module-dir %t/dir-flang %s  2>&1
! RUN: cat %t/dir-flang/m.mod | FileCheck %s --check-prefix=LARGE

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang -fc1)
!-----------------------------------------
! RUN: rm -rf %t/dir-flang  && mkdir -p %t/dir-flang && %flang_fc1 -fsyntax-only -module-dir %t/dir-flang %s  2>&1
! RUN: cat %t/dir-flang/m.mod | FileCheck %s --check-prefix=NOLARGE
! RUN: rm -rf %t/dir-flang  && mkdir -p %t/dir-flang && %flang_fc1 -fsyntax-only -flarge-sizes -module-dir %t/dir-flang %s  2>&1
! RUN: cat %t/dir-flang/m.mod | FileCheck %s --check-prefix=LARGE

! NOLARGE: real(4)::z(1_8:10_8)
! NOLARGE-NEXT: integer(4),parameter::size_kind=4_4

! LARGE: real(4)::z(1_8:10_8)
! LARGE-NEXT: integer(4),parameter::size_kind=8_4

module m
  implicit none
  real :: z(10)
  integer, parameter :: size_kind = kind(ubound(z, 1)) !-flarge-sizes
end
