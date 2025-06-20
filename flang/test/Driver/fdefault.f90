! Ensure argument -fdefault* work as expected.
! TODO: Add checks when actual codegen is possible for this family

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: rm -rf %t/dir-flang  && mkdir -p %t/dir-flang && %flang -fsyntax-only -module-dir %t/dir-flang %s  2>&1
! RUN: cat %t/dir-flang/m.mod | FileCheck %s --check-prefix=NOOPTION
! RUN: rm -rf %t/dir-flang  && mkdir -p %t/dir-flang && %flang -fsyntax-only -fdefault-real-8 -module-dir %t/dir-flang %s  2>&1
! RUN: cat %t/dir-flang/m.mod | FileCheck %s --check-prefix=REAL8
! RUN: rm -rf %t/dir-flang  && mkdir -p %t/dir-flang && %flang -fsyntax-only -fdefault-real-8 -fdefault-double-8 -module-dir %t/dir-flang %s  2>&1
! RUN: cat %t/dir-flang/m.mod | FileCheck %s --check-prefix=DOUBLE8
! RUN: not %flang -fsyntax-only -fdefault-double-8 %s  2>&1 | FileCheck %s --check-prefix=ERROR

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang -fc1)
!-----------------------------------------
! RUN: rm -rf %t/dir-flang  && mkdir -p %t/dir-flang && %flang_fc1 -fsyntax-only -module-dir %t/dir-flang %s  2>&1
! RUN: cat %t/dir-flang/m.mod | FileCheck %s --check-prefix=NOOPTION
! RUN: rm -rf %t/dir-flang  && mkdir -p %t/dir-flang && %flang_fc1 -fsyntax-only -fdefault-real-8 -module-dir %t/dir-flang %s  2>&1
! RUN: cat %t/dir-flang/m.mod | FileCheck %s --check-prefix=REAL8
! RUN: rm -rf %t/dir-flang  && mkdir -p %t/dir-flang && %flang_fc1 -fsyntax-only -fdefault-real-8 -fdefault-double-8 -module-dir %t/dir-flang %s  2>&1
! RUN: cat %t/dir-flang/m.mod | FileCheck %s --check-prefix=DOUBLE8
! RUN: not %flang_fc1 -fsyntax-only -fdefault-double-8 %s  2>&1 | FileCheck %s --check-prefix=ERROR

! NOOPTION: integer(4),parameter::real_kind=4_4
! NOOPTION-NEXT: intrinsic::kind
! NOOPTION-NEXT: integer(4),parameter::double_kind=8_4

! REAL8: integer(4),parameter::real_kind=8_4
! REAL8-NEXT: intrinsic::kind
! REAL8-NEXT: integer(4),parameter::double_kind=16_4

! DOUBLE8: integer(4),parameter::real_kind=8_4
! DOUBLE8-NEXT: intrinsic::kind
! DOUBLE8-NEXT: integer(4),parameter::double_kind=8_4

! ERROR: Use of `-fdefault-double-8` requires `-fdefault-real-8`

module m
  implicit none
  real :: x
  double precision :: y
  integer, parameter :: real_kind = kind(x)            !-fdefault-real-8
  integer, parameter :: double_kind = kind(y)          !-fdefault-double-8
end
