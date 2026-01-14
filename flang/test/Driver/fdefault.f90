! Ensure argument -fdefault* work as expected.
! TODO: Add checks when actual codegen is possible for this family
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:    %flang_fc1 -fsyntax-only -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=NOOPTION
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:   %flang_fc1 -fsyntax-only -fdefault-integer-4 -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=INTEGER4
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:   %flang_fc1 -fsyntax-only -fdefault-real-4 -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=REAL4
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:   %flang_fc1 -fsyntax-only -fdefault-integer-4 -fdefault-real-4 \
! RUN:       -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=BOTH4
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:   %flang_fc1 -fsyntax-only -fdefault-integer-8 -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=INTEGER8
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:   %flang -fsyntax-only -fdefault-real-8 -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=REAL8
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:   %flang_fc1 -fsyntax-only -fdefault-integer-8 -fdefault-real-8 \
! RUN:       -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=BOTH8
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:   %flang_fc1 -fsyntax-only -fdefault-real-8 -fdefault-double-8 \
! RUN:       -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=DOUBLE8
!
! RUN: not %flang_fc1 -fsyntax-only -fdefault-double-8 %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=ERROR
!
! The last occurrence of -fdefault-* "wins"
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:   %flang_fc1 -fsyntax-only -fdefault-integer-4 -fdefault-integer-8 \
! RUN:       -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=INTEGER8
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:   %flang_fc1 -fsyntax-only -fdefault-integer-8 -fdefault-integer-4 \
! RUN:       -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=INTEGER4
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:   %flang_fc1 -fsyntax-only -fdefault-real-4 -fdefault-real-8 \
! RUN:       -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=REAL8
!
! RUN: rm -rf %t && mkdir -p %t && \
! RUN:   %flang_fc1 -fsyntax-only -fdefault-real-8 -fdefault-real-4 \
! RUN:       -module-dir %t %s
! RUN: cat %t/m.mod | FileCheck %s --check-prefix=REAL4
!
! NOOPTION: integer(4),parameter::integer_kind=4_4
! NOOPTION-NEXT: intrinsic::kind
! NOOPTION-NEXT: integer(4),parameter::real_kind=4_4
! NOOPTION-NEXT: integer(4),parameter::double_kind=8_4
!
! INTEGER4: integer(4),parameter::integer_kind=4_4
! INTEGER4-NEXT: intrinsic::kind
! INTEGER4-NEXT: integer(4),parameter::real_kind=4_4
! INTEGER4-NEXT: integer(4),parameter::double_kind=8_4
!
! REAL4: integer(4),parameter::integer_kind=4_4
! REAL4-NEXT: intrinsic::kind
! REAL4-NEXT: integer(4),parameter::real_kind=4_4
! REAL4-NEXT: integer(4),parameter::double_kind=8_4
!
! BOTH4: integer(4),parameter::integer_kind=4_4
! BOTH4-NEXT: intrinsic::kind
! BOTH4-NEXT: integer(4),parameter::real_kind=4_4
! BOTH4-NEXT: integer(4),parameter::double_kind=8_4
!
! INTEGER8: integer(8),parameter::integer_kind=8_8
! INTEGER8-NEXT: intrinsic::kind
! INTEGER8-NEXT: integer(8),parameter::real_kind=4_8
! INTEGER8-NEXT: integer(8),parameter::double_kind=8_8
!
! REAL8: integer(4),parameter::integer_kind=4_4
! REAL8-NEXT: intrinsic::kind
! REAL8-NEXT: integer(4),parameter::real_kind=8_4
! REAL8-NEXT: integer(4),parameter::double_kind=16_4
!
! BOTH8: integer(8),parameter::integer_kind=8_8
! BOTH8-NEXT: intrinsic::kind
! BOTH8-NEXT: integer(8),parameter::real_kind=8_8
! BOTH8-NEXT: integer(8),parameter::double_kind=16_8
!
! DOUBLE8: integer(4),parameter::integer_kind=4_4
! DOUBLE8-NEXT: intrinsic::kind
! DOUBLE8-NEXT: integer(4),parameter::real_kind=8_4
! DOUBLE8-NEXT: integer(4),parameter::double_kind=8_4
!
! ERROR: Use of `-fdefault-double-8` requires `-fdefault-real-8`

module m
  implicit none
  integer :: i
  real :: x
  double precision :: y
  integer, parameter :: integer_kind = kind(i)         !-fdefault-integer-*
  integer, parameter :: real_kind = kind(x)            !-fdefault-real-*
  integer, parameter :: double_kind = kind(y)          !-fdefault-double-*
end
