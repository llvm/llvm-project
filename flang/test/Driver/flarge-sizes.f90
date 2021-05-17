! Ensure argument -flarge-sizes works as expected.
! TODO: Add checks when actual codegen is possible.

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: rm -rf %t/dir-flang-new  && mkdir -p %t/dir-flang-new && %flang -fsyntax-only -module-dir %t/dir-flang-new %s  2>&1
! RUN: cat %t/dir-flang-new/m.mod | FileCheck %s --check-prefix=NOLARGE
! RUN: rm -rf %t/dir-flang-new  && mkdir -p %t/dir-flang-new && %flang -fsyntax-only -flarge-sizes -module-dir %t/dir-flang-new %s  2>&1
! RUN: cat %t/dir-flang-new/m.mod | FileCheck %s --check-prefix=LARGE

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: rm -rf %t/dir-flang-new  && mkdir -p %t/dir-flang-new && %flang_fc1 -fsyntax-only -module-dir %t/dir-flang-new %s  2>&1
! RUN: cat %t/dir-flang-new/m.mod | FileCheck %s --check-prefix=NOLARGE
! RUN: rm -rf %t/dir-flang-new  && mkdir -p %t/dir-flang-new && %flang_fc1 -fsyntax-only -flarge-sizes -module-dir %t/dir-flang-new %s  2>&1
! RUN: cat %t/dir-flang-new/m.mod | FileCheck %s --check-prefix=LARGE

!-----------------------------------------
! EXPECTED OUTPUT WITHOUT -flarge-sizes
!-----------------------------------------
! NOLARGE: real(4)::z(1_8:10_8)
! NOLARGE-NEXT: integer(4),parameter::size_kind=4_4

!-----------------------------------------
! EXPECTED OUTPUT FOR -flarge-sizes
!-----------------------------------------
! LARGE: real(4)::z(1_8:10_8)
! LARGE-NEXT: integer(4),parameter::size_kind=8_4

module m
  implicit none
  real :: z(10)
  integer, parameter :: size_kind = kind(ubound(z, 1)) !-flarge-sizes
end
