
!-----------
! RUN LINES
!-----------
! RUN: %flang --version 2>&1 | FileCheck %s
! RUN: not %flang --versions 2>&1 | FileCheck %s --check-prefix=ERROR

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! CHECK: flang-new version
! CHECK-NEXT: Target:
! CHECK-NEXT: Thread model:
! CHECK-NEXT: InstalledDir:

! ERROR: flang-new: error: unsupported option '--versions'; did you mean '--version'?
