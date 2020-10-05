! RUN: %flang-new --version 2>&1 | FileCheck %s
! RUN: not %flang-new --versions 2>&1 | FileCheck %s --check-prefix=ERROR

! REQUIRES: new-flang-driver

! CHECK:flang-new version 
! CHECK-NEXT:Target:
! CHECK-NEXT:Thread model:
! CHECK-NEXT:InstalledDir:

! ERROR: flang-new: error: unsupported option '--versions'; did you mean '--version'?
