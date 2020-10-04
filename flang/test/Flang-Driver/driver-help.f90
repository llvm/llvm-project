! RUN: %flang-new -help 2>&1 | FileCheck %s
! RUN: %flang-new -fc1 -help 2>&1 | FileCheck %s
! RUN: not %flang-new -helps 2>&1 | FileCheck %s --check-prefix=ERROR

! REQUIRES: new-flang-driver

! CHECK:USAGE: flang-new
! CHECK-EMPTY:
! CHECK-NEXT:OPTIONS:
! CHECK-NEXT: -help     Display available options
! CHECK-NEXT: --version Print version information

! ERROR: error: unknown argument '-helps'; did you mean '-help'
