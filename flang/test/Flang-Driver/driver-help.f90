! RUN: %flang-new -help 2>&1 | FileCheck %s --check-prefix=HELP
! RUN: %flang-new -fc1 -help 2>&1 | FileCheck %s --check-prefix=HELP-FC1
! RUN: not %flang-new -helps 2>&1 | FileCheck %s --check-prefix=ERROR

! REQUIRES: new-flang-driver

! HELP:USAGE: flang-new
! HELP-EMPTY:
! HELP-NEXT:OPTIONS:
! HELP-NEXT: -fcolor-diagnostics    Enable colors in diagnostics
! HELP-NEXT: -fno-color-diagnostics Disable colors in diagnostics
! HELP-NEXT: -help     Display available options
! HELP-NEXT: --version Print version information

! HELP-FC1:USAGE: flang-new
! HELP-FC1-EMPTY:
! HELP-FC1-NEXT:OPTIONS:
! HELP-FC1-NEXT: -help     Display available options
! HELP-FC1-NEXT: --version Print version information

! ERROR: flang-new: error: unknown argument '-helps'; did you mean '-help'
