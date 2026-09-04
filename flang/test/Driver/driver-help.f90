! RUN: %flang -help 2>&1 | FileCheck %s --check-prefix=HELP
! RUN: not %flang -helps 2>&1 | FileCheck %s --check-prefix=ERROR

! RUN: %flang_fc1 -help 2>&1 | FileCheck %s --check-prefix=HELP-FC1
! RUN: not %flang_fc1 -helps 2>&1 | FileCheck %s --check-prefix=ERROR

! HELP:USAGE: flang
! HELP-EMPTY:
! HELP-NEXT:OPTIONS:
! HELP: -ffp-sum-reassociation
! HELP: Enable Fortran-standard compliant reassociation within individual REAL and COMPLEX sum expressions
! HELP: may change exact floating-point results
! HELP-NOT: -freal-sum-reassociation

! HELP-FC1:USAGE: flang
! HELP-FC1-EMPTY:
! HELP-FC1-NEXT:OPTIONS:
! HELP-FC1: -ffp-sum-reassociation
! HELP-FC1: Enable Fortran-standard compliant reassociation within individual REAL and COMPLEX sum expressions
! HELP-FC1: may change exact floating-point results
! HELP-FC1-NOT: -freal-sum-reassociation

! ERROR: error: unknown argument '-helps'; did you mean '-help'
