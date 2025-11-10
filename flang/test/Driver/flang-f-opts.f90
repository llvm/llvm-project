! Test for errors and warnings generated when parsing driver options. You can
! use this file for relatively small tests and to avoid creating new test files.

! RUN: %flang -### -S -O4 -ffp-contract=on %s 2>&1 | FileCheck %s

! CHECK: warning: the argument 'on' is not supported for option 'ffp-contract='. Mapping to 'ffp-contract=off'
! CHECK: warning: -O4 is equivalent to -O3
! CHECK-LABEL: "-fc1"
! CHECK: -ffp-contract=off
! CHECK: -O3

! RUN: %flang -### -S -fprofile-generate %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-GENERATE-LLVM %s
! CHECK-PROFILE-GENERATE-LLVM: "-fprofile-generate"
! RUN: %flang -### -S -fprofile-use=%S %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE-DIR %s
! CHECK-PROFILE-USE-DIR: "-fprofile-use={{.*}}"

! RUN: %flang -### -fbuiltin %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=WARN-BUILTIN
! WARN-BUILTIN: warning: '-fbuiltin' is not valid for Fortran

! RUN: %flang -### -fno-builtin %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=WARN-NO-BUILTIN
! WARN-NO-BUILTIN: warning: '-fno-builtin' is not valid for Fortran

! RUN: %flang -### -fbuiltin -fno-builtin %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=WARN-BUILTIN-MULTIPLE
! WARN-BUILTIN-MULTIPLE: warning: '-fbuiltin' is not valid for Fortran
! WARN-BUILTIN-MULTIPLE: warning: '-fno-builtin' is not valid for Fortran

! When emitting an error with a suggestion, ensure that the diagnostic message
! uses '-Xflang' instead of '-Xclang'. This is typically emitted when an option
! that is available for `flang -fc1` is passed to `flang`. We use -complex-range
! since it is only available for fc1. If this option is ever exposed to `flang`,
! a different option will have to be used in the test below.
!
! RUN: not %flang -### -complex-range=full %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix UNKNOWN-SUGGEST
!
! UNKNOWN-SUGGEST: error: unknown argument '-complex-range=full';
! UNKNOWN-SUGGEST-SAME: did you mean '-Xflang -complex-range=full'
!
! RUN: not %flang -### -not-an-option %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix UNKNOWN-NO-SUGGEST
!
! UNKNOWN-NO-SUGGEST: error: unknown argument: '-not-an-option'{{$}}
