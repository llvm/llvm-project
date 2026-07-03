! This only checks that the command line is correctly passed on to the
! -record-command-line option FC1 option and that the latter does not complain
! about anything. The correct lowering to a module attribute and beyond will
! be checked in other tests.
!
! RUN: %flang -### -target x86_64-unknown-linux -frecord-command-line %s 2>&1 | FileCheck --check-prefix=CHECK-RECORD %s
! RUN: %flang -### -target x86_64-unknown-macosx -frecord-command-line %s 2>&1 | FileCheck --check-prefix=CHECK-RECORD %s
! RUN: not %flang -### -target x86_64-unknown-windows -frecord-command-line %s 2>&1 | FileCheck --check-prefix=CHECK-RECORD-ERROR %s

! RUN: %flang -### -target x86_64-unknown-linux -fno-record-command-line %s 2>&1 | FileCheck --check-prefix=CHECK-NO-RECORD %s
! RUN: %flang -### -target x86_64-unknown-macosx -fno-record-command-line %s 2>&1 | FileCheck --check-prefix=CHECK-NO-RECORD %s
! RUN: %flang -### -target x86_64-unknown-windows -fno-record-command-line %s 2>&1 | FileCheck --check-prefix=CHECK-NO-RECORD %s

! CHECK-RECORD: "-record-command-line"
! CHECK-NO-RECORD-NOT: "-record-command-line"
! CHECK-RECORD-ERROR: error: unsupported option '-frecord-command-line' for target
