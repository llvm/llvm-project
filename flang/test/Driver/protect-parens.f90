! RUN: %flang -### %s -o %t 2>&1 | FileCheck %s -check-prefix=PROTECT
! RUN: %flang -fprotect-parens -### %s -o %t 2>&1 | FileCheck %s -check-prefix=PROTECT
! RUN: %flang -fno-protect-parens -### %s -o %t 2>&1 | FileCheck %s -check-prefix=NO-PROTECT
! RUN: %flang -Ofast -### %s -o %t 2>&1 | FileCheck %s -check-prefix=NO-PROTECT
! RUN: %flang -Ofast -fprotect-parens -### %s -o %t 2>&1 | FileCheck %s -check-prefix=PROTECT

! Note: -fprotect-parens is not passed to the frontend, because it's the
! default. Only -fno-protect-parens is passed to turn off the default.
! Thus, in case of PROTECT, we don't want to have any -f[no-]protect-parens
! options.

! PROTECT-NOT: "-f{{.*}}protect-parens"
! NO-PROTECT: "-fno-protect-parens"
