! RUN: %flang -fprotect-parens -### %s -o %t 2>&1 | FileCheck %s -check-prefix=PROTECT
! RUN: %flang -fno-protect-parens -### %s -o %t 2>&1 | FileCheck %s -check-prefix=NO-PROTECT
! RUN: %flang -### %s -o %t 2>&1 | FileCheck %s -check-prefix=DEFAULT
! RUN: %flang -Ofast -### %s -o %t 2>&1 | FileCheck %s -check-prefix=OFAST
! RUN: %flang -Ofast -fprotect-parens -### %s -o %t 2>&1 | FileCheck %s -check-prefix=OFAST-PROTECT

! Note: -fprotect-parens is not passed to the frontend, because it's the
! default. Only -fno-protect-parens is passed to turn off the default.
! PROTECT-NOT: "-f{{.*}}protect-parens"
! NO-PROTECT: "-fno-protect-parens"
! DEFAULT-NOT: "-f{{.*}}protect-parens"
! OFAST: "-fno-protect-parens"
! OFAST-PROTECT-NOT: "-f{{.*}}protect-parens"
