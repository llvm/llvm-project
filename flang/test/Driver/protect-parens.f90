! RUN: %flang -fprotect-parens -### %s -o %t 2>&1 | FileCheck %s -check-prefix=PROTECT
! RUN: %flang -fno-protect-parens -### %s -o %t 2>&1 | FileCheck %s -check-prefix=NO-PROTECT
! RUN: %flang -### %s -o %t 2>&1 | FileCheck %s -check-prefix=DEFAULT
! RUN: %flang -Ofast -### %s -o %t 2>&1 | FileCheck %s -check-prefix=OFAST
! RUN: %flang -Ofast -fprotect-parens -### %s -o %t 2>&1 | FileCheck %s -check-prefix=OFAST-PROTECT

! PROTECT: "-fprotect-parens"
! NO-PROTECT: "-fno-protect-parens"
! DEFAULT: "-fprotect-parens"
! OFAST: "-fno-protect-parens"
! OFAST-PROTECT: "-fprotect-parens"
