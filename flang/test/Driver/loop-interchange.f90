! RUN: %flang -### -S -floop-interchange %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-INTERCHANGE %s
! RUN: %flang -### -S -fno-loop-interchange %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-INTERCHANGE %s
! RUN: %flang -### -S -O0 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-INTERCHANGE %s
! RUN: %flang -### -S -O1 %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-INTERCHANGE %s
! RUN: %flang -### -S -O2 %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-INTERCHANGE %s
! RUN: %flang -### -S -O3 %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-INTERCHANGE %s
! RUN: %flang -### -S -Os %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-INTERCHANGE %s
! RUN: %flang -### -S -Oz %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-INTERCHANGE %s
! CHECK-LOOP-INTERCHANGE: "-floop-interchange"
! CHECK-NO-LOOP-INTERCHANGE-NOT: "-floop-interchange"

program test
end program
