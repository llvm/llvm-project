! RUN: %flang -### -S -floop-interchange %s 2>&1 | FileCheck -check-prefix=CHECK-LOOP-INTERCHANGE %s
! RUN: %flang -### -S -fno-loop-interchange %s 2>&1 | FileCheck -check-prefix=CHECK-NO-LOOP-INTERCHANGE %s
! CHECK-LOOP-INTERCHANGE: "-floop-interchange"
! CHECK-NO-LOOP-INTERCHANGE: "-fno-loop-interchange"

program test
end program
