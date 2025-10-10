! RUN: %flang -fno-fast-real-mod -### -c %s 2>&1 | FileCheck %s -check-prefix CHECK-NO-FAST-REAL-MOD

! CHECK-NO-FAST-REAL-MOD: "-fno-fast-real-mod"

program test
    ! nothing to be done in here
end program test
