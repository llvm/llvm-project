! RUN: %flang -ffast-real-mod -### -c %s 2>&1 | FileCheck %s -check-prefix CHECK-FAST-REAL-MOD
! RUN: %flang -fno-fast-real-mod -### -c %s 2>&1 | FileCheck %s -check-prefix CHECK-NO-FAST-REAL-MOD

! CHECK-FAST-REAL-MOD: "-ffast-real-mod"
! CHECK-NO-FAST-REAL-MOD: "-fno-fast-real-mod"

program test
    ! nothing to be done in here
end program test
