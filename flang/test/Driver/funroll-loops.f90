! RUN: %flang -### -funroll-loops %s 2>&1 | FileCheck %s -check-prefix UNROLL
! RUN: %flang -### -fno-unroll-loops %s 2>&1 | FileCheck %s -check-prefix NO-UNROLL

! UNROLL: "-funroll-loops"
! NO-UNROLL: "-fno-unroll-loops"
