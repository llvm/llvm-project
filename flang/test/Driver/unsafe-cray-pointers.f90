! RUN: %flang -funsafe-cray-pointers -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-ON
! RUN: %flang -fno-unsafe-cray-pointers -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-OFF

! CHECK-ON: "-mmlir" "-unsafe-cray-pointers"
! CHECK-OFF-NOT: "-mmlir" "-unsafe-cray-pointers"
