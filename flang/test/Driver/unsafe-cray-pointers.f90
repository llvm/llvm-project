! RUN: %flang -funsafe-cray-pointers -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-ON
! RUN: %flang -fno-unsafe-cray-pointers -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-OFF

! CHECK-ON: "-mmlir" "-funsafe-cray-pointers"
! CHECK-OFF-NOT: "-mmlir" "-funsafe-cray-pointers"
