! RUN: %flang -funsafe-cray-pointers -### %s 2>&1 | FileCheck %s

! CHECK: "-mmlir -funsafe-cray-pointers"
! CHECK-NOT: "-mmlir -funsafe-cray-pointers"
