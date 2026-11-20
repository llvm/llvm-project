! Check that the driver passes through -f[no-]check-already-allocated:
! RUN: %flang -### -S -fcheck-already-allocated %s -o - 2>&1 | FileCheck %s --check-prefix=ON
! RUN: %flang -### -S -fno-check-already-allocated %s -o - 2>&1 | FileCheck %s --check-prefix=OFF

! Check that the compiler accepts -f[no-]check-already-allocated:
! RUN: %flang_fc1 -emit-hlfir -fcheck-already-allocated %s -o -
! RUN: %flang_fc1 -emit-hlfir -fno-check-already-allocated %s -o -

! ON: "-fc1"{{.*}}"-fcheck-already-allocated"

! OFF: "-fc1"{{.*}}"-fno-check-already-allocated"
