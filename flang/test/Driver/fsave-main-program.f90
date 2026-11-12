! Check that the driver passes through -fsave-main-program:
! RUN: %flang -### -S -fsave-main-program %s -o - 2>&1 | FileCheck %s
! CHECK: "-fc1"{{.*}}"-fsave-main-program"

! RUN: %flang -### -S -fno-save-main-program %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK2
! CHECK2: "-fc1"{{.*}}"-fno-save-main-program"

! Check that the compiler accepts -fsave-main-program:
! RUN: %flang_fc1 -emit-hlfir -fsave-main-program %s -o -
