! Check that the driver passes through -fsave-main-program:
! RUN: %flang -### -S -fsave-main-program %s -o - 2>&1 | FileCheck %s
! Check that the compiler accepts -fsave-main-program:
! RUN: %flang_fc1 -emit-hlfir -fsave-main-program %s -o -
! CHECK: "-fc1"{{.*}}"-fsave-main-program"
