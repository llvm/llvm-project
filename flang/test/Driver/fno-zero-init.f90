! Check that the driver passes through -fno-zero-init-global-without-init:
! RUN: %flang -### -S -fno-zero-init-global-without-init %s -o - 2>&1 | FileCheck %s
! Check that the compiler accepts -fno-zero-init-global-without-init:
! RUN: %flang_fc1 -emit-hlfir -fno-zero-init-global-without-init %s -o - 
! CHECK: "-fc1"{{.*}}"-fno-zero-init-global-without-init"
