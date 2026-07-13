! Check that the driver passes through -f[no-]init-global-zero:
! RUN: %flang -### -S -finit-global-zero %s -o - 2>&1 | FileCheck --check-prefix=CHECK-POS %s
! RUN: %flang -### -S -fno-init-global-zero %s -o - 2>&1 | FileCheck --check-prefix=CHECK-NEG %s
! Check that the compiler accepts -f[no-]init-global-zero:
! RUN: %flang_fc1 -emit-hlfir -finit-global-zero %s -o -
! RUN: %flang_fc1 -emit-hlfir -fno-init-global-zero %s -o -

! CHECK-POS: "-fc1"{{.*}}"-finit-global-zero"
! CHECK-NEG: "-fc1"{{.*}}"-fno-init-global-zero"
