! RUN: %flang -v -S -emit-llvm -o - %s --target=aarch64-linux-gnu -fno-pie 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-STATIC,CHECK-STATIC-IR

! RUN: %flang -v -S -emit-llvm -o - %s --target=aarch64-linux-gnu 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-PIE-LEVEL2,CHECK-PIE-LEVEL2-IR
! RUN: %flang -v -S -emit-llvm -o - %s --target=aarch64-linux-gnu -fpie 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-PIE-LEVEL1,CHECK-PIE-LEVEL1-IR
! RUN: %flang -v -S -emit-llvm -o - %s --target=aarch64-linux-gnu -fPIE 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-PIE-LEVEL2,CHECK-PIE-LEVEL2-IR

! RUN: %flang -v -S -emit-llvm -o - %s --target=aarch64-linux-gnu -fpic 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-PIC-LEVEL1,CHECK-PIC-LEVEL1-IR
! RUN: %flang -v -S -emit-llvm -o - %s --target=aarch64-linux-gnu -fPIC 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-PIC-LEVEL2,CHECK-PIC-LEVEL2-IR

! RUN: %flang -v -### -o - %s --target=i386-apple-darwin -mdynamic-no-pic 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC-NO-PIC-32
! RUN: %flang -v -### -o - %s --target=x86_64-apple-darwin -mdynamic-no-pic 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC-NO-PIC-64

! RUN: %flang -v -### -o - %s --target=arm-none-eabi -fropi 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-ROPI
! RUN: %flang -v -### -o - %s --target=arm-none-eabi -frwpi 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-RWPI
! RUN: %flang -v -### -o - %s --target=arm-none-eabi -fropi -frwpi 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-ROPI-RWPI


! CHECK: -fc1


!! -fno-pie.
! CHECK-STATIC: -mrelocation-model static
! CHECK-STATIC-NOT: -pic

! CHECK-STATIC-IR-NOT: {{PIE|PIC}} Level


!! -fpic.
! CHECK-PIC-LEVEL1: -mrelocation-model pic -pic-level 1
! CHECK-PIC-LEVEL1-NOT: -pic-is-pie

! CHECK-PIC-LEVEL1-IR-NOT: "PIE Level"
! CHECK-PIC-LEVEL1-IR: !"PIC Level", i32 1}
! CHECK-PIC-LEVEL1-IR-NOT: "PIE Level"


!! -fPIC.
! CHECK-PIC-LEVEL2: -mrelocation-model pic -pic-level 2
! CHECK-PIC-LEVEL2-NOT: -pic-is-pie

! CHECK-PIC-LEVEL2-IR-NOT: "PIE Level"
! CHECK-PIC-LEVEL2-IR: !"PIC Level", i32 2}
! CHECK-PIC-LEVEL2-IR-NOT: "PIE Level"


!! -fpie.
! CHECK-PIE-LEVEL1: -mrelocation-model pic -pic-level 1 -pic-is-pie
! CHECK-PIE-LEVEL1-IR: !"PIC Level", i32 1}
! CHECK-PIE-LEVEL1-IR: !"PIE Level", i32 1}


!! -fPIE.
! CHECK-PIE-LEVEL2: -mrelocation-model pic -pic-level 2 -pic-is-pie
! CHECK-PIE-LEVEL2-IR: !"PIC Level", i32 2}
! CHECK-PIE-LEVEL2-IR: !"PIE Level", i32 2}


!! -mdynamic-no-pic
! CHECK-DYNAMIC-NO-PIC-32: "-mrelocation-model" "dynamic-no-pic"
! CHECK-DYNAMIC-NO-PIC-32-NOT: "-pic-level"
! CHECK-DYNAMIC-NO-PIC-32-NOT: "-pic-is-pie"

! CHECK-DYNAMIC-NO-PIC-64: "-mrelocation-model" "dynamic-no-pic" "-pic-level" "2"
! CHECK-DYNAMIC-NO-PIC-64-NOT: "-pic-is-pie"


!! -fropi -frwpi
! CHECK-ROPI: "-mrelocation-model" "ropi"
! CHECK-ROPI-NOT: "-pic

! CHECK-RWPI: "-mrelocation-model" "rwpi"
! CHECK-RWPI-NOT: "-pic

! CHECK-ROPI-RWPI: "-mrelocation-model" "ropi-rwpi"
! CHECK-ROPI-RWPI-NOT: "-pic
