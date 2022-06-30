! Verify that in contrast to Clang, Flang does not default to generating position independent executables/code

!-------------
! RUN COMMANDS
!-------------
! RUN: %flang -### %s --target=aarch64-linux-gnu 2>&1 | FileCheck %s --check-prefix=CHECK-NOPIE
! RUN: %flang -### %s --target=aarch64-linux-gnu -fno-pie 2>&1 | FileCheck %s --check-prefix=CHECK-NOPIE

! RUN: %flang -### %s --target=aarch64-linux-gnu -fpie 2>&1 | FileCheck %s --check-prefix=CHECK-PIE

!----------------
! EXPECTED OUTPUT
!----------------
! CHECK-NOPIE: "-fc1"
! CHECk-NOPIE-NOT: "-fpic"
! CHECK-NOPIE: "{{.*}}ld"
! CHECK-NOPIE-NOT: "-pie"

! CHECK-PIE: "-fc1"
!! TODO Once Flang supports `-fpie`, it //should// use -fpic when invoking `flang -fc1`. Update the following line once `-fpie` is
! available.
! CHECk-PIE-NOT: "-fpic"
! CHECK-PIE: "{{.*}}ld"
! CHECK-PIE-NOT: "-pie"
