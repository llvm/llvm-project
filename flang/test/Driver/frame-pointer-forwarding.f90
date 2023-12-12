! Test that flang-new forwards -fno-omit-frame-pointer and -fomit-frame-pointer Flang frontend
! RUN: %flang -fno-omit-frame-pointer --target=x86-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s
! CHECK: "-mframe-pointer=all"

! RUN: %flang -fno-omit-frame-pointer --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NONLEAFFP
! CHECK-NONLEAFFP: "-mframe-pointer=non-leaf"

! RUN: %flang -fomit-frame-pointer --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NONEFP
! CHECK-NONEFP: "-mframe-pointer=none"
