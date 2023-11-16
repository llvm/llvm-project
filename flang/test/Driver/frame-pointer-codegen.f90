! Test that flang-new forwards -fno-omit-frame-pointer and -fomit-frame-pointer Flang frontend
! RUN: %flang -fno-omit-frame-pointer -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s
! CHECK: "-mframe-pointer=non-leaf"

! RUN: %flang -fomit-frame-pointer -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NONEFP
! CHECK-NONEFP: "-mframe-pointer=none"

! RUN: %flang -fomit-frame-pointer -fsyntax-only -### -Xflang -mframe-pointer=all %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-ALLFP
! CHECK-ALLFP: "-mframe-pointer=all"
