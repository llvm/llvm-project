! Test that flang forwards -fno-omit-frame-pointer and -fomit-frame-pointer Flang frontend
! RUN: %flang --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NOVALUE
! CHECK-NOVALUE: "-fc1"{{.*}}"-mframe-pointer=non-leaf"

! RUN: %flang -fomit-frame-pointer --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NONEFP
! CHECK-NONEFP: "-fc1"{{.*}}"-mframe-pointer=none"

! RUN: %flang -fno-omit-frame-pointer --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NONLEAFFP
! CHECK-NONLEAFFP: "-fc1"{{.*}}"-mframe-pointer=non-leaf"

! RUN: %flang -fno-omit-frame-pointer --target=x86-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-ALLFP
! CHECK-ALLFP: "-fc1"{{.*}}"-mframe-pointer=all"
