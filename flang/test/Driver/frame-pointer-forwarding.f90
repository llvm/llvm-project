! Test that flang forwards -fno-omit-frame-pointer and -fomit-frame-pointer to Flang frontend

! Default behavior in aarch64, -mframe-pointer=non-leaf-no-reserve.
! RUN: %flang --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NOVALUE
! CHECK-NOVALUE: "-fc1"{{.*}}"-mframe-pointer=non-leaf-no-reserve"

! RUN: %flang -fomit-frame-pointer --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NONEFP
! CHECK-NONEFP: "-fc1"{{.*}}"-mframe-pointer=none"

! RUN: %flang -fno-omit-frame-pointer --target=aarch64-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-NONLEAFFP
! CHECK-NONLEAFFP: "-fc1"{{.*}}"-mframe-pointer=non-leaf-no-reserve"

! RUN: %flang -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer --target=aarch64-none-none -### %s 2>&1 | FileCheck %s --check-prefixes=FC1,FRAME-POINTER-ALL

! Default behavior in x86_64, -mframe-pointer=all at -O0 level.
! RUN: %flang -fno-omit-frame-pointer --target=x86-none-none -fsyntax-only -### %s -o %t 2>&1  | FileCheck %s --check-prefix=CHECK-ALLFP
! CHECK-ALLFP: "-fc1"{{.*}}"-mframe-pointer=all"

! RUN: %flang -O0 --target=x86_64-unknown-linux-gnu -### %s 2>&1 | FileCheck %s --check-prefixes=FC1,FRAME-POINTER-ALL
! RUN: %flang -O0 -momit-leaf-frame-pointer --target=x86_64-unknown-linux-gnu -### %s 2>&1 | FileCheck %s --check-prefixes=FC1,FRAME-POINTER-NON-LEAF
! RUN: %flang -O0 -mno-omit-leaf-frame-pointer --target=x86_64-unknown-linux-gnu -### %s 2>&1 | FileCheck %s --check-prefixes=FC1,FRAME-POINTER-ALL
! RUN: %flang -O0 -fomit-frame-pointer -momit-leaf-frame-pointer --target=x86_64-unknown-linux-gnu -### %s 2>&1 | FileCheck %s --check-prefixes=FC1,FRAME-POINTER-NONE
! RUN: %flang -fno-omit-frame-pointer -momit-leaf-frame-pointer -mno-omit-leaf-frame-pointer --target=x86_64-unknown-linux-gnu -### %s 2>&1 | FileCheck %s --check-prefixes=FC1,FRAME-POINTER-ALL

! At -O1/-O2/-O3 level, default is -mframe-pointer=none.
! RUN: %flang --target=x86_64-unknown-linux-gnu -O2 -### %s 2>&1 | FileCheck %s --check-prefixes=FC1,FRAME-POINTER-NONE

! With -fno-omit-frame-pointer and -mno-omit-leaf-frame-pointer, -mframe-pointer=all.
! RUN: %flang -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer --target=x86_64-unknown-linux-gnu -O2 -### %s 2>&1 | FileCheck %s --check-prefixes=FC1,FRAME-POINTER-ALL

! Without -fno-omit-frame-pointer the leaf option is silently allowed but has no effect, matching Clang's behavior.
! RUN: %flang -momit-leaf-frame-pointer --target=x86_64-unknown-linux-gnu -O2 -### %s -o %t 2>&1  | FileCheck %s --check-prefixes=FC1,FRAME-POINTER-NONE
! RUN: %flang -fno-omit-frame-pointer --target=x86_64-unknown-linux-gnu -O2 -### %s -o %t 2>&1 | FileCheck %s --check-prefixes=FC1,FRAME-POINTER-ALL

! RUN: %flang -fno-omit-frame-pointer -momit-leaf-frame-pointer --target=x86_64-unknown-linux-gnu -O2 -### %s 2>&1 | FileCheck %s --check-prefixes=FC1,FRAME-POINTER-NON-LEAF

! FC1: "-fc1"
! FRAME-POINTER-ALL-SAME: "-mframe-pointer=all"
! FRAME-POINTER-NON-LEAF-SAME: "-mframe-pointer=non-leaf-no-reserve"
! FRAME-POINTER-NONE-SAME: "-mframe-pointer=none"
