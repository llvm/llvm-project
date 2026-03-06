! Check that the driver passes through -f[no-]realloc-lhs:
! RUN: %flang -### -S -frealloc-lhs %s -o - 2>&1 | FileCheck %s --check-prefix=ON
! RUN: %flang -### -S -fno-realloc-lhs %s -o - 2>&1 | FileCheck %s --check-prefix=OFF

! Check that the compiler accepts -f[no-]realloc-lhs:
! RUN: %flang_fc1 -emit-hlfir -frealloc-lhs %s -o -
! RUN: %flang_fc1 -emit-hlfir -fno-realloc-lhs %s -o -

! ON: "-fc1"{{.*}}"-frealloc-lhs"

! OFF: "-fc1"{{.*}}"-fno-realloc-lhs"
