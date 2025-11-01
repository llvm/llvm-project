! Check that the driver passes through -finit-global-zero:
! RUN: %flang -### -S -finit-local-zero %s -o - 2>&1 | FileCheck %s
      
! Check that the compiler accepts -finit-local-zero:
! RUN: %flang_fc1 -emit-hlfir -finit-local-zero %s -o -


! CHECK: "-fc1"{{.*}}"-finit-local-zero"
