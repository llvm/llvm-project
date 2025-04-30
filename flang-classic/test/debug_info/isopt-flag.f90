!RUN: %flang -O0 -g -S -emit-llvm %s -o - | FileCheck %s --check-prefix=FALSE
!RUN: %flang -O2 -g -S -emit-llvm %s -o - | FileCheck %s --check-prefix=TRUE

!FALSE: isOptimized: false
!TRUE: isOptimized: true

SUBROUTINE sub()
END SUBROUTINE
