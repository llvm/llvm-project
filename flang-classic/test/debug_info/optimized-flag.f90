!RUN: %flang -O0 -g -S -emit-llvm %s -o - | FileCheck %s --check-prefix=UNOPT
!RUN: %flang -O2 -g -S -emit-llvm %s -o - | FileCheck %s --check-prefix=OPT

!UNOPT-NOT: DISPFlagOptimized
!OPT: DISPFlagOptimized

SUBROUTINE sub()
END SUBROUTINE
