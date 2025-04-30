       ! RUN: %flang -S -emit-llvm %s -o - | FileCheck %s
      PROGRAM MINIMAL
      IMPLICIT NONE
      REAL OLD(1)
      POINTER(IOLD, OLD)
       ! CHECK: %struct.STATICS1 = type <{ ptr  }>
      DATA IOLD/-1/
       ! CHECK: @.STATICS1 = internal global %struct.STATICS1 <{ ptr inttoptr (i64 -1 to ptr) }>
      END PROGRAM MINIMAL
