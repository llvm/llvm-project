! Check that -ftime-report flag is passed as-is to fc1. The value of the flag
! is only checked there. This behavior intentionally mirrors that of clang.
!
! RUN: %flang -### -c -ftime-report %s 2>&1 | FileCheck %s --check-prefix=CHECK-DRIVER

! TODO: Currently, detailed timing of LLVM IR optimization and code generation
! passes is not supported. When that is done, add more checks here to make sure
! the output is as expected.

! RUN: %flang -c -ftime-report -O0 %s 2>&1 | FileCheck %s --check-prefix=CHECK-COMMON
! RUN: %flang -c -ftime-report -O1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-COMMON

! CHECK-DRIVER: "-ftime-report"

! CHECK-COMMON: Flang execution timing report
! CHECK-COMMON: MLIR generation
! CHECK-COMMON: MLIR translation/optimization
! CHECK-COMMON: LLVM IR generation
! CHECK-COMMON: LLVM IR optimizations
! CHECK-COMMON: Assembly/Object code generation

end program
