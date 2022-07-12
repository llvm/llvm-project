// RUN: %clang -mfunction-return= -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VALID %s
// RUN: not %clang -mfunction-return -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-INVALID %s

// RUN: %clang -mfunction-return=keep -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-KEEP %s
// RUN: %clang -mfunction-return=thunk-extern -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-EXTERN %s

// RUN: %clang -mfunction-return=keep -mfunction-return=thunk-extern -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-EXTERN %s
// RUN: %clang -mfunction-return=thunk-extern -mfunction-return=keep -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-KEEP %s

// CHECK-VALID:   "-mfunction-return="
// CHECK-INVALID: error: unknown argument: '-mfunction-return'

// CHECK-KEEP:       "-mfunction-return=keep"
// CHECK-KEEP-NOT:   "-mfunction-return=thunk-extern"
// CHECK-EXTERN:     "-mfunction-return=thunk-extern"
// CHECK-EXTERN-NOT: "-mfunction-return=keep"
