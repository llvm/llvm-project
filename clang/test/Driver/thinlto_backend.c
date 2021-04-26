// XFAIL: aix

// RUN: %clang -O2 %s -flto=thin -c -o %t.o
// RUN: llvm-lto -thinlto -o %t %t.o

// -fthinlto_index should be passed to cc1
// RUN: %clang -O2 -o %t1.o -x ir %t.o -c -fthinlto-index=%t.thinlto.bc -### \
// RUN:     2>&1 | FileCheck %s -check-prefix=CHECK-THINLTOBE-ACTION
// CHECK-THINLTOBE-ACTION: -fthinlto-index=
// CHECK-THINLTOBE-ACTION-SAME: {{"?-x"? "?ir"?}}

// Check that this also works without -x ir.
// RUN: %clang -O2 -o %t1.o %t.o -c -fthinlto-index=%t.thinlto.bc -### 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-THINLTOBE-ACTION

// -save-temps should be passed to cc1
// RUN: %clang -O2 -o %t1.o -x ir %t.o -c -fthinlto-index=%t.thinlto.bc -save-temps -### 2>&1 | FileCheck %s -check-prefix=CHECK-SAVE-TEMPS -check-prefix=CHECK-SAVE-TEMPS-CWD
// RUN: %clang -O2 -o %t1.o -x ir %t.o -c -fthinlto-index=%t.thinlto.bc -save-temps=cwd -### 2>&1 | FileCheck %s -check-prefix=CHECK-SAVE-TEMPS -check-prefix=CHECK-SAVE-TEMPS-CWD
// RUN: %clang -O2 -o %t1.o -x ir %t.o -c -fthinlto-index=%t.thinlto.bc -save-temps=obj -### 2>&1 | FileCheck %s -check-prefix=CHECK-SAVE-TEMPS -check-prefix=CHECK-SAVE-TEMPS-OBJ
// CHECK-SAVE-TEMPS-NOT: -emit-llvm-bc
// CHECK-SAVE-TEMPS-CWD: -save-temps=cwd
// CHECK-SAVE-TEMPS-OBJ: -save-temps=obj
// CHECK-SAVE-TEMPS-NOT: -emit-llvm-bc

// Ensure clang driver gives the expected error for incorrect input type
// RUN: not %clang -O2 -o %t1.o %s -c -fthinlto-index=%t.thinlto.bc 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-WARNING
// CHECK-WARNING: error: option '-fthinlto-index={{.*}}' requires input to be LLVM bitcode
