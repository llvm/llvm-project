// RUN: %clang -O2 %s -flto=thin -c -o %t.o
// RUN: llvm-lto -thinlto -o %t %t.o

// -fthinlto_index should be passed to cc1
// RUN: %clang -O2 -o %t1.o -x ir %t.o -c -fthinlto-index=%t.thinlto.bc -### 2>&1 | FileCheck %s -check-prefix=CHECK-THINLTOBE-ACTION
// CHECK-THINLTOBE-ACTION: -fthinlto-index=

// Ensure clang driver gives the expected error for incorrect input type
// RUN: not %clang -O2 -o %t1.o %s -c -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s -check-prefix=CHECK-WARNING
// CHECK-WARNING: error: invalid argument '-fthinlto-index={{.*}}' only allowed with '-x ir'
