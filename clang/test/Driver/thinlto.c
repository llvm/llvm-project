// -flto=thin causes a switch to llvm-bc object files.
// RUN: %clang -ccc-print-phases -c %s -flto=thin 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILE-ACTIONS < %t %s
//
// CHECK-COMPILE-ACTIONS: 2: compiler, {1}, ir
// CHECK-COMPILE-ACTIONS: 3: backend, {2}, lto-bc

// RUN: %clang -ccc-print-phases %s -flto=thin 2> %t
// RUN: FileCheck -check-prefix=CHECK-COMPILELINK-ACTIONS < %t %s
//
// CHECK-COMPILELINK-ACTIONS: 0: input, "{{.*}}thinlto.c", c
// CHECK-COMPILELINK-ACTIONS: 1: preprocessor, {0}, cpp-output
// CHECK-COMPILELINK-ACTIONS: 2: compiler, {1}, ir
// CHECK-COMPILELINK-ACTIONS: 3: backend, {2}, lto-bc
// CHECK-COMPILELINK-ACTIONS: 4: linker, {3}, image
