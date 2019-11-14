// RUN: %clang -cc1 -fvisibility default -DSTORAGE="extern" -o - -emit-interface-stubs -std=c99 -xc %s | \
// RUN: FileCheck -check-prefix=CHECK-EXTERN %s

// RUN: %clang -fvisibility=default -DSTORAGE="extern" -O0 -o - -c -std=c99 \
// RUN: -xc %s | llvm-nm - 2>&1 | FileCheck -check-prefix=CHECK-EXTERN %s

// RUN: %clang -cc1 -fvisibility default -DSTORAGE="extern" -o - -emit-interface-stubs -std=c99 -xc %s | \
// RUN: FileCheck -check-prefix=CHECK-EXTERN2 %s

// RUN: %clang -fvisibility=default -DSTORAGE="extern" -O0 -o - -c -std=c99 -xc %s | llvm-nm - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-EXTERN2 %s

// RUN: %clang -cc1 -fvisibility default -DSTORAGE="static" -o - -emit-interface-stubs -std=c99 -xc %s | \
// RUN: FileCheck -check-prefix=CHECK-STATIC %s

// RUN: %clang -fvisibility=default -DSTORAGE="static" -O0 -o - -c -std=c99 -xc %s | llvm-nm - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-STATIC %s

// CHECK-EXTERN-NOT: foo
// CHECK-STATIC-NOT: foo
// CHECK-STATIC-NOT: bar

// We want to emit extern function symbols.
// CHECK-EXTERN2: bar
STORAGE int foo;
STORAGE int bar() { return 42; }
