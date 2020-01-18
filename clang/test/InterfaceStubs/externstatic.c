// REQUIRES: x86-registered-target
// RUN: %clang -cc1 -fvisibility default -DSTORAGE="extern" -o - -emit-interface-stubs -std=c99 -xc %s | \
// RUN: FileCheck -check-prefix=CHECK-EXTERN %s

// RUN: %clang -cc1 -triple x86_64 -fvisibility default -DSTORAGE=extern -O0 -o - -emit-obj -std=c99 \
// RUN:   %s | llvm-nm - 2>&1 | FileCheck -check-prefix=CHECK-EXTERN %s

// RUN: %clang -cc1 -fvisibility default -DSTORAGE="extern" -o - -emit-interface-stubs -std=c99 -xc %s | \
// RUN: FileCheck -check-prefix=CHECK-EXTERN2 %s

// RUN: %clang -cc1 -triple x86_64 -fvisibility default -DSTORAGE=extern -O0 -o - -emit-obj -std=c99 \
// RUN:   %s | llvm-nm - 2>&1 | FileCheck -check-prefix=CHECK-EXTERN2 %s

// RUN: %clang -cc1 -fvisibility default -DSTORAGE="static" -o - -emit-interface-stubs -std=c99 -xc %s | \
// RUN: FileCheck -check-prefix=CHECK-STATIC %s

// RUN: %clang -cc1 -triple x86_64 -fvisibility default -DSTORAGE=static -O0 -o - -emit-obj -std=c99 \
// RUN:   %s | llvm-nm - 2>&1 | count 0

// CHECK-EXTERN-NOT: foo
// CHECK-STATIC-NOT: foo
// CHECK-STATIC-NOT: bar

// We want to emit extern function symbols.
// CHECK-EXTERN2: bar
STORAGE int foo;
STORAGE int bar() { return 42; }
