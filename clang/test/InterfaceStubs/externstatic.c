// REQUIRES: x86-registered-target
// RUN: %clang -DSTORAGE="extern" -target x86_64-unknown-linux-gnu -o - \
// RUN: -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-yaml-elf-v1 -std=c99 -xc %s | \
// RUN: FileCheck -check-prefix=CHECK-EXTERN %s
// RUN: %clang -DSTORAGE="extern" -target x86_64-linux-gnu -O0 -o - -c -std=c99 \
// RUN: -xc %s | llvm-nm - 2>&1 | FileCheck -check-prefix=CHECK-EXTERN %s

// RUN: %clang -DSTORAGE="extern" -target x86_64-unknown-linux-gnu -o - \
// RUN: -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-yaml-elf-v1 -std=c99 -xc %s | \
// RUN: FileCheck -check-prefix=CHECK-EXTERN2 %s
// RUN: %clang -DSTORAGE="extern" -target x86_64-linux-gnu -O0 -o - -c -std=c99 \
// RUN: -xc %s | llvm-nm - 2>&1 | FileCheck -check-prefix=CHECK-EXTERN2 %s

// RUN: %clang -DSTORAGE="static" -target x86_64-unknown-linux-gnu -o - \
// RUN: -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-yaml-elf-v1 -std=c99 -xc %s | \
// RUN: FileCheck -check-prefix=CHECK-STATIC %s
// RUN: %clang -DSTORAGE="static" -target x86_64-linux-gnu -O0 -o - -c -std=c99 \
// RUN: -xc %s | llvm-nm - 2>&1 | FileCheck -check-prefix=CHECK-STATIC %s

// CHECK-EXTERN-NOT: foo
// CHECK-STATIC-NOT: foo
// CHECK-STATIC-NOT: bar

// We want to emit extern function symbols.
// CHECK-EXTERN2: bar
STORAGE int foo;
STORAGE int bar() { return 42; }
