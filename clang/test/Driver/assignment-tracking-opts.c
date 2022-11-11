// RUN: %clang -### -S %s -g -target x86_64-linux-gnu 2>&1 | FileCheck --check-prefix=CHECK-NO-AT %s
// RUN: %clang -### -S %s -g -target x86_64-linux-gnu 2>&1              \
// RUN:                      -Xclang -fexperimental-assignment-tracking \
// RUN: | FileCheck --check-prefix=CHECK-AT %s

// CHECK-NO-AT-NOT: "-mllvm" "-experimental-assignment-tracking"
// CHECK-NO-AT-NOT: "-fexperimental-assignment-tracking"

// CHECK-AT: "-mllvm" "-experimental-assignment-tracking"
// CHECK-AT: "-fexperimental-assignment-tracking"
