// RUN: %clang -### -c --target=x86_64 -fxray-instrument %s 2>&1 | FileCheck %s
// RUN: %clang -### -c --target=x86_64 -fxray-instrument -fxray-function-index -fno-xray-function-index %s 2>&1 | FileCheck %s --check-prefix=DISABLED

// CHECK-NOT:  "-fxray-function-index"
// DISABLED:   "-fno-xray-function-index"
