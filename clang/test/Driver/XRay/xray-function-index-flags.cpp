// This test ensures that when we invoke the clang compiler, that the -cc1
// options respect the -fno-xray-function-index flag we provide in the
// invocation. The default should be to *include* the function index.
//
// RUN: %clang -### -fxray-instrument -target x86_64 -c %s 2>&1 | FileCheck %s
// RUN: %clang -### -fxray-instrument -target x86_64 -fxray-function-index -c %s 2>&1 | FileCheck %s

// CHECK-NOT:  -fno-xray-function-index

// RUN: %clang -### -fxray-instrument -target x86_64 -fno-xray-function-index -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-DISABLED

// CHECK-DISABLED:  -fno-xray-function-index
