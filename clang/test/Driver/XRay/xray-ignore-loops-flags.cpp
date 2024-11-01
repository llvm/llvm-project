// This test ensures that when we invoke the clang compiler, that the -cc1
// options include the -fxray-ignore-loops flag we provide in the
// invocation.
//
// RUN: %clang -### -c --target=x86_64 -fxray-ignore-loops %s 2>&1 | FileCheck %s
// CHECK:  -fxray-ignore-loops
