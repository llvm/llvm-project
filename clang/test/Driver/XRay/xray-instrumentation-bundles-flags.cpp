// This test ensures that when we invoke the clang compiler, that the -cc1
// options include the -fxray-instrumentation-bundle= flag we provide in the
// invocation.
//
// RUN: %clang -### -c --target=aarch64 -fxray-instrument -fxray-instrumentation-bundle=function %s 2>&1 | FileCheck %s
// CHECK:  "-fxray-instrumentation-bundle=function"
