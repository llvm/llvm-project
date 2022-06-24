// This test ensures that when we invoke the clang compiler, that the -cc1
// options include the -fxray-instrumentation-bundle= flag we provide in the
// invocation.

// RUN: %clang -fxray-instrument -fxray-instrumentation-bundle=function -### \
// RUN:   -c -o - %s 2>&1 | FileCheck %s
// CHECK:  -fxray-instrumentation-bundle=function

// REQUIRES: linux || freebsd
// REQUIRES: amd64 || x86_64 || x86_64h || arm || aarch64 || arm64
