// This test ensures that when we invoke the clang compiler, that the -cc1
// options include the -fxray-instrumentation-bundle= flag we provide in the
// invocation.
//
// RUN: %clang -fxray-instrument -fxray-instrumentation-bundle=function -### \
// RUN:     -x c++ -std=c++11 -emit-llvm -c -o - %s 2>&1 \
// RUN:     | FileCheck %s
// CHECK:  -fxray-instrumentation-bundle=function
//
// REQUIRES-ANY: linux, freebsd
// REQUIRES-ANY: amd64, x86_64, x86_64h, arm, aarch64, arm64
