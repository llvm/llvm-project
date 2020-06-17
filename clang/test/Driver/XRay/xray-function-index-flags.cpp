// This test ensures that when we invoke the clang compiler, that the -cc1
// options respect the -fno-xray-function-index flag we provide in the
// invocation. The default should be to *include* the function index.
//
// RUN: %clang -fxray-instrument -fxray-function-index -target x86_64-linux- -### \
// RUN:     -x c++ -std=c++11 -emit-llvm -c -o - %s 2>&1 \
// RUN:     | FileCheck %s
// CHECK-NOT:  -fno-xray-function-index
//
// RUN: %clang -fxray-instrument -target x86_64-linux- -### \
// RUN:     -x c++ -std=c++11 -emit-llvm -c -o - %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CHECK-DEFAULT
// CHECK-DEFAULT-NOT:  -fno-xray-function-index
//
// RUN: %clang -fxray-instrument -fno-xray-function-index -target x86_64-linux- -### \
// RUN:     -x c++ -std=c++11 -emit-llvm -c -o - %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix CHECK-DISABLED
// CHECK-DISABLED:  -fno-xray-function-index
//
// REQUIRES: x86_64 || x86_64h
