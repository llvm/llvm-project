// This test ensures that when we invoke the clang compiler, that the -cc1
// options include the -fxray-ignore-loops flag we provide in the
// invocation.

// RUN: %clang -fxray-instrument -fxray-ignore-loops -target x86_64-linux- -### \
// RUN:   -x c++ -emit-llvm -c -o - %s 2>&1 | FileCheck %s
// CHECK:  -fxray-ignore-loops

// REQUIRES: x86_64 || x86_64h
