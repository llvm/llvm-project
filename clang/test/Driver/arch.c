// RUN: %clang -target armv7a-unknown-linux-gnueabi -S -emit-llvm %s -o - | FileCheck %s --check-prefix=V7
// RUN: %clang -target armv8a-unknown-linux-gnueabi -S -emit-llvm %s -o - | FileCheck %s --check-prefix=V8

// V7: target triple = "armv7-unknown-linux-gnueabi"
// V8: target triple = "armv8a-unknown-linux-gnueabi"
