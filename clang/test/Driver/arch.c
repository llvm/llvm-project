// RUN: %clang -target armv7a-unknown-linux-gnueabi -S -emit-llvm %s -o - | FileCheck %s

// CHECK: target triple = "armv7-unknown-linux-gnueabi"
