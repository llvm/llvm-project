// RUN: %clang_cc1 -flto-linker-scripts -triple aarch64-linux-gnu -emit-llvm < %s | FileCheck %s
// REQUIRES: aarch64-registered-target
// CHECK: @x = global i32 0, section ".bss"
int x;
