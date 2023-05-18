// RUN: %clang_cc1 -triple=i686-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s
// PR2691

// CHECK: weak
void init_IRQ(void) __attribute__((weak, alias("native_init_IRQ")));
void native_init_IRQ(void) {}
