// RUN: %clang_cc1 %s -fvisibility=default -emit-llvm -o - | FileCheck %s

// CHECK: hidden global
int X __attribute__ ((__visibility__ ("hidden"))) = 123;
