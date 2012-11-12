// RUN: %clang_cc1 %s -emit-llvm -o - | grep nounwind

void bar() { asm (""); }
