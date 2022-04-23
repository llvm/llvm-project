// RUN: %clang_cc1 -fsanitize=shift-exponent,shift-base -triple=x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

void foo(int* c, int* shamt) {
  *c = 1 << (*c << *shamt);
}
