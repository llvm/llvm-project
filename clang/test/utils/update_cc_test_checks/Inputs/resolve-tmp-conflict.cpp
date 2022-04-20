// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

void foo(int a) {
  int &tmp0 = a;
  int &&tmp1 = 1;
  tmp1 = a;
  return;
}
