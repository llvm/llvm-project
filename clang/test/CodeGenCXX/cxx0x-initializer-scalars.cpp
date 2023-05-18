// RUN: %clang_cc1 -std=c++11 -S -emit-llvm -o - %s | FileCheck %s

void f()
{
  // CHECK: store i32 0
  int i{};
}
