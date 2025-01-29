// RUN: %clang_cc1 -std=c++11 -emit-llvm -o - %s | FileCheck %s

void f()
{
  // CHECK: store i32 0
  int i{};
}
