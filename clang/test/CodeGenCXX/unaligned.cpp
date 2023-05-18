// RUN: %clang_cc1 -x c++ -std=c++11 -triple x86_64-unknown-linux-gnu -fms-extensions -emit-llvm < %s | FileCheck %s

int foo() {
  // CHECK: ret i32 1
  return alignof(__unaligned int);
}
