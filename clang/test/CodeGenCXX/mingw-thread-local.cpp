// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple x86_64-w64-mingw32 | FileCheck %s

extern thread_local int var;

int get() {
  return var;
}

// CHECK: declare extern_weak void @_ZTH3var()
