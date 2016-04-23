// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -x c++ %S/Inputs/pr27445.h -emit-pch -o %t.pch
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions %s -include-pch %t.pch -emit-llvm -o - | FileCheck %s

class A;
void fn1(A &) {}

class __declspec(dllexport) A {
  int operator=(A) { return field_; }
  void (*on_arena_allocation_)(Info);
  int field_;
};

// CHECK: %class.A = type { void (%struct.Info*)*, i32 }
// CHECK: %struct.Info = type { i32 (...)** }
