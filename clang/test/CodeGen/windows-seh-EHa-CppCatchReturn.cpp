// RUN: %clang_cc1 -triple x86_64-windows -fasync-exceptions -fcxx-exceptions -fexceptions -fms-extensions -x c++ -Wno-implicit-function-declaration -S -emit-llvm %s -o - | FileCheck %s

// CHECK: define dso_local void @"?foo@@YAXXZ
// CHECK: invoke void @llvm.seh.try.begin()
// CHECK-NOT: llvm.seh.scope.begin
// CHECK-NOT: llvm.seh.scope.end

// FIXME: Do we actually need llvm.seh.scope*?
void foo() {
  try {}
  catch (...) {
  return;
  }
}
