// RUN: %clang_cc1 -triple x86_64-windows -fasync-exceptions -fcxx-exceptions -fexceptions -fms-extensions -x c++ -Wno-implicit-function-declaration -emit-llvm %s -o - | FileCheck %s

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

__declspec(noreturn) void bar();
class baz {
public:
  ~baz();
};

// CHECK: define dso_local void @"?qux@@YAXXZ
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK-NOT: llvm.seh.try
// CHECK-NOT: llvm.seh.scope.end

// We don't need to generate llvm.seh.scope.end for unreachable.
void qux() {
  baz a;
  bar();
}
