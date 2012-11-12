// RUN: %clang_cc1 -g -emit-llvm %s -o - | FileCheck %s
// Do not use function name to create named metadata used to hold
// local variable info. For example. llvm.dbg.lv.~A is an invalid name.

// CHECK-NOT: llvm.dbg.lv.~A
class A {
public:
  ~A() { int i = 0; i++; }
};

int foo(int i) {
  A a;
  return 0;
}
