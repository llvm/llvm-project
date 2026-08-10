// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -triple i686-windows -emit-llvm-only -fcoverage-mapping -dump-coverage-mapping -fprofile-instrument=clang %s | FileCheck %s

struct A {
  virtual ~A();
};

// CHECK: ?PR32761@@YAXXZ:
// CHECK-NEXT: File 0, [[@LINE+1]]:16 -> [[@LINE+3]]:2 = #0
void PR32761() {
  A a;
}
