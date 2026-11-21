// RUN: %clang_cc1 -triple riscv32 -emit-llvm -fcf-protection=branch -mcf-branch-label-scheme=func-sig %s -o - | FileCheck %s

class Base {
public:
  virtual void f(int);
};
void call_virtual_dispatch(Base *b, int x) {
  b->f(x);
}
// CHECK: !riscv_lpad_label [[LABEL:![0-9]+]]
// CHECK: [[LABEL]] = !{i32 {{[1-9][0-9]*}}}