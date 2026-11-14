// RUN: %clang_cc1 -triple riscv32 -emit-llvm -fcf-protection=branch -mcf-branch-label-scheme=func-sig %s -o - | FileCheck %s

class Base {
public:
  virtual ~Base();
};
class Derived : public Base {
public:
  ~Derived() override;
};

void destroy(Base *b) {
  delete b;
}

// CHECK: !riscv_lpad_label [[LABEL:![0-9]+]]
// CHECK: [[LABEL]] = !{i32 {{[1-9][0-9]*}}}