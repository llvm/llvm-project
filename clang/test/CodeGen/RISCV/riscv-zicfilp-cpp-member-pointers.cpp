// RUN: %clang_cc1 -triple riscv32 -emit-llvm -fcf-protection=branch -mcf-branch-label-scheme=func-sig %s -o - | FileCheck %s

class Base {
public:
  void nonvirtual_method(int);
  virtual void virtual_method(int);
};
class Derived : public Base {
public:
  void virtual_method(int) override;
};

void call_nonvirtual(Base *b, void (Base::*pm)(int), int x) { (b->*pm)(x); }
void call_virtual(Base *b, void (Base::*pm)(int), int x) { (b->*pm)(x); }

void call_converted(Derived *d, int x) {
  void (Base::*pm_base)(int) = &Base::virtual_method;
  void (Derived::*pm_derived)(int) = pm_base; // valid Base::* -> Derived::* conversion
  (d->*pm_derived)(x); // dispatches to Derived::virtual_method at runtime
}

void (Base::*global_pm)(int) = &Base::nonvirtual_method;
void call_through_variable(Base *b, int x) { (b->*global_pm)(x); }

// CHECK-COUNT-4: !riscv_lpad_label [[LABEL:![0-9]+]]
// CHECK: [[LABEL]] = !{i32 {{[1-9][0-9]*}}}