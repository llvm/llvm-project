// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// rdar://7268289

class t {
public:
  virtual void foo(void);
  void bar(void);
};

void
t::bar(void) {
// CHECK: _ZN1t3barEv{{.*}} align 2
}

void
t::foo(void) {
// CHECK: _ZN1t3fooEv{{.*}} align 2
}
