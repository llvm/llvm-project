// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-nvcall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-nvcall,cfi-cast-strict -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-STRICT %s

struct A {
  virtual void f();
};

struct B : A {
  int i;
  void g();
};

struct C : A {
  void g();
};

// CHECK-LABEL: @bg
// CHECK-STRICT-LABEL: @bg
extern "C" void bg(B *b) {
  // CHECK: call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-STRICT: call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  b->g();
}

// CHECK-LABEL: @cg
// CHECK-STRICT-LABEL: @cg
extern "C" void cg(C *c) {
  // http://clang.llvm.org/docs/ControlFlowIntegrity.html#strictness
  // In this case C's layout is the same as its base class, so we allow
  // c to be of type A in non-strict mode.

  // CHECK: call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1A")
  // CHECK-STRICT: call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1C")
  c->g();
}
