// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - %s | FileCheck %s
// UNSUPPORTED: ppc64be

// CHECK-LABEL: define void @foo_no_mempcy() #0
extern "C" void foo_no_mempcy() __attribute__((no_builtin("memcpy"))) {}

// CHECK-LABEL: define void @foo_no_mempcy_twice() #0
extern "C" void foo_no_mempcy_twice() __attribute__((no_builtin("memcpy"))) __attribute__((no_builtin("memcpy"))) {}

// CHECK-LABEL: define void @foo_no_builtins() #1
extern "C" void foo_no_builtins() __attribute__((no_builtin)) {}

// CHECK-LABEL: define void @foo_no_mempcy_memset() #2
extern "C" void foo_no_mempcy_memset() __attribute__((no_builtin("memset", "memcpy"))) {}

// CHECK-LABEL: define void @separate_attrs() #2
extern "C" void separate_attrs() __attribute__((no_builtin("memset"))) __attribute__((no_builtin("memcpy"))) {}

// CHECK-LABEL: define void @separate_attrs_ordering() #2
extern "C" void separate_attrs_ordering() __attribute__((no_builtin("memcpy"))) __attribute__((no_builtin("memset"))) {}

struct A {
  virtual int foo() const __attribute__((no_builtin("memcpy"))) { return 1; }
  virtual ~A();
};

struct B : public A {
  int foo() const override __attribute__((no_builtin("memmove"))) { return 2; }
  virtual ~B();
};

// CHECK-LABEL: define void @call_a_foo(%struct.A* %a) #3
extern "C" void call_a_foo(A *a) {
  // CHECK: %call = call i32 %2(%struct.A* %0)
  a->foo(); // virtual call is not annotated
}

// CHECK-LABEL: define void @call_b_foo(%struct.B* %b) #3
extern "C" void call_b_foo(B *b) {
  // CHECK: %call = call i32 %2(%struct.B* %0)
  b->foo(); // virtual call is not annotated
}

// CHECK-LABEL: define void @call_foo_no_mempcy() #3
extern "C" void call_foo_no_mempcy() {
  // CHECK: call void @foo_no_mempcy() #6
  foo_no_mempcy(); // call gets annotated with "no-builtin-memcpy"
}

A::~A() {} // Anchoring A so A::foo() gets generated
B::~B() {} // Anchoring B so B::foo() gets generated

// CHECK-LABEL: define linkonce_odr i32 @_ZNK1A3fooEv(%struct.A* %this) unnamed_addr #0 comdat align 2
// CHECK-LABEL: define linkonce_odr i32 @_ZNK1B3fooEv(%struct.B* %this) unnamed_addr #5 comdat align 2

// CHECK:     attributes #0 = {{{.*}}"no-builtin-memcpy"{{.*}}}
// CHECK-NOT: attributes #0 = {{{.*}}"no-builtin-memmove"{{.*}}}
// CHECK-NOT: attributes #0 = {{{.*}}"no-builtin-memset"{{.*}}}
// CHECK:     attributes #1 = {{{.*}}"no-builtins"{{.*}}}
// CHECK:     attributes #2 = {{{.*}}"no-builtin-memcpy"{{.*}}"no-builtin-memset"{{.*}}}
// CHECK-NOT: attributes #2 = {{{.*}}"no-builtin-memmove"{{.*}}}
// CHECK:     attributes #5 = {{{.*}}"no-builtin-memmove"{{.*}}}
// CHECK-NOT: attributes #5 = {{{.*}}"no-builtin-memcpy"{{.*}}}
// CHECK-NOT: attributes #5 = {{{.*}}"no-builtin-memset"{{.*}}}
// CHECK:     attributes #6 = { "no-builtin-memcpy" }
