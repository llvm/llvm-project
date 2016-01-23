// Test that we don't emit a bit set entry for a speculative (available_externally) vtable.
// This does not happen in the Microsoft ABI.
// RUN: %clang_cc1 -triple x86_64-unknown-linux -O1 -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -emit-llvm -o - %s | FileCheck  %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -O1 -fsanitize=cfi-vcall -fsanitize-trap=cfi-vcall -fsanitize-cfi-cross-dso -emit-llvm -o - %s | FileCheck  %s

class A {
 public:
  virtual ~A();
};

A a;

// CHECK: @_ZTV1A ={{.*}} available_externally
// CHECK-NOT: !{{.*}} = !{!{{.*}}, [4 x i8*]* @_ZTV1A, i64 16}
