// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s -fcxx-exceptions                                         | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s -fcxx-exceptions -fexperimental-new-constant-interpreter | FileCheck %s


/// The read from a used to succeed, causing the entire if statement to vanish.
extern void e();
int somefunc() {
  auto foo = [a = false]() mutable {
    if (a)
      e();
  };
  foo();
}

// CHECK: call void @_Z1ev()
