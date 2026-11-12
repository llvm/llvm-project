// RUN: %clang_cc1 %s -triple %itanium_abi_triple -fcxx-exceptions -fms-compatibility -emit-llvm -o - | FileCheck %s

extern "C" {
  void f();

  // In MS mode we don't validate the exception specification.
  void f() throw() {
  }
}

// PR18661: Clang would fail to emit function definition with mismatching
// exception specification, even though it was just treated as a warning.

// CHECK: define {{.*}}void @f()
