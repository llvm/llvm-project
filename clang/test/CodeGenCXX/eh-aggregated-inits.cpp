// Check that initialization of the only one memcpy-able struct member will not
// be performed twice after successful non-trivial initializtion of the second
// member.
//
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -O0 -fno-elide-constructors -emit-llvm %s -o - | FileCheck %s

int globId = 0;

struct ImplicitCopy {
  int id;

  ImplicitCopy() { id = 10; }
  ~ImplicitCopy() { id = 20; }
};

struct ExplicitCopy {
  int id;

  ExplicitCopy() { id = 15; }
  ExplicitCopy(const ExplicitCopy &x) { id = 25; }
  ~ExplicitCopy() { id = 35; }
};

struct Container {
  ImplicitCopy o1; // memcpy-able member.
  ExplicitCopy o2; // non-trivial initialization.

  Container() { globId = 1000; }
  ~Container() { globId = 2000; }
};

int main() {
  try {
    Container c1;
    // CHECK-DAG: call void @llvm.memcpy
    // CHECK-DAG: declare void @llvm.memcpy
    // CHECK-NOT: @llvm.memcpy
    Container c2(c1);

    return 2;
  }
  catch (...) {
    return 1;
  }
  return 0;
}
