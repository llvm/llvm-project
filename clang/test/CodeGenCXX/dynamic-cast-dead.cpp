// RUN: %clang_cc1 -I%S %s -O3 -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -std=c++11 -o - | FileCheck %s
struct A { virtual ~A(); };
struct B : A { };

void foo(A* a) {
  // CHECK-NOT: call {{.*}} @__dynamic_cast
  B* b = dynamic_cast<B*>(a);
}
