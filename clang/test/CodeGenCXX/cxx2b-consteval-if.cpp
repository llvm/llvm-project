// RUN: %clang_cc1 -std=c++2b %s -emit-llvm -o - | FileCheck %s

void should_be_used_1();
void should_be_used_2();
void should_be_used_3();
constexpr void should_not_be_used() {}

constexpr void f() {
  if consteval {
    should_not_be_used(); // CHECK-NOT: call {{.*}}should_not_be_used
  } else {
    should_be_used_1();  // CHECK: call {{.*}}should_be_used_1
  }

  if !consteval {
    should_be_used_2();  // CHECK: call {{.*}}should_be_used_2
  }

  if !consteval {
    should_be_used_3();  // CHECK: call {{.*}}should_be_used_3
  } else {
    should_not_be_used(); // CHECK-NOT: call {{.*}}should_not_be_used
  }
}

void g() {
  f();
}
