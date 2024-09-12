// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void should_be_used_1();
void should_be_used_2();
void should_be_used_3();
constexpr void should_not_be_used() {}

constexpr void f() {
  if consteval {
    should_not_be_used(); // CHECK-NOT: call {{.*}}should_not_be_used
  } else {
    should_be_used_1(); // CHECK: call {{.*}}should_be_used_1
  }

  if !consteval {
    should_be_used_2(); // CHECK: call {{.*}}should_be_used_2
  } else {
    should_not_be_used(); // CHECK-NOT: call {{.*}}should_not_be_used
  }

  if consteval {
    should_not_be_used(); // CHECK-NOT: call {{.*}}should_not_be_used
  }

  if !consteval {
    should_be_used_3(); // CHECK: call {{.*}}should_be_used_3
  }
}

void g() {
  f();
}
