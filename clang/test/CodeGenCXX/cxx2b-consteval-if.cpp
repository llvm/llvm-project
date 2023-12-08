// RUN: %clang_cc1 -std=c++23 %s -emit-llvm -o - | FileCheck %s

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

namespace GH55638 {

constexpr bool is_constant_evaluated() noexcept {
  if consteval { return true; } else { return false; }
}

constexpr int compiletime(int) {
   return 2;
}

constexpr int runtime(int) {
   return 1;
}

constexpr int test(int x) {
  if(is_constant_evaluated())
    return compiletime(x);  // CHECK-NOT: call {{.*}}compiletime
   return runtime(x);  // CHECK: call {{.*}}runtime
}

int f(int x) {
  x = test(x);
  return x;
}

}
