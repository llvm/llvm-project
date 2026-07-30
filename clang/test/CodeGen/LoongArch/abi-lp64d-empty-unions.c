// RUN: %clang_cc1 -triple loongarch64 -target-feature +f -target-feature +d -target-abi lp64d -emit-llvm %s -o - | \
// RUN:   FileCheck --check-prefix=CHECK-C %s
// RUN: %clang_cc1 -triple loongarch64 -target-feature +f -target-feature +d -target-abi lp64d -emit-llvm %s -o - -x c++ | \
// RUN:   FileCheck --check-prefix=CHECK-CXX %s

#include <stdint.h>

// CHECK-C: define{{.*}} void @test1()
// CHECK-CXX: define{{.*}} i64 @_Z5test12u1(i64{{[^,]*}})
union u1 { };
union u1 test1(union u1 a) {
  return a;
}

struct s1 {
  union u1 u;
  int i;
  float f;
};

// CHECK-C: define{{.*}} { i32, float } @test2(i32{{[^,]*}}, float{{[^,]*}})
// CHECK-CXX: define{{.*}} [2 x i64] @_Z5test22s1([2 x i64]{{[^,]*}})
struct s1 test2(struct s1 a) {
  return a;
}
