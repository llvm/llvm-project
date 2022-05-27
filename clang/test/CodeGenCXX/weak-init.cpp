// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s

extern const int W __attribute__((weak)) = 99;
const int S = 77;

// CHECK: @C1 = {{.*}} 77
extern const int C1 = S;

// CHECK: %0 = load {{.*}} @W
// CHECK-NEXT: store {{.*}} %0, {{.*}} @C2
extern const int C2 = W;
