// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -emit-llvm -o - %s | FileCheck %s

// C++ is the only language that accepts the logical operators on vector
// operands.  A lane is true when its value is not zero, which is an ordinary
// floating-point comparison; `-0.0` compares equal to zero and a NaN is true.
typedef long double V __attribute__((vector_size(32)));

extern "C" int not_(V a) { return (!a)[0]; }
// CHECK-LABEL: define {{.*}} @not_(
// CHECK: fcmp oeq <2 x x86_fp80> {{.*}}, zeroinitializer

extern "C" int land_(V a, V b) { return (a && b)[0]; }
// CHECK-LABEL: define {{.*}} @land_(
// CHECK: fcmp une <2 x x86_fp80> {{.*}}, zeroinitializer
// CHECK: fcmp une <2 x x86_fp80> {{.*}}, zeroinitializer
// CHECK: and <2 x i1>

extern "C" int lor_(V a, V b) { return (a || b)[0]; }
// CHECK-LABEL: define {{.*}} @lor_(
// CHECK: fcmp une <2 x x86_fp80> {{.*}}, zeroinitializer
// CHECK: fcmp une <2 x x86_fp80> {{.*}}, zeroinitializer
// CHECK: or <2 x i1>
