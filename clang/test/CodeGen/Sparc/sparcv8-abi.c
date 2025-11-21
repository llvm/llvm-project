// RUN: %clang_cc1 -triple sparc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define{{.*}} { float, float } @p(ptr noundef byval({ float, float }) align 4 %a, ptr noundef byval({ float, float }) align 4 %b) #0 {
float __complex__
p (float __complex__  a, float __complex__  b)
{
  return 0;
}

// CHECK-LABEL: define{{.*}} { double, double } @q(ptr noundef byval({ double, double }) align 8 %a, ptr noundef byval({ double, double }) align 8 %b) #0 {
double __complex__
q (double __complex__  a, double __complex__  b)
{
  return 0;
}

// CHECK-LABEL: define{{.*}} { i64, i64 } @r(ptr noundef byval({ i64, i64 }) align 8 %a, ptr noundef byval({ i64, i64 }) align 8 %b) #0 {
long long __complex__
r (long long __complex__  a, long long __complex__  b)
{
  return 0;
}
