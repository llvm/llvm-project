// RUN: %clang_cc1 -triple sparc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define{{.*}} { float, float } @p({ float, float }* byval({ float, float }) align 4 %a, { float, float }* byval({ float, float }) align 4 %b) #0 {
float __complex__
p (float __complex__  a, float __complex__  b)
{
}

// CHECK-LABEL: define{{.*}} { double, double } @q({ double, double }* byval({ double, double }) align 8 %a, { double, double }* byval({ double, double }) align 8 %b) #0 {
double __complex__
q (double __complex__  a, double __complex__  b)
{
}

// CHECK-LABEL: define{{.*}} { i64, i64 } @r({ i64, i64 }* byval({ i64, i64 }) align 8 %a, { i64, i64 }* byval({ i64, i64 }) align 8 %b) #0 {
long long __complex__
r (long long __complex__  a, long long __complex__  b)
{
}
