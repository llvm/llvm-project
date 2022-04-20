// RUN: %clang_cc1 -no-opaque-pointers -triple i386-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,NONCOFF
// RUN: %clang_cc1 -no-opaque-pointers -triple i386-unknown-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,COFF

__attribute((aligned(16))) float a[128];
union {int a[4]; __attribute((aligned(16))) float b[4];} b;

// CHECK: @a = {{.*}}zeroinitializer, align 16
// CHECK: @b = {{.*}}zeroinitializer, align 16

long long int test5[1024];
// CHECK-DAG: @test5 = {{.*}}global [1024 x i64] zeroinitializer, align 8

// PR5279 - Reduced alignment on typedef.
typedef int myint __attribute__((aligned(1)));

void test1(myint *p) {
  *p = 0;
}
// CHECK: @test1(
// CHECK: store i32 0, i32* {{.*}}, align 1
// CHECK: ret void

int test1a(myint *p) {
  return *p;
}
// CHECK: @test1a(
// CHECK: load i32, i32* {{.*}}, align 1
// CHECK: ret i32


// PR5279 - Reduced alignment on typedef.
typedef float __attribute__((vector_size(16), aligned(4))) packedfloat4;

void test2(packedfloat4 *p) {
  *p = (packedfloat4) { 3.2f, 2.3f, 0.1f, 0.0f };
}
// CHECK: @test2(
// CHECK: store <4 x float> {{.*}}, align 4
// CHECK: ret void


// PR5279 - Reduced alignment on typedef.
typedef float __attribute__((ext_vector_type(3), aligned(4))) packedfloat3;
void test3(packedfloat3 *p) {
  *p = (packedfloat3) { 3.2f, 2.3f, 0.1f };
}
// CHECK: @test3(
// CHECK: %{{.*}} = bitcast <3 x float>* %{{.*}} to <4 x float>*
// CHECK: store <4 x float> {{.*}}, align 4
// CHECK: ret void



typedef float __attribute__((vector_size(16), aligned(64))) float4align64;

// rdar://10639962 - Typedef alignment lost in p[]-style dereferencing
void test4(float4align64 *p) {
  p[0] = (float4align64){ 3.2f, 2.3f, 0.1f, 0.0f };
}
// CHECK: @test4(
// CHECK: store <4 x float> {{.*}}, <4 x float>* {{.*}}, align 64

// PR24944 - Typedef alignment not honored on no-op cast.
typedef float __attribute__((vector_size(16), aligned(16))) float4align16;
typedef float __attribute__((vector_size(16), aligned(2))) float4align2;
void test6(float4align64 *p) {
    float4align64 vec = *(float4align2*) p;
}
// CHECK-LABEL: @test6
// CHECK:       load <4 x float>, <4 x float>* {{.*}}, align 2

typedef int __attribute__((ext_vector_type(200 * 16))) BigVecTy;
void test7(void) {
  BigVecTy V;
}
// CHECK-LABEL: @test7
// NONCOFF: alloca <3200 x i32>, align 16384
// COFF: alloca <3200 x i32>, align 8192
