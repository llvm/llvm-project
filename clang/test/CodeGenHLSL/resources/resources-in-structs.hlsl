// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }

// Single resource field in struct.
struct A {
  RWBuffer<float> Buf;
};

// Global variable for resource a.Buf
//
// CHECK-DAG: @a.Buf = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK-DAG: @[[aBufStr:.*]] = private unnamed_addr constant [6 x i8] c"a.Buf\00", align 1
[[vk::binding(0)]]
A a : register(u0);

// Resource array in struct.
struct B {
  RWBuffer<float> Bufs[10];
};

// Resource arrays do not have a global, they are initialized on demand. Just check the string name is generated correctly.
//
// CHECK-DAG: @[[bBufsStr:.*]] = private unnamed_addr constant [7 x i8] c"b.Bufs\00", align 1
[[vk::binding(2)]]
B b : register(u2);

// Resources with counters
struct C {
  StructuredBuffer<float> BufMany[3][2];
  StructuredBuffer<float> BufOne;
};

// CHECK-DAG: @[[cBufOne:.*]] = private unnamed_addr constant [9 x i8] c"c.BufOne\00", align 1
// CHECK-DAG: @[[cBufMany:.*]] = private unnamed_addr constant [10 x i8] c"c.BufMany\00", align 1

[[vk::binding(10)]] 
C c : register(t10);

// Check that a.Buf is initialized from binding
//
// CHECK: define internal void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}}(%"class.hlsl::RWBuffer") align 4 @a.Buf, i32 noundef 0, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[aBufStr]])
// CHECK-NEXT: ret void

// Check that c.BufOne is initialized from binding with counter
//
// CHECK: define internal void @__cxx_global_var_init.3()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @hlsl::StructuredBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr dead_on_unwind writable sret(%"class.hlsl::StructuredBuffer") align 4 @c.BufOne, i32 noundef 16, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[cBufOne]])
// CHECK-NEXT: ret void

// CHECK: define internal void @main()()
// CHECK: %[[TMP1:.*]] = alloca %"class.hlsl::RWBuffer", align 4
// CHECK: %[[TMP2:.*]] = alloca %"class.hlsl::StructuredBuffer", align 4
[numthreads(1, 1, 1)]
void main() {

// CHECK: %[[PTR1:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int) const(ptr noundef nonnull align 4 dereferenceable(4) @a.Buf, i32 noundef 0)
// CHECK-NEXT: store float 0x3FF3AE1480000000, ptr %[[PTR1]], align 4
  a.Buf[0] = 1.230f;

// Resource array access - first create the resource from binding, then access the element and store to it.
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 %[[TMP1]], i32 noundef 2, i32 noundef 0, i32 noundef 10, i32 noundef 5, ptr noundef @[[bBufsStr]])
// CHECK-NEXT: %[[PTR2:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int) const(ptr {{.*}} %[[TMP1]], i32 noundef 0)
// CHECK-NEXT: store float 0x40123D70A0000000, ptr %[[PTR2]], align 4
  b.Bufs[5][0] = 4.56f;

// CHECK: %[[PTR3:.*]] = call {{.*}} ptr @hlsl::StructuredBuffer<float>::operator[](unsigned int) const(ptr noundef nonnull align 4 dereferenceable(4) @c.BufOne, i32 noundef 0)
// CHECK-NEXT: load float, ptr %[[PTR3]], align 4
  float x = c.BufOne[0];

// Resource with counter array access - first create the resource from binding, then access the element and store to it.
// CHECK: call void @hlsl::StructuredBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::StructuredBuffer") align 4 %[[TMP2]], i32 noundef 10, i32 noundef 0, i32 noundef 6, i32 noundef 5, ptr noundef @c.BufMany.str)
// CHECK-NEXT: %[[PTR4:.*]] = call {{.*}} ptr @hlsl::StructuredBuffer<float>::operator[](unsigned int) const(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP2]], i32 noundef 0)
// CHECK-NEXT: load float, ptr %[[PTR4]], align 4
  float f = c.BufMany[2][1][0];
}
