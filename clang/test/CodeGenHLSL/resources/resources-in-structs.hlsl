// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }

// Single resource field in struct.
struct A {
  RWBuffer<float> Buf;
};

// Global variable for resource a.Buf
//
// CHECK: @a.Buf = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK: @[[aBufStr:.*]] = private unnamed_addr constant [6 x i8] c"a.Buf\00", align 1
[[vk::binding(0)]]
A a : register(u0);

// Resource array in struct.
struct B {
  RWBuffer<float> Bufs[10];
};

// Resource arrays do not have a global, they are initialized on demand. Just check the string name is generated correctly.
//
// CHECK: @[[bBufsStr:.*]] = private unnamed_addr constant [7 x i8] c"b.Bufs\00", align 1
[[vk::binding(2)]]
B b : register(u2);

// Check that a.Buf is initialized from binding
//
// CHECK: define internal void @__cxx_global_var_init()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}}(%"class.hlsl::RWBuffer") align 4 @a.Buf, i32 noundef 0, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[aBufStr]])
// CHECK-NEXT: ret void

// CHECK: define internal void @main()()
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[TMP:.*]] = alloca %"class.hlsl::RWBuffer", align 4
[numthreads(1, 1, 1)]
void main() {

// CHECK-NEXT: %[[PTR:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int)(ptr noundef nonnull align 4 dereferenceable(4) @a.Buf, i32 noundef 0) #5
// CHECK-NEXT: store float 0x3FF3AE1480000000, ptr %[[PTR]], align 4
  a.Buf[0] = 1.230f;

// Resource array access - first create the resource from binding, then access the element and store to it.
// CHECK-NEXT: call void @hlsl::RWBuffer<float>::__createFromBinding(unsigned int, unsigned int, int, unsigned int, char const*)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWBuffer") align 4 %[[TMP]], i32 noundef 2, i32 noundef 0, i32 noundef 10, i32 noundef 5, ptr noundef @[[bBufsStr]])
// CHECK-NEXT: %[[PTR2:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int)(ptr {{.*}} %[[TMP]], i32 noundef 0)
// CHECK-NEXT: store float 0x40123D70A0000000, ptr %[[PTR2]], align 4
// CHECK-NEXT: ret void
  b.Bufs[5][0] = 4.56f;
}
