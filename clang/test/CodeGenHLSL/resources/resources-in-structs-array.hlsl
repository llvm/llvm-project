// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }

// Array of structs with resources
struct A {
  RWBuffer<float> Buf;
};

// CHECK: @arrayOfA.0.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[arrayOfA0BufStr:.*]] = private unnamed_addr constant [15 x i8] c"arrayOfA.0.Buf\00"
// CHECK: @arrayOfA.1.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[arrayOfA1BufStr:.*]] = private unnamed_addr constant [15 x i8] c"arrayOfA.1.Buf\00"

[[vk::binding(0, 1)]]
A arrayOfA[2] : register(u0, space1);

// Nested struct arrays with resources
struct G {
  A multiArray[2][2];
};

// CHECK: @gArray.0.multiArray.0.0.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[gArray0MultiArray00BufStr:.*]] = private unnamed_addr constant [28 x i8] c"gArray.0.multiArray.0.0.Buf\00"
// CHECK: @gArray.0.multiArray.0.1.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[gArray0MultiArray01BufStr:.*]] = private unnamed_addr constant [28 x i8] c"gArray.0.multiArray.0.1.Buf\00"
// CHECK: @gArray.0.multiArray.1.0.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[gArray0MultiArray10BufStr:.*]] = private unnamed_addr constant [28 x i8] c"gArray.0.multiArray.1.0.Buf\00"
// CHECK: @gArray.0.multiArray.1.1.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[gArray0MultiArray11BufStr:.*]] = private unnamed_addr constant [28 x i8] c"gArray.0.multiArray.1.1.Buf\00"
// CHECK: @gArray.1.multiArray.0.0.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[gArray1MultiArray00BufStr:.*]] = private unnamed_addr constant [28 x i8] c"gArray.1.multiArray.0.0.Buf\00"
// CHECK: @gArray.1.multiArray.0.1.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[gArray1MultiArray01BufStr:.*]] = private unnamed_addr constant [28 x i8] c"gArray.1.multiArray.0.1.Buf\00"
// CHECK: @gArray.1.multiArray.1.0.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[gArray1MultiArray10BufStr:.*]] = private unnamed_addr constant [28 x i8] c"gArray.1.multiArray.1.0.Buf\00"
// CHECK: @gArray.1.multiArray.1.1.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[gArray1MultiArray11BufStr:.*]] = private unnamed_addr constant [28 x i8] c"gArray.1.multiArray.1.1.Buf\00"

[[vk::binding(10, 2)]]
G gArray[2] : register(u10, space2);

// Array of structs that contain a resource array
struct B {
  RWStructuredBuffer<float> ManyBufs[2];
};

B bArray[2];

// CHECK: @[[bArray1ManyBufsStr:.*]] = private unnamed_addr constant [18 x i8] c"bArray.1.ManyBufs\00", align 1

// Make sure the global single resources are initialized from binding; resource arrays are initialized on access.
//
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @arrayOfA.0.Buf,
// CHECK-SAME: i32 noundef 0, i32 noundef 1, i32 noundef 1, i32 noundef 0, ptr noundef @[[arrayOfA0BufStr]])
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @arrayOfA.1.Buf,
// CHECK-SAME: i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 0, ptr noundef @[[arrayOfA1BufStr]])

// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @gArray.0.multiArray.0.0.Buf,
// CHECK-SAME: i32 noundef 10, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @[[gArray0MultiArray00BufStr]])
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @gArray.0.multiArray.0.1.Buf,
// CHECK-SAME: i32 noundef 11, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @[[gArray0MultiArray01BufStr]])
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @gArray.0.multiArray.1.0.Buf,
// CHECK-SAME: i32 noundef 12, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @[[gArray0MultiArray10BufStr]])
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @gArray.0.multiArray.1.1.Buf,
// CHECK-SAME: i32 noundef 13, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @[[gArray0MultiArray11BufStr]])
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @gArray.1.multiArray.0.0.Buf,
// CHECK-SAME: i32 noundef 14, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @[[gArray1MultiArray00BufStr]])
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @gArray.1.multiArray.0.1.Buf,
// CHECK-SAME: i32 noundef 15, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @[[gArray1MultiArray01BufStr]])
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @gArray.1.multiArray.1.0.Buf,
// CHECK-SAME: i32 noundef 16, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @[[gArray1MultiArray10BufStr]])
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @gArray.1.multiArray.1.1.Buf,
// CHECK-SAME: i32 noundef 17, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @[[gArray1MultiArray11BufStr]])

// CHECK: define internal void @main()()
// CHECK-NEXT: entry:
[numthreads(1, 1, 1)]
void main() {
// CHECK-NEXT: %[[TMP:.*]] = alloca %"class.hlsl::RWStructuredBuffer", align 4

// CHECK-NEXT: %[[PTR1:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int) const(ptr {{.*}} @arrayOfA.1.Buf, i32 noundef 0)
// CHECK-NEXT: store float 1.000000e+00, ptr %[[PTR1]]
  arrayOfA[1].Buf[0] = 1.0f;

// CHECK-NEXT: %[[PTR2:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int) const(ptr {{.*}} @gArray.1.multiArray.1.0.Buf, i32 noundef 0)
// CHECK-NEXT: store float 2.000000e+00, ptr %[[PTR2]]
  gArray[1].multiArray[1][0].Buf[0] = 2.0f;

// CHECK-NEXT: %[[PTR3:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int) const(ptr {{.*}} @gArray.0.multiArray.0.1.Buf, i32 noundef 0)
// CHECK-NEXT: store float 3.000000e+00, ptr %[[PTR3]]
  gArray[0].multiArray[0][1].Buf[0] = 3.0f;

// Resource array access - first create the resource from binding, then access the element and store to it.
// CHECK-NEXT: call void @hlsl::RWStructuredBuffer<float>::__createFromImplicitBindingWithImplicitCounter(unsigned int, unsigned int, int, unsigned int, char const*, unsigned int)
// CHECK-SAME: (ptr {{.*}} sret(%"class.hlsl::RWStructuredBuffer") align 4 %[[TMP]], i32 noundef 2, i32 noundef 0, i32 noundef 2, i32 noundef 1, ptr noundef @bArray.1.ManyBufs.str, i32 noundef 3)
// CHECK-NEXT: %[[PTR4:.*]] = call {{.*}} ptr @hlsl::RWStructuredBuffer<float>::operator[](unsigned int) const(ptr {{.*}} %[[TMP]], i32 noundef 0)
// CHECK-NEXT: store float 4.000000e+00, ptr %[[PTR4]], align 4
  bArray[1].ManyBufs[1][0] = 4.0f;
}
