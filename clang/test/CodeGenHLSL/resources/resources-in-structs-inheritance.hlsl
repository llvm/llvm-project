// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }
// CHECK: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", float, 0, 0) }
// CHECK: %"class.hlsl::SamplerState" = type { target("dx.Sampler", 0) }
// CHECK: %"class.hlsl::StructuredBuffer.0" = type { target("dx.RawBuffer", i32, 0, 0) }

// Simple inheritance
struct A {
  RWBuffer<float> Buf;
};

struct C : A {
  RWBuffer<float> Buf2;
};

// Global variables for resources c.A::Buf and c.Buf2
// (Looks like llvm-cxxfilt doesn't demangle names with `::`.)
//
// CHECK: @"_ZL8c.A::Buf" = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[cABufStr:.*]] = private unnamed_addr constant [9 x i8] c"c.A::Buf\00"
// CHECK: @c.Buf2 = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[cBuf2Str:.*]] = private unnamed_addr constant [7 x i8] c"c.Buf2\00"

[[vk::binding(3)]]
C c : register(u3);

// Global variables for resources d.A::Buf and d.A.Buf
//
// CHECK: @"_ZL8d.A::Buf" = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[dABufStr1:.*]] = private unnamed_addr constant [9 x i8] c"d.A::Buf\00"
// CHECK: @d.A.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[dABufStr2:.*]] = private unnamed_addr constant [8 x i8] c"d.A.Buf\00"

// Inheritance with same named field
struct D : A {
    A A;
};
D d;

// Multiple resources kinds and inheritance
class B {
  StructuredBuffer<int> SrvBufs[2];
};

class E : B {
};

class F : E {
  A a;
  StructuredBuffer<float> SrvBuf;
  SamplerState Samp;
};

// Global variables for resources f.a.Buf, f.SrvBuf and f.Samp.
// Resource array f.E::B::SrvBufs does not have a global, it is initialized on demand.
//
// CHECK: @f.a.Buf = internal global %"class.hlsl::RWBuffer" poison
// CHECK: @[[fABufStr:.*]] = private unnamed_addr constant [8 x i8] c"f.a.Buf\00"
// CHECK: @f.SrvBuf = internal global %"class.hlsl::StructuredBuffer" poison
// CHECK: @[[fSrvBufStr:.*]] = private unnamed_addr constant [9 x i8] c"f.SrvBuf\00"
// CHECK: @f.Samp = internal global %"class.hlsl::SamplerState" poison
// CHECK: @[[fSampStr:.*]] = private unnamed_addr constant [7 x i8] c"f.Samp\00"
// CHECK: @[[fEBSrvBufStr:.*]] = private unnamed_addr constant [16 x i8] c"f.E::B::SrvBufs\00"

[[vk::binding(10)]]
F f : register(t0) : register(u20) : register(s3);

// Make sure they are initialized from binding
//
// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @"_ZL8c.A::Buf",
// CHECK-SAME: i32 noundef 3, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[cABufStr]])

// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @c.Buf2,
// CHECK-SAME: i32 noundef 4, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[cBuf2Str]])

// CHECK: call void @hlsl::RWBuffer<float>::__createFromImplicitBinding({{.*}})(ptr {{.*}} @"_ZL8d.A::Buf",
// CHECK-SAME: i32 noundef 0, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[dABufStr1]])

// CHECK: call void @hlsl::RWBuffer<float>::__createFromImplicitBinding({{.*}})(ptr {{.*}} @d.A.Buf,
// CHECK-SAME: i32 noundef 1, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[dABufStr2]])

// CHECK: call void @hlsl::RWBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @f.a.Buf,
// CHECK-SAME: i32 noundef 20, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[fABufStr]])

// CHECK: call void @hlsl::StructuredBuffer<float>::__createFromBinding({{.*}})(ptr {{.*}} @f.SrvBuf,
// CHECK-SAME: i32 noundef 2, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[fSrvBufStr]])

// CHECK: call void @hlsl::SamplerState::__createFromBinding({{.*}})(ptr {{.*}} @f.Samp,
// CHECK-SAME: i32 noundef 3, i32 noundef 0, i32 noundef 1, i32 noundef 0, ptr noundef @[[fSampStr]])

// CHECK: define internal void @main()()
// CHECK-NEXT: entry:
[numthreads(1, 1, 1)]
void main() {
// CHECK-NEXT: %i = alloca i32
// CHECK-NEXT: %[[TMP:.*]] = alloca %"class.hlsl::StructuredBuffer.0"
// CHECK-NEXT: %a = alloca float

// CHECK-NEXT: %[[PTR1:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int)(ptr {{.*}} @"_ZL8c.A::Buf", i32 noundef 0)
// CHECK-NEXT: store float 0x3FF3AE1480000000, ptr %[[PTR1:]]
  c.Buf[0] = 1.230f;

// CHECK-NEXT: %[[PTR2:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int)(ptr {{.*}} @c.Buf2, i32 noundef 0)
// CHECK-NEXT: store float 0x4002B851E0000000, ptr %[[PTR2:]]
  c.Buf2[0] = 2.340f;

// CHECK-NEXT: %[[PTR3:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int)(ptr {{.*}} @"_ZL8d.A::Buf", i32 noundef 0)
// CHECK-NEXT: store float 0x400B9999A0000000, ptr %[[PTR3:]]
  d.Buf[0] = 3.450f;

// CHECK-NEXT: %[[PTR4:.*]] = call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int)(ptr {{.*}} @d.A.Buf, i32 noundef 0)
// CHECK-NEXT: store float 0x40123D70A0000000, ptr %[[PTR4:]]
  d.A.Buf[0] = 4.560f;

// Resource array access - initilized on demand:
// CHECK-NEXT: call void @hlsl::StructuredBuffer<int>::__createFromBinding({{.*}})(ptr {{.*}} %[[TMP]],
// CHECK-SAME: i32 noundef 0, i32 noundef 0, i32 noundef 2, i32 noundef 0, ptr noundef @[[fEBSrvBufStr]])
// CHECK-NEXT: %[[PTR5:.*]] = call {{.*}} ptr @hlsl::StructuredBuffer<int>::operator[](unsigned int) const(ptr {{.*}} %[[TMP]], i32 noundef 1)
// CHECK-NEXT: %[[VAL1:.*]] = load i32, ptr %[[PTR5]]
// CHECK-NEXT: store i32 %[[VAL1]], ptr %i
  int i = f.SrvBufs[0][1];

// CHECK-NEXT: %[[PTR6:.*]] = call {{.*}} ptr @hlsl::StructuredBuffer<float>::operator[](unsigned int) const({{.*}} @f.SrvBuf, i32 noundef 0)
// CHECK-NEXT: %[[VAL2:.*]] = load float, ptr %[[PTR6]]
// CHECK-NEXT: store float %[[VAL2]], ptr %a
  float a = f.SrvBuf[0];

// CHECK: [[PTR7:.*]]= call {{.*}} ptr @hlsl::RWBuffer<float>::operator[](unsigned int)(ptr {{.*}} @f.a.Buf, i32 noundef 0)
// CHECK: store float %{{.*}}, ptr %call6
  f.a.Buf[0] = (float)i + a;
}
