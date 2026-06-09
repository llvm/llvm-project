// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -finclude-default-header -O3 -emit-llvm -o - %s | FileCheck %s
// CHECK: [[Buf:@.*]] = private unnamed_addr constant [4 x i8] c"Buf\00"
// CHECK: [[Buf2:@.*]] = private unnamed_addr constant [5 x i8] c"Buf2\00"
// CHECK: [[Buf3:@.*]] = private unnamed_addr constant [5 x i8] c"Buf3\00"
// CHECK: [[CB:@.*]] = private unnamed_addr constant [3 x i8] c"CB\00"
// CHECK: [[CB2:@.*]] = private unnamed_addr constant [4 x i8] c"CB2\00"
// CHECK: [[Buf4:@.*]] = private unnamed_addr constant [5 x i8] c"Buf4\00"
// CHECK: [[Buf5:@.*]] = private unnamed_addr constant [5 x i8] c"Buf5\00"
// CHECK: [[Buf6:@.*]] = private unnamed_addr constant [5 x i8] c"Buf6\00"

[[vk::binding(23, 102)]] StructuredBuffer<float> Buf;
[[vk::binding(14, 1)]] StructuredBuffer<float> Buf2 : register(t23, space102);
[[vk::binding(14)]] StructuredBuffer<float> Buf3 : register(t23, space102);

[[vk::binding(1, 2)]] cbuffer CB {
  float a;
};

[[vk::binding(10,20)]] cbuffer CB2 {
  float b;
};


[[vk::binding(24, 103)]] Buffer<int> Buf4;
[[vk::binding(25, 104)]] RWBuffer<int2> Buf5;
[[vk::binding(26, 105)]] RWStructuredBuffer<float> Buf6;

[numthreads(1,1,1)]
void main() {
// CHECK: call {{.*}} @llvm.spv.resource.handlefrombinding{{.*}}(i32 102, i32 23, {{.*}} [[Buf]])
// CHECK: call {{.*}} @llvm.spv.resource.handlefrombinding{{.*}}(i32 1, i32 14, {{.*}} [[Buf2]])
// CHECK: call {{.*}} @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 14, {{.*}} [[Buf3]])
// CHECK: call {{.*}} @llvm.spv.resource.handlefrombinding{{.*}}(i32 2, i32 1, {{.*}} [[CB]])
// CHECK: call {{.*}} @llvm.spv.resource.handlefrombinding{{.*}}(i32 20, i32 10, {{.*}} [[CB2]])
// CHECK: call {{.*}} @llvm.spv.resource.handlefrombinding{{.*}}(i32 103, i32 24, {{.*}} [[Buf4]])
// CHECK: call {{.*}} @llvm.spv.resource.handlefrombinding{{.*}}(i32 104, i32 25, {{.*}} [[Buf5]])
// CHECK: call {{.*}} @llvm.spv.resource.handlefrombinding{{.*}}(i32 105, i32 26, {{.*}} [[Buf6]])
  float f1 = Buf.Load(0);
  float f2 = Buf2.Load(0);
  float f3 = Buf3.Load(0);
  int i = Buf4.Load(0);
  Buf5[0] = i;
  Buf6[0] = f1+f2+f3+a+b;
}
