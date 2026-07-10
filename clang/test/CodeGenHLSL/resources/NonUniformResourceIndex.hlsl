// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute -emit-llvm -disable-llvm-passes -o - %s \
// RUN:   | llvm-cxxfilt | FileCheck -DTARGET=dx %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-compute -emit-llvm -disable-llvm-passes -o - %s \
// RUN:   | llvm-cxxfilt | FileCheck -DTARGET=spv %s

RWBuffer<float> A[10];

[numthreads(4,1,1)]
void main(uint GI : SV_GroupID) {
  // CHECK: %[[GI:.*]] = load i32, ptr %GI.addr
  // CHECK: %[[NURI_1:.*]] = call i32 @llvm.[[TARGET]].resource.nonuniformindex(i32 %[[GI]])
  // CHECK: call void @hlsl::RWBuffer<float>::__createFromImplicitBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // CHECK-SAME: (ptr {{.*}}, i32 noundef 0, i32 noundef 0, i32 noundef 10, i32 noundef %[[NURI_1]], ptr noundef @A.str)
  float a = A[NonUniformResourceIndex(GI)][0];

  // CHECK: %[[GI:.*]] = load i32, ptr %GI.addr
  // CHECK: %[[ADD:.*]] = add i32 %[[GI]], 1
  // CHECK: %[[NURI_2:.*]] = call i32 @llvm.[[TARGET]].resource.nonuniformindex(i32 %[[ADD]])
  // CHECK: %[[MOD:.*]] = urem i32 %[[NURI_2]], 10
  // CHECK: call void @hlsl::RWBuffer<float>::__createFromImplicitBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // CHECK-SAME: (ptr {{.*}}, i32 noundef 0, i32 noundef 0, i32 noundef 10, i32 noundef %[[MOD]], ptr noundef @A.str)
  float b = A[NonUniformResourceIndex(GI + 1) % 10][0];

  // CHECK: %[[GI:.*]] = load i32, ptr %GI.addr
  // CHECK: %[[NURI_3:.*]] = call i32 @llvm.[[TARGET]].resource.nonuniformindex(i32 %[[GI]])
  // CHECK: %[[MUL:.*]] = mul i32 3, %[[NURI_3]]
  // CHECK: %[[ADD2:.*]] = add i32 10, %[[MUL]]
  // CHECK: call void @hlsl::RWBuffer<float>::__createFromImplicitBinding(unsigned int, unsigned int, int, unsigned int, char const*)
  // CHECK-SAME: (ptr {{.*}}, i32 noundef 0, i32 noundef 0, i32 noundef 10, i32 noundef %[[ADD2]], ptr noundef @A.str)
  float c = A[10 + 3 * NonUniformResourceIndex(GI)][0];
  A[0][0] = a + b + c;
}
