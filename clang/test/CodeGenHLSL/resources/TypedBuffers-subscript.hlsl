// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -o - -O0 %s | FileCheck %s --check-prefixes=DXIL,CHECK
// RUN: %clang_cc1 -triple spirv1.6-pc-vulkan1.3-compute -fspv-use-unknown-image-format -emit-llvm -o - -O0 %s | FileCheck %s --check-prefixes=SPIRV,CHECK

Buffer<int> In;
RWBuffer<int> Out;

[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {
  // CHECK: define void @main()

  // DXIL: %[[INPTR:.*]] = call {{.*}} ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_0_0_1t(target("dx.TypedBuffer", i32, 0, 0, 1) %{{.*}}, i32 %{{.*}})
  // SPIRV: %[[INPTR:.*]] = call {{.*}} ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_1_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 1, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[LOAD:.*]] = load i32, ptr {{.*}}%[[INPTR]]
  // DXIL: %[[OUTPTR:.*]] = call {{.*}} ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
  // SPIRV: %[[OUTPTR:.*]] = call {{.*}} ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: store i32 %[[LOAD]], ptr {{.*}}%[[OUTPTR]]
  Out[GI] = In[GI];

  // DXIL: %[[INPTR:.*]] = call {{.*}} ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
  // SPIRV: %[[INPTR:.*]] = call {{.*}} ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[LOAD:.*]] = load i32, ptr {{.*}}%[[INPTR]]
  // DXIL: %[[OUTPTR:.*]] = call {{.*}} ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
  // SPIRV: %[[OUTPTR:.*]] = call {{.*}} ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: store i32 %[[LOAD]], ptr {{.*}}%[[OUTPTR]]
  Out[GI + 1] = Out[GI];
}
