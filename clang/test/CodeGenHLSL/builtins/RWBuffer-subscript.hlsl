// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -o - -O0 %s | FileCheck %s --check-prefixes=DXC,CHECK
// RUN: %clang_cc1 -triple spirv1.6-pc-vulkan1.3-compute -emit-llvm -o - -O0 %s | FileCheck %s --check-prefixes=SPIRV,CHECK

RWBuffer<int> In;
RWBuffer<int> Out;

[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {
  // CHECK: define void @main()

  // DXC: %[[INPTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(1) @llvm.dx.resource.getpointer.p1.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
  // SPIRV: %[[INPTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[LOAD:.*]] = load i32, ptr addrspace({{[0-9]+}}) %[[INPTR]]
  // DXC: %[[OUTPTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(1) @llvm.dx.resource.getpointer.p1.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
  // SPIRV: %[[OUTPTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: store i32 %[[LOAD]], ptr addrspace({{[0-9]+}}) %[[OUTPTR]]
  Out[GI] = In[GI];

  // DXC: %[[INPTR:.*]] = call ptr addrspace(1) @llvm.dx.resource.getpointer.p1.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
  // SPIRV: %[[INPTR:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[LOAD:.*]] = load i32, ptr addrspace({{[0-9]+}}) %[[INPTR]]
  // DXC: %[[OUTPTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(1) @llvm.dx.resource.getpointer.p1.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
  // SPIRV: %[[OUTPTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: store i32 %[[LOAD]], ptr addrspace({{[0-9]+}}) %[[OUTPTR]]
  Out[GI] = In.Load(GI);
}
