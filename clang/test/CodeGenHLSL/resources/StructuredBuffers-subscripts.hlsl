// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -o - -O0 %s | FileCheck %s -check-prefixes=DXIL,CHECK
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -emit-llvm -o - -O0 %s | FileCheck %s -check-prefixes=SPV,CHECK

struct S {
  float f;
};

StructuredBuffer<int> In;
RWStructuredBuffer<int> Out1;
RWStructuredBuffer<S> RWSB3;

#ifndef __SPIRV__
// RasterizerOrderedStructuredBuffer are not implement in SPIR-V yet.
RasterizerOrderedStructuredBuffer<int> Out2;
#endif

[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {
  // CHECK: define void @main()

  // CHECK: %[[TMP:.*]] = alloca %struct.S, align 1

  // DXIL: %[[INPTR:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_0_0t.i32(target("dx.RawBuffer", i32, 0, 0) %{{.*}}, i32 %{{.*}})
  // SPV: %[[INPTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_0t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[LOAD:.*]] = load i32, ptr {{.*}}%[[INPTR]]
  // DXIL: %[[OUT1PTR:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %{{.*}}, i32 %{{.*}})
  // SPV: %[[OUT1PTR:.*]] =  call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %{{.*}}, i32 %{{.*}})
  // CHECK: store i32 %[[LOAD]], ptr {{.*}}%[[OUT1PTR]]
  Out1[GI] = In[GI];

#ifndef __SPIRV__
  // DXIL: %[[INPTR:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_0_0t.i32(target("dx.RawBuffer", i32, 0, 0) %{{.*}}, i32 %{{.*}})
  // DXIL: %[[LOAD:.*]] = load i32, ptr %[[INPTR]]
  // DXIL: %[[OUT2PTR:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_1t.i32(target("dx.RawBuffer", i32, 1, 1) %{{.*}}, i32 %{{.*}})
  // DXIL: store i32 %[[LOAD]], ptr %[[OUT2PTR]]
  Out2[GI] = In[GI];
#endif

  // SPV: %[[OUTPTR:.*]] = call noundef align 1 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t.i32(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %{{.*}}, i32 %{{.*}})
  // SPV: %[[INPTR:.*]] = call noundef align 1 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t.i32(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %{{.*}}, i32 %{{.*}})
  // SPV: call void @llvm.memcpy.p11.p11.i64(ptr addrspace(11) align 1 %[[OUTPTR]], ptr addrspace(11) align 1 %[[INPTR]], i64 4, i1 false)
  // SPV: call void @llvm.memcpy.p0.p11.i64(ptr align 1 %[[TMP]], ptr addrspace(11) align 1 %[[OUTPTR]], i64 4, i1 false)

  // DXIL: %[[OUTPTR:.*]] = call noundef nonnull align 1 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_s_struct.Ss_1_0t.i32(target("dx.RawBuffer", %struct.S, 1, 0) %{{.*}}, i32 %{{.*}})
  // DXIL: %[[INPTR:.*]] = call noundef nonnull align 1 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_s_struct.Ss_1_0t.i32(target("dx.RawBuffer", %struct.S, 1, 0) %{{.*}}, i32 %{{.*}})
  // DXIL: call void @llvm.memcpy.p0.p0.i32(ptr align 1 %[[OUTPTR]], ptr align 1 %[[INPTR]], i32 4, i1 false)
  // DXIL: call void @llvm.memcpy.p0.p0.i32(ptr align 1 %[[TMP]], ptr align 1 %[[OUTPTR]], i32 4, i1 false)
  RWSB3[0] = RWSB3[1];
}
