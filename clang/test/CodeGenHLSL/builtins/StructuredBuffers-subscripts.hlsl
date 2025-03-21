// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -o - -O0 %s | FileCheck %s

struct S {
  float f;
};

StructuredBuffer<int> In;
RWStructuredBuffer<int> Out1;
RWStructuredBuffer<S> RWSB3;
RasterizerOrderedStructuredBuffer<int> Out2;

[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {
  // CHECK: define void @main()

  // CHECK: %[[INPTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(1) @llvm.dx.resource.getpointer.p1.tdx.RawBuffer_i32_0_0t(target("dx.RawBuffer", i32, 0, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[LOAD:.*]] = load i32, ptr addrspace(1) %[[INPTR]]
  // CHECK: %[[OUT1PTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(1) @llvm.dx.resource.getpointer.p1.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: store i32 %[[LOAD]], ptr addrspace(1) %[[OUT1PTR]]
  Out1[GI] = In[GI];

  // CHECK: %[[INPTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(1) @llvm.dx.resource.getpointer.p1.tdx.RawBuffer_i32_0_0t(target("dx.RawBuffer", i32, 0, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[LOAD:.*]] = load i32, ptr addrspace(1) %[[INPTR]]
  // CHECK: %[[OUT2PTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(1) @llvm.dx.resource.getpointer.p1.tdx.RawBuffer_i32_1_1t(target("dx.RawBuffer", i32, 1, 1) %{{.*}}, i32 %{{.*}})
  // CHECK: store i32 %[[LOAD]], ptr addrspace(1) %[[OUT2PTR]]
  Out2[GI] = In[GI];

  // The addrspacecast comes from `S::operator=` member function, which expects
  // parameters in address space 0. This is why hlsl_device is a sub address
  // space of the default address space.
  // CHECK: %[[INPTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(1) @llvm.dx.resource.getpointer.p1.tdx.RawBuffer_s_struct.Ss_1_0t(target("dx.RawBuffer", %struct.S, 1, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[INCAST:.*]] = addrspacecast ptr addrspace(1) %[[INPTR]] to ptr
  // CHECK: %[[OUTPTR:.*]] = call noundef align 4 dereferenceable(4) ptr addrspace(1) @llvm.dx.resource.getpointer.p1.tdx.RawBuffer_s_struct.Ss_1_0t(target("dx.RawBuffer", %struct.S, 1, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[OUTCAST:.*]] = addrspacecast ptr addrspace(1) %[[OUTPTR]] to ptr
  // CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %[[OUTCAST]], ptr align 4 %[[INCAST]], i32 4, i1 false)
  RWSB3[0] = RWSB3[1];
}
