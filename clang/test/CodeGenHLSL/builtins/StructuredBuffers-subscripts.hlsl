// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -o - -O0 %s | FileCheck %s

StructuredBuffer<int> In;
RWStructuredBuffer<int> Out1;
RasterizerOrderedStructuredBuffer<int> Out2;

[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {
  // CHECK: define void @main()

  // CHECK: %[[INPTR:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_0_0t(target("dx.RawBuffer", i32, 0, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[LOAD:.*]] = load i32, ptr %[[INPTR]]
  // CHECK: %[[OUT1PTR:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: store i32 %[[LOAD]], ptr %[[OUT1PTR]]
  Out1[GI] = In[GI];

  // CHECK: %[[INPTR:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_0_0t(target("dx.RawBuffer", i32, 0, 0) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[LOAD:.*]] = load i32, ptr %[[INPTR]]
  // CHECK: %[[OUT2PTR:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_1t(target("dx.RawBuffer", i32, 1, 1) %{{.*}}, i32 %{{.*}})
  // CHECK: store i32 %[[LOAD]], ptr %[[OUT2PTR]]
  Out2[GI] = In[GI];
}
