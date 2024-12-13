// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -o - -O0 %s | FileCheck %s

RWBuffer<int> In;
RWBuffer<int> Out;

[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {
  // CHECK: define void @main()
  // CHECK: %[[INPTR:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
  // CHECK: %[[LOAD:.*]] = load i32, ptr %[[INPTR]]
  // CHECK: %[[OUTPTR:.*]] = call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
  // CHECK: store i32 %[[LOAD]], ptr %[[OUTPTR]]
  Out[GI] = In[GI];
}
