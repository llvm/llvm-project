// RUN: %clang --driver-mode=dxc -T cs_6_0 -Fo x.dxil %s | FileCheck %s
// CHECK-NOT: llvm.loop.mustprogress

StructuredBuffer<uint4> X : register(t0);
StructuredBuffer<float4> In : register(t1);
RWStructuredBuffer<float4> Out : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
  for (uint I = 0; I < X[dispatch_thread_id].x; ++I) {
    Out[dispatch_thread_id] = In[dispatch_thread_id];
  }
}
