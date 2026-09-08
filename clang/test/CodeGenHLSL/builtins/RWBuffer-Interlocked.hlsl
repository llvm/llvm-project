// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan1.3-compute %s -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Regression coverage for free-function interlocked operations on a typed
// resource subscript (RWBuffer<T>[i]). This exercises the
// LangAS::hlsl_device branch of the dest-argument address-space check in
// SemaHLSL and ensures the atomicrmw is emitted on the pointer returned by
// resource.getpointer for a TypedBuffer (as opposed to the RawBuffer path
// covered by the ByteAddressBuffer tests). InterlockedMin is called once on
// a signed buffer and once on an unsigned buffer so that the signed/unsigned
// atomicrmw selection is pinned to an exactly-named resource handle type
// (spirv.SignedImage vs spirv.Image). Add new intrinsics here as more
// InterlockedX operations gain resource support.

RWBuffer<int> Out : register(u0);
RWBuffer<uint> UOut : register(u1);

// CHECK-LABEL: define void @main
// DXCHECK:  %[[PTR1:.*]] = call {{.*}} @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
// DXCHECK:  atomicrmw add ptr %[[PTR1]], i32 1 syncscope("device") monotonic
// DXCHECK:  %[[PTR2:.*]] = call {{.*}} @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
// DXCHECK:  atomicrmw or ptr %[[PTR2]], i32 1 syncscope("device") monotonic
// DXCHECK:  %[[PTR3:.*]] = call {{.*}} @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
// DXCHECK:  atomicrmw xor ptr %[[PTR3]], i32 1 syncscope("device") monotonic
// DXCHECK:  %[[PTR4:.*]] = call {{.*}} @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
// DXCHECK:  atomicrmw min ptr %[[PTR4]], i32 1 syncscope("device") monotonic
// DXCHECK:  %[[PTR5:.*]] = call {{.*}} @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_0t.i32(target("dx.TypedBuffer", i32, 1, 0, 0) %{{.*}}, i32 %{{.*}})
// DXCHECK:  atomicrmw umin ptr %[[PTR5]], i32 1 syncscope("device") monotonic
// DXCHECK:  %[[PTR6:.*]] = call {{.*}} @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
// DXCHECK:  atomicrmw and ptr %[[PTR6]], i32 1 syncscope("device") monotonic
// SPVCHECK: %[[PTR1:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, {{.*}}) %{{.*}}, i32 %{{.*}})
// SPVCHECK: atomicrmw add ptr addrspace(11) %[[PTR1]], i32 1 syncscope("device") monotonic
// SPVCHECK: %[[PTR2:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, {{.*}}) %{{.*}}, i32 %{{.*}})
// SPVCHECK: atomicrmw or ptr addrspace(11) %[[PTR2]], i32 1 syncscope("device") monotonic
// SPVCHECK: %[[PTR3:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, {{.*}}) %{{.*}}, i32 %{{.*}})
// SPVCHECK: atomicrmw xor ptr addrspace(11) %[[PTR3]], i32 1 syncscope("device") monotonic
// SPVCHECK: %[[PTR4:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, {{.*}}) %{{.*}}, i32 %{{.*}})
// SPVCHECK: atomicrmw min ptr addrspace(11) %[[PTR4]], i32 1 syncscope("device") monotonic
// SPVCHECK: %[[PTR5:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.Image", i32, {{.*}}) %{{.*}}, i32 %{{.*}})
// SPVCHECK: atomicrmw umin ptr addrspace(11) %[[PTR5]], i32 1 syncscope("device") monotonic
// SPVCHECK: %[[PTR6:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, {{.*}}) %{{.*}}, i32 %{{.*}})
// SPVCHECK: atomicrmw and ptr addrspace(11) %[[PTR6]], i32 1 syncscope("device") monotonic
[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
  InterlockedAdd(Out[id.x], 1);
  InterlockedOr(Out[id.x], 1);
  InterlockedXor(Out[id.x], 1);
  InterlockedMin(Out[id.x], 1);
  InterlockedMin(UOut[id.x], 1u);
  InterlockedAnd(Out[id.x], 1);
}
