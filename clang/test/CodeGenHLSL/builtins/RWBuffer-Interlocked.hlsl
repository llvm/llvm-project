// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan1.3-compute %s -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Regression coverage for free-function InterlockedAdd/InterlockedOr on a
// typed resource subscript (RWBuffer<int>[i]). This exercises the
// LangAS::hlsl_device branch of the dest-argument address-space check in
// SemaHLSL and ensures the atomicrmw is emitted on the pointer returned by
// resource.getpointer for a TypedBuffer (as opposed to the RawBuffer path
// covered by the ByteAddressBuffer tests). Add new intrinsics here as more
// InterlockedX operations gain resource support.

RWBuffer<int> Out : register(u0);

// CHECK-LABEL: define void @main
// DXCHECK:  %[[PTR1:.*]] = call {{.*}} @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
// DXCHECK:  atomicrmw add ptr %[[PTR1]], i32 1 monotonic
// DXCHECK:  %[[PTR2:.*]] = call {{.*}} @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) %{{.*}}, i32 %{{.*}})
// DXCHECK:  atomicrmw or ptr %[[PTR2]], i32 1 monotonic
// SPVCHECK: %[[PTR1:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.{{Image|SignedImage}}", i32, {{.*}}) %{{.*}}, i32 %{{.*}})
// SPVCHECK: atomicrmw add ptr addrspace(11) %[[PTR1]], i32 1 monotonic
// SPVCHECK: %[[PTR2:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.{{Image|SignedImage}}", i32, {{.*}}) %{{.*}}, i32 %{{.*}})
// SPVCHECK: atomicrmw or ptr addrspace(11) %[[PTR2]], i32 1 monotonic
[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
  InterlockedAdd(Out[id.x], 1);
  InterlockedOr(Out[id.x], 1);
}
