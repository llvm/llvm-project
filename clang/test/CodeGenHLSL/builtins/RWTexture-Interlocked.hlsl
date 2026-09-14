// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan1.3-compute %s -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// Regression coverage for free-function interlocked operations on a texture
// subscript (RWTexture2D<T>[i], RWTexture2DArray<T>[i]). This is the texture
// counterpart of RWBuffer-Interlocked.hlsl: the atomicrmw has to be emitted on
// the pointer returned by resource.getpointer with the whole coordinate vector,
// since that is what DXILResourceAccess splits into the coordinate operands of
// the DXIL AtomicBinOp op. InterlockedMin is called once on a signed texture
// and once on an unsigned one so that the signed/unsigned atomicrmw selection
// is pinned to an exactly-named resource handle type (spirv.SignedImage vs
// spirv.Image). Add new intrinsics here as more InterlockedX operations gain
// resource support.

RWTexture2D<int> Out : register(u0);
RWTexture2DArray<uint> UOut : register(u1);

// CHECK-LABEL: define void @main
// DXCHECK:  %[[PTR1:.*]] = call {{.*}} @llvm.dx.resource.getpointer.{{.*}}(target("dx.Texture", i32, 1, 0, 1, 2) %{{.*}}, <2 x i32> %{{.*}})
// DXCHECK:  atomicrmw add ptr %[[PTR1]], i32 1 syncscope("device") monotonic
// DXCHECK:  %[[PTR2:.*]] = call {{.*}} @llvm.dx.resource.getpointer.{{.*}}(target("dx.Texture", i32, 1, 0, 1, 2) %{{.*}}, <2 x i32> %{{.*}})
// DXCHECK:  atomicrmw min ptr %[[PTR2]], i32 1 syncscope("device") monotonic
// DXCHECK:  %[[PTR3:.*]] = call {{.*}} @llvm.dx.resource.getpointer.{{.*}}(target("dx.Texture", i32, 1, 0, 0, 7) %{{.*}}, <3 x i32> %{{.*}})
// DXCHECK:  atomicrmw or ptr %[[PTR3]], i32 1 syncscope("device") monotonic
// DXCHECK:  %[[PTR4:.*]] = call {{.*}} @llvm.dx.resource.getpointer.{{.*}}(target("dx.Texture", i32, 1, 0, 0, 7) %{{.*}}, <3 x i32> %{{.*}})
// DXCHECK:  atomicrmw xor ptr %[[PTR4]], i32 1 syncscope("device") monotonic
// DXCHECK:  %[[PTR5:.*]] = call {{.*}} @llvm.dx.resource.getpointer.{{.*}}(target("dx.Texture", i32, 1, 0, 0, 7) %{{.*}}, <3 x i32> %{{.*}})
// DXCHECK:  atomicrmw umin ptr %[[PTR5]], i32 1 syncscope("device") monotonic
// SPVCHECK: %[[PTR1:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, {{.*}}) %{{.*}}, <2 x i32> %{{.*}})
// SPVCHECK: atomicrmw add ptr addrspace(11) %[[PTR1]], i32 1 syncscope("device") monotonic
// SPVCHECK: %[[PTR2:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, {{.*}}) %{{.*}}, <2 x i32> %{{.*}})
// SPVCHECK: atomicrmw min ptr addrspace(11) %[[PTR2]], i32 1 syncscope("device") monotonic
// SPVCHECK: %[[PTR3:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.Image", i32, {{.*}}) %{{.*}}, <3 x i32> %{{.*}})
// SPVCHECK: atomicrmw or ptr addrspace(11) %[[PTR3]], i32 1 syncscope("device") monotonic
// SPVCHECK: %[[PTR4:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.Image", i32, {{.*}}) %{{.*}}, <3 x i32> %{{.*}})
// SPVCHECK: atomicrmw xor ptr addrspace(11) %[[PTR4]], i32 1 syncscope("device") monotonic
// SPVCHECK: %[[PTR5:.*]] = call {{.*}} @llvm.spv.resource.getpointer.{{.*}}(target("spirv.Image", i32, {{.*}}) %{{.*}}, <3 x i32> %{{.*}})
// SPVCHECK: atomicrmw umin ptr addrspace(11) %[[PTR5]], i32 1 syncscope("device") monotonic
[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
  InterlockedAdd(Out[id.xy], 1);
  InterlockedMin(Out[id.xy], 1);
  InterlockedOr(UOut[id], 1u);
  InterlockedXor(UOut[id], 1u);
  InterlockedMin(UOut[id], 1u);
}
