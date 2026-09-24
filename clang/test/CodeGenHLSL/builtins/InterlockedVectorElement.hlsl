// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXCHECK

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPVCHECK

// The destination of an atomic can name one element of a vector. Check that
// the atomic gets the address of that element and not the address of the whole
// vector. The tests call the builtins directly because the HLSL functions do
// not yet bind a reference to a vector element.

groupshared int4 gs;
RWStructuredBuffer<int4> Buf : register(u0);

// A component name gives a constant index, so the offset folds into the GEP.
// CHECK-LABEL: define {{.*}}void @{{.*}}test_component
// CHECK: atomicrmw add ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @gs, {{i32|i64}} 8), i32 %{{.*}} syncscope("workgroup") monotonic
export void test_component(int v) {
  __builtin_hlsl_interlocked_add(gs.z, v);
}

// A constant subscript names the same element and gives the same address.
// CHECK-LABEL: define {{.*}}void @{{.*}}test_subscript
// CHECK: atomicrmw add ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @gs, {{i32|i64}} 8), i32 %{{.*}} syncscope("workgroup") monotonic
export void test_subscript(int v) {
  __builtin_hlsl_interlocked_add(gs[2], v);
}

// A variable subscript indexes by the element type, not by the vector type.
// CHECK-LABEL: define {{.*}}void @{{.*}}test_dynamic
// CHECK: [[ELT:%.*]] = getelementptr i32, ptr addrspace(3) @gs, i32 %{{.*}}
// CHECK: atomicrmw add ptr addrspace(3) [[ELT]], i32 %{{.*}} syncscope("workgroup") monotonic
export void test_dynamic(int i, int v) {
  __builtin_hlsl_interlocked_add(gs[i], v);
}

// Compare-store takes the element address the same way.
// CHECK-LABEL: define {{.*}}void @{{.*}}test_cmpstore
// CHECK: cmpxchg ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @gs, {{i32|i64}} 8), i32 %{{.*}}, i32 %{{.*}} syncscope("workgroup") monotonic monotonic
export void test_cmpstore(int cmp, int v) {
  __builtin_hlsl_interlocked_compare_store(gs.z, cmp, v);
}

// An element of a buffer element gets the byte offset of the component.
// CHECK-LABEL: define {{.*}}void @{{.*}}test_resource
// DXCHECK:  [[ELT:%.*]] = getelementptr i32, ptr %{{.*}}, i32 2
// DXCHECK:  cmpxchg ptr [[ELT]], i32 %{{.*}}, i32 %{{.*}} syncscope("device") monotonic monotonic
// SPVCHECK: [[ELT:%.*]] = getelementptr i32, ptr addrspace(11) %{{.*}}, i64 2
// SPVCHECK: cmpxchg ptr addrspace(11) [[ELT]], i32 %{{.*}}, i32 %{{.*}} syncscope("device") monotonic monotonic
export void test_resource(int cmp, int v) {
  __builtin_hlsl_interlocked_compare_store(Buf[0].z, cmp, v);
}
