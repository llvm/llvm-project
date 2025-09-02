// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -O0 %s -o - | FileCheck %s --check-prefixes=CHECK,INLINE

// Make sure global variable for ctors exist for lib profile.
// CHECK:@llvm.global_ctors

RWBuffer<float> Buffer;

[shader("compute")]
[numthreads(1,1,1)]
void FirstEntry() {}

// CHECK: define void @FirstEntry()
// CHECK-NEXT: entry:
// NOINLINE-NEXT:   call void @_GLOBAL__sub_I_GlobalConstructorLib.hlsl()
// NOINLINE-NEXT:   call void @_Z10FirstEntryv()
// Verify inlining leaves only calls to "llvm." intrinsics
// INLINE-NOT:   call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// CHECK: ret void

[shader("compute")]
[numthreads(1,1,1)]
void SecondEntry() {}

// CHECK: define void @SecondEntry()
// CHECK-NEXT: entry:
// NOINLINE-NEXT:   call void @_GLOBAL__sub_I_GlobalConstructorLib.hlsl()
// NOINLINE-NEXT:   call void @_Z11SecondEntryv()
// Verify inlining leaves only calls to "llvm." intrinsics
// INLINE-NOT:   call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// CHECK: ret void


// Verify the constructors are alwaysinline
// NOINLINE: ; Function Attrs: {{.*}}alwaysinline
// NOINLINE-NEXT: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIfEC1EU9_Res_u_CTfu17__hlsl_resource_t(ptr {{.*}} %this, target("dx.TypedBuffer", float, 1, 0, 0) %handle) {{.*}} [[CtorAttr:\#[0-9]+]]
// NOINLINE: ; Function Attrs: {{.*}}alwaysinline
// NOINLINE-NEXT: define linkonce_odr hidden void @_ZN4hlsl8RWBufferIfEC2EU9_Res_u_CTfu17__hlsl_resource_t(ptr {{.*}} %this, target("dx.TypedBuffer", float, 1, 0, 0) %handle) {{.*}} [[CtorAttr]]

// NOINLINE: ; Function Attrs: {{.*}}alwaysinline
// NOINLINE-NEXT: define internal void @_GLOBAL__sub_I_GlobalConstructorLib.hlsl() [[InitAttr:\#[0-9]+]]

// NOINLINE-DAG: attributes [[InitAttr]] = {{.*}} alwaysinline
// NOINLINE-DAG: attributes [[CtorAttr]] = {{.*}} alwaysinline
