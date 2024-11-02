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


// Verify the constructor is alwaysinline
// NOINLINE: ; Function Attrs: {{.*}}alwaysinline
// NOINLINE-NEXT: define linkonce_odr void @_ZN4hlsl8RWBufferIfEC2Ev({{.*}} [[CtorAttr:\#[0-9]+]]

// NOINLINE: ; Function Attrs: {{.*}}alwaysinline
// NOINLINE-NEXT: define internal void @_GLOBAL__sub_I_GlobalConstructorLib.hlsl() [[InitAttr:\#[0-9]+]]

// NOINLINE-DAG: attributes [[InitAttr]] = {{.*}} alwaysinline
// NOINLINE-DAG: attributes [[CtorAttr]] = {{.*}} alwaysinline
