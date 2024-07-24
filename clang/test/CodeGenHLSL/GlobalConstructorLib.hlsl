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
// NOINLINE-NEXT:   call void @"?FirstEntry@@YAXXZ"()
// Verify inlining leaves only calls to "llvm." intrinsics
// INLINE-NOT:   call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// CHECK: ret void

[shader("compute")]
[numthreads(1,1,1)]
void SecondEntry() {}

// CHECK: define void @SecondEntry()
// CHECK-NEXT: entry:
// NOINLINE-NEXT:   call void @_GLOBAL__sub_I_GlobalConstructorLib.hlsl()
// NOINLINE-NEXT:   call void @"?SecondEntry@@YAXXZ"()
// Verify inlining leaves only calls to "llvm." intrinsics
// INLINE-NOT:   call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// CHECK: ret void


// Verify the constructor is alwaysinline
// NOINLINE: ; Function Attrs: {{.*}}alwaysinline
// NOINLINE-NEXT: define internal void @_GLOBAL__sub_I_GlobalConstructorLib.hlsl() [[IntAttr:\#[0-9]+]]

// NOINLINE: attributes [[IntAttr]] = {{.*}} alwaysinline
