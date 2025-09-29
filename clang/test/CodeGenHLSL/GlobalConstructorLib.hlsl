// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes %s -o - | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -O0 %s -o - | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,INLINE

// Make sure global variable for ctors exist for lib profile.
// CHECK:@llvm.global_ctors

RWBuffer<float> Buffer;

[shader("compute")]
[numthreads(1,1,1)]
void FirstEntry() {}

// CHECK: define void @FirstEntry()
// CHECK-NEXT: entry:
// NOINLINE-NEXT:   call void @_GLOBAL__sub_I_GlobalConstructorLib.hlsl()
// NOINLINE-NEXT:   call void @FirstEntry()
// Verify inlining leaves only calls to "llvm." intrinsics
// INLINE-NOT:   call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// CHECK: ret void

[shader("compute")]
[numthreads(1,1,1)]
void SecondEntry() {}

// CHECK: define void @SecondEntry()
// CHECK-NEXT: entry:
// NOINLINE-NEXT:   call void @_GLOBAL__sub_I_GlobalConstructorLib.hlsl()
// NOINLINE-NEXT:   call void @SecondEntry()
// Verify inlining leaves only calls to "llvm." intrinsics
// INLINE-NOT:   call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// CHECK: ret void


// Verify the constructors are alwaysinline
// NOINLINE: ; Function Attrs: {{.*}}alwaysinline
// NOINLINE-NEXT: define linkonce_odr hidden void @hlsl::RWBuffer<float>::RWBuffer()({{.*}}){{.*}} [[CtorAttr:\#[0-9]+]]

// NOINLINE: ; Function Attrs: {{.*}}alwaysinline
// NOINLINE-NEXT: define linkonce_odr hidden void @hlsl::RWBuffer<float>::RWBuffer(hlsl::RWBuffer<float> const&)({{.*}}){{.*}} [[CtorAttr]]

// NOINLINE: ; Function Attrs: {{.*}}alwaysinline
// NOINLINE-NEXT: define linkonce_odr hidden void @hlsl::RWBuffer<float>::RWBuffer()(ptr noundef nonnull align 4 dereferenceable(4) %this){{.*}} [[CtorAttr:\#[0-9]+]]

// NOINLINE: ; Function Attrs: {{.*}}alwaysinline
// NOINLINE-NEXT: define linkonce_odr hidden void @hlsl::RWBuffer<float>::RWBuffer(hlsl::RWBuffer<float> const&)({{.*}}){{.*}} [[CtorAttr]]

// NOINLINE: ; Function Attrs: {{.*}}alwaysinline
// NOINLINE-NEXT: define internal void @_GLOBAL__sub_I_GlobalConstructorLib.hlsl() [[InitAttr:\#[0-9]+]]

// NOINLINE-DAG: attributes [[InitAttr]] = {{.*}} alwaysinline
// NOINLINE-DAG: attributes [[CtorAttr]] = {{.*}} alwaysinline
