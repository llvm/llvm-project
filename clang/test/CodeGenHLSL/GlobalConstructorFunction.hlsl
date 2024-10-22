// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -O0 %s -o - | FileCheck %s --check-prefixes=CHECK,INLINE

int i;

__attribute__((constructor)) void call_me_first(void) {
  i = 12;
}

__attribute__((constructor)) void then_call_me(void) {
  i = 13;
}

__attribute__((destructor)) void call_me_last(void) {
  i = 0;
}

[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {}

// Make sure global variable for ctors/dtors removed.
// CHECK-NOT:@llvm.global_ctors
// CHECK-NOT:@llvm.global_dtors

// CHECK: define void @main()
// CHECK-NEXT: entry:
// Verify function constructors are emitted
// NOINLINE-NEXT:   call void @_Z13call_me_firstv()
// NOINLINE-NEXT:   call void @_Z12then_call_mev()
// NOINLINE-NEXT:   %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
// NOINLINE-NEXT:   call void @_Z4mainj(i32 %0)
// NOINLINE-NEXT:   call void @_Z12call_me_lastv(
// NOINLINE-NEXT:   ret void

// Verify constructor calls are inlined when AlwaysInline is run
// INLINE-NEXT:   alloca
// INLINE-NEXT:   store i32 12
// INLINE-NEXT:   store i32 13
// INLINE-NEXT:   %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
// INLINE-NEXT:   store i32 %
// INLINE-NEXT:   store i32 0
// INLINE:   ret void
