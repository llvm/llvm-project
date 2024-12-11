// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s | FileCheck %s

[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {
  main(GI - 1);
}

// For HLSL entry functions, we are generating a C-export function that wraps
// the C++-mangled entry function. The wrapper function can be used to populate
// semantic parameters and provides the expected void(void) signature that
// drivers expect for entry points.

//CHECK: define void @main() #[[#ENTRY_ATTR:]] {
//CHECK-NEXT: entry:
//CHECK-NEXT:   %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
//CHECK-NEXT:   call void @_Z4mainj(i32 %0)
//CHECK-NEXT:   ret void
//CHECK-NEXT: }

// Verify that the entry had the expected dx.shader attribute

//CHECK: attributes #[[#ENTRY_ATTR]] = { {{.*}}"hlsl.shader"="compute"{{.*}} }
