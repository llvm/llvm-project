// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -o - -O0 %s | FileCheck %s

RWBuffer<int> In;
RWBuffer<int> Out;

[numthreads(1,1,1)]
void main(unsigned GI : SV_GroupIndex) {
  Out[GI] = In[GI];
}

// Even at -O0 the subscript operators get inlined. The -O0 IR is a bit messy
// and confusing to follow so the match here is pretty weak.

// CHECK: define void @main()
// Verify inlining leaves only calls to "llvm." intrinsics
// CHECK-NOT:   call {{[^@]*}} @{{[^l][^l][^v][^m][^\.]}}
// CHECK: ret void
