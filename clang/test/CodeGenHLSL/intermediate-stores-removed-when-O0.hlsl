// RUN: not %clang_cc1 -triple dxilv1.7-unknown-shadermodel6.7-compute -O0 -S -hlsl-entry main -finclude-default-header -o - -x hlsl %s 2>&1


// Issue caused by issue https://github.com/llvm/llvm-project/issues/168604
// CHECK: error: Unsupported intrinsic llvm.experimental.noalias.scope.decl for DXIL lowering

// CHECK-NOT: Load of {{.*}} is not a global resource handle

RWBuffer<int> In : register(u0);
RWBuffer<int> Out : register(u1);

[numthreads(1,1,1)]
void main(uint GI : SV_GroupIndex) {
    Out[GI] = In[GI];
}

