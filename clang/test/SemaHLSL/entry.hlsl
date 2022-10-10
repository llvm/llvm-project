// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry foo -DWITH_NUM_THREADS -ast-dump -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -hlsl-entry foo  -o - %s  -verify


// Make sure add HLSLShaderAttr along with HLSLNumThreadsAttr.
// CHECK:HLSLNumThreadsAttr 0x{{.*}} <line:10:2, col:18> 1 1 1
// CHECK:HLSLShaderAttr 0x{{.*}} <line:13:1> Compute

#ifdef WITH_NUM_THREADS
[numthreads(1,1,1)]
#endif
// expected-error@+1 {{missing numthreads attribute for compute shader entry}}
void foo() {

}
