// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s 
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-mesh -x hlsl -ast-dump -o - %s | FileCheck %s 
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-amplification -x hlsl -ast-dump -o - %s | FileCheck %s 
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -o - %s | FileCheck %s 
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -ast-dump -o - %s -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-vertex -x hlsl -ast-dump -o - %s -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-hull -x hlsl -ast-dump -o - %s -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-domain -x hlsl -ast-dump -o - %s -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s -DFAIL -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel5.0-compute -x hlsl -ast-dump -o - %s -DFAIL -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel4.0-compute -x hlsl -ast-dump -o - %s -DFAIL -verify

#if __SHADER_TARGET_STAGE == __SHADER_STAGE_COMPUTE || __SHADER_TARGET_STAGE == __SHADER_STAGE_MESH || __SHADER_TARGET_STAGE == __SHADER_STAGE_AMPLIFICATION || __SHADER_TARGET_STAGE == __SHADER_STAGE_LIBRARY
#ifdef FAIL
#if __SHADER_TARGET_MAJOR == 6
// expected-error@+1 {{'numthreads' attribute requires exactly 3 arguments}}
[numthreads]
// expected-error@+1 {{'numthreads' attribute requires exactly 3 arguments}}
[numthreads()]
// expected-error@+1 {{'numthreads' attribute requires exactly 3 arguments}}
[numthreads(1,2,3,4)]
// expected-error@+1 {{'numthreads' attribute requires an integer constant}}
[numthreads("1",2,3)]
// expected-error@+1 {{argument 'X' to numthreads attribute cannot exceed 1024}}
[numthreads(-1,2,3)]
// expected-error@+1 {{argument 'Y' to numthreads attribute cannot exceed 1024}}
[numthreads(1,-2,3)]
// expected-error@+1 {{argument 'Z' to numthreads attribute cannot exceed 1024}}
[numthreads(1,2,-3)]
// expected-error@+1 {{total number of threads cannot exceed 1024}}
[numthreads(1024,1024,1024)]
#elif __SHADER_TARGET_MAJOR == 5
// expected-error@+1 {{argument 'Z' to numthreads attribute cannot exceed 64}}
[numthreads(1,2,68)]
#else
// expected-error@+1 {{argument 'Z' to numthreads attribute cannot exceed 1}}
[numthreads(1,2,2)]
// expected-error@+1 {{total number of threads cannot exceed 768}}
[numthreads(1024,1,1)]
#endif
#endif
// CHECK: HLSLNumThreadsAttr 0x{{[0-9a-fA-F]+}} <line:{{[0-9]+}}:2, col:18> 1 2 1
[numthreads(1,2,1)]
int entry() {
 return 1;
}

// expected-warning@+1 {{'numthreads' attribute only applies to global functions}}
[numthreads(1,1,1)]
struct Fido {
  // expected-warning@+1 {{'numthreads' attribute only applies to global functions}}
  [numthreads(1,1,1)]
  void wag() {}
};

#else
// expected-error-re@+1 {{attribute 'numthreads' is unsupported in {{[A-Za-z]+}} shaders, requires Compute, Amplification, Mesh or Library}}
[numthreads(1,1,1)]
int main() {
 return 1;
}
#endif


