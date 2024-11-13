// RUN: %clang_cc1 -triple dxil-unknown-shadermodel6.6-compute -S -finclude-default-header -o - %s | FileCheck %s

// The purpose of this test is to ensure that the AST writer
// only emits struct bodies when within the context of a 
// larger object that is being outputted on the RHS.


// note that "{ <4 x float> }" in the check below is a struct type, but only the
// body is emitted on the RHS because we are already in the context of a
// target extension type definition (class.hlsl::StructuredBuffer)
// CHECK: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", { <4 x float> }, 0, 0), %struct.mystruct }
// CHECK: %struct.mystruct = type { <4 x float> }
// CHECK: %dx.types.Handle = type { ptr }
// CHECK: %dx.types.ResBind = type { i32, i32, i32, i8 }
// CHECK: %dx.types.ResourceProperties = type { i32, i32 }

struct mystruct
{
    float4 Color;
};

StructuredBuffer<mystruct> my_buffer : register(t2, space4);

export float4 test()
{
    return my_buffer[0].Color;
}

[numthreads(1,1,1)]
void main() {}
