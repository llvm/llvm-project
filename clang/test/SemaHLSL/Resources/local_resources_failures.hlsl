// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -verify

Texture2D<float4> gTex0 : register(t0);
Texture2D<float4> gTex1 : register(t1);
SamplerState gSampler : register(s0);

StructuredBuffer<float4> gSB : register(t2);
RWStructuredBuffer<float4> gRW : register(u0);

Texture2D<float4> gTexArray[4] : register(t10);

//--------------------------------------------------------------
// FAIL 1: static local resource
//--------------------------------------------------------------
// This causes an assert in DXC
float4 Fail_Static(float2 uv)
{
    static Texture2D<float4> tex = gTex0;
    // expected-error@-1 {{static resource}}
    return tex.Sample(gSampler, uv);
}

//--------------------------------------------------------------
// FAIL 2: arithmetic on resource
//--------------------------------------------------------------

float Fail_Arithmetic()
{
    Texture2D<float4> tex = gTex0;
    tex = tex + 1;
    // expected-error@-1 {{scalar, vector, or matrix expected}}
    return tex;
    // expected-error@-1 {{cannot initialize return object of type 'float' with an lvalue of type 'Texture2D<float4>'}}
}

//--------------------------------------------------------------
// FAIL 3: comparison of resources
//--------------------------------------------------------------

bool Fail_Compare()
{
    Texture2D<float4> a = gTex0;
    Texture2D<float4> b = gTex1;
    return a == b;
    // expected-error@-1 {{operator cannot be used with built-in type 'Texture2D<vector<float, 4> >'}}
}

//--------------------------------------------------------------
// FAIL 4: conversion to bool
//--------------------------------------------------------------

bool Fail_Bool()
{
    Texture2D<float4> tex = gTex0;
    return tex;
    // expected-error@-1 {{cannot initialize return object of type 'bool' with an lvalue of type 'Texture2D<float4>'}}
}

//--------------------------------------------------------------
// FAIL 5: cast from resource
//--------------------------------------------------------------

uint Fail_Cast()
{
    Texture2D<float4> tex = gTex0;
    return (uint)tex;
    // expected-error@-1 {{cannot convert from 'Texture2D<float4>' to 'uint'}}
}

//--------------------------------------------------------------
// FAIL 6: addition of resources
//--------------------------------------------------------------

float Fail_Add()
{
    Texture2D<float4> tex = gTex0;
    return tex + tex;
    // expected-error@-1 {{scalar, vector, or matrix expected}}
}


//--------------------------------------------------------------
// FAIL 7: default parameter on resource
//--------------------------------------------------------------

float4 Fail_DefaultParam(Texture2D<float4> tex = gTex0, float2 uv)
    // expected-error@-1 {{missing default argument on parameter 'uv'}}
    // expected-note@-2 {{candidate function not viable: requires 2 arguments, but 1 was provided}}
    // note, this note above does not get emitted when -verify is passed as an option.
{
    return tex.Sample(gSampler, uv);
}

//--------------------------------------------------------------
// FAIL 8: reinterpret cast
//--------------------------------------------------------------

float4 Fail_Reinterpret(float2 uv)
{
    Texture2D<float4> tex = gTex0;
    return ((Texture2D<float4>)gSampler).Sample(gSampler, uv);
    // expected-error@-1 {{cannot convert from 'SamplerState' to 'Texture2D<float4>'}}
}

//--------------------------------------------------------------
// FAIL 9: RWStructuredBuffer with resource type
//--------------------------------------------------------------

void Fail_LocalBuffer()
{
    RWStructuredBuffer<Texture2D<float4> > badBuffer;
    // expected-error@-1 {{object 'Texture2D<float4>' is not allowed in builtin template parameters}}
}

//--------------------------------------------------------------
// FAIL 10: wave uniformity violation
//--------------------------------------------------------------

float4 Fail_WaveUniform(float2 uv)
{
    Texture2D<float4> tex = gTex0;
    // I presume the reason we don't see this expected error is that 
    // DxilCondenseResources is responsible for emitting this.
    // So, it will not run when -verify is passed to the compiler since
    // DXIL IR is not the intended output in that situation.
    // However, in DXC, we do expect an error here.
    if(WaveActiveAllTrue(true))
        // expected-error@-1 {{local resource not guaranteed to map to unique global resource}}
        tex = gTex1;
    return tex.Sample(gSampler, uv);
}

//--------------------------------------------------------------
// Entry point calling all fail functions to prevent DCE
//--------------------------------------------------------------

[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = float2(0,0);
    float4 r = 0;
    gTex0 = tex;
    // FAIL 1
    r += Fail_Static(uv);

    // FAIL 2
    Fail_Arithmetic();

    // FAIL 3
    Fail_Compare();

    // FAIL 4
    Fail_Bool();

    // FAIL 5
    Fail_Cast();

    // FAIL 6
    Fail_Add();

    // FAIL 7
    // note, this error does not get emitted when -verify is passed as an option.
    // expected-error@+1{{no matching function for call to 'Fail_DefaultParam'}}
    Fail_DefaultParam(gTex0, uv);

    // FAIL 8
    Fail_Reinterpret(uv);

    // FAIL 9
    Fail_LocalBuffer();

    // FAIL 10
    r += Fail_WaveUniform(uv);
}
