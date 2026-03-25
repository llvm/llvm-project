// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -verify

//==============================================================
// COMPREHENSIVE LOCAL RESOURCE VARIABLE TEST SUITE
//
// Structure
//   PASS TESTS
//   ENTRYPOINT
//
// Goal
//   Exhaustively document legal and illegal uses of HLSL
//   resource variables declared in local scope.
//
//==============================================================

//--------------------------------------------------------------
// Global resources
//--------------------------------------------------------------

Texture2D<float4> gTex0 : register(t0);
Texture2D<float4> gTex1 : register(t1);
Texture2D<float4> gTex2 : register(t2);

SamplerState gSampler : register(s0);

RWTexture2D<float4> gOut : register(u0);

StructuredBuffer<float4> gSB : register(t3);
RWStructuredBuffer<float4> gRW : register(u1);

Texture2D<float4> gTexArray[4] : register(t10);


//==============================================================
// PASS TESTS
//==============================================================


//--------------------------------------------------------------
// PASS 0
//--------------------------------------------------------------

groupshared Texture2D<float4> sharedTex;

float4 Use_SharedTex(float2 uv)
{
    return gTex0.Sample(gSampler, uv);
}

//--------------------------------------------------------------
// PASS 1
//--------------------------------------------------------------

float4 Fail_TernaryInit(bool cond, float2 uv)
{
    Texture2D<float4> tex = cond ? gTex0 : gTex1;

    return tex.Sample(gSampler, uv);
}
//--------------------------------------------------------------
// PASS 2
//--------------------------------------------------------------

void Fail_LoopVar()
{
    for(Texture2D<float4> tex = gTex0; false;)
    {
    }
}

//--------------------------------------------------------------
// PASS 3
//--------------------------------------------------------------

float4 Fail_ExpressionInit(float2 uv)
{
    Texture2D<float4> tex = (true ? gTex0 : gTex1);
    return tex.Sample(gSampler, uv);
}
//--------------------------------------------------------------
// PASS 4
//--------------------------------------------------------------

struct FailSharedStruct
{
    Texture2D<float4> tex;
};

groupshared FailSharedStruct sharedStruct;

float4 Use_FailSharedStruct(float2 uv)
{
    return gTex0.Sample(gSampler, uv);
}

//--------------------------------------------------------------
// PASS 5
//--------------------------------------------------------------

struct FailStruct
{
    Texture2D<float4> tex;
};

float4 Fail_StructArray(float2 uv)
{
    FailStruct s[2];

    s[0].tex = gTex0;
    return s[0].tex.Sample(gSampler, uv);
}

//--------------------------------------------------------------
// PASS 6
//--------------------------------------------------------------
groupshared Texture2D<float4> Fail_Shared;

float4 Use_Fail_Shared(float2 uv)
{
    return gTex0.Sample(gSampler, uv);
}

//--------------------------------------------------------------
// PASS 7
//--------------------------------------------------------------
Texture2D<float4> Fail_ReturnLocal_Uninitialized() 
{
  Texture2D<float4> tex; // uninitialized local resource 
  return tex;
}

//--------------------------------------------------------------
// PASS 8
//--------------------------------------------------------------
Texture2D<float4> Fail_ReturnLocal() 
{
  Texture2D<float4> tex = gTex0; 
  return tex; 
}


//--------------------------------------------------------------
// PASS 9
//--------------------------------------------------------------
struct S { Texture2D<float4> arr[2]; };
float4 Pass_StructArray(float2 uv)
{
    S s;
    s.arr[0] = gTex0;
    return s.arr[0].Sample(gSampler, uv);
}

//--------------------------------------------------------------
// PASS 10
//--------------------------------------------------------------

float4 Pass_LocalArray(float2 uv)
{
    Texture2D<float4> arr[2];

    arr[0] = gTex0;
    return arr[0].Sample(gSampler, uv);
}

//--------------------------------------------------------------
// PASS 11
// Uninitialized use
//--------------------------------------------------------------

float4 Pass_Uninitialized(float2 uv)
{
    Texture2D<float4> tex; 
    

    return tex.Sample(gSampler, uv);
}

//--------------------------------------------------------------
// PASS 12
// Simple local alias
//--------------------------------------------------------------

float4 Pass_Alias(float2 uv)
{
    Texture2D<float4> tex = gTex0;
    return tex.Sample(gSampler, uv);
}


//--------------------------------------------------------------
// PASS 13
// Reassignment
//--------------------------------------------------------------

float4 Pass_Reassign(float2 uv)
{
    Texture2D<float4> tex = gTex0;
    tex = gTex1;
    return tex.Sample(gSampler, uv);
}


//--------------------------------------------------------------
// PASS 14
// Control flow aliasing
//--------------------------------------------------------------

float4 Pass_IfAlias(bool cond, float2 uv)
{
    Texture2D<float4> tex;

    if (cond)
        tex = gTex0;
    else
        tex = gTex1;

    return tex.Sample(gSampler, uv);
}


//--------------------------------------------------------------
// PASS 15
// Loop aliasing
//--------------------------------------------------------------

float4 Pass_Loop(float2 uv)
{
    float4 sum = 0;

    for(int i=0;i<4;i++)
    {
        Texture2D<float4> tex = gTexArray[i];
        sum += tex.Sample(gSampler, uv);
    }

    return sum;
}



//--------------------------------------------------------------
// PASS 16
// Struct containing resource
//--------------------------------------------------------------

struct PassStruct
{
    Texture2D<float4> tex;
    SamplerState samp;
};

float4 Pass_Struct(float2 uv)
{
    PassStruct s;
    s.tex = gTex0;
    s.samp = gSampler;

    return s.tex.Sample(s.samp, uv);
}



//--------------------------------------------------------------
// PASS 17
// Passing resource through multiple functions
//--------------------------------------------------------------

float4 Pass_Level2(Texture2D<float4> tex, float2 uv)
{
    return tex.Sample(gSampler, uv);
}

float4 Pass_Level1(Texture2D<float4> tex, float2 uv)
{
    return Pass_Level2(tex, uv);
}

float4 Pass_FunctionForward(float2 uv)
{
    Texture2D<float4> tex = gTex1;
    return Pass_Level1(tex, uv);
}



//--------------------------------------------------------------
// PASS 18
// Resource merge via conditional assignments
// (SSA-style PHI equivalent)
//--------------------------------------------------------------

float4 Pass_PhiMerge(bool cond, float2 uv)
{
    Texture2D<float4> tex;

    if(cond)
        tex = gTex0;
    else
        tex = gTex2;

    return tex.Sample(gSampler, uv);
}



//--------------------------------------------------------------
// PASS 19
// Nested scope shadowing
//--------------------------------------------------------------

float4 Pass_Shadow(float2 uv)
{
    Texture2D<float4> tex = gTex0;

    {
        Texture2D<float4> tex = gTex1;
        return tex.Sample(gSampler, uv);
    }
}



//--------------------------------------------------------------
// PASS 20
// Resource in switch
//--------------------------------------------------------------

float4 Pass_Switch(int v, float2 uv)
{
    Texture2D<float4> tex = gTex0;

    switch(v)
    {
        case 1: tex = gTex1; break;
        case 2: tex = gTex2; break;
    }

    return tex.Sample(gSampler, uv);
}



//--------------------------------------------------------------
// PASS 21
// Bindless descriptor indexing
//--------------------------------------------------------------

float4 Pass_Bindless(uint idx, float2 uv)
{
    Texture2D<float4> tex = gTexArray[idx];
    return tex.Sample(gSampler, uv);
}



//--------------------------------------------------------------
// PASS 22
// Resource with wave operations
//--------------------------------------------------------------

float4 Pass_WaveUse(float2 uv)
{
    Texture2D<float4> tex = gTex0;

    float4 v = tex.Sample(gSampler, uv);

    uint active = WaveActiveCountBits(true);

    return v * active;
}



//--------------------------------------------------------------
// PASS 23
// Resource alias used inside nested loops
//--------------------------------------------------------------

float4 Pass_NestedLoops(float2 uv)
{
    float4 sum = 0;

    for(int i=0;i<2;i++)
    for(int j=0;j<2;j++)
    {
        Texture2D<float4> tex = gTexArray[i+j];
        sum += tex.Sample(gSampler, uv);
    }

    return sum;
}



//--------------------------------------------------------------
// PASS 24
// Resource lifetime across blocks
//--------------------------------------------------------------

float4 Pass_BlockLifetime(float2 uv)
{
    Texture2D<float4> tex;

    {
        tex = gTex1;
    }

    return tex.Sample(gSampler, uv);
}


//--------------------------------------------------------------
// PASS 25
// Deep nested PHI merges
//--------------------------------------------------------------

float4 Pass_DeepPhi(bool a, bool b, float2 uv)
{
    Texture2D<float4> tex;

    if(a)
    {
        if(b)
            tex = gTex0;
        else
            tex = gTex1;
    }
    else
    {
        tex = gTex2;
    }

    return tex.Sample(gSampler, uv);
}



//--------------------------------------------------------------
// PASS 26
// Loop-carried resource value
//--------------------------------------------------------------

float4 Pass_LoopCarried(int iterations, float2 uv)
{
    Texture2D<float4> tex = gTex0;

    for(int i=0;i<iterations;i++)
    {
        tex = gTexArray[i & 3];
    }

    return tex.Sample(gSampler, uv);
}



//--------------------------------------------------------------
// PASS 27
// Resource alias chain
//--------------------------------------------------------------

float4 Pass_AliasChain(float2 uv)
{
    Texture2D<float4> a = gTex0;
    Texture2D<float4> b = a;
    Texture2D<float4> c = b;

    return c.Sample(gSampler, uv);
}



//--------------------------------------------------------------
// PASS 28
// Resource inside nested structs
//--------------------------------------------------------------

struct PassNestedInner
{
    Texture2D<float4> tex;
};

struct PassNestedOuter
{
    PassNestedInner inner;
};

float4 Pass_NestedStruct(float2 uv)
{
    PassNestedOuter s;

    s.inner.tex = gTex1;

    return s.inner.tex.Sample(gSampler, uv);
}



//--------------------------------------------------------------
// PASS 29
// Resource forwarded through multiple struct layers
//--------------------------------------------------------------

struct PassForwardA { Texture2D<float4> tex; };
struct PassForwardB { PassForwardA a; };

float4 Pass_ForwardStructLayers(float2 uv)
{
    PassForwardB b;
    b.a.tex = gTex2;

    return b.a.tex.Sample(gSampler, uv);
}



//--------------------------------------------------------------
// PASS 30
// Resource alias inside switch fallthrough
//--------------------------------------------------------------

float4 Pass_SwitchFallthrough(int v, float2 uv)
{
    Texture2D<float4> tex = gTex0;

    switch(v)
    {
        case 0:
            tex = gTex1;
        case 1:
            tex = gTex2;
            break;
    }

    return tex.Sample(gSampler, uv);
}



//--------------------------------------------------------------
// PASS 31
// Resource used after early-return path merge
//--------------------------------------------------------------

float4 Pass_EarlyReturn(bool cond, float2 uv)
{
    Texture2D<float4> tex = gTex0;

    if(cond)
        return tex.Sample(gSampler, uv);

    tex = gTex1;

    return tex.Sample(gSampler, uv);
}


//--------------------------------------------------------------
// PASS 32
// Resource alias across nested blocks
//--------------------------------------------------------------

float4 Pass_NestedBlocks(float2 uv)
{
    Texture2D<float4> tex;

    {
        tex = gTex1;

        {
            tex = gTex2;
        }
    }

    return tex.Sample(gSampler, uv);
}



//--------------------------------------------------------------
// PASS 33
// Resource assigned via bindless selection
//--------------------------------------------------------------

float4 Pass_BindlessSelection(uint a, uint b, float2 uv)
{
    Texture2D<float4> tex;

    tex = gTexArray[a];
    tex = gTexArray[b];

    return tex.Sample(gSampler, uv);
}


//==============================================================
// ENTRY POINT
//==============================================================


[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    float2 uv = float2(tid.xy)/256.0;

    float4 r = 0;
    r += Pass_TernaryInit(true, uv);
    Pass_LoopVar();
    r += Pass_ExpressionInit(uv);
    r += Use_FailSharedStruct(uv);
    r += Pass_StructArray(uv);
    r += Use_Fail_Shared(uv);
    Texture2D<float4> mytex = Fail_ReturnLocal_Uninitialized();
    Texture2D<float4> mytex2 = Fail_ReturnLocal();
    r += Pass_StructArray(uv);
    r += Pass_LocalArray(uv);
    r += Pass_Uninitialized(uv);
    r += Pass_Alias(uv);
    r += Pass_Reassign(uv);
    r += Pass_IfAlias(true,uv);
    r += Pass_Loop(uv);
    r += Pass_Struct(uv);
    r += Pass_FunctionForward(uv);
    r += Pass_PhiMerge(true,uv);
    r += Pass_Shadow(uv);
    r += Pass_Switch(1,uv);
    r += Pass_Bindless(0,uv);
    r += Pass_WaveUse(uv);
    r += Pass_NestedLoops(uv);
    r += Pass_BlockLifetime(uv);
    r += Pass_DeepPhi(true, false, uv);
    r += Pass_LoopCarried(15, uv);
    r += Pass_AliasChain(uv);
    r += Pass_NestedStruct(uv);
    r += Pass_ForwardStructLayers(uv);
    r += Pass_SwitchFallthrough(0, uv);
    r += Pass_EarlyReturn(true, uv);
    r += Pass_NestedBlocks(uv);
    r += Pass_BindlessSelection(2, 3, uv);

    gOut[tid.xy] = r;
}