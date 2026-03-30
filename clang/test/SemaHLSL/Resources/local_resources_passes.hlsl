// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -emit-llvm -verify

//==============================================================
// GLOBAL RESOURCES
//==============================================================

RWByteAddressBuffer gBuf0 : register(u0);
RWByteAddressBuffer gBuf1 : register(u1);
RWByteAddressBuffer gBuf2 : register(u2);

RWByteAddressBuffer gOut  : register(u3);

RWByteAddressBuffer gBufArray[4] : register(u10);


//==============================================================
// HELPERS
//==============================================================

uint DoStore(RWByteAddressBuffer buf, uint offset, uint value)
{
    buf.Store(offset, value);
    return value;
}


//==============================================================
// PASS TESTS
//==============================================================

// PASS 0
groupshared RWByteAddressBuffer sharedBuf;

uint Use_Shared(uint idx)
{
    return DoStore(gBuf0, idx * 4, 1);
}

// PASS 1
uint Pass_TernaryInit(bool cond, uint idx)
{
    // expected-warning@+1{{assignment of 'cond ? gBuf0 : gBuf1' to local resource 'buf' is not to the same unique global resource}}
    RWByteAddressBuffer buf = cond ? gBuf0 : gBuf1;
    return DoStore(buf, idx * 4, 2);
}

// PASS 2
void Pass_LoopVar()
{
    for(RWByteAddressBuffer buf = gBuf0; false;)
    {
    }
}

// PASS 3
uint Pass_ExpressionInit(uint idx)
{
    RWByteAddressBuffer buf = (true ? gBuf0 : gBuf1);
    return DoStore(buf, idx * 4, 3);
}

// PASS 4
struct PassBufStruct { RWByteAddressBuffer buf; };

groupshared PassBufStruct sharedStruct;

uint Use_PassSharedStruct(uint idx)
{
    return DoStore(gBuf0, idx * 4, 4);
}

// PASS 5
uint Pass_StructArray(uint idx)
{
    PassBufStruct s[2];
    s[0].buf = gBuf0;
    return DoStore(s[0].buf, idx * 4, 5);
}

// PASS 6
groupshared RWByteAddressBuffer Pass_Shared;

uint Use_Pass_Shared(uint idx)
{
    return DoStore(gBuf0, idx * 4, 6);
}

// PASS 7
RWByteAddressBuffer Pass_ReturnLocal_Uninitialized()
{
    RWByteAddressBuffer buf;
    return buf;
}

// PASS 8
RWByteAddressBuffer Pass_ReturnLocal()
{
    RWByteAddressBuffer buf = gBuf0;
    return buf;
}

// PASS 9
struct S { RWByteAddressBuffer arr[2]; };

uint Pass_StructArrayAssignment(uint idx)
{
    S s;
    s.arr[0] = gBuf0;
    return DoStore(s.arr[0], idx * 4, 9);
}

// PASS 10
uint Pass_LocalArray(uint idx)
{
    RWByteAddressBuffer arr[2];
    arr[0] = gBuf0;
    return DoStore(arr[0], idx * 4, 10);
}

// PASS 11
uint Pass_Uninitialized(uint idx)
{
    RWByteAddressBuffer buf;
    return DoStore(buf, idx * 4, 11);
}

// PASS 12
uint Pass_Alias(uint idx)
{
    RWByteAddressBuffer buf = gBuf0;
    return DoStore(buf, idx * 4, 12);
}

// PASS 13
uint Pass_Reassign(uint idx)
{
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf = gBuf0;
    // expected-warning@+1{{assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource}}
    buf = gBuf1;
    return DoStore(buf, idx * 4, 13);
}

// PASS 14
uint Pass_IfAlias(bool cond, uint idx)
{
    RWByteAddressBuffer buf;
    // expected-warning@+1{{assignment of 'cond ? gBuf0 : gBuf1' to local resource 'buf' is not to the same unique global resource}}
    buf = cond ? gBuf0 : gBuf1;
    return DoStore(buf, idx * 4, 14);
}

// PASS 15
uint Pass_Loop(uint idx)
{
    uint sum = 0;
    for(unsigned int i=0;i<4;i++)
    {    
        RWByteAddressBuffer buf = gBufArray[i];
        sum += DoStore(buf, idx * 4 + i * 4, 15);
    }
    return sum;
}

// PASS 16
struct PassStruct
{
    RWByteAddressBuffer buf;
};

uint Pass_Struct(uint idx)
{
    PassStruct s;
    s.buf = gBuf0;
    return DoStore(s.buf, idx * 4, 16);
}

// PASS 17
uint Pass_Level2(RWByteAddressBuffer buf, uint idx)
{
    return DoStore(buf, idx * 4, 17);
}

uint Pass_Level1(RWByteAddressBuffer buf, uint idx)
{
    return Pass_Level2(buf, idx);
}

uint Pass_FunctionForward(uint idx)
{
    RWByteAddressBuffer buf = gBuf1;
    return Pass_Level1(buf, idx);
}

// PASS 18
uint Pass_PhiMerge(bool cond, uint idx)
{
    RWByteAddressBuffer buf;
    // expected-warning@+1{{assignment of 'cond ? gBuf0 : gBuf2' to local resource 'buf' is not to the same unique global resource}}
    buf = cond ? gBuf0 : gBuf2;
    return DoStore(buf, idx * 4, 18);
}

// PASS 19
uint Pass_Shadow(uint idx)
{
    RWByteAddressBuffer buf = gBuf0;
    {
        RWByteAddressBuffer buf = gBuf1;
        return DoStore(buf, idx * 4, 19);
    }
}

// PASS 20
uint Pass_Switch(int v, uint idx)
{
    // expected-note@+2{{variable 'buf' is declared here}}
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf = gBuf0;

    switch(v)
    {
        // expected-warning@+1{{assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource}}
        case 1: buf = gBuf1; break;
        // expected-warning@+1{{assignment of 'gBuf2' to local resource 'buf' is not to the same unique global resource}}
        case 2: buf = gBuf2; break;
    }

    return DoStore(buf, idx * 4, 20);
}

// PASS 21
uint Pass_Bindless(uint idx)
{
    RWByteAddressBuffer buf = gBufArray[idx & 3];
    return DoStore(buf, idx * 4, 21);
}

// PASS 22
uint Pass_WaveUse(uint idx)
{
    RWByteAddressBuffer buf = gBuf0;
    uint active = WaveActiveCountBits(true);
    return DoStore(buf, idx * 4, active);
}

// PASS 23
uint Pass_NestedLoops(uint idx)
{
    uint sum = 0;
    for(unsigned int i=0;i<2;i++)
    for(unsigned int j=0;j<2;j++)
    {        
        RWByteAddressBuffer buf = gBufArray[i+j];
        sum += DoStore(buf, idx * 4 + (i+j)*4, 23);
    }
    return sum;
}

// PASS 24
uint Pass_BlockLifetime(uint idx)
{
    RWByteAddressBuffer buf;
    {
        buf = gBuf1;
    }
    return DoStore(buf, idx * 4, 24);
}

// PASS 25
uint Pass_DeepPhi(bool a, bool b, uint idx)
{
    RWByteAddressBuffer buf;

    if(a)
        // expected-warning@+1{{assignment of 'b ? gBuf0 : gBuf1' to local resource 'buf' is not to the same unique global resource}}
        buf = b ? gBuf0 : gBuf1;
    else
        buf = gBuf2;

    return DoStore(buf, idx * 4, 25);
}

// PASS 26
uint Pass_LoopCarried(int iterations, uint idx)
{
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf = gBuf0;

    for(int i=0;i<iterations;i++)
        // expected-warning@+1{{assignment of 'gBufArray[i & 3]' to local resource 'buf' is not to the same unique global resource}}
        buf = gBufArray[i & 3];

    return DoStore(buf, idx * 4, 26);
}

// PASS 27
uint Pass_AliasChain(uint idx)
{
    RWByteAddressBuffer a = gBuf0;
    RWByteAddressBuffer b = a;
    RWByteAddressBuffer c = b;

    return DoStore(c, idx * 4, 27);
}

// PASS 28
struct PassNestedInner { RWByteAddressBuffer buf; };
struct PassNestedOuter { PassNestedInner inner; };

uint Pass_NestedStruct(uint idx)
{
    PassNestedOuter s;
    s.inner.buf = gBuf1;
    return DoStore(s.inner.buf, idx * 4, 28);
}

// PASS 29
struct PassForwardA { RWByteAddressBuffer buf; };
struct PassForwardB { PassForwardA a; };

uint Pass_ForwardStructLayers(uint idx)
{
    PassForwardB b;
    b.a.buf = gBuf2;
    return DoStore(b.a.buf, idx * 4, 29);
}

// PASS 30
uint Pass_SwitchFallthrough(int v, uint idx)
{
    // expected-note@+2{{variable 'buf' is declared here}}
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf = gBuf0;

    switch(v)
    {
        // expected-warning@+1{{assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource}}
        case 0: buf = gBuf1;
        // expected-warning@+1{{assignment of 'gBuf2' to local resource 'buf' is not to the same unique global resource}}
        case 1: buf = gBuf2; break;
    }

    return DoStore(buf, idx * 4, 30);
}

// PASS 31
uint Pass_EarlyReturn(bool cond, uint idx)
{
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf = gBuf0;

    if(cond)
        return DoStore(buf, idx * 4, 31);
    // expected-warning@+1{{assignment of 'gBuf1' to local resource 'buf' is not to the same unique global resource}}
    buf = gBuf1;
    return DoStore(buf, idx * 4, 31);
}

// PASS 32
uint Pass_NestedBlocks(uint idx)
{
    // expected-note@+1{{variable 'buf' is declared here}}
    RWByteAddressBuffer buf;

    {
        buf = gBuf1;
        {
            // expected-warning@+1{{assignment of 'gBuf2' to local resource 'buf' is not to the same unique global resource}}
            buf = gBuf2;
        }
    }

    return DoStore(buf, idx * 4, 32);
}

// PASS 33
uint Pass_BindlessSelection(uint a, uint b, uint idx)
{
    RWByteAddressBuffer buf;

    buf = gBufArray[a & 3];
    buf = gBufArray[b & 3];

    return DoStore(buf, idx * 4, 33);
}


//==============================================================
// ENTRY POINT
//==============================================================

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x + tid.y * 8;

    uint r = 0;

    r += Pass_TernaryInit(true, idx);
    Pass_LoopVar();
    r += Pass_ExpressionInit(idx);
    r += Use_PassSharedStruct(idx);
    r += Pass_StructArray(idx);
    r += Use_Pass_Shared(idx);
    RWByteAddressBuffer tmp0 = Pass_ReturnLocal_Uninitialized();
    RWByteAddressBuffer tmp1 = Pass_ReturnLocal();
    r += Pass_StructArray(idx);
    r += Pass_LocalArray(idx);
    r += Pass_Uninitialized(idx);
    r += Pass_Alias(idx);
    r += Pass_Reassign(idx);
    r += Pass_IfAlias(true, idx);
    r += Pass_Loop(idx);
    r += Pass_Struct(idx);
    r += Pass_FunctionForward(idx);
    r += Pass_PhiMerge(true, idx);
    r += Pass_Shadow(idx);
    r += Pass_Switch(1, idx);
    r += Pass_Bindless(idx);
    r += Pass_WaveUse(idx);
    r += Pass_NestedLoops(idx);
    r += Pass_BlockLifetime(idx);
    r += Pass_DeepPhi(true, false, idx);
    r += Pass_LoopCarried(15, idx);
    r += Pass_AliasChain(idx);
    r += Pass_NestedStruct(idx);
    r += Pass_ForwardStructLayers(idx);
    r += Pass_SwitchFallthrough(0, idx);
    r += Pass_EarlyReturn(true, idx);
    r += Pass_NestedBlocks(idx);
    r += Pass_BindlessSelection(2, 3, idx);

    gOut.Store(idx * 4, r);
}