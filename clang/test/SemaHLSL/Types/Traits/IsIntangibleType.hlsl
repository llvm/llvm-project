// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -fnative-half-type -verify %s
// expected-no-diagnostics

_Static_assert(__builtin_hlsl_is_intangible(__hlsl_resource_t), "");
// no need to check array of __hlsl_resource_t, arrays of sizeless types are not supported

_Static_assert(!__builtin_hlsl_is_intangible(int), "");
_Static_assert(!__builtin_hlsl_is_intangible(float3), "");
_Static_assert(!__builtin_hlsl_is_intangible(half[4]), "");

typedef __hlsl_resource_t Res;
_Static_assert(__builtin_hlsl_is_intangible(const Res), "");
// no need to check array of Res, arrays of sizeless types are not supported

struct ABuffer {
    const int i[10];
    __hlsl_resource_t h;
};
_Static_assert(__builtin_hlsl_is_intangible(ABuffer), "");
_Static_assert(__builtin_hlsl_is_intangible(ABuffer[10]), "");

struct MyStruct {
    half2 h2;
    int3 i3;
};
_Static_assert(!__builtin_hlsl_is_intangible(MyStruct), "");
_Static_assert(!__builtin_hlsl_is_intangible(MyStruct[10]), "");

class MyClass {
    int3 ivec;
    float farray[12];
    MyStruct ms;
    ABuffer buf;
};
_Static_assert(__builtin_hlsl_is_intangible(MyClass), "");
_Static_assert(__builtin_hlsl_is_intangible(MyClass[2]), "");

union U {
    double d[4];
    Res buf;
};
_Static_assert(__builtin_hlsl_is_intangible(U), "");
_Static_assert(__builtin_hlsl_is_intangible(U[100]), "");

class MyClass2 {
    int3 ivec;
    float farray[12];
    U u;
};
_Static_assert(__builtin_hlsl_is_intangible(MyClass2), "");
_Static_assert(__builtin_hlsl_is_intangible(MyClass2[5]), "");

class Simple {
    int a;
};

template<typename T> struct TemplatedBuffer {
    T a;
    __hlsl_resource_t h;
};
_Static_assert(__builtin_hlsl_is_intangible(TemplatedBuffer<int>), "");

struct MyStruct2 : TemplatedBuffer<float> {
    float x;
};
_Static_assert(__builtin_hlsl_is_intangible(MyStruct2), "");

struct MyStruct3 {
    const TemplatedBuffer<float> TB[10];
};
_Static_assert(__builtin_hlsl_is_intangible(MyStruct3), "");

template<typename T> struct SimpleTemplate {
    T a;
};
_Static_assert(__builtin_hlsl_is_intangible(SimpleTemplate<__hlsl_resource_t>), "");
_Static_assert(!__builtin_hlsl_is_intangible(SimpleTemplate<float>), "");

_Static_assert(__builtin_hlsl_is_intangible(RWBuffer<float>), "");
_Static_assert(__builtin_hlsl_is_intangible(StructuredBuffer<Simple>), "");
