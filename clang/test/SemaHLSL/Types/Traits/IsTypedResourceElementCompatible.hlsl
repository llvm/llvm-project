// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -fnative-half-type -verify %s
// expected-no-diagnostics

struct oneInt {
    int i;
};

struct twoInt {
   int aa;
   int ab;
};

struct threeInts {
  oneInt o;
  twoInt t;
};

struct oneFloat {
    float f;
};
struct depthDiff {
  int i;
  oneInt o;
  oneFloat f;
};

struct notHomogenous{     
  int i;
  float f;
};

struct EightElements {
  twoInt x[2];
  twoInt y[2];
};

struct EightHalves {
half x[8]; 
};

struct intVec {
  int2 i;
};

struct oneIntWithVec {
  int i;
  oneInt i2;
  int2 i3;
};

struct weirdStruct {
  int i;
  intVec iv;
};

_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(int), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(float), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(float4), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(double2), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(oneInt), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(oneFloat), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(twoInt), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(threeInts), "");
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(notHomogenous), "");
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(depthDiff), "");
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(EightElements), "");
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(EightHalves), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(oneIntWithVec), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(weirdStruct), "");
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(RWBuffer<int>), "");


// arrays not allowed
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(half[4]), "");

template<typename T> struct TemplatedBuffer {
    T a;
    __hlsl_resource_t h;
};
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(TemplatedBuffer<int>), "");

struct MyStruct1 : TemplatedBuffer<float> {
    float x;
};
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(MyStruct1), "");

struct MyStruct2 {
    const TemplatedBuffer<float> TB[10];
};
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(MyStruct2), "");

template<typename T> struct SimpleTemplate {
    T a;
};

// though the element type is incomplete, the type trait should still technically return true
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(SimpleTemplate<__hlsl_resource_t>), "");

_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(SimpleTemplate<float>), "");


typedef int myInt;

struct TypeDefTest {
    int x;
    myInt y;
};

_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(TypeDefTest), "");


