// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -fnative-half-type -verify %s
// expected-no-diagnostics


_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(int), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(float), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(float4), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(double2), "");

// types must be complete
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(RWBuffer<int>), "");
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(__hlsl_resource_t), "");

struct notComplete;
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(notComplete), "");


struct s {
    int x;
};

struct Empty {};

template<typename T> struct TemplatedBuffer {
    T a;
};

template<typename T> struct TemplatedVector {
    vector<T, 4> v;
};

// structs not allowed
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(s), "");
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(Empty), "");
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(TemplatedBuffer<int>), "");
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(TemplatedVector<int>), "");

// arrays not allowed
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(half[4]), "");

typedef vector<int, 8> int8;
// too many elements
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(int8), "");

typedef int MyInt;
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(MyInt), "");

// bool and enums not allowed
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(bool), "");
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(vector<bool, 2>), "");

enum numbers { one, two, three };

_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(numbers), "");

// size exceeds 16 bytes, and exceeds element count limit after splitting 64 bit types into 32 bit types
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(double3), "");

