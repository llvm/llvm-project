// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -fnative-half-type -verify %s
// expected-no-diagnostics


_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(int), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(float), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(float4), "");
_Static_assert(__builtin_hlsl_is_typed_resource_element_compatible(double2), "");

_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(RWBuffer<int>), "");

struct s {
    int x;
};

// structs not allowed
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(s), "");

// arrays not allowed
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(half[4]), "");

typedef vector<int, 8> int8;
// too many elements
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(int8), "");

// size exceeds 16 bytes, and exceeds element count limit after splitting 64 bit types into 32 bit types
_Static_assert(!__builtin_hlsl_is_typed_resource_element_compatible(double3), "");

