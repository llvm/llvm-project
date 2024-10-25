// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -fnative-half-type -verify %s

// expected-note@+1{{forward declaration of 'notComplete'}}
struct notComplete;
// expected-error@+1{{incomplete type 'notComplete' where a complete type is required}}
_Static_assert(!__builtin_hlsl_is_line_vector_layout_compatible(notComplete), "");


// types must be complete
_Static_assert(__builtin_hlsl_is_line_vector_layout_compatible(__hlsl_resource_t), "");
