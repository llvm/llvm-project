// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only -verify %s

// Some bad declarations
hlsl::vector ShouldWorkSomeday; // expected-error{{use of alias template 'hlsl::vector' requires template arguments}}
// expected-note@*:* {{template declaration from hidden source: template <class element = float, int element_count = 4> using vector = element __attribute__((ext_vector_type(element_count)))}}

hlsl::vector<1> BadVec; // expected-error{{template argument for template type parameter must be a type}}
// expected-note@*:* {{template parameter from hidden source: class element = float}}

hlsl::vector<int, float> AnotherBadVec; // expected-error{{template argument for non-type template parameter must be an expression}}
// expected-note@*:* {{template parameter from hidden source: int element_count = 4}}

hlsl::vector<int, 2, 3> YABV; // expected-error{{too many template arguments for alias template 'vector'}}
// expected-note@*:* {{template declaration from hidden source: template <class element = float, int element_count = 4> using vector = element __attribute__((ext_vector_type(element_count)))}}

// This code is rejected by clang because clang puts the HLSL built-in types
// into the HLSL namespace.
namespace hlsl {
  struct vector {}; // expected-error {{redefinition of 'vector'}}
}

// This code is rejected by dxc because dxc puts the HLSL built-in types
// into the global space, but clang will allow it even though it will shadow the
// vector template.
struct vector {}; // expected-note {{candidate found by name lookup is 'vector'}}

vector<int,2> VecInt2; // expected-error {{reference to 'vector' is ambiguous}}

// expected-note@*:* {{candidate found by name lookup is 'hlsl::vector'}}
