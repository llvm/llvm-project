// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only -verify %s

// Some bad declarations
hlsl::matrix ShouldWorkSomeday; // expected-error{{use of alias template 'hlsl::matrix' requires template arguments}}
// expected-note@*:* {{template declaration from hidden source: template <class element = float, int rows_count = 4, int cols_count = 4> requires rows_count <= 4 && cols_count <= 4 using matrix = element __attribute__((matrix_type(rows_count, cols_count)))}}

hlsl::matrix<1,1,1> BadMat; // expected-error{{template argument for template type parameter must be a type}}
// expected-note@*:* {{template parameter from hidden source: class element = float}}

hlsl::matrix<int, float,4> AnotherBadMat; // expected-error{{template argument for non-type template parameter must be an expression}}
// expected-note@*:* {{template parameter from hidden source: int rows_count = 4}}

hlsl::matrix<int, 2, 3, 2> YABV; // expected-error{{too many template arguments for alias template 'matrix'}}
// expected-note@*:* {{template declaration from hidden source: template <class element = float, int rows_count = 4, int cols_count = 4> requires rows_count <= 4 && cols_count <= 4 using matrix = element __attribute__((matrix_type(rows_count, cols_count)))}}

// This code is rejected by clang because clang puts the HLSL built-in types
// into the HLSL namespace.
namespace hlsl {
  struct matrix {}; // expected-error {{redefinition of 'matrix'}}
}

// This code is rejected by dxc because dxc puts the HLSL built-in types
// into the global space, but clang will allow it even though it will shadow the
// matrix template.
struct matrix {}; // expected-note {{candidate found by name lookup is 'matrix'}}

matrix<int,2,2> matInt2x2; // expected-error {{reference to 'matrix' is ambiguous}}

// expected-note@*:* {{candidate found by name lookup is 'hlsl::matrix'}}
