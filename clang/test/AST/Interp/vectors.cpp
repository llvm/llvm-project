// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s

typedef int __attribute__((vector_size(16))) VI4;
constexpr VI4 A = {1,2,3,4};
static_assert(A[0] == 1, ""); // ref-error {{not an integral constant expression}}
static_assert(A[1] == 2, ""); // ref-error {{not an integral constant expression}}
static_assert(A[2] == 3, ""); // ref-error {{not an integral constant expression}}
static_assert(A[3] == 4, ""); // ref-error {{not an integral constant expression}}


/// FIXME: It would be nice if the note said 'vector' instead of 'array'.
static_assert(A[12] == 4, ""); // ref-error {{not an integral constant expression}} \
                               // expected-error {{not an integral constant expression}} \
                               // expected-note {{cannot refer to element 12 of array of 4 elements in a constant expression}}


/// VectorSplat casts
typedef __attribute__(( ext_vector_type(4) )) float float4;
constexpr float4 vec4_0 = (float4)0.5f;
static_assert(vec4_0[0] == 0.5, ""); // ref-error {{not an integral constant expression}}
static_assert(vec4_0[1] == 0.5, ""); // ref-error {{not an integral constant expression}}
static_assert(vec4_0[2] == 0.5, ""); // ref-error {{not an integral constant expression}}
static_assert(vec4_0[3] == 0.5, ""); // ref-error {{not an integral constant expression}}
constexpr int vec4_0_discarded = ((float4)12.0f, 0);


/// ImplicitValueInitExpr of vector type
constexpr float4 arr4[2] = {
  {1,2,3,4},
};
static_assert(arr4[0][0] == 1, ""); // ref-error {{not an integral constant expression}}
static_assert(arr4[0][1] == 2, ""); // ref-error {{not an integral constant expression}}
static_assert(arr4[0][2] == 3, ""); // ref-error {{not an integral constant expression}}
static_assert(arr4[0][3] == 4, ""); // ref-error {{not an integral constant expression}}
static_assert(arr4[1][0] == 0, ""); // ref-error {{not an integral constant expression}}
static_assert(arr4[1][0] == 0, ""); // ref-error {{not an integral constant expression}}
static_assert(arr4[1][0] == 0, ""); // ref-error {{not an integral constant expression}}
static_assert(arr4[1][0] == 0, ""); // ref-error {{not an integral constant expression}}


/// From constant-expression-cxx11.cpp
namespace Vector {
  typedef int __attribute__((vector_size(16))) VI4;
  constexpr VI4 f(int n) {
    return VI4 { n * 3, n + 4, n - 5, n / 6 };
  }
  constexpr auto v1 = f(10);
  static_assert(__builtin_vectorelements(v1) == (16 / sizeof(int)), "");

  typedef double __attribute__((vector_size(32))) VD4;
  constexpr VD4 g(int n) {
    return (VD4) { n / 2.0, n + 1.5, n - 5.4, n * 0.9 };
  }
  constexpr auto v2 = g(4);
  static_assert(__builtin_vectorelements(v2) == (32 / sizeof(double)), "");
}

/// FIXME: We need to support BitCasts between vector types.
namespace {
  typedef float __attribute__((vector_size(16))) VI42;
  constexpr VI42 A2 = A; // expected-error {{must be initialized by a constant expression}}
}
