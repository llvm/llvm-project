// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s

typedef int __attribute__((vector_size(16))) VI4;
constexpr VI4 A = {1,2,3,4};
static_assert(A[0] == 1, "");
static_assert(A[1] == 2, "");
static_assert(A[2] == 3, "");
static_assert(A[3] == 4, "");


/// FIXME: It would be nice if the note said 'vector' instead of 'array'.
static_assert(A[12] == 4, ""); // both-error {{not an integral constant expression}} \
                               // both-note {{cannot refer to element 12 of array of 4 elements in a constant expression}}


/// VectorSplat casts
typedef __attribute__(( ext_vector_type(4) )) float float4;
constexpr float4 vec4_0 = (float4)0.5f;
static_assert(vec4_0[0] == 0.5, "");
static_assert(vec4_0[1] == 0.5, "");
static_assert(vec4_0[2] == 0.5, "");
static_assert(vec4_0[3] == 0.5, "");
constexpr int vec4_0_discarded = ((float4)12.0f, 0);


/// ImplicitValueInitExpr of vector type
constexpr float4 arr4[2] = {
  {1,2,3,4},
};
static_assert(arr4[0][0] == 1, "");
static_assert(arr4[0][1] == 2, "");
static_assert(arr4[0][2] == 3, "");
static_assert(arr4[0][3] == 4, "");
static_assert(arr4[1][0] == 0, "");
static_assert(arr4[1][0] == 0, "");
static_assert(arr4[1][0] == 0, "");
static_assert(arr4[1][0] == 0, "");


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

namespace BoolToSignedIntegralCast{
  typedef __attribute__((__ext_vector_type__(4))) unsigned int int4;
  constexpr int4 intsT = (int4)true;
  static_assert(intsT[0] == -1, "");
  static_assert(intsT[1] == -1, "");
  static_assert(intsT[2] == -1, "");
  static_assert(intsT[3] == -1, "");
}

namespace VectorElementExpr {
  typedef int int2 __attribute__((ext_vector_type(2)));
  typedef int int4 __attribute__((ext_vector_type(4)));
  constexpr int oneElt = int4(3).x;
  static_assert(oneElt == 3);

  constexpr int2 twoElts = ((int4){11, 22, 33, 44}).yz;
  static_assert(twoElts.x == 22, "");
  static_assert(twoElts.y == 33, "");
}

namespace Temporaries {
  typedef __attribute__((vector_size(16))) int vi4a;
  typedef __attribute__((ext_vector_type(4))) int vi4b;
  struct S {
    vi4a v;
    vi4b w;
  };
  int &&s = S().w[1];
}

#ifdef __SIZEOF_INT128__
namespace bigint {
  typedef __attribute__((__ext_vector_type__(4))) __int128 bigint4;
  constexpr bigint4 A = (bigint4)true;
  static_assert(A[0] == -1, "");
  static_assert(A[1] == -1, "");
  static_assert(A[2] == -1, "");
  static_assert(A[3] == -1, "");
}
#endif
