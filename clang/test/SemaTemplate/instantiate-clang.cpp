// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test template instantiation for Clang-specific features.

// ---------------------------------------------------------------------
// Vector types
// ---------------------------------------------------------------------
typedef __attribute__(( ext_vector_type(2) )) double double2;
typedef __attribute__(( ext_vector_type(4) )) double double4;

template<typename T>
struct ExtVectorAccess0 {
  void f(T v1, double4 v2) {
    v1.xy = v2.yx;
  }
};

template struct ExtVectorAccess0<double2>;
template struct ExtVectorAccess0<double4>;

typedef __attribute__(( ext_vector_type(2) )) double double2;

template<typename T, typename U, int N, int M>
struct ShuffleVector0 {
  void f(T t, U u, double2 a, double2 b) {
    (void)__builtin_shufflevector(t, u, N, M); // expected-error{{index}}
    (void)__builtin_shufflevector(a, b, N, M); // expected-error{{index}}
    (void)__builtin_shufflevector(a, b, 2, 1);
  }
};

template struct ShuffleVector0<double2, double2, 2, 1>;
template struct ShuffleVector0<double2, double2, 4, 3>; // expected-note{{instantiation}}


