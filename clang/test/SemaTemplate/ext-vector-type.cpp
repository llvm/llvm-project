// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T, unsigned Length> 
struct make1 { 
  typedef T __attribute__((ext_vector_type(Length))) type; 
};

void test_make1() {
  make1<int, 5>::type x;
  x.x = 4;
}

template<typename T, unsigned Length> 
struct make2 { 
  typedef T __attribute__((ext_vector_type(Length))) type; // expected-error{{zero vector size}}
};

int test_make2() {
  make2<int, 0> x; // expected-note{{in instantiation of}} 
}

template<typename T, unsigned Length> 
struct make3 { 
  typedef T __attribute__((ext_vector_type(Length))) type; // expected-error{{invalid vector element type 's'}}
};

struct s {};

int test_make3() {
  make3<s, 3>x; // expected-note{{in instantiation of}} 
}

template<typename T, T Length> 
struct make4 { 
  typedef T __attribute__((ext_vector_type(Length))) type; 
};

int test_make4() {
  make4<int, 4>::type x;
  x.w = 7;
}

typedef int* int_ptr;
template<unsigned Length>
struct make5 {
  typedef int_ptr __attribute__((ext_vector_type(Length))) type; // expected-error{{invalid vector element type}}
};

template<int Length>
struct make6 {
  typedef int __attribute__((ext_vector_type(Length))) type;
};

int test_make6() {
  make6<4>::type x;
  x.w = 7;

  make6<2>::type y;
  y.x = -1;
  y.w = -1; // expected-error{{vector component access exceeds type}}
}

namespace Deduction {
  template<typename T> struct X0;

  template<typename T, unsigned N>
  struct X0<T __attribute__((ext_vector_type(N)))> {
    static const unsigned value = 0;
  };

  template<typename T>
  struct X0<T __attribute__((ext_vector_type(4)))> {
    static const unsigned value = 1;
  };

  template<unsigned N>
  struct X0<float __attribute__((ext_vector_type(N)))> {
    static const unsigned value = 2;
  };

  template<>
  struct X0<float __attribute__((ext_vector_type(4)))> {
    static const unsigned value = 3;
  };

  typedef int __attribute__((ext_vector_type(2))) int2;
  typedef int __attribute__((ext_vector_type(4))) int4;
  typedef float __attribute__((ext_vector_type(2))) float2;
  typedef float __attribute__((ext_vector_type(4))) float4;

  int array0[X0<int2>::value == 0? 1 : -1];
  int array1[X0<int4>::value == 1? 1 : -1];
  int array2[X0<float2>::value == 2? 1 : -1];
  int array3[X0<float4>::value == 3? 1 : -1];
}
