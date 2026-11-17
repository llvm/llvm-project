// Verify that errors are emitted for all kinds of invalid HLX objects

// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -fsyntax-only -triple hexagon -target-cpu hexagonv85 \
// RUN: -target-feature +hlx -verify %s

typedef short HLX_Vector __attribute__((__vector_size__(2048))) __attribute__((aligned(2048)));
typedef long HLX_VectorPair __attribute__((__vector_size__(4096))) __attribute__((aligned(2048)));

HLX_Vector sum; // expected-error{{global, static or variable sized objects with HLX types are not allowed}}
static HLX_Vector sum_static; // expected-error{{global, static or variable sized objects with HLX types are not allowed}}
HLX_Vector sum_arr[10]; // expected-error{{global, static or variable sized objects with HLX types are not allowed}}

typedef struct hlx_struct {
  int p;
  HLX_VectorPair q; // expected-error{{aggregate objects with HLX types are not allowed}}
  HLX_Vector r; // expected-error{{aggregate objects with HLX types are not allowed}}
  static HLX_Vector s; // expected-error{{global, static or variable sized objects with HLX types are not allowed}}
} st0;

typedef struct {
  int a;
  st0 b;
} st1; // No error for parent struct, does not have a member of HLX type

typedef struct {
  HLX_Vector arr[10]; // expected-error{{aggregate objects with HLX types are not allowed}}
  HLX_Vector flexible_arr[]; // expected-error{{aggregate objects with HLX types are not allowed}}
} st2;

template <typename T, unsigned size>
struct st3 { // Dependent on T
  T p; // expected-error2{{aggregate objects with HLX types are not allowed}}
  T arr[size]; // expected-error2{{aggregate objects with HLX types are not allowed}}
  static T q; // expected-error2{{global, static or variable sized objects with HLX types are not allowed}}
};

typedef union hlx_union {
  HLX_Vector p; // expected-error{{aggregate objects with HLX types are not allowed}}
  HLX_VectorPair q; // expected-error{{aggregate objects with HLX types are not allowed}}
  static HLX_Vector r; // expected-error{{global, static or variable sized objects with HLX types are not allowed}}
} un0;

typedef union {
  HLX_Vector arr[10]; // expected-error{{aggregate objects with HLX types are not allowed}}
  HLX_Vector flexible_arr[]; // expected-error{{aggregate objects with HLX types are not allowed}}
} un1;

class cl0 {
  int p;
  HLX_VectorPair q; // expected-error{{aggregate objects with HLX types are not allowed}}
  HLX_Vector r; // expected-error{{aggregate objects with HLX types are not allowed}}
  HLX_Vector arr[10]; // expected-error{{aggregate objects with HLX types are not allowed}}
  static HLX_Vector s; // expected-error{{global, static or variable sized objects with HLX types are not allowed}}
};

class cl1 {
  st1 val; // No error for parent class, does not have a member of HLX type
};

template <typename T, unsigned size>
class cl2 { // Dependent on T
  T p; // expected-error2{{aggregate objects with HLX types are not allowed}}
  T arr[size]; // expected-error2{{aggregate objects with HLX types are not allowed}}
  static T q; // expected-error2{{global, static or variable sized objects with HLX types are not allowed}}
};

HLX_Vector foo(int n, int k, HLX_Vector (*dst)[n]) { // No error for pointer to variable sized array
  struct st3<int, 10> recordE;
  struct st3<HLX_Vector, 10> recordF; // expected-note{{in instantiation of template class}}
  struct st3<HLX_VectorPair, 10> recordG; // expected-note{{in instantiation of template class}}
  HLX_Vector a[n][6]; // expected-error{{global, static or variable sized objects with HLX types are not allowed}}
  HLX_Vector b[n]; // expected-error{{global, static or variable sized objects with HLX types are not allowed}}
  dst = &b;
  return sum;
}

HLX_Vector bar(HLX_VectorPair v) { // No error for function parameter
  cl2<int, 10> C;
  cl2<HLX_Vector, 10> D; // expected-note{{in instantiation of template class}}
  cl2<HLX_VectorPair, 10> E; // expected-note{{in instantiation of template class}}
  HLX_Vector a[10]; // No error for local array with known size
  HLX_VectorPair b; // No error for local variable
  static HLX_Vector static_vec; // expected-error{{global, static or variable sized objects with HLX types are not allowed}}
  return a[0];
}
