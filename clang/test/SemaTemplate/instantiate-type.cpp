// RUN: %clang_cc1 -fsyntax-only -verify %s

int* f(int);
float *f(...);

template<typename T>
struct X {
  typedef typeof(T*) typeof_type;
  typedef typeof(f(T())) typeof_expr;
};

int *iptr0;
float *fptr0;
X<int>::typeof_type &iptr1 = iptr0;

X<int>::typeof_expr &iptr2 = iptr0;
X<float*>::typeof_expr &fptr1 = fptr0;

namespace rdar13094134 {
  template <class>
  class X {
    typedef struct {
      Y *y; // expected-error{{unknown type name 'Y'}}
    } Y; 
  };

  X<int> xi;
}
