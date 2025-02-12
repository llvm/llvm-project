// RUN: %clang_cc1 -std=c++20 -Wno-all -fexperimental-bounds-safety-attributes -verify %s
#include <ptrcheck.h>
typedef unsigned size_t;
class MyClass {
  size_t m;
  int q[__counted_by(m)]; // expected-error{{arrow notation not allowed for struct member in count parameter}}

  // The error is because we do not have FAM support right now.  Previously, this example will crash.
};

namespace value_dependent_assertion_violation {
  // Running attribute-only mode on the C++ code below crashes
  // previously.
  void g(int * __counted_by(n) * p, size_t n);

  template<typename T>
  struct S {
    void f(T p) {
      g(nullptr, sizeof(p));
    }
  };

  // expected-error@-4{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
  template struct S<int>; // expected-note{{in instantiation of member function 'value_dependent_assertion_violation::S<int>::f' requested here}}
}
