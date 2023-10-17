// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

namespace PR61118 {

union S {
  struct {
    int a;
  };
};

void f(int x, auto) {
  const S result { // expected-error {{field designator (null) does not refer to any field in type 'const S'}}
    .a = x
  };
}

void g(void) {
  f(0, 0); // expected-note {{in instantiation of function template specialization 'PR61118::f<int>' requested here}}
}

} // end namespace PR61118

namespace GH62156 {
union U1 {
   int x;
   float y;
};

struct NonTrivial {
  NonTrivial();
  ~NonTrivial();
};

union U2 {
   NonTrivial x;
   float y;
};

void f() {
   U1 u{.x=2,  // expected-note {{previous initialization is here}}
        .y=1}; // expected-error {{initializer partially overrides prior initialization of this subobject}}
   new U2{.x = NonTrivial{}, // expected-note {{previous initialization is here}}
          .y=1}; // expected-error {{initializer would partially override prior initialization of object of type 'NonTrivial' with non-trivial destruction}}
}
}
