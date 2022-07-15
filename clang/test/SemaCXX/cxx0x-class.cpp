// RUN: %clang_cc1 -Wno-uninitialized -fsyntax-only -verify -std=c++11 -Wno-error=static-float-init %s 

int vs = 0;

class C {
public:
  struct NestedC {
    NestedC(int);
  };

  int i = 0;
  static int si = 0; // expected-error {{non-const static data member must be initialized out of line}}
  static const NestedC ci = 0; // expected-error {{static data member of type 'const C::NestedC' must be initialized out of line}}
  static const int nci = vs; // expected-error {{in-class initializer for static data member is not a constant expression}}
  static const int vi = 0;
  static const volatile int cvi = 0; // expected-error {{static const volatile data member must be initialized out of line}}
};

namespace rdar8367341 {
  float foo(); // expected-note {{here}}

  struct A {
    static const float x = 5.0f; // expected-warning {{requires 'constexpr'}} expected-note {{add 'constexpr'}}
    static const float y = foo(); // expected-warning {{requires 'constexpr'}} expected-note {{add 'constexpr'}}
    static constexpr float x2 = 5.0f;
    static constexpr float y2 = foo(); // expected-error {{must be initialized by a constant expression}} expected-note {{non-constexpr function 'foo'}}
  };
}


namespace Foo {
  // Regression test -- forward declaration of Foo should not cause error about
  // nonstatic data member.
  class Foo;
  class Foo {
    int x;
    int y = x;
  };
}

// Instantiating another default member initializer while parsing one should
// not cause us to mess up the 'this' override.
template<typename> struct DefaultMemberTemplate { int n = 0; };
class DefaultMemberInitSelf {
  DefaultMemberTemplate<int> t = {};
  int *p = &t.n;
};

namespace composed_templates {
  // Regression test -- obtaining the type from composed templates should not
  // require out-of-line definition.
  template <typename T> struct Zero { static const typename T::type value = 0; };
  struct Integer { using type = int; };
  template struct Zero<Integer>;
}
