// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-show-option -verify %s
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-show-option -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-show-option -verify -std=c++11 %s

// C++98 [basic.lookup.classref]p1:
//   In a class member access expression (5.2.5), if the . or -> token is
//   immediately followed by an identifier followed by a <, the identifier must
//   be looked up to determine whether the < is the beginning of a template
//   argument list (14.2) or a less-than operator. The identifier is first
//   looked up in the class of the object expression. If the identifier is not
//   found, it is then looked up in the context of the entire postfix-expression
//   and shall name a class or function template. If the lookup in the class of
//   the object expression finds a template, the name is also looked up in the
//   context of the entire postfix-expression and
//    -- if the name is not found, the name found in the class of the object
//       expression is used, otherwise
//    -- if the name is found in the context of the entire postfix-expression
//       and does not name a class template, the name found in the class of the
//       object expression is used, otherwise
//    -- if the name found is a class template, it must refer to the same
//       entity as the one found in the class of the object expression,
//       otherwise the program is ill-formed.

// From PR 7247
template<typename T>
struct set{};
#if __cplusplus <= 199711L
// expected-note@-2 {{lookup from the current scope refers here}}
#endif
struct Value {
  template<typename T>
  void set(T value) {}
#if __cplusplus <= 199711L
  // expected-note@-2 {{lookup in the object type 'Value' refers here}}
#endif

  void resolves_to_same() {
    Value v;
    v.set<double>(3.2);
  }
};
void resolves_to_different() {
  {
    Value v;
    // The fact that the next line is a warning rather than an error is an
    // extension.
    v.set<double>(3.2);
#if __cplusplus <= 199711L
    // expected-warning@-2 {{lookup of 'set' in member access expression is ambiguous; using member of 'Value'}}
#endif
  }
  {
    int set;  // Non-template.
    Value v;
    v.set<double>(3.2);
  }
}

namespace rdar9915664 {
  struct A {
    template<typename T> void a();
  };

  struct B : A { };

  struct C : A { };

  struct D : B, C {
    A &getA() { return static_cast<B&>(*this); }

    void test_a() {
      getA().a<int>();
    }
  };
}

namespace PR11856 {
  template<typename T> T end(T);

  template <typename T>
  void Foo() {
    T it1;
    if (it1->end < it1->end) {
    }
  }

  template<typename T> T *end(T*);

  class X { };
  template <typename T>
  void Foo2() {
    T it1;
    if (it1->end < it1->end) {
    }

    X *x;
    if (x->end < 7) {  // expected-error{{no member named 'end' in 'PR11856::X'}}
    }
  }
}
