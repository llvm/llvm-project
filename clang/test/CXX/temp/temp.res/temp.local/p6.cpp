// RUN: %clang_cc1 -verify %s -fcxx-exceptions -std=c++1y

namespace N {}

template<typename T, // expected-note {{declared here}}
         typename T> struct X {}; // expected-error {{declaration of 'T' shadows template parameter}}

template<typename T> struct Y { // expected-note 17{{declared here}}
  template<typename T> struct A {}; // expected-error {{declaration of 'T' shadows template parameter}}

  struct B {
    template<typename> struct T {}; // FIXME: desired-error {{declaration of 'T' shadows template parameter}}
  };
  struct C {
    template<typename> void T(); // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct D {
    struct T {}; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct E {
    typedef int T; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct F {
    using T = int; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct G {
    int T; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct H {
    static int T; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct I {
    void T(); // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct J {
    enum T { e }; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct K {
    enum E { T }; // expected-error {{declaration of 'T' shadows template parameter}}
  };

  void a() {
    extern int T; // expected-error {{declaration of 'T' shadows template parameter}}
  }
  void b() {
    int T; // expected-error {{declaration of 'T' shadows template parameter}}
  }
  void c() {
    try {}
    catch (int T) {} // expected-error {{declaration of 'T' shadows template parameter}}
  }
  void d() {
    void T(); // expected-error {{declaration of 'T' shadows template parameter}}
  }
  void e() {
    namespace T = N; // expected-error {{declaration of 'T' shadows template parameter}}
  }

  // FIXME: These diagnostics are poorly worded. Lookup for the elaborated type
  // specifier finds the template parameter in this case, which is ill-formed
  // because it's not a struct.
  void f() {
    struct T *p; // expected-error {{declaration of 'T' shadows template parameter}}
  }
  friend struct T; // expected-error {{declaration of 'T' shadows template parameter}}
};

template<int T> struct Z { // expected-note 15{{declared here}}
  template<typename T> struct A {}; // expected-error {{declaration of 'T' shadows template parameter}}

  struct B {
    template<typename> struct T {}; // FIXME: desired-error {{declaration of 'T' shadows template parameter}}
  };
  struct C {
    template<typename> void T(); // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct D {
    struct T {}; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct E {
    typedef int T; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct F {
    using T = int; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct G {
    int T; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct H {
    static int T; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct I {
    void T(); // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct J {
    enum T { e }; // expected-error {{declaration of 'T' shadows template parameter}}
  };
  struct K {
    enum E { T }; // expected-error {{declaration of 'T' shadows template parameter}}
  };

  void a() {
    extern int T; // expected-error {{declaration of 'T' shadows template parameter}}
  }
  void b() {
    int T; // expected-error {{declaration of 'T' shadows template parameter}}
  }
  void c() {
    try {}
    catch (int T) {} // expected-error {{declaration of 'T' shadows template parameter}}
  }
  void d() {
    void T(); // expected-error {{declaration of 'T' shadows template parameter}}
  }
  void e() {
    namespace T = N; // expected-error {{declaration of 'T' shadows template parameter}}
  }

  // These cases are valid when 'T' is a non-type template parameter, as T
  // names an injected struct ::T, which doesn't shadow the template parameter.
  void f() {
    struct T *p;
  }
  friend struct T;
};

template<typename T> // expected-note {{declared here}}
void f(int T) {} // expected-error {{declaration of 'T' shadows template parameter}}

// FIXME: These are ill-formed: a template-parameter shall not have the same name as the template name.
namespace A {
  template<typename T> struct T {};
}
namespace B {
  template<typename T> void T() {}
}
namespace C {
  template<typename T> int T;
}
