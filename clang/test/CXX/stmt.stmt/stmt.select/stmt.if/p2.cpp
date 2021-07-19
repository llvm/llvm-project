// RUN: %clang_cc1 -std=c++1z -verify %s
// RUN: %clang_cc1 -std=c++1z -verify %s -DUNDEFINED

#ifdef UNDEFINED
// "used but not defined" errors don't get produced if we have more interesting
// errors.
namespace std_example {
  template <typename T, typename... Rest> void g(T &&p, Rest &&... rs) {
    // use p
    if constexpr(sizeof...(rs) > 0)
      g(rs...);
  }
  void use_g() {
    g(1, 2, 3);
  }

  static int x(); // no definition of x required
  int f() {
    if constexpr (true)
      return 0;
    else if (x())
      return x();
    else
      return -x();
  }
}

namespace odr_use_in_selected_arm {
  static int x(); // expected-warning {{is not defined}}
  int f() {
    if constexpr (false)
      return 0;
    else if (x()) // expected-note {{here}}
      return x();
    else
      return -x();
  }
}
#else
namespace ccce {

  struct S {
  };
  void f() {
    if (5) {}
    if constexpr (5) {
    }
  }
  template<int N> void g() {
    if constexpr (N) {
    }
  }
  template void g<5>();
  void h() {
    if constexpr (4.3) { //expected-warning {{implicit conversion from 'double' to 'bool' changes value}}
    }
    constexpr void *p = nullptr;
    if constexpr (p) {
    }
  }

  void not_constant(int b, S s) { //  expected-note 2{{declared here}}
    if constexpr (bool(b)) {      // expected-error {{constexpr if condition is not a constant expression}} expected-note {{cannot be used in a constant expression}}
    }
    if constexpr (b) { // expected-error {{constexpr if condition is not a constant expression}} expected-note {{cannot be used in a constant expression}}
    }
    if constexpr (s) { // expected-error {{value of type 'ccce::S' is not contextually convertible to 'bool'}}
    }

    constexpr S constexprS;
    if constexpr (constexprS) { // expected-error {{value of type 'const ccce::S' is not contextually convertible to 'bool'}}
    }
  }
}

namespace generic_lambda {
  // Substituting for T produces a hard error here, even if substituting for
  // the type of x would remove the error.
  template<typename T> void f() {
    [](auto x) {
      if constexpr (sizeof(T) == 1 && sizeof(x) == 1)
        T::error(); // expected-error 2{{'::'}}
    } (0);
  }

  template<typename T> void g() {
    [](auto x) {
      if constexpr (sizeof(T) == 1)
        if constexpr (sizeof(x) == 1)
          T::error(); // expected-error {{'::'}}
    } (0);
  }

  void use() {
    f<int>(); // expected-note {{instantiation of}}
    f<char>(); // expected-note {{instantiation of}}
    g<int>(); // ok
    g<char>(); // expected-note {{instantiation of}}
  }
}

namespace potentially_discarded_branch_target {
  void in_switch(int n) {
    switch (n)
      case 4: if constexpr(sizeof(n) == 4) return;
    if constexpr(sizeof(n) == 4)
      switch (n) case 4: return;
    switch (n) {
      if constexpr (sizeof(n) == 4) // expected-note 2{{constexpr if}}
        case 4: return; // expected-error {{cannot jump}}
      else
        default: break; // expected-error {{cannot jump}}
    }
  }

  template<typename T>
  void in_switch_tmpl(int n) {
    switch (n) {
      if constexpr (sizeof(T) == 4) // expected-note 2{{constexpr if}}
        case 4: return; // expected-error {{cannot jump}}
      else
        default: break; // expected-error {{cannot jump}}
    }
  }

  void goto_scope(int n) {
    goto foo; // expected-error {{cannot jump}}
    if constexpr(sizeof(n) == 4) // expected-note {{constexpr if}}
      foo: return;
bar:
    if constexpr(sizeof(n) == 4)
      goto bar; // ok
  }

  template<typename T>
  void goto_scope(int n) {
    goto foo; // expected-error {{cannot jump}}
    if constexpr(sizeof(n) == 4) // expected-note {{constexpr if}}
      foo: return;
bar:
    if constexpr(sizeof(n) == 4)
      goto bar; // ok
  }

  void goto_redef(int n) {
a:  if constexpr(sizeof(n) == 4) // expected-error {{redefinition}} expected-note {{constexpr if}}
      a: goto a; // expected-note 2{{previous}}
    else
      a: goto a; // expected-error {{redefinition}} expected-error {{cannot jump}}
  }

  void evil_things() {
    goto evil_label; // expected-error {{cannot jump}}
    if constexpr (true || ({evil_label: false;})) {} // expected-note {{constexpr if}}

    if constexpr (true) // expected-note {{constexpr if}}
      goto surprise; // expected-error {{cannot jump}}
    else
      surprise: {}
  }
}
#endif
