// RUN: %clang_cc1 -std=c++26 -verify=expected,old-interp %s
// RUN: %clang_cc1 -std=c++26 -verify -fexperimental-new-constant-interpreter %s

constexpr void foo(int) {}
void bar(int) {} // expected-note 7 {{declared here}}

consteval { foo(1); }
consteval { // expected-error {{could not evaluate consteval block}}
  bar(1); // expected-note {{non-constexpr function 'bar' cannot be used in a constant expression}}
}

struct X {
  consteval { foo(2); }
  consteval { // expected-error {{could not evaluate consteval block}}
    bar(2); // expected-note {{non-constexpr function 'bar' cannot be used in a constant expression}}
  }

  void f() {
    consteval { foo(3); }
    consteval { // expected-error {{could not evaluate consteval block}}
      bar(3); // expected-note {{non-constexpr function 'bar' cannot be used in a constant expression}}
    }
  }
};

void f1() {
  consteval { foo(4); }
  consteval { // expected-error {{could not evaluate consteval block}}
    bar(4); // expected-note {{non-constexpr function 'bar' cannot be used in a constant expression}}
  }
}

union U {
  consteval { foo(5); }
  consteval { // expected-error {{could not evaluate consteval block}}
    bar(5); // expected-note {{non-constexpr function 'bar' cannot be used in a constant expression}}
  }
};

consteval {
  consteval { // expected-error {{could not evaluate consteval block}}
    foo(6);
    bar(6); // expected-note {{non-constexpr function 'bar' cannot be used in a constant expression}}
  }
}

template <typename T>
constexpr void t1() {
  consteval { foo(5); }
  consteval { // expected-error {{could not evaluate consteval block}}
    bar(5); // expected-note {{non-constexpr function 'bar' cannot be used in a constant expression}}
  }
}

template void t1<int>(); // expected-note {{in instantiation of function template specialization 't1<int>'}}

template <typename ...Ts>
void t2() {
  consteval { Ts t; } // expected-error {{consteval block contains unexpanded parameter pack 'Ts'}}
}

void f2() {
  int x1; // expected-note {{'x1' declared here}}
  consteval { // expected-note {{consteval block begins here}}
    decltype(x1) y = 4;
    (void)&x1; // expected-error {{cannot capture variable 'x1' in consteval block}}
  }

  static int x2;
  consteval {
    decltype(x2) y = 4;
    (void)&x2;
  }

  constexpr int x3 = 4; // expected-note {{'x3' declared here}}
  consteval { // expected-note {{consteval block begins here}}
    decltype(x3) y = 4;
    (void)&x3; // expected-error {{cannot capture variable 'x3' in consteval block}}
  }

  static constexpr int x4 = 4;
  consteval {
    decltype(x4) y = 4;
    (void)&x4;
  }
}

struct S {
  int x1;
  static int x2;
  static constexpr int x3 = 4;

  void f() {
    consteval { // old-interp-error {{could not evaluate consteval block}}
      decltype(x1) y = 4;
      (void) &x1; // expected-error {{cannot capture 'this' in consteval block}} old-interp-note {{implicit use of 'this' pointer}}
    }

    consteval {
      decltype(x2) y = 4;
      (void) &x2;
    }

    consteval {
      decltype(x3) y = 4;
      (void) &x3;
    }
  }
};

constexpr void g() {}

consteval { return; }
consteval { return g(); }
consteval { return static_cast<void>(4); }

consteval { return 4; } // expected-error {{consteval block should not return a value}}
consteval { return "foobar"; } // expected-error {{consteval block should not return a value}}

// FIXME: We should diagnose this; this is https://github.com/llvm/llvm-project/issues/188661.
consteval { return {}; }

// FIXME: The diagnostics for these are weird, probably due to the same issue.
consteval { return {1}; } // expected-error {{could not evaluate consteval block}} expected-note {{subexpression not valid in a constant expression}}
consteval { return {1, 2}; } // expected-error {{could not evaluate consteval block}} expected-note {{subexpression not valid in a constant expression}}
consteval { return {"foobar"}; } // expected-error {{could not evaluate consteval block}} expected-note {{subexpression not valid in a constant expression}}
