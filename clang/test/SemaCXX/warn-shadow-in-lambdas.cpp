// RUN: %clang_cc1 -std=c++14 -verify=expected,cxx14 -fsyntax-only -Wshadow -D AVOID %s
// RUN: %clang_cc1 -std=c++14 -verify=expected,cxx14 -fsyntax-only -Wshadow -Wshadow-uncaptured-local %s
// RUN: %clang_cc1 -std=c++14 -verify=expected,cxx14 -fsyntax-only -Wshadow-all %s
// RUN: %clang_cc1 -std=c++17 -verify -fsyntax-only -Wshadow-all %s
// RUN: %clang_cc1 -std=c++20 -verify -fsyntax-only -Wshadow-all %s

void foo(int param) { // expected-note 1+ {{previous declaration is here}}
  int var = 0; // expected-note 1+ {{previous declaration is here}}

  // Avoid warnings for variables that aren't implicitly captured.
  {
#ifdef AVOID
    auto f1 = [=] { int var = 1; };  // no warning
    auto f2 = [&] { int var = 2; };  // no warning
    auto f3 = [=] (int param) { ; }; // no warning
    auto f4 = [&] (int param) { ; }; // no warning
    auto f5 = [=] { static int var = 1; };  // no warning
    auto f6 = [&] { static int var = 2; };  // no warning
#else
    auto f1 = [=] { int var = 1; };  // expected-warning {{declaration shadows a local variable}}
    auto f2 = [&] { int var = 2; };  // expected-warning {{declaration shadows a local variable}}
    auto f3 = [=] (int param) { ; }; // expected-warning {{declaration shadows a local variable}}
    auto f4 = [&] (int param) { ; }; // expected-warning {{declaration shadows a local variable}}
    auto f5 = [=] { static int var = 1; };  // expected-warning {{declaration shadows a local variable}}
    auto f6 = [&] { static int var = 2; };  // expected-warning {{declaration shadows a local variable}}
#endif
  }

  // Warn for variables that are implicitly captured.
  {
    auto f1 = [=] () {
      {
        int var = 1; // expected-warning {{declaration shadows a local variable}}
      }
      int x = var; // expected-note {{variable 'var' is captured here}}
    };
    auto f2 = [&]
#ifdef AVOID
      (int param) {
#else
      (int param) { // expected-warning {{declaration shadows a local variable}}
#endif
      int x = var; // expected-note {{variable 'var' is captured here}}
      int var = param; // expected-warning {{declaration shadows a local variable}}
    };
  }

  // Warn for variables that are explicitly captured when a lambda has a default
  // capture specifier.
  {
    auto f1 = [=, &var] () { // expected-note {{variable 'var' is captured here}}
      int x = param; // expected-note {{variable 'param' is captured here}}
      int var = 0; // expected-warning {{declaration shadows a local variable}}
      int param = 0; // expected-warning {{declaration shadows a local variable}}
    };
  }

  // Warn normally inside of lambdas.
  auto l1 = [] { // expected-note {{previous declaration is here}}
      int x = 1; // expected-note {{previous declaration is here}}
      { int x = 2; } // expected-warning {{declaration shadows a local variable}}
  };
  auto l2 = [] (int x) { // expected-note {{previous declaration is here}}
    { int x = 1; } // expected-warning {{declaration shadows a local variable}}
  };

  // Avoid warnings for variables that aren't explicitly captured.
  {
#ifdef AVOID
    auto f1 = [] { int var = 1; }; // no warning
    auto f2 = [] (int param) { ; }; // no warning
    auto f3 = [param] () { int var = 1; }; // no warning
    auto f4 = [var] (int param) { ; }; // no warning
    auto f5 = [param] () { static int var = 1; }; // no warning
    auto f6 = [] { static int var = 1; }; // no warning
#else
    auto f1 = [] { int var = 1; }; // expected-warning {{declaration shadows a local variable}}
    auto f2 = [] (int param) { ; }; // expected-warning {{declaration shadows a local variable}}
    auto f3 = [param] () { int var = 1; }; // expected-warning {{declaration shadows a local variable}}
    auto f4 = [var] (int param) { ; }; // expected-warning {{declaration shadows a local variable}}
    auto f5 = [param] () { static int var = 1; }; // expected-warning {{declaration shadows a local variable}}
    auto f6 = [] { static int var = 1; }; // expected-warning {{declaration shadows a local variable}}
#endif
  };

  // Warn for variables that are explicitly captured.
  {
    auto f1 = [var] () { // expected-note {{variable 'var' is explicitly captured here}}
      int var = 1; // expected-warning {{declaration shadows a local variable}}
    };
    auto f2 = [param]   // expected-note {{variable 'param' is explicitly captured here}}
     (int param) { ; }; // expected-error {{a lambda parameter cannot shadow an explicitly captured entity}}
  }

  // Warn for variables defined in the capture list.
  auto l3 = [z = var] { // expected-note {{previous declaration is here}}
#ifdef AVOID
    int var = 1; // no warning
#else
    int var = 1; // expected-warning {{declaration shadows a local variable}}
#endif
    { int z = 1; } // expected-warning {{declaration shadows a local variable}}
  };
#ifdef AVOID
  auto l4 = [var = param] (int param) { ; }; // no warning
#else
  auto l4 = [var = param](int param) { ; }; // expected-warning 2{{declaration shadows a local variable}}
#endif

  // Make sure that inner lambdas work as well.
  auto l5 = [var, l1] { // expected-note {{variable 'l1' is explicitly captured here}}
    auto l1 = [] { // expected-warning {{declaration shadows a local variable}}
#ifdef AVOID
      int var = 1; // no warning
#else
      int var = 1; // expected-warning {{declaration shadows a local variable}}
#endif
    };
#ifdef AVOID
    auto f1 = [] { int var = 1; }; // no warning
    auto f2 = [=] { int var = 1; }; // no warning
#else
    auto f1 = [] { int var = 1; }; // expected-warning {{declaration shadows a local variable}}
    auto f2 = [=] { int var = 1; }; // expected-warning {{declaration shadows a local variable}}
#endif
    auto f3 = [var] // expected-note {{variable 'var' is explicitly captured here}}
      { int var = 1; }; // expected-warning {{declaration shadows a local variable}}
    auto f4 = [&] {
      int x = var; // expected-note {{variable 'var' is captured here}}
      int var = 2; // expected-warning {{declaration shadows a local variable}}
    };
  };
  auto l6 = [&] {
    auto f1 = [param] { // expected-note {{variable 'param' is explicitly captured here}}
      int param = 0; // expected-warning {{declaration shadows a local variable}}
    };
  };
  auto l7 = [&] {
    auto f1 = [param] { // expected-note {{variable 'param' is explicitly captured here}}
      static int param = 0; // expected-warning {{declaration shadows a local variable}}
    };
  };

  // Generic lambda arguments should work.
#ifdef AVOID
  auto g1 = [](auto param) { ; }; // no warning
  auto g2 = [=](auto param) { ; }; // no warning
#else
  auto g1 = [](auto param) { ; }; // expected-warning {{declaration shadows a local variable}}
  auto g2 = [=](auto param) { ; }; // expected-warning {{declaration shadows a local variable}}
#endif
  auto g3 = [param] // expected-note {{variable 'param' is explicitly captured here}}
   (auto param) { ; }; // expected-error {{a lambda parameter cannot shadow an explicitly captured entity}}
}

void avoidWarningWhenRedefining() {
  int a = 1;
  auto l = [b = a] { // expected-note {{previous definition is here}}
    // Don't warn on redefinitions.
    int b = 0; // expected-error {{redefinition of 'b'}}
  };
}

namespace GH61105 {
void f() {
  int y = 0;
  int x = 0;
#if __cplusplus >= 202002L
  auto l1 = [y]<typename y>(y) { return 0; }; // expected-error {{declaration of 'y' shadows template parameter}} \
                                              // expected-note {{template parameter is declared here}}
  auto l2 = [=]<typename y>() { int a = y; return 0; }; // expected-error {{'y' does not refer to a value}} \
                                                        // expected-note {{declared here}}
  auto l3 = [&, y]<typename y, typename>(y) { int a = x; return 0; }; // expected-error {{declaration of 'y' shadows template parameter}} \
                                                                      // expected-note {{template parameter is declared here}}
  auto l4 = [x, y]<typename y, int x>() { return 0; }; // expected-error {{declaration of 'y' shadows template parameter}} \
                                                       // expected-error {{declaration of 'x' shadows template parameter}} \
                                                       // expected-note 2{{template parameter is declared here}}
  auto l5 = []<typename y>(y) { return 0; }; // No diagnostic
#endif
}
}

namespace GH71976 {
#ifdef AVOID
struct A {
  int b = 5;
  int foo() {
    return [b = b]() { return b; }(); // no -Wshadow diagnostic, init-capture does not shadow b due to not capturing this
  }
};

struct B {
  int a;
  void foo() {
    auto b = [a = this->a] {}; // no -Wshadow diagnostic, init-capture does not shadow a due to not capturing his
  }
};

struct C {
  int b = 5;
  int foo() {
    return [a = b]() {
      return [=, b = a]() { // no -Wshadow diagnostic, init-capture does not shadow b due to outer lambda
        return b;
      }();
    }();
  }
};

#else
struct A {
  int b = 5; // expected-note {{previous}}
  int foo() {
    return [b = b]() { return b; }(); // expected-warning {{declaration shadows a field}}
  }
};

struct B {
  int a; // expected-note {{previous}}
  void foo() {
    auto b = [a = this->a] {}; // expected-warning {{declaration shadows a field}}
  }
};

struct C {
  int b = 5; // expected-note {{previous}}
  int foo() {
    return [a = b]() {
      return [=, b = a]() { // expected-warning {{declaration shadows a field}}
        return b;
      }();
    }();
  }
};

struct D {
  int b = 5; // expected-note {{previous}}
  int foo() {
    return [b = b, this]() { return b; }(); // expected-warning {{declaration shadows a field}}
  }
};

struct E {
  int b = 5;
  int foo() {
    return [a = b]() { // expected-note {{previous}}
      return [=, a = a]() { // expected-warning {{shadows a local}}
        return a;
      }();
    }();
  }
};

#endif

struct S {
    int a ;
};

int foo() {
#ifdef AVOID
  auto [a] = S{0}; // cxx14-warning {{decomposition declarations are a C++17 extension}}
  [a = a] () { // No warning with basic -Wshadow due to uncaptured-local classification
  }();
#else
  auto [a] = S{0}; // cxx14-warning {{decomposition declarations are a C++17 extension}} expected-note {{previous declaration is here}}
  [a = a] () { // expected-warning {{declaration shadows a structured binding}}
  }();
#endif
}

}
