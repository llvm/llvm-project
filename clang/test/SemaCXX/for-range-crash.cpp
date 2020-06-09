// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s

// Ensure that we don't crash if errors are suppressed by an error limit.
// RUN: not %clang_cc1 -fsyntax-only -std=c++17 -ferror-limit=1 %s

error e; // expected-error {{unknown type name}}

template <typename>
class Bar {
  Bar<int> *variables_to_modify;
  foo() { // expected-error {{C++ requires a type specifier for all declarations}}
    for (auto *c : *variables_to_modify)
      delete c;
  }
};

void foo() {
  int a;
  struct X; // expected-note {{forward declaration}}
  for (X x // expected-error {{incomplete type}}
      : a) { // expected-error {{range expression of type 'int'}}
    constexpr int n = sizeof(x);
  }

  struct S { int x, y; };
  for (S [x, y] // expected-error {{must be 'auto'}}
      : a) { // expected-error {{range expression}}
    typename decltype(x)::a b;
  }
}
