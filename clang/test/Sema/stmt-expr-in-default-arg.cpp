// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++20

void foo() {
  void fn(int i, int = ({ 1; })); // expected-error {{default argument may not use a GNU statement expression}}

  auto a = [](int = ({ 1; })) {}; // expected-error {{default argument may not use a GNU statement expression}}

  auto b = []<int = ({ 1; })>(){}; // expected-error {{default non-type template argument may not use a GNU statement expression}}

  void fn(int i, int j = ({{}, {}, {,}}), int k = ""); // expected-error {{default argument may not use a GNU statement expression}} expected-error {{cannot initialize a parameter of type 'int' with an lvalue of type 'const char[1]'}} expected-note {{passing argument to parameter 'k' here}}
}

template <int foo = ({ 1; })> // expected-error {{default non-type template argument may not use a GNU statement expression}}
void f() {}

template <int bar = ({ 1; })> // expected-error {{default non-type template argument may not use a GNU statement expression}}
class S {};

template <typename Callable>
int bar(Callable &&Call) {
  return Call();
}

int baz() {
  auto l = [](int a = ({ int x = 12; x; })) { // expected-error {{default argument may not use a GNU statement expression}}
    return 1;
  };
  return bar(l);
}
