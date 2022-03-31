// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx2b -std=c++2b %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx20 -std=c++20 %s

//cxx2b-no-diagnostics

struct S {
  constexpr int operator[](int i) {
    return i;
  }
  constexpr int operator[](int a, int b) { // cxx20-error {{overloaded 'operator[]' cannot have more than one parameter before C++2b}}
    return a + b;
  }
  constexpr int operator[]() { // cxx20-error {{overloaded 'operator[]' cannot have no parameter before C++2b}}
    return 42;
  }
};

struct Defaults {
  constexpr int operator[](int i = 0) { // cxx20-error {{overloaded 'operator[]' cannot have a defaulted parameter before C++2b}}
    return 0;
  }
  constexpr int operator[](int a, int b, int c = 0) { // cxx20-error {{overloaded 'operator[]' cannot have a defaulted parameter before C++2b}}\
                                                         // cxx20-error {{cannot have more than one parameter before C++2b}}
    return 0;
  }
};

template <typename... T>
struct T1 {
  constexpr auto operator[](T &&...arg); // cxx20-error {{overloaded 'operator[]' cannot have no parameter before C++2b}} \
                                           // cxx20-error {{overloaded 'operator[]' cannot have more than one parameter before C++2b}}
};

T1<> t10;         // cxx20-note {{requested here}}
T1<int, int> t12; // cxx20-note {{requested here}}
T1<int> t11;

struct Variadic {
  constexpr int operator[](auto &&...arg) { return 0; }
};

void f() {
  S s;
  (void)s[0];
  (void)s[1, 2]; // cxx20-warning {{left operand of comma operator has no effect}}\
                   // cxx20-warning {{top-level comma expression in array subscript is deprecated in C++20 and unsupported in C++2b}}
  (void)S{}[];   // cxx20-error {{expected expression}}

  (void)Defaults{}[1];
  (void)Defaults{}[];     // cxx20-error {{expected expression}}
  (void)Defaults{}[1, 2]; // cxx20-warning {{left operand of comma operator has no effect}}\
                            // cxx20-warning {{top-level comma expression in array subscript is deprecated in C++20 and unsupported in C++2b}}

  Variadic{}[]; // cxx20-error {{expected expression}}
  Variadic{}[1];
  Variadic{}[1, 2]; // cxx20-warning {{left operand of comma operator has no effect}}\
                       // cxx20-warning {{top-level comma expression in array subscript is deprecated in C++20 and unsupported in C++2b}}
}
