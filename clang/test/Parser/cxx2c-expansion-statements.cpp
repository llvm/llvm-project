// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify
namespace std {
template <typename T>
struct initializer_list {
  const T* a;
  const T* b;
  initializer_list(T*, T*) {}
};
}

void bad() {
  template for; // expected-error {{expected '(' after 'for'}}
  template for (); // expected-error {{expected expression}} expected-error {{expected ';' in 'for' statement specifier}} expected-error {{expansion statement must be range-based}}
  template for (;); // expected-error {{expected ';' in 'for' statement specifier}} expected-error {{expansion statement must be range-based}}
  template for (;;); // expected-error {{expansion statement must be range-based}}
  template for (int x;;); // expected-error {{expansion statement must be range-based}}
  template for (x : {1}); // expected-error {{expansion statement requires type for expansion variable}}
  template for (: {1}); // expected-error {{expected expression}} expected-error {{expected ';' in 'for' statement specifier}} expected-error {{expansion statement must be range-based}}
  template for (auto y : {1})]; // expected-error {{expected expression}}
  template for (auto y : {1}; // expected-error {{expected ')'}} expected-note {{to match this '('}}

  template for (extern auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'extern'}}
  template for (extern static auto y : {1, 2}); // expected-error {{cannot combine with previous 'extern' declaration specifier}} expected-error {{expansion variable 'y' may not be declared 'extern'}}
  template for (static auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'static'}}
  template for (thread_local auto y : {1, 2}); // expected-error {{'thread_local' variables must have global storage}}
  template for (static thread_local auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'thread_local'}}
  template for (__thread auto y : {1, 2}); // expected-error {{'__thread' variables must have global storage}}
  template for (static __thread auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'static'}}
  template for (constinit auto y : {1, 2}); // expected-error {{local variable cannot be declared 'constinit'}}
  template for (consteval auto y : {1, 2});  // expected-error {{consteval can only be used in function declarations}}
  template for (int x; extern auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'extern'}}
  template for (int x; extern static auto y : {1, 2}); // expected-error {{cannot combine with previous 'extern' declaration specifier}} expected-error {{expansion variable 'y' may not be declared 'extern'}}
  template for (int x; static auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'static'}}
  template for (int x; thread_local auto y : {1, 2}); // expected-error {{'thread_local' variables must have global storage}}
  template for (int x; static thread_local auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'thread_local'}}
  template for (int x; __thread auto y : {1, 2}); // expected-error {{'__thread' variables must have global storage}}
  template for (int x; static __thread auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'static'}}
  template for (int x; constinit auto y : {1, 2}); // expected-error {{local variable cannot be declared 'constinit'}}
  template for (int x; consteval auto y : {1, 2});  // expected-error {{consteval can only be used in function declarations}}
  template for (auto y : {abc, -+, }); // expected-error {{use of undeclared identifier 'abc'}} expected-error 2 {{expected expression}}
  template while (true) {} // expected-error {{expected '<' after 'template'}}
  template for (auto y : {{1}, {2}, {3, {4}}, {{{5}}}});
}

void good() {
  template for (auto y : {});
  template for (auto y : {1, 2});
  template for (int x; auto y : {1, 2});
  template for (int x; int y : {1, 2});
  template for (int x; constexpr auto y : {1, 2});
  template for (int x; constexpr int y : {1, 2});
  template for (constexpr int a : {1, 2}) {
    template for (constexpr int b : {1, 2}) {
      template for (constexpr int c : {1, 2});
    }
  }
}
