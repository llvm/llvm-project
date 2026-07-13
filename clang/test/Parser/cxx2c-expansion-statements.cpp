// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify=expected,brace -Wexpansion-stmt-braced-body
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
  template for (); // expected-error {{expected expression}} expected-error {{expected ';' in 'for' statement specifier}} expected-error {{expansion statement must use the syntax of a range-based for loop}}
  template for (;); // expected-error {{expected ';' in 'for' statement specifier}} expected-error {{expansion statement must use the syntax of a range-based for loop}}
  template for (;;); // expected-error {{expansion statement must use the syntax of a range-based for loop}}
  template for (int x;;); // expected-error {{expansion statement must use the syntax of a range-based for loop}}
  template for (x : {1});
  // expected-error@-1 {{expansion statement requires type for expansion variable}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (: {1}); // expected-error {{expected expression}} expected-error {{expected ';' in 'for' statement specifier}} expected-error {{expansion statement must use the syntax of a range-based for loop}}
  template for (auto y : {1})]; // expected-error {{expected expression}}
  template for (auto y : {1};
  // expected-error@-1 {{expected ')'}}
  // expected-note@-2 {{to match this '('}}
  // brace-warning@-3 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (extern auto y : {1, 2});
  // expected-error@-1 {{expansion variable 'y' may not be declared 'extern'}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (register auto y : {1, 2});
  // expected-error@-1 {{expansion variable 'y' may not be declared 'register'}}
  // expected-error@-2 {{ISO C++17 does not allow 'register' storage class specifier}}
  // brace-warning@-3 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (__private_extern__ auto y : {1, 2});
  // expected-error@-1 {{expansion variable 'y' may not be declared 'extern'}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (extern static auto y : {1, 2});
  // expected-error@-1 {{cannot combine with previous 'extern' declaration specifier}}
  // expected-error@-2 {{expansion variable 'y' may not be declared 'extern'}}
  // brace-warning@-3 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (static auto y : {1, 2});
  // expected-error@-1 {{expansion variable 'y' may not be declared 'static'}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (thread_local auto y : {1, 2});
  // expected-error@-1 {{'thread_local' variables must have global storage}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (static thread_local auto y : {1, 2});
  // expected-error@-1 {{expansion variable 'y' may not be declared 'thread_local'}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (__thread auto y : {1, 2});
  // expected-error@-1 {{'__thread' variables must have global storage}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (static __thread auto y : {1, 2});
  // expected-error@-1 {{expansion variable 'y' may not be declared 'static'}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (constinit auto y : {1, 2});
  // expected-error@-1 {{local variable cannot be declared 'constinit'}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (consteval auto y : {1, 2});
  // expected-error@-1 {{consteval can only be used in function declarations}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; extern auto y : {1, 2});
  // expected-error@-1 {{expansion variable 'y' may not be declared 'extern'}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; extern static auto y : {1, 2});
  // expected-error@-1 {{cannot combine with previous 'extern' declaration specifier}}
  // expected-error@-2 {{expansion variable 'y' may not be declared 'extern'}}
  // brace-warning@-3 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; static auto y : {1, 2});
  // expected-error@-1 {{expansion variable 'y' may not be declared 'static'}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; thread_local auto y : {1, 2});
  // expected-error@-1 {{'thread_local' variables must have global storage}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; static thread_local auto y : {1, 2});
  // expected-error@-1 {{expansion variable 'y' may not be declared 'thread_local'}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; __thread auto y : {1, 2});
  // expected-error@-1 {{'__thread' variables must have global storage}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; static __thread auto y : {1, 2});
  // expected-error@-1 {{expansion variable 'y' may not be declared 'static'}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; constinit auto y : {1, 2});
  // expected-error@-1 {{local variable cannot be declared 'constinit'}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; consteval auto y : {1, 2});
  // expected-error@-1 {{consteval can only be used in function declarations}}
  // brace-warning@-2 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (auto y : {abc, -+, });
  // expected-error@-1 {{use of undeclared identifier 'abc'}}
  // expected-error@-2 {{expected expression}}
  // brace-warning@-3 {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (3 : "error") // expected-error {{expansion statement declaration must declare a variable}} \
                                expected-error {{expansion statement must use the syntax of a range-based for loop}}
    ;
  template while (true) {} // expected-error {{expected '<' after 'template'}}
  ; // Semicolon for synchronisation; otherwise, the parser skips over next statement...
  template do {} while (true); // expected-error {{expected '<' after 'template'}}
  for template (int x : {}) {} // expected-error {{'for template' is invalid; use 'template for' instead}}
  template for (int x : {1})
    [ // brace-warning {{ISO C++ requires a compound statement to be the body of expansion statement}}
      []] {}
}

void good() {
  template for (auto y : {}); // brace-warning {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (auto y : {1, 2}); // brace-warning {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; auto y : {1, 2}); // brace-warning {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; int y : {1, 2}); // brace-warning {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; constexpr auto y : {1, 2}); // brace-warning {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (int x; constexpr int y : {1, 2}); // brace-warning {{ISO C++ requires a compound statement to be the body of expansion statement}}
  template for (constexpr int a : {1, 2}) {
    template for (constexpr int b : {1, 2}) {
      template for (constexpr int c : {1, 2}); // brace-warning {{ISO C++ requires a compound statement to be the body of expansion statement}}
    }
  }
}

void trailing_comma() {
  template for (int x : {1, 2,}) {}
  template for (int x : {,}) {} // expected-error {{expected expression}}
}
