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
  template for (); // expected-error {{expected expression}} expected-error {{expected ';' in 'for' statement specifier}} expected-error {{expansion statement must be a range-based for loop}} expected-error {{TODO (expansion statements)}}
  template for (;); // expected-error {{expected ';' in 'for' statement specifier}} expected-error {{expansion statement must be a range-based for loop}} expected-error {{TODO (expansion statements)}}
  template for (;;); // expected-error {{expansion statement must be a range-based for loop}} expected-error {{TODO (expansion statements)}}
  template for (int x;;); // expected-error {{expansion statement must be a range-based for loop}} expected-error {{TODO (expansion statements)}}
  template for (x : {1}); // expected-error {{expansion statement requires type for expansion variable}} expected-error {{TODO (expansion statements)}}
  template for (: {1}); // expected-error {{expected expression}} expected-error {{expected ';' in 'for' statement specifier}} expected-error {{expansion statement must be a range-based for loop}} expected-error {{TODO (expansion statements)}}
  template for (auto y : {1})]; // expected-error {{expected expression}} expected-error {{TODO (expansion statements)}}
  template for (auto y : {1}; // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{TODO (expansion statements)}}
  template for (extern auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'extern'}} expected-error {{TODO (expansion statements)}}
  template for (extern static auto y : {1, 2}); // expected-error {{cannot combine with previous 'extern' declaration specifier}} expected-error {{expansion variable 'y' may not be declared 'extern'}} expected-error {{TODO (expansion statements)}}
  template for (static auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'static'}} expected-error {{TODO (expansion statements)}}
  template for (thread_local auto y : {1, 2}); // expected-error {{'thread_local' variables must have global storage}} expected-error {{TODO (expansion statements)}}
  template for (static thread_local auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'thread_local'}} expected-error {{TODO (expansion statements)}}
  template for (__thread auto y : {1, 2}); // expected-error {{'__thread' variables must have global storage}} expected-error {{TODO (expansion statements)}}
  template for (static __thread auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'static'}} expected-error {{TODO (expansion statements)}}
  template for (constinit auto y : {1, 2}); // expected-error {{local variable cannot be declared 'constinit'}} expected-error {{TODO (expansion statements)}}
  template for (consteval auto y : {1, 2});  // expected-error {{consteval can only be used in function declarations}} expected-error {{TODO (expansion statements)}}
  template for (int x; extern auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'extern'}} expected-error {{TODO (expansion statements)}}
  template for (int x; extern static auto y : {1, 2}); // expected-error {{cannot combine with previous 'extern' declaration specifier}} expected-error {{expansion variable 'y' may not be declared 'extern'}} expected-error {{TODO (expansion statements)}}
  template for (int x; static auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'static'}} expected-error {{TODO (expansion statements)}}
  template for (int x; thread_local auto y : {1, 2}); // expected-error {{'thread_local' variables must have global storage}} expected-error {{TODO (expansion statements)}}
  template for (int x; static thread_local auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'thread_local'}} expected-error {{TODO (expansion statements)}}
  template for (int x; __thread auto y : {1, 2}); // expected-error {{'__thread' variables must have global storage}} expected-error {{TODO (expansion statements)}}
  template for (int x; static __thread auto y : {1, 2}); // expected-error {{expansion variable 'y' may not be declared 'static'}} expected-error {{TODO (expansion statements)}}
  template for (int x; constinit auto y : {1, 2}); // expected-error {{local variable cannot be declared 'constinit'}} expected-error {{TODO (expansion statements)}}
  template for (int x; consteval auto y : {1, 2});  // expected-error {{consteval can only be used in function declarations}} expected-error {{TODO (expansion statements)}}
  template for (auto y : {abc, -+, }); // expected-error {{use of undeclared identifier 'abc'}} expected-error {{expected expression}} expected-error {{TODO (expansion statements)}}
  template for (3 : "error") // expected-error {{expansion statement declaration must declare a variable}} \
                                expected-error {{expansion statement must be a range-based for loop}} expected-error {{TODO (expansion statements)}}
    ;
  template while (true) {} // expected-error {{expected '<' after 'template'}}
}

void good() {
  template for (auto y : {}); // expected-error {{TODO (expansion statements)}}
  template for (auto y : {1, 2}); // expected-error {{TODO (expansion statements)}}
  template for (int x; auto y : {1, 2}); // expected-error {{TODO (expansion statements)}}
  template for (int x; int y : {1, 2}); // expected-error {{TODO (expansion statements)}}
  template for (int x; constexpr auto y : {1, 2}); // expected-error {{TODO (expansion statements)}}
  template for (int x; constexpr int y : {1, 2}); // expected-error {{TODO (expansion statements)}}
  template for (constexpr int a : {1, 2}) { // expected-error {{TODO (expansion statements)}}
    template for (constexpr int b : {1, 2}) { // expected-error {{TODO (expansion statements)}}
      template for (constexpr int c : {1, 2}); // expected-error {{TODO (expansion statements)}}
    }
  }
}

void trailing_comma() {
  template for (int x : {1, 2,}) {} // expected-error {{TODO (expansion statements)}}
  template for (int x : {,}) {} // expected-error {{expected expression}} expected-error {{TODO (expansion statements)}}
}
