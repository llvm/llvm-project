// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// expected-error@+2 {{non-templated function cannot have a requires clause}}
void f1(int a)
  requires true;
template <typename T>
auto f2(T a) -> bool
  requires true; // OK

// expected-error@+4 {{trailing return type must appear before trailing requires clause}}
template <typename T>
auto f3(T a)
  requires true
-> bool;

// expected-error@+2{{trailing requires clause can only be used when declaring a function}}
void (*pf)()
  requires true;

// expected-error@+1{{trailing requires clause can only be used when declaring a function}}
void g(int (*)() requires true);

// expected-error@+1{{expected expression}}
auto *p = new void(*)(char)
  requires true;

namespace GH61748 {
template<typename T>
struct S {
  // expected-error@+1 {{non-template friend declaration with a requires clause must be a definition}}
  friend void declared_friend() requires(sizeof(T) > 1);
  // OK, is a definition.
  friend void defined_friend() requires(sizeof(T) > 1){}
  // OK, is a member.
  void member() requires(sizeof(T) > 1);
};

template<typename T>
void ContainingFunction() {
  // expected-error@+1 {{non-templated function cannot have a requires clause}}
  void bad() requires(sizeof(T) > 1);
  // expected-error@+1 {{function definition is not allowed here}}
  void still_bad() requires(sizeof(T) > 1) {}

}

void NonTemplContainingFunction() {
  // expected-error@+1 {{non-templated function cannot have a requires clause}}
  (void)[]() requires (sizeof(int)>1){};
  // OK, a template.
  auto X = [](auto) requires (sizeof(int)>1){};
  // OK, a template.
  auto Y = []<typename T>(T t) requires (sizeof(int)>1){};

  X(1);
  Y(1);
}

template<typename T>
union U {
  void f() requires true;
};
}
