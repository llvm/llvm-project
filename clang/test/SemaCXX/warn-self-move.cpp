// RUN: %clang_cc1 -fsyntax-only -Wself-move -std=c++11 -verify %s

// definitions for std::move
namespace std {
inline namespace foo {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T> typename remove_reference<T>::type &&move(T &&t);
}
}

void int_test() {
  int x = 5;
  x = std::move(x);  // expected-warning{{explicitly moving}}
  (x) = std::move(x);  // expected-warning{{explicitly moving}}

  x = static_cast<int&&>(x);  // expected-warning{{explicitly moving}}
  (x) = static_cast<int&&>(x);  // expected-warning{{explicitly moving}}

  using std::move;
  x = move(x); // expected-warning{{explicitly moving}} \
                   expected-warning {{unqualified call to 'std::move}}
}

int global;
void global_int_test() {
  global = std::move(global);  // expected-warning{{explicitly moving}}
  (global) = std::move(global);  // expected-warning{{explicitly moving}}

  global = static_cast<int&&>(global);  // expected-warning{{explicitly moving}}
  (global) = static_cast<int&&>(global);  // expected-warning{{explicitly moving}}

  using std::move;
  global = move(global); // expected-warning{{explicitly moving}} \
                             expected-warning {{unqualified call to 'std::move}}
}

class field_test {
  int x;
  field_test(field_test&& other) {
    x = std::move(x);  // expected-warning{{explicitly moving}}
    x = static_cast<int&&>(x);  // expected-warning{{explicitly moving}}
    x = std::move(other.x);
    x = static_cast<int&&>(other.x);
    other.x = std::move(x);
    other.x = static_cast<int&&>(x);
    other.x = std::move(other.x);  // expected-warning{{explicitly moving}}
    other.x = static_cast<int&&>(other.x);  // expected-warning{{explicitly moving}}
  }
  void withSuggest(int x) {
    x = static_cast<int&&>(x); // expected-warning{{explicitly moving variable of type 'int' to itself; did you mean to move to member 'x'?}}
    x = std::move(x); // expected-warning{{explicitly moving variable of type 'int' to itself; did you mean to move to member 'x'?}}
  }
};

struct A {};
struct B { A a; };
struct C { C() {}; ~C() {} };
void struct_test() {
  A a;
  a = std::move(a);  // expected-warning{{explicitly moving}}
  a = static_cast<A&&>(a);  // expected-warning{{explicitly moving}}

  B b;
  b = std::move(b);  // expected-warning{{explicitly moving}}
  b = static_cast<B&&>(b);  // expected-warning{{explicitly moving}}
  b.a = std::move(b.a);  // expected-warning{{explicitly moving}}
  b.a = static_cast<A&&>(b.a);  // expected-warning{{explicitly moving}}

  C c;
  c = std::move(c);  // expected-warning{{explicitly moving}}
  c = static_cast<C&&>(c);  // expected-warning{{explicitly moving}}
}
