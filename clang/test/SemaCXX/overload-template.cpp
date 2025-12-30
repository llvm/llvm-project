// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -verify -fsyntax-only %s

enum copy_traits { movable = 1 };

template <int>
struct optional_ctor_base {};
template <typename T>
struct ctor_copy_traits {
  // this would produce a c++98-compat warning, which would erroneously get the
  // no-matching-function-call error's notes attached to it (or suppress those
  // notes if this diagnostic was suppressed, as it is in this case)
  static constexpr int traits = copy_traits::movable;
};
template <typename T>
struct optional : optional_ctor_base<ctor_copy_traits<T>::traits> {
  template <typename U>
  constexpr optional(U&& v);
};
struct A {};
struct XA {
  XA(const A&);
};
struct B {};
struct XB {
  XB(const B&);
  XB(const optional<B>&);
};
struct YB : XB {
  using XB::XB;
};
void InsertRow(const XA&, const YB&); // expected-note {{candidate function not viable: no known conversion from 'int' to 'const XA' for 1st argument}}
void ReproducesBugSimply() {
  InsertRow(3, B{}); // expected-error {{no matching function for call to 'InsertRow'}}
}

#if __cplusplus >= 202302L
namespace overloadCheck{
  template<typename T>
  concept AlwaysTrue = true;

  struct S {
    int f(AlwaysTrue auto) { return 1; }
    void f(this S&&, auto) {}

    void g(auto) {}
    int g(this S&&,AlwaysTrue auto) {return 1;}

    int h(AlwaysTrue auto) { return 1; } //expected-note {{previous definition is here}}
    int h(this S&&,AlwaysTrue auto) { // expected-error {{class member cannot be redeclared}} 
      return 1;
    }
  };

  int main() {
    int x = S{}.f(0);
    int y = S{}.g(0);
  }
}
#endif

namespace GH93076 {
template <typename ...a> int b(a..., int); // expected-note-re 3 {{candidate function template not viable: no known conversion from 'int ()' to 'int' for {{.*}} argument}}
int d() {
  (void)b<int, int>(0, 0, d); // expected-error {{no matching function for call to 'b'}}
  (void)b<int, int>(0, d, 0); // expected-error {{no matching function for call to 'b'}}
  (void)b<int, int>(d, 0, 0); // expected-error {{no matching function for call to 'b'}}
  return 0;
 }
}
