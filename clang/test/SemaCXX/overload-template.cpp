// RUN: %clang_cc1 -fsyntax-only -verify %s

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

