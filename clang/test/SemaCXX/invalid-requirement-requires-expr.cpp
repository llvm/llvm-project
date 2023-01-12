// RUN: %clang_cc1 %s -I%S -std=c++2a -verify

// RequiresExpr contains invalid requirement. (Eg. Highly recurisive template).
template<int x>
struct A { static constexpr bool far(); };
class B {
    bool data_member;
    friend struct A<1>;
};

template<>
constexpr bool A<0>::far() { return true; }

template<int x>
constexpr bool A<x>::far() {
    return requires(B b) {
      b.data_member;
      requires A<x-1>::far(); //expected-note 3{{in instantiation}} // expected-note 6{{while}} expected-note {{contexts in backtrace; use -ftemplate-backtrace-limit=0 to see all}}
      // expected-error@-1{{recursive template instantiation exceeded maximum depth}}
    };
}
static_assert(A<1>::far());
static_assert(!A<10001>::far()); // expected-note {{in instantiation of member function}}
