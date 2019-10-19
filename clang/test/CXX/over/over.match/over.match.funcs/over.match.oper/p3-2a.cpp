// RUN: %clang_cc1 -std=c++2a -verify %s
// RUN: %clang_cc1 -std=c++2a -verify -Wall -DNO_ERRORS %s

#ifndef NO_ERRORS
namespace bullet3 {
  // the built-in candidates include all of the candidate operator fnuctions
  // [...] that, compared to the given operator

  // - do not have the same parameter-type-list as any non-member candidate

  enum E { e };

  // Suppress both builtin operator<=>(E, E) and operator<(E, E).
  void operator<=>(E, E); // expected-note {{while rewriting}}
  bool cmp = e < e; // expected-error {{invalid operands to binary expression ('void' and 'int')}}

  // None of the other bullets have anything to test here. In principle we
  // need to suppress both builtin operator@(A, B) and operator@(B, A) when we
  // see a user-declared reversible operator@(A, B), and we do, but that's
  // untestable because the only built-in reversible candidates are
  // operator<=>(E, E) and operator==(E, E) for E an enumeration type, and
  // those are both symmetric anyway.
}

namespace bullet4 {
  // The rewritten candidate set is determined as follows:

  template<int> struct X {};
  X<1> x1;
  X<2> x2;

  struct Y {
    int operator<=>(X<2>) = delete; // #1member
    bool operator==(X<2>) = delete; // #2member
  };
  Y y;

  // - For the relational operators, the rewritten candidates include all
  //   non-rewritten candidates for the expression x <=> y.
  int operator<=>(X<1>, X<2>) = delete; // #1

  // expected-note@#1 5{{candidate function has been explicitly deleted}}
  // expected-note@#1 5{{candidate function (with reversed parameter order) not viable: no known conversion from 'X<1>' to 'X<2>' for 1st argument}}
  bool lt = x1 < x2; // expected-error {{selected deleted operator '<=>'}}
  bool le = x1 <= x2; // expected-error {{selected deleted operator '<=>'}}
  bool gt = x1 > x2; // expected-error {{selected deleted operator '<=>'}}
  bool ge = x1 >= x2; // expected-error {{selected deleted operator '<=>'}}
  bool cmp = x1 <=> x2; // expected-error {{selected deleted operator '<=>'}}

  // expected-note@#1member 5{{candidate function has been explicitly deleted}}
  // expected-note@#1 5{{candidate function not viable: no known conversion from 'bullet4::Y' to 'X<1>' for 1st argument}}
  // expected-note@#1 5{{candidate function (with reversed parameter order) not viable: no known conversion from 'bullet4::Y' to 'X<2>' for 1st argument}}
  bool mem_lt = y < x2; // expected-error {{selected deleted operator '<=>'}}
  bool mem_le = y <= x2; // expected-error {{selected deleted operator '<=>'}}
  bool mem_gt = y > x2; // expected-error {{selected deleted operator '<=>'}}
  bool mem_ge = y >= x2; // expected-error {{selected deleted operator '<=>'}}
  bool mem_cmp = y <=> x2; // expected-error {{selected deleted operator '<=>'}}

  // - For the relational and three-way comparison operators, the rewritten
  //   candidates also include a synthesized candidate, with the order of the
  //   two parameters reversed, for each non-rewritten candidate for the
  //   expression y <=> x.

  // expected-note@#1 5{{candidate function (with reversed parameter order) has been explicitly deleted}}
  // expected-note@#1 5{{candidate function not viable: no known conversion from 'X<2>' to 'X<1>' for 1st argument}}
  bool rlt = x2 < x1; // expected-error {{selected deleted operator '<=>'}}
  bool rle = x2 <= x1; // expected-error {{selected deleted operator '<=>'}}
  bool rgt = x2 > x1; // expected-error {{selected deleted operator '<=>'}}
  bool rge = x2 >= x1; // expected-error {{selected deleted operator '<=>'}}
  bool rcmp = x2 <=> x1; // expected-error {{selected deleted operator '<=>'}}

  // expected-note@#1member 5{{candidate function (with reversed parameter order) has been explicitly deleted}}
  // expected-note@#1 5{{candidate function not viable: no known conversion from 'X<2>' to 'X<1>' for 1st argument}}
  // expected-note@#1 5{{candidate function (with reversed parameter order) not viable: no known conversion from 'bullet4::Y' to 'X<1>' for 2nd argument}}
  bool mem_rlt = x2 < y; // expected-error {{selected deleted operator '<=>'}}
  bool mem_rle = x2 <= y; // expected-error {{selected deleted operator '<=>'}}
  bool mem_rgt = x2 > y; // expected-error {{selected deleted operator '<=>'}}
  bool mem_rge = x2 >= y; // expected-error {{selected deleted operator '<=>'}}
  bool mem_rcmp = x2 <=> y; // expected-error {{selected deleted operator '<=>'}}

  // For the != operator, the rewritten candidates include all non-rewritten
  // candidates for the expression x == y
  int operator==(X<1>, X<2>) = delete; // #2

  // expected-note@#2 2{{candidate function has been explicitly deleted}}
  // expected-note@#2 2{{candidate function (with reversed parameter order) not viable: no known conversion from 'X<1>' to 'X<2>' for 1st argument}}
  bool eq = x1 == x2; // expected-error {{selected deleted operator '=='}}
  bool ne = x1 != x2; // expected-error {{selected deleted operator '=='}}

  // expected-note@#2member 2{{candidate function has been explicitly deleted}}
  // expected-note@#2 2{{candidate function not viable: no known conversion from 'bullet4::Y' to 'X<1>' for 1st argument}}
  // expected-note@#2 2{{candidate function (with reversed parameter order) not viable: no known conversion from 'bullet4::Y' to 'X<2>' for 1st argument}}
  bool mem_eq = y == x2; // expected-error {{selected deleted operator '=='}}
  bool mem_ne = y != x2; // expected-error {{selected deleted operator '=='}}

  // For the equality operators, the rewritten candidates also include a
  // synthesized candidate, with the order of the two parameters reversed, for
  // each non-rewritten candidate for the expression y == x

  // expected-note@#2 2{{candidate function (with reversed parameter order) has been explicitly deleted}}
  // expected-note@#2 2{{candidate function not viable: no known conversion from 'X<2>' to 'X<1>' for 1st argument}}
  bool req = x2 == x1; // expected-error {{selected deleted operator '=='}}
  bool rne = x2 != x1; // expected-error {{selected deleted operator '=='}}

  // expected-note@#2member 2{{candidate function (with reversed parameter order) has been explicitly deleted}}
  // expected-note@#2 2{{candidate function not viable: no known conversion from 'X<2>' to 'X<1>' for 1st argument}}
  // expected-note@#2 2{{candidate function (with reversed parameter order) not viable: no known conversion from 'bullet4::Y' to 'X<1>' for 2nd argument}}
  bool mem_req = x2 == y; // expected-error {{selected deleted operator '=='}}
  bool mem_rne = x2 != y; // expected-error {{selected deleted operator '=='}}

  // For all other operators, the rewritten candidate set is empty.
  X<3> operator+(X<1>, X<2>) = delete; // expected-note {{no known conversion from 'X<2>' to 'X<1>'}}
  X<3> reversed_add = x2 + x1; // expected-error {{invalid operands}}
}

// Various C++17 cases that are known to be broken by the C++20 rules.
namespace problem_cases {
  // We can have an ambiguity between an operator and its reversed form. This
  // wasn't intended by the original "consistent comparison" proposal, and we
  // allow it as extension, picking the non-reversed form.
  struct A {
    bool operator==(const A&); // expected-note {{ambiguity is between a regular call to this operator and a call with the argument order reversed}}
  };
  bool cmp_non_const = A() == A(); // expected-warning {{ambiguous}}

  struct B {
    virtual bool operator==(const B&) const;
  };
  struct D : B {
    bool operator==(const B&) const override; // expected-note {{operator}}
  };
  bool cmp_base_derived = D() == D(); // expected-warning {{ambiguous}}

  template<typename T> struct CRTPBase {
    bool operator==(const T&) const; // expected-note {{operator}}
  };
  struct CRTP : CRTPBase<CRTP> {};
  bool cmp_crtp = CRTP() == CRTP(); // expected-warning {{ambiguous}}

  // We can select a non-rewriteable operator== for a != comparison, when there
  // was a viable operator!= candidate we could have used instead.
  //
  // Rejecting this seems OK on balance.
  using UBool = signed char; // ICU uses this.
  struct ICUBase {
    virtual UBool operator==(const ICUBase&) const;
    UBool operator!=(const ICUBase &arg) const { return !operator==(arg); }
  };
  struct ICUDerived : ICUBase {
    UBool operator==(const ICUBase&) const override; // expected-note {{declared here}}
  };
  bool cmp_icu = ICUDerived() != ICUDerived(); // expected-error {{not 'bool'}}
}

#else // NO_ERRORS

namespace problem_cases {
  // We can select a reversed candidate where we used to select a non-reversed
  // one, and in the worst case this can dramatically change the meaning of the
  // program. Make sure we at least warn on the worst cases under -Wall.
  struct iterator;
  struct const_iterator {
    const_iterator(iterator);
    bool operator==(const const_iterator&) const;
  };
  struct iterator {
    bool operator==(const const_iterator &o) const { // expected-warning {{all paths through this function will call itself}}
      return o == *this;
    }
  };
}
#endif // NO_ERRORS
