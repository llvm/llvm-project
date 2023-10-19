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
  // expected-note@#1 5{{candidate function not viable: no known conversion from 'Y' to 'X<1>' for 1st argument}}
  // expected-note@#1 5{{candidate function (with reversed parameter order) not viable: no known conversion from 'Y' to 'X<2>' for 1st argument}}
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
  // expected-note@#1 5{{candidate function (with reversed parameter order) not viable: no known conversion from 'Y' to 'X<1>' for 2nd argument}}
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
  // expected-note@#2 2{{candidate function not viable: no known conversion from 'Y' to 'X<1>' for 1st argument}}
  // expected-note@#2 2{{candidate function (with reversed parameter order) not viable: no known conversion from 'Y' to 'X<2>' for 1st argument}}
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
  // expected-note@#2 2{{candidate function (with reversed parameter order) not viable: no known conversion from 'Y' to 'X<1>' for 2nd argument}}
  bool mem_req = x2 == y; // expected-error {{selected deleted operator '=='}}
  bool mem_rne = x2 != y; // expected-error {{selected deleted operator '=='}}

  // For all other operators, the rewritten candidate set is empty.
  X<3> operator+(X<1>, X<2>) = delete; // expected-note {{no known conversion from 'X<2>' to 'X<1>'}}
  X<3> reversed_add = x2 + x1; // expected-error {{invalid operands}}
}

namespace PR44627 {
  namespace ADL {
    struct type {};
    bool operator==(type lhs, int rhs) {
      return true;
    }
  }

  bool b1 = ADL::type() == 0;
  bool b2 = 0 == ADL::type();
}

namespace P2468R2 {
// Problem cases prior to P2468R2 but now intentionally rejected.
struct SymmetricNonConst {
  bool operator==(const SymmetricNonConst&); // expected-note {{ambiguity is between a regular call to this operator and a call with the argument order reversed}}
  // expected-note@-1 {{mark 'operator==' as const or add a matching 'operator!=' to resolve the ambiguity}}
};
bool cmp_non_const = SymmetricNonConst() == SymmetricNonConst(); // expected-warning {{ambiguous}}

struct SymmetricConst {
  bool operator==(const SymmetricConst&) const;
};
bool cmp_const = SymmetricConst() == SymmetricConst();

struct SymmetricNonConstWithoutConstRef {
  bool operator==(SymmetricNonConstWithoutConstRef);
};
bool cmp_non_const_wo_ref = SymmetricNonConstWithoutConstRef() == SymmetricNonConstWithoutConstRef();

struct B {
  virtual bool operator==(const B&) const;
};
struct D : B {
  bool operator==(const B&) const override; // expected-note {{operator}}
};
bool cmp_base_derived = D() == D(); // expected-warning {{ambiguous}}

// Reversed "3" not used because we find "2".
// Rewrite != from "3" but warn that "chosen rewritten candidate must return cv-bool".
using UBool = signed char;
struct ICUBase {
  virtual UBool operator==(const ICUBase&) const; // 1.
  UBool operator!=(const ICUBase &arg) const { return !operator==(arg); } // 2.
};
struct ICUDerived : ICUBase {
  // 3.
  UBool operator==(const ICUBase&) const override; // expected-note {{declared here}}
};
bool cmp_icu = ICUDerived() != ICUDerived(); // expected-warning {{ISO C++20 requires return type of selected 'operator==' function for rewritten '!=' comparison to be 'bool', not 'UBool' (aka 'signed char')}}
// Accepted by P2468R2.
// 1
struct S {
  bool operator==(const S&) { return true; }
  bool operator!=(const S&) { return false; }
};
bool ts = S{} != S{};
// 2
template<typename T> struct CRTPBase {
  bool operator==(const T&) const;
  bool operator!=(const T&) const;
};
struct CRTP : CRTPBase<CRTP> {};
bool cmp_crtp = CRTP() == CRTP();
bool cmp_crtp2 = CRTP() != CRTP();
// https://github.com/llvm/llvm-project/issues/57711
namespace issue_57711 {
template <class T>
bool compare(T l, T r)
    requires requires { l == r; } {
  return l == r;
}

void test() {
  compare(CRTP(), CRTP()); // previously this was a hard error (due to SFINAE failure).
}
}
// 3
template <bool>
struct GenericIterator {
  using ConstIterator = GenericIterator<true>;
  using NonConstIterator = GenericIterator<false>;
  GenericIterator() = default;
  GenericIterator(const NonConstIterator&);

  bool operator==(ConstIterator) const;
  bool operator!=(ConstIterator) const;
};
using Iterator = GenericIterator<false>;

bool biter = Iterator{} == Iterator{};

// Intentionally rejected by P2468R2
struct ImplicitInt {
  ImplicitInt();
  ImplicitInt(int*);
  bool operator==(const ImplicitInt&) const; // expected-note {{candidate function (with reversed parameter order)}}
  operator int*() const;
};
bool implicit_int = nullptr != ImplicitInt{}; // expected-error {{use of overloaded operator '!=' is ambiguous (with operand types 'std::nullptr_t' and 'ImplicitInt')}}
                                              // expected-note@-1 4 {{built-in candidate operator!=}}

// https://eel.is/c++draft/over.match.oper#example-2
namespace example {
struct A {};
template<typename T> bool operator==(A, T);     // 1. expected-note {{candidate function template not viable: no known conversion from 'int' to 'A' for 1st argument}}
bool a1 = 0 == A();                             // OK, calls reversed 1
template<typename T> bool operator!=(A, T);
bool a2 = 0 == A();  // expected-error {{invalid operands to binary expression ('int' and 'A')}}

struct B {
  bool operator==(const B&);    // 2
  // expected-note@-1 {{ambiguity is between a regular call to this operator and a call with the argument order reversed}}
};
struct C : B {
  C();
  C(B);
  bool operator!=(const B&);    // 3
};
bool c1 = B() == C(); // OK, calls 2; reversed 2 is not a candidate because search for operator!= in C finds 3
bool c2 = C() == B();  // Search for operator!= inside B never finds 3. expected-warning {{ISO C++20 considers use of overloaded operator '==' (with operand types 'C' and 'B') to be ambiguous despite there being a unique best viable function}}

struct D {};
template<typename T> bool operator==(D, T);     // 4
inline namespace N {
  template<typename T> bool operator!=(D, T);   // 5
}
bool d1 = 0 == D();  // OK, calls reversed 4; 5 does not forbid 4 as a rewrite target as "search" does not look inside inline namespaces.
} // namespace example

namespace template_tests {
namespace template_head_does_not_match {
struct A {};
template<typename T, class U = int> bool operator==(A, T);
template <class T> bool operator!=(A, T);
bool x = 0 == A(); // Ok. Use rewritten candidate.
}

namespace template_with_different_param_name_are_equivalent {
struct A {};
template<typename T> bool operator==(A, T); // expected-note {{candidate function template not viable: no known conversion from 'int' to 'A' for 1st argument}}
template <typename U> bool operator!=(A, U);
bool x = 0 == A(); // expected-error {{invalid operands to binary expression ('int' and 'A')}}
}

namespace template_and_non_template {
struct A {
template<typename T> bool operator==(const T&);
// expected-note@-1{{mark 'operator==' as const or add a matching 'operator!=' to resolve the ambiguity}}
// expected-note@-2{{ambiguity is between a regular call to this operator and a call with the argument order reversed}}
};
bool a = A() == A(); // expected-warning {{ambiguous despite there being a unique best viable function}}

struct B {
template<typename T> bool operator==(const T&) const;
bool operator!=(const B&);
};
bool b = B() == B(); // ok. No rewrite due to const.

struct C {};
template <class T=int>
bool operator==(C, int);
bool operator!=(C, int);
bool c = 0 == C(); // Ok. Use rewritten candidate as the non-template 'operator!=' does not correspond to template 'operator=='
}
} // template_tests

namespace using_decls {
namespace simple {
struct C {};
bool operator==(C, int); // expected-note {{candidate function not viable: no known conversion from 'int' to 'C' for 1st argument}}
bool a = 0 == C(); // Ok. Use rewritten candidate.
namespace other_ns { bool operator!=(C, int); }
bool b = 0 == C(); // Ok. Use rewritten candidate.
using other_ns::operator!=;
bool c = 0 == C(); // Rewrite not possible. expected-error {{invalid operands to binary expression ('int' and 'C')}}
}
namespace templated {
struct C {};
template<typename T>
bool operator==(C, T); // expected-note {{candidate function template not viable: no known conversion from 'int' to 'C' for 1st argument}}
bool a = 0 == C(); // Ok. Use rewritten candidate.
namespace other_ns { template<typename T> bool operator!=(C, T); }
bool b = 0 == C(); // Ok. Use rewritten candidate.
using other_ns::operator!=;
bool c = 0 == C(); // Rewrite not possible. expected-error {{invalid operands to binary expression ('int' and 'C')}}
} // templated
} // using_decls

// FIXME(GH58185): Match requires clause.
namespace match_requires_clause {
template<int x>
struct A {
bool operator==(int) requires (x==1); // 1.
bool operator!=(int) requires (x==2); // 2.
};
int a1 = 0 == A<1>(); // Should not find 2 as the requires clause does not match. \
                      // expected-error {{invalid operands to binary expression ('int' and 'A<1>')}}
}

namespace static_operators {
// Verify no crash.
struct X { 
  bool operator ==(X const&); // expected-note {{ambiguity is between a regular call}}
                              // expected-note@-1 {{mark 'operator==' as const or add a matching 'operator!=' to resolve the ambiguity}}
  static bool operator !=(X const&, X const&); // expected-error {{overloaded 'operator!=' cannot be a static member function}}
};
bool x = X() == X(); // expected-warning {{ambiguous}}
}
} // namespace P2468R2

namespace GH53954{
namespace friend_template_1 {
struct P {
    template <class T>
    friend bool operator==(const P&, const T&); // expected-note {{candidate}} \
                                                // expected-note {{ambiguous candidate function with reversed arguments}}
};
struct A : public P {};
struct B : public P {};
bool check(A a, B b) { return a == b; } // expected-warning {{use of overloaded operator '==' (with operand types 'A' and 'B') to be ambiguous}}
}

namespace friend_template_2 {
struct P {
    template <class T>
    friend bool operator==(const T&, const P&); // expected-note {{candidate}} \
                                                // expected-note {{ambiguous candidate function with reversed arguments}}
};
struct A : public P {};
struct B : public P {};
bool check(A a, B b) { return a == b; } // expected-warning {{use of overloaded operator '==' (with operand types 'A' and 'B') to be ambiguous}}
}

namespace member_template {
struct P {
  template<class S>
  bool operator==(const S &) const; // expected-note {{candidate}} \
                                    // expected-note {{ambiguous candidate function with reversed arguments}}
};
struct A : public P {};
struct B : public P {};
bool check(A a, B b) { return a == b; } // expected-warning {{use of overloaded operator '==' (with operand types 'A' and 'B') to be ambiguous}}
}

namespace non_member_template_1 {
struct P {};
template<class S>
bool operator==(const P&, const S &); // expected-note {{candidate}} \
                                      // expected-note {{ambiguous candidate function with reversed arguments}}

struct A : public P {};
struct B : public P {};
bool check(A a, B b) { return a == b; } // expected-warning {{use of overloaded operator '==' (with operand types 'A' and 'B') to be ambiguous}}
}
}

namespace non_member_template_2 {
struct P {};
template<class S>
bool operator==(const S&, const P&); // expected-note {{candidate}} \
                                     // expected-note {{ambiguous candidate function with reversed arguments}}

struct A : public P {};
struct B : public P {};
bool check(A a, B b) { return a == b; } // expected-warning {{use of overloaded operator '==' (with operand types 'A' and 'B') to be ambiguous}}
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
