// RUN: %clang_cc1 -fsyntax-only -verify %s 

struct S; // expected-note 8 {{forward declaration of 'S'}}

struct NoConv {};
struct Bad { template<class T> Bad(T v) noexcept(noexcept(member_ = v)) {} int member_; };

void f() {
  __is_pod(S); // expected-error{{incomplete type 'S' used in type trait expression}}
  __is_pod(S[]); // expected-error{{incomplete type 'S' used in type trait expression}}

  __is_trivially_copyable(S); // expected-error{{incomplete type 'S' used in type trait expression}}
  __is_trivially_copyable(S[]); // expected-error{{incomplete type 'S' used in type trait expression}}

  __is_trivially_relocatable(S); // expected-error{{incomplete type 'S' used in type trait expression}}
  __is_trivially_relocatable(S[]); // expected-error{{incomplete type 'S' used in type trait expression}}

  static_assert(!__reference_constructs_from_temporary(S, NoConv&&)); // expected-error{{incomplete type 'S' used in type trait expression}}
  static_assert(!__reference_constructs_from_temporary(Bad, S)); // expected-error{{incomplete type 'S' used in type trait expression}}
}
