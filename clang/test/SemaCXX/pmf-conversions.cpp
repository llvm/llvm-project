// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only %s -verify

struct S {
  int a;
  void method();
  void method_overload();
  void method_overload(int);
};

using pmf_type = void (S::*)();
using pm_type = int S::*;
using pf_type = void (*)(S*);
using pf_type_mismatched = void (*)(S*, int);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpmf-conversions"

// constexpr pmf conversions are not supported yet.
constexpr pf_type method_constexpr = reinterpret_cast<pf_type>(&S::method); // expected-error {{constexpr variable 'method_constexpr' must be initialized by a constant expression}} expected-note {{reinterpret_cast is not allowed in a constant expression}}
pf_type method = reinterpret_cast<pf_type>(&S::method);

void pmf_convert_no_object(pmf_type method, pm_type field) {
  (void)reinterpret_cast<pf_type>(&S::method);
  (void)reinterpret_cast<pf_type>(method);
  (void)reinterpret_cast<pf_type>(((method)));
  (void)(pf_type)(&S::method);
  (void)(pf_type)(method);
  (void)reinterpret_cast<pf_type_mismatched>(&S::method);
  (void)reinterpret_cast<pf_type_mismatched>(method);
  (void)reinterpret_cast<pf_type>(&S::a); // expected-error {{reinterpret_cast from 'int S::*' to 'pf_type' (aka 'void (*)(S *)') is not allowed}}
  (void)reinterpret_cast<pf_type>(field); // expected-error {{reinterpret_cast from 'pm_type' (aka 'int S::*') to 'pf_type' (aka 'void (*)(S *)') is not allowed}}
}

void pmf_convert_with_base(S* p, S& r, pmf_type method, pm_type field) {
  (void)reinterpret_cast<pf_type>(p->*(&S::method));
  (void)reinterpret_cast<pf_type>(((p)->*((&S::method))));
  (void)reinterpret_cast<pf_type>(p->*method);
  (void)reinterpret_cast<pf_type>(((p)->*(method)));
  (void)reinterpret_cast<pf_type>(p->*(static_cast<pmf_type>(&S::method_overload)));
  (void)(pf_type)(p->*(&S::method));
  (void)(pf_type)(p->*method);
  (void)reinterpret_cast<pf_type_mismatched>(p->*method);
  (void)reinterpret_cast<pf_type>(r.*method);
  (void)reinterpret_cast<pf_type_mismatched>(r.*method);
  (void)reinterpret_cast<pf_type>(p->*(&S::a));
  (void)reinterpret_cast<pf_type>(p->*field);
}

#pragma clang diagnostic pop

void pmf_convert_warning(S *p, pmf_type method) {
  (void)reinterpret_cast<pf_type>(method); // expected-warning {{converting the bound member function 'pmf_type' (aka 'void (S::*)()') to a function pointer 'pf_type' (aka 'void (*)(S *)') is a GNU extension}}
  (void)reinterpret_cast<pf_type>(p->*method); // expected-warning {{converting the bound member function '<bound member function type>' to a function pointer 'pf_type' (aka 'void (*)(S *)') is a GNU extension}}
}
