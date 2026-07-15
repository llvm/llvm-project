// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.MemoryUnsafeCastChecker -verify %s

#include "mock-types.h"

struct Base : RefCountable { virtual ~Base() {} };
struct Derived : Base { int extra; };

struct S {
  Base& m_ref;

  void f() {
    // this->m_ref contains CXXThisExpr as descendant -> suppressed
    auto& dref = static_cast<Derived&>(this->m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}

    // Pointer equivalent using exact match — IS flagged:
    auto* dptr = static_cast<Derived*>(&this->m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  }

  void f_reinterpret() {
    auto& dref = reinterpret_cast<Derived&>(this->m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}

    auto* dptr = reinterpret_cast<Derived*>(&this->m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  }

  void f_dynamic() {
    auto& dref = dynamic_cast<Derived&>(this->m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}

    auto* dptr = dynamic_cast<Derived*>(&this->m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  }

  void f_cstyle() {
    auto& dref = (Derived&)this->m_ref;
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}

    auto* dptr = (Derived*)&this->m_ref;
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  }
};
