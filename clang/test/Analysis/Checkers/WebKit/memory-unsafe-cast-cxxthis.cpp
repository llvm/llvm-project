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

// Member access with an implicit `this`
struct S2 {
  Base& m_ref;

  void f() {
    auto& dref = static_cast<Derived&>(m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}

    auto* dptr = static_cast<Derived*>(&m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  }

  void f_reinterpret() {
    auto& dref = reinterpret_cast<Derived&>(m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}

    auto* dptr = reinterpret_cast<Derived*>(&m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  }

  void f_dynamic() {
    auto& dref = dynamic_cast<Derived&>(m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}

    auto* dptr = dynamic_cast<Derived*>(&m_ref);
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  }

  void f_cstyle() {
    auto& dref = (Derived&)m_ref;
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}

    auto* dptr = (Derived*)&m_ref;
    // expected-warning@-1 {{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  }
};

// A cast of `this` to an unrelated
// further-derived class is still an unsafe downcast and must be flagged.
struct T1 : Derived {
  void bogus_ptr();
};
struct GrandchildPtr : T1 { int more; };
void T1::bogus_ptr() {
  auto* g = static_cast<GrandchildPtr*>(this);
  // expected-warning@-1 {{Unsafe cast from base type 'T1' to derived type 'GrandchildPtr'}}
}

struct T2 : Derived {
  void bogus_ref();
};
struct GrandchildRef : T2 { int more; };
void T2::bogus_ref() {
  auto& g = static_cast<GrandchildRef&>(*this);
  // expected-warning@-1 {{Unsafe cast from base type 'T2' to derived type 'GrandchildRef'}}
}
