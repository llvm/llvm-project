// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedLocalVarsChecker -verify %s

#include "mock-types.h"

class Derived;
Derived* [[clang::annotate_type("webkit.pointerconversion")]] annotatedCast(RefCountable*);
template <typename T, typename U> T* unknownCast(U* u);
void destroyEverything();

class Derived : public RefCountable {
public:
  Derived* asDerived() { return static_cast<Derived*>(this); }
  Derived& asDerivedRef() { return *static_cast<Derived*>(this); }
  Derived* maybe(Derived* other) { return other ? other : static_cast<Derived*>(this); }
  Derived* cStyle() { return (Derived*)this; }
  Derived* reinterp() { return reinterpret_cast<Derived*>(this); }
  Derived* dyn() { return dynamic_cast<Derived*>(this); }
  const Derived* constThis() const { return static_cast<const Derived*>(this); }
  Derived* constCast() const { return const_cast<Derived*>(static_cast<const Derived*>(this)); }
  Derived& cStyleRef() { return (Derived&)*this; }
  RefCountable* upcast() { return this; }
  Derived* viaHelper() { return downcast<Derived>(this); }
  Derived& viaHelperRef() { return downcast<Derived>(*this); }
  Derived* viaAnnotated() { return annotatedCast(this); }
  Derived* viaUnknown() { return unknownCast<Derived>(this); }
  virtual Derived* virt() { return this; }

  // Non-trivial bodies which destroy other objects. The returned pointer is
  // still the receiver itself.
  Derived* deleteOtherThenReturn() { delete m_owned; m_owned = nullptr; return static_cast<Derived*>(this); }
  Derived* clearThenReturn() { m_child = nullptr; return static_cast<Derived*>(this); }
  Derived* callUnknownThenReturn() { destroyEverything(); return downcast<Derived>(this); }

  RefPtr<Derived> m_child;
  Derived* m_owned { nullptr };
};

void testSafeReceiver(Ref<Derived> ref, RefPtr<Derived> refPtr) {
  {
    Derived* d = ref->asDerived();
    d->method();
  }
  {
    Derived& d = ref->asDerivedRef();
    d.method();
  }
  {
    Derived* d = refPtr->asDerived();
    d->method();
  }
  {
    Derived* d = ref->cStyle();
    d->method();
  }
  {
    Derived* d = ref->reinterp();
    d->method();
  }
  {
    Derived* d = ref->dyn();
    d->method();
  }
  {
    const Derived* d = ref->constThis();
    d->constMethod();
  }
  {
    Derived* d = ref->constCast();
    d->method();
  }
  {
    Derived& d = ref->cStyleRef();
    d.method();
  }
  {
    RefCountable* d = ref->upcast();
    d->method();
  }
  {
    Derived* d = ref->viaHelper();
    d->method();
  }
  {
    Derived& d = ref->viaHelperRef();
    d.method();
  }
  {
    Derived* d = ref->viaAnnotated();
    d->method();
  }
  {
    Derived* d = ref->viaUnknown(); // Not a known pointer conversion function.
    // expected-warning@-1{{Local variable 'd' is a raw pointer to RefPtr-capable type 'Derived'}}
    d->method();
  }
  {
    Derived* d = ref->virt(); // Virtual functions may be overridden.
    // expected-warning@-1{{Local variable 'd' is a raw pointer to RefPtr-capable type 'Derived'}}
    d->method();
  }
}

void testNonTrivialBody(Ref<Derived> ref) {
  {
    Derived* d = ref->deleteOtherThenReturn();
    d->method();
  }
  {
    Derived* d = ref->clearThenReturn();
    d->method();
  }
  {
    Derived* d = ref->callUnknownThenReturn();
    d->method();
  }
}

struct Holder {
  Derived* m_raw;
  void testUnsafeReceiver() {
    {
      Derived* d = m_raw->asDerived();
      // expected-warning@-1{{Local variable 'd' is a raw pointer to RefPtr-capable type 'Derived'}}
      d->method();
    }
    {
      Derived* d = m_raw->viaHelper();
      // expected-warning@-1{{Local variable 'd' is a raw pointer to RefPtr-capable type 'Derived'}}
      d->method();
    }
    {
      Derived* d = m_raw->maybe(m_raw);
      // expected-warning@-1{{Local variable 'd' is a raw pointer to RefPtr-capable type 'Derived'}}
      d->method();
    }
  }
};

struct NonTrivialHolder {
  Derived* m_raw;
  void testUnsafeReceiver() {
    {
      Derived* d = m_raw->deleteOtherThenReturn();
      // expected-warning@-1{{Local variable 'd' is a raw pointer to RefPtr-capable type 'Derived'}}
      d->method();
    }
    {
      Derived* d = m_raw->clearThenReturn();
      // expected-warning@-1{{Local variable 'd' is a raw pointer to RefPtr-capable type 'Derived'}}
      d->method();
    }
    {
      Derived* d = m_raw->callUnknownThenReturn();
      // expected-warning@-1{{Local variable 'd' is a raw pointer to RefPtr-capable type 'Derived'}}
      d->method();
    }
  }
};
