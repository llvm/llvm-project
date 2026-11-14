// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

class Derived;
Derived* [[clang::annotate_type("webkit.pointerconversion")]] annotatedCast(RefCountable*);
template <typename T, typename U> T* unknownCast(U* u);
void destroyEverything();

void consume(RefCountable*);
void consume(const RefCountable*);
void consumeRef(RefCountable&);

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
  consume(ref->asDerived());
  consumeRef(ref->asDerivedRef());
  consume(refPtr->asDerived());
  consume(ref->cStyle());
  consume(ref->reinterp());
  consume(ref->dyn());
  consume(ref->constThis());
  consume(ref->constCast());
  consumeRef(ref->cStyleRef());
  consume(ref->upcast());
  consume(ref->viaHelper());
  consumeRef(ref->viaHelperRef());
  consume(ref->viaAnnotated());
  consume(ref->viaUnknown()); // Not a known pointer conversion function.
  // expected-warning@-1{{Function argument 'ref->viaUnknown()' (to 'consume') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  consume(ref->virt()); // Virtual functions may be overridden.
  // expected-warning@-1{{Function argument 'ref->virt()' (to 'consume') is a raw pointer to RefPtr-capable type 'RefCountable'}}
}

void testNonTrivialBody(Ref<Derived> ref) {
  consume(ref->deleteOtherThenReturn());
  consume(ref->clearThenReturn());
  consume(ref->callUnknownThenReturn());
}

struct Holder {
  Derived* m_raw;
  void testUnsafeReceiver() {
    consume(m_raw->asDerived());
    // expected-warning@-1{{Function argument 'this->m_raw->asDerived()' (to 'consume') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    consume(m_raw->maybe(m_raw));
    // expected-warning@-1{{Function argument 'this->m_raw->maybe(this->m_raw)' (to 'consume') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  }
};

struct NonTrivialHolder {
  Derived* m_raw;
  void testUnsafeReceiver() {
    consume(m_raw->deleteOtherThenReturn());
    // expected-warning@-1{{Function argument 'this->m_raw' (parameter 'this' to 'Derived::deleteOtherThenReturn') is a raw pointer to RefPtr-capable type 'Derived'}}
    // expected-warning@-2{{Function argument 'this->m_raw->deleteOtherThenReturn()' (to 'consume') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    consume(m_raw->clearThenReturn());
    // expected-warning@-1{{Function argument 'this->m_raw' (parameter 'this' to 'Derived::clearThenReturn') is a raw pointer to RefPtr-capable type 'Derived'}}
    // expected-warning@-2{{Function argument 'this->m_raw->clearThenReturn()' (to 'consume') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    consume(m_raw->callUnknownThenReturn());
    // expected-warning@-1{{Function argument 'this->m_raw' (parameter 'this' to 'Derived::callUnknownThenReturn') is a raw pointer to RefPtr-capable type 'Derived'}}
    // expected-warning@-2{{Function argument 'this->m_raw->callUnknownThenReturn()' (to 'consume') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  }
};
