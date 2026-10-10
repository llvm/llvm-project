// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedLocalVarsChecker -verify %s

#include "mock-types.h"

class Voice : public RefCountable { };

template <typename T> T* unsafeCast(T& t) { return &t; }

Voice* provide();
void consume(Voice&);

template <typename F> void map(F f) {
  Ref<Voice> voice = adoptRef(*provide());
  f(voice);
}

// The Ref parameter guards the raw pointer for the duration of the call.
void concreteGuarded(Ref<Voice>& guard) {
  Voice* v = guard.ptr();
  consume(*v);
}

// Same code, but the receiver of ptr() is only resolved in the instantiation.
template <typename T> void tmplGuarded(T& guard) {
  Voice* v = guard.ptr();
  consume(*v);
}

void lambdaGuarded() {
  map([](auto& guard) {
    Voice* v = guard.ptr();
    consume(*v);
  });
}

void concreteLambdaGuarded() {
  map([](Ref<Voice>& guard) {
    Voice* v = guard.ptr();
    consume(*v);
  });
}

// The initializer is only known to be unsafe once unsafeCast() is resolved.
void lambdaUnsafe() {
  map([](auto& guard) {
    Voice* v = unsafeCast(guard.get());
    // expected-warning@-1{{Local variable 'v' is a raw pointer to RefPtr-capable type 'Voice'}}
    consume(*v);
  });
}

// Mutating the guardian invalidates the raw pointer. The assignment is only
// visible in the instantiated body, so the instantiation has to be the decl
// the guardian is looked up in.
void concreteLambdaMutatedGuardian() {
  map([](Ref<Voice>& guard) {
    Voice* v = guard.ptr();
    // expected-warning@-1{{Local variable 'v' is a raw pointer to RefPtr-capable type 'Voice'}}
    guard = adoptRef(*provide());
    consume(*v);
  });
}

void lambdaMutatedGuardian() {
  map([](auto& guard) {
    Voice* v = guard.ptr();
    // expected-warning@-1{{Local variable 'v' is a raw pointer to RefPtr-capable type 'Voice'}}
    guard = adoptRef(*provide());
    consume(*v);
  });
}

// A pattern which is never instantiated generates no code to diagnose.
template <typename T> void neverInstantiated(T& guard) {
  Voice* v = unsafeCast(guard.get());
  consume(*v);
}

void instantiate() {
  Ref<Voice> voice = adoptRef(*provide());
  tmplGuarded(voice);
}
