// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

class Voice : public RefCountable { };

class Wrapper : public RefCountable {
public:
  static Ref<Wrapper> create(Voice&);
};

template <typename T> Ref<T> protect(T& t) { return Ref<T>(t); }
template <typename T> T* unsafeCast(T& t) { return &t; }

Voice* provide();
void consume(Voice&);

template <typename F> void map(F f) {
  Ref<Voice> voice = adoptRef(*provide());
  f(voice.get());
}

void genericLambdaWithProtectedArg() {
  // The call to protect() is unresolved in the body of the generic lambda but
  // returns a Ref<Voice> in its instantiation.
  map([](auto& voice) { return Wrapper::create(protect(voice)); });
}

void concreteLambdaWithProtectedArg() {
  map([](Voice& voice) { return Wrapper::create(protect(voice)); });
}

void genericLambdaWithUnsafeArg() {
  map([](auto& voice) { consume(*unsafeCast(voice)); });
  // expected-warning@-1{{Function argument '*unsafeCast(voice)' (to 'consume') is a raw reference to RefPtr-capable type 'Voice'}}
}

void genericLambdaWithUnsafeCall() {
  map([](auto&) { consume(*provide()); });
  // expected-warning@-1{{Function argument '*provide()' (to 'consume') is a raw reference to RefPtr-capable type 'Voice'}}
}

void genericLambdaWithProtectCall() {
  map([](auto&) { consume(protect(*provide())); });
}
void genericLambdaWithUnsafeCaptureInit() {
  // The capture initializer belongs to the enclosing scope, not the pattern.
  map([w = Wrapper::create(*provide())](auto&) { });
  // expected-warning@-1{{Function argument '*provide()' (to 'Wrapper::create') is a raw reference to RefPtr-capable type 'Voice'}}
}
