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

// Wrapper::create is qualified, so it is resolved in the pattern even though
// its argument is still type-dependent there. The argument resolves to a
// Ref<Voice> in the instantiation and is therefore safe.
template <typename T> void protectedArg(T& voice) {
  Wrapper::create(protect(voice));
}

// The argument is only known to be unsafe once unsafeCast() is resolved.
template <typename T> void unsafeArg(T& voice) {
  consume(*unsafeCast(voice));
  // expected-warning@-1{{Function argument '*unsafeCast(voice)' (to 'consume') is a raw reference to RefPtr-capable type 'Voice'}}
}

// Nothing in this call is dependent, so the pattern alone is enough.
template <typename T> void unsafeCall(T&) {
  consume(*provide());
  // expected-warning@-1{{Function argument '*provide()' (to 'consume') is a raw reference to RefPtr-capable type 'Voice'}}
}

// A pattern which is never instantiated generates no code to diagnose.
template <typename T> void neverInstantiated(T& voice) {
  Wrapper::create(protect(voice));
  consume(*unsafeCast(voice));
}

template <typename U> struct Holder {
  template <typename T> void protectedArg(T& voice) {
    Wrapper::create(protect(voice));
  }
  template <typename T> void unsafeArg(T& voice) {
    consume(*unsafeCast(voice));
    // expected-warning@-1{{Function argument '*unsafeCast(voice)' (to 'consume') is a raw reference to RefPtr-capable type 'Voice'}}
  }
};

void instantiate() {
  Ref<Voice> voice = adoptRef(*provide());
  protectedArg(voice.get());
  unsafeArg(voice.get());
  unsafeCall(voice.get());
  Holder<int> holder;
  holder.protectedArg(voice.get());
  holder.unsafeArg(voice.get());
}
