// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s -verify

@interface A
- knownMethod;
@end

@interface B
- unknownMethod;
@end

@interface C : A
- knownMethod;
@end

template<typename T> struct RetainPtr {
  explicit operator T*() const;
};

void methodCallToSpecific(RetainPtr<A> a) {
  [a knownMethod];
  [a unknownMethod]; // expected-warning{{'A' may not respond to 'unknownMethod'}}
}

void explicitCast(RetainPtr<A> a, RetainPtr<B> b, RetainPtr<C> c) {
  (void)(A*)a;
  (void)(A*)b; // expected-error{{cannot convert 'RetainPtr<B>' to 'A *' without a conversion operator}}
  (void)(A*)c;
  (void)(C*)a;
  (void)static_cast<A*>(a);
  (void)static_cast<A*>(b);  // expected-error{{cannot convert 'RetainPtr<B>' to 'A *' without a conversion operator}}
  (void)static_cast<A*>(c);
}

struct Incomplete; // expected-note{{forward declaration}}

void methodCallToIncomplete(Incomplete &incomplete) {
  [incomplete knownMethod]; // expected-error{{incomplete receiver type 'Incomplete'}}
}

struct IdPtr {
  explicit operator id() const;
};

void methodCallToId(IdPtr a) {
  [a knownMethod];
  [a unknownMethod];
}

void explicitCast(IdPtr a) {
  (void)(A*)a;
  (void)static_cast<A*>(a);
}
