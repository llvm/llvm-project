// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak %s -verify

__attribute__((objc_root_class))
@interface NSObject
@end
  
@class Forward;
@class Forward2;

// Tests for generic arguments.

@interface PC1<T> : NSObject
- (T) get;
- (void) set: (T) v; // expected-note 4 {{parameter}}
@end

void test1a(PC1<__weak id> *obj) { // expected-error {{type argument '__weak id' cannot be qualified with '__weak'}}
  id x = [obj get];
  [obj set: x];
}

void test1b(PC1<__strong id> *obj) { // expected-error {{type argument '__strong id' cannot be qualified with '__strong'}}
  id x = [obj get];
  [obj set: x];
}

void test1c(PC1<id> *obj) {
  id x = [obj get];
  [obj set: x];
}

// Test that this doesn't completely kill downstream type-checking.
void test1d(PC1<__weak Forward*> *obj) { // expected-error {{type argument 'Forward *__weak' cannot be qualified with '__weak'}}
  Forward2 *x = [obj get]; // expected-warning {{incompatible}}
  [obj set: x]; // expected-warning {{incompatible}}
}

void test1e(PC1<__strong Forward*> *obj) { // expected-error {{type argument 'Forward *__strong' cannot be qualified with '__strong'}}
  Forward2 *x = [obj get]; // expected-warning {{incompatible}}
  [obj set: x]; // expected-warning {{incompatible}}
}

void test1f(PC1<Forward*> *obj) {
  Forward2 *x = [obj get]; // expected-warning {{incompatible}}
  [obj set: x]; // expected-warning {{incompatible}}
}

// Typedefs are fine, just silently ignore them.
typedef __strong id StrongID;
void test1g(PC1<StrongID> *obj) {
  Forward2 *x = [obj get];
  [obj set: x];
}

typedef __strong Forward *StrongForward;
void test1h(PC1<StrongForward> *obj) {
  Forward2 *x = [obj get]; // expected-warning {{incompatible}}
  [obj set: x]; // expected-warning {{incompatible}}
}

// These aren't really ARC-specific, but they're the same basic idea.
void test1i(PC1<const id> *obj) { // expected-error {{type argument 'const id' cannot be qualified with 'const'}}
  id x = [obj get];
  [obj set: x];
}

void test1j(PC1<volatile id> *obj) { // expected-error {{type argument 'volatile id' cannot be qualified with 'volatile'}}
  id x = [obj get];
  [obj set: x];
}

void test1k(PC1<__attribute__((address_space(256))) id> *obj) { // expected-error {{type argument '__attribute__((address_space(256))) id' cannot be qualified with '__attribute__((address_space(256)))'}}
  id x = [obj get];
  [obj set: x];
}

// Tests for generic parameter bounds.

@interface PC2<T : __strong id> // expected-error {{type bound '__strong id' for type parameter 'T' cannot be qualified with '__strong'}}
@end

@interface PC3<T : __weak id> // expected-error {{type bound '__weak id' for type parameter 'T' cannot be qualified with '__weak'}}
@end

@interface PC4<T : __strong Forward*> // expected-error {{type bound 'Forward *__strong' for type parameter 'T' cannot be qualified with '__strong'}}
@end

@interface PC5<T : __weak Forward*> // expected-error {{type bound 'Forward *__weak' for type parameter 'T' cannot be qualified with '__weak'}}
@end

@interface PC6<T : StrongID> // expected-error {{type bound 'StrongID' (aka '__strong id') for type parameter 'T' cannot be qualified with '__strong'}}
@end

@interface PC7<T : StrongForward> // expected-error {{type bound 'StrongForward' (aka 'Forward *__strong') for type parameter 'T' cannot be qualified with '__strong'}}
@end

// These aren't really ARC-specific, but they're the same basic idea.
@interface PC8<T : const id> // expected-error {{type bound 'const id' for type parameter 'T' cannot be qualified with 'const'}}
@end

@interface PC9<T : volatile id> // expected-error {{type bound 'volatile id' for type parameter 'T' cannot be qualified with 'volatile'}}
@end

@interface PC10<T : __attribute__((address_space(256))) id> // expected-error {{type bound '__attribute__((address_space(256))) id' for type parameter 'T' cannot be qualified with '__attribute__((address_space(256)))'}}
@end