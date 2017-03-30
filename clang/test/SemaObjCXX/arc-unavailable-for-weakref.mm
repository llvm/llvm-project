// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -fsyntax-only -fobjc-arc -verify %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -fsyntax-only -fobjc-weak -verify %s
// rdar://9693477

__attribute__((objc_arc_weak_reference_unavailable))
@interface NSOptOut1072  // expected-note {{class is declared here}}
@end

@interface sub : NSOptOut1072 @end // expected-note 2 {{class is declared here}}

int main() {
  __weak sub *w2; // expected-error {{class is incompatible with __weak references}}

  __weak NSOptOut1072 *ns1; // expected-error {{class is incompatible with __weak references}}

  id obj;

  ns1 = (__weak sub *)obj; // expected-error {{assignment of a weak-unavailable object to a __weak object}} \
                           // expected-error {{class is incompatible with __weak references}} \
                           // expected-error {{explicit ownership qualifier on cast result has no effect}}
}

// rdar://9732636
__attribute__((objc_arc_weak_reference_unavailable))
@interface NOWEAK
+ (id) new;
@end

NOWEAK * Test1() {
  NOWEAK * strong1 = [NOWEAK new];
  __weak id weak1;
  weak1 = strong1; // expected-error {{assignment of a weak-unavailable object to a __weak object}}

  __weak id weak2 = strong1; // expected-error {{assignment of a weak-unavailable object to a __weak object}}
  return (__weak id)strong1; // expected-error {{cast of weak-unavailable object of type 'NOWEAK *' to a __weak object of type '__weak id'}} \
                             // expected-error {{explicit ownership qualifier on cast result has no effect}}
}

@protocol P @end
@protocol P1 @end

NOWEAK<P, P1> * Test2() {
  NOWEAK<P, P1> * strong1 = 0;
  __weak id<P> weak1;
  weak1 = strong1; // expected-error {{assignment of a weak-unavailable object to a __weak object}}

  __weak id<P> weak2 = strong1; // expected-error {{assignment of a weak-unavailable object to a __weak object}}
  return (__weak id<P, P1>)strong1; // expected-error {{cast of weak-unavailable object of type 'NOWEAK<P,P1> *' to a __weak object of type '__weak id<P,P1>'}} \
                                    // expected-error {{explicit ownership qualifier on cast result has no effect}}
}

