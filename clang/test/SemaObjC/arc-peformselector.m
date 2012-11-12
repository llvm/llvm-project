// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify %s
// rdar://9659270

@interface NSObject
- (id)copy; // expected-note {{method 'copy' declared here}}
- (id) test __attribute__((ns_returns_retained)); // expected-note {{method 'test' declared here}}
+ (id) new ; // expected-note {{method 'new' declared here}}
- (id) init __attribute__((ns_returns_not_retained));
- (id)PlusZero;
- (id)PlusOne __attribute__((ns_returns_retained)); // expected-note {{method 'PlusOne' declared here}}
@end

@interface I : NSObject
{
  SEL sel1;
}
- (id)performSelector:(SEL)aSelector;
- (id)performSelector:(SEL)aSelector withObject:(id)object;
- (id)performSelector:(SEL)aSelector withObject:(id)object1 withObject:(id)object2;
@end

@implementation I
- (id) Meth {
  return [self performSelector : @selector(copy)]; // expected-error {{performSelector names a selector which retains the object}}
  return [self performSelector : @selector(test)]; // expected-error {{performSelector names a selector which retains the object}}
  return [self performSelector : @selector(new)]; // expected-error {{performSelector names a selector which retains the object}}
  return [self performSelector : @selector(init)];
  return [self performSelector : sel1]; // expected-warning {{performSelector may cause a leak because its selector is unknown}} \
					// expected-note {{used here}}

  return [self performSelector : @selector(PlusZero)];
  return [self performSelector : @selector(PlusOne)]; // expected-error {{performSelector names a selector which retains the object}}
}

- (id)performSelector:(SEL)aSelector { return 0; }
- (id)performSelector:(SEL)aSelector withObject:(id)object { return 0; }
- (id)performSelector:(SEL)aSelector withObject:(id)object1 withObject:(id)object2 { return 0; }
@end
