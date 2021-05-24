// RUN: %clang_cc1 -triple arm64-apple-ios11 -fsyntax-only -verify %s

@protocol P

@property (nonatomic) int reqProp __attribute__((availability(ios, introduced=12.0))); // expected-note 2 {{is here}}



@optional
@property (nonatomic) int myProp __attribute__((availability(ios, introduced=12.0))); // expected-note {{has been marked as being introduced in}}

@optional
@property (nonatomic, readonly) int depProp __attribute__((availability(ios, introduced=8.0, deprecated=12.0))); // expected-note {{protocol method is here}}

@optional
@property (nonatomic) int obsProp __attribute__((availability(ios, introduced=8.0, obsoleted=12.0)));

@optional
- (void) unavaibleInClass __attribute__((availability(ios, introduced=12.0))); // expected-note {{method is here}}

@end

@interface X <P>

@property (nonatomic) int myProp __attribute__((availability(ios, introduced=13.0))); // expected-note 2 {{has been marked as being introduced in}}

@property (nonatomic) int reqProp __attribute__((availability(ios, introduced=13.0))); // expected-warning 2 {{method introduced after the protocol method it implements on iOS}}

@property (nonatomic, readonly) int depProp __attribute__((availability(ios, introduced=8.0, deprecated=10.0))); // expected-warning {{method deprecated before the protocol method it implements on iOS (12.0 vs. 10.0)}} expected-note {{been explicitly marked deprecated here}}

@property (nonatomic) int obsProp __attribute__((availability(ios, introduced=8.0, obsoleted=10.0))); // expected-note {{been explicitly marked unavailable here}}

- (void) unavaibleInClass __attribute__((availability(ios, unavailable))); // expected-warning {{method cannot be unavailable on iOS when the protocol method it implements is available}}

@end


void test(X *x) {
  int i =  x.myProp;  // expected-warning {{'myProp' is only available on iOS 13.0 or newer}} expected-note {{enclose}}
  x.myProp = i;       // expected-warning {{'setMyProp:' is only available on iOS 13.0 or newer}} expected-note {{enclose}}
  int i2 = x.depProp; // expected-warning {{'depProp' is deprecated: first deprecated in iOS 10.0}}
  int i3 = x.obsProp; // expected-error {{'obsProp' is unavailable: obsoleted in iOS 10.0}}
}

void testProto(id<P> x) {
  int i = x.myProp; // expected-warning {{'myProp' is only available on iOS 12.0 or newer}} expected-note {{enclose}}
}
