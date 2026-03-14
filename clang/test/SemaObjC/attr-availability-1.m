// RUN: %clang_cc1 -triple x86_64-apple-darwin9.0.0 -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++ -std=c++11 -triple x86_64-apple-darwin9.0.0 -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++ -std=c++03 -triple x86_64-apple-darwin9.0.0 -fsyntax-only -verify %s

#if !__has_feature(attribute_availability_with_version_underscores)
# error "missing feature"
#endif

@protocol P
- (void)proto_method __attribute__((availability(macosx,introduced=10_1,deprecated=10_2))); // expected-note 2 {{'proto_method' has been explicitly marked deprecated here}}
@end

@interface A <P>
- (void)method __attribute__((availability(macosx,introduced=10_1,deprecated=10_2))); // expected-note {{'method' has been explicitly marked deprecated here}}

- (void)overridden __attribute__((availability(macosx,introduced=10_3))); // expected-note{{overridden method is here}}
- (void)overridden2 __attribute__((availability(macosx,introduced=10_3)));
- (void)overridden3 __attribute__((availability(macosx,deprecated=10_3)));
- (void)overridden4 __attribute__((availability(macosx,deprecated=10_3))); // expected-note{{overridden method is here}}
- (void)overridden5 __attribute__((availability(macosx,unavailable)));
- (void)overridden6 __attribute__((availability(macosx,introduced=10_3))); // expected-note{{overridden method is here}}
@end

@interface B : A
- (void)method; // NOTE: we expect 'method' to *not* inherit availability.
- (void)overridden __attribute__((availability(macosx,introduced=10_4))); // expected-warning{{overriding method introduced after overridden method on macOS (10.4 vs. 10.3)}}
- (void)overridden2 __attribute__((availability(macosx,introduced=10_2)));
- (void)overridden3 __attribute__((availability(macosx,deprecated=10_4)));
- (void)overridden4 __attribute__((availability(macosx,deprecated=10_2))); // expected-warning{{overriding method deprecated before overridden method on macOS (10.3 vs. 10.2)}}
- (void)overridden5 __attribute__((availability(macosx,introduced=10_3)));
- (void)overridden6 __attribute__((availability(macosx,unavailable))); // expected-warning{{overriding method cannot be unavailable on macOS when its overridden method is available}}
@end

void f(A *a, B *b) {
  [a method]; // expected-warning{{'method' is deprecated: first deprecated in macOS 10.2}}
  [b method]; // no-warning
  [a proto_method]; // expected-warning{{'proto_method' is deprecated: first deprecated in macOS 10.2}}
  [b proto_method]; // expected-warning{{'proto_method' is deprecated: first deprecated in macOS 10.2}}
}

// Warn about using a deprecated method when that method is re-implemented in a
// subclass where the redeclared method is not deprecated.
@interface C
- (void) method __attribute__((availability(macosx,introduced=10_1,deprecated=10_2))); // expected-note {{'method' has been explicitly marked deprecated here}}
@end

@interface D : C
- (void) method;
@end

@interface E : D
- (void) method;
@end

@implementation D
- (void) method {
  [super method]; // expected-warning {{'method' is deprecated: first deprecated in macOS 10.2}}
}
@end

@implementation E
- (void) method {
  [super method]; // no-warning
}
@end

@class NSMutableArray;

@interface NSDictionary
+ (instancetype)dictionaryWithObjectsAndKeys:(id)firstObject, ... __attribute__((sentinel(0,1)));
@end

@class NSString;

extern NSString *NSNibTopLevelObjects __attribute__((availability(macosx,introduced=10_0 ,deprecated=10_8,message="" )));
id NSNibOwner, topNibObjects;

@interface AppDelegate (SIEImport) // expected-error {{cannot find interface declaration for 'AppDelegate'}}

-(void)__attribute__((ibaction))importFromSIE:(id)sender;

@end

@implementation AppDelegate (SIEImport) // expected-error {{cannot find interface declaration for 'AppDelegate'}}

-(void)__attribute__((ibaction))importFromSIE:(id)sender {

 NSMutableArray *topNibObjects;
 NSDictionary *nibLoadDict = [NSDictionary dictionaryWithObjectsAndKeys:self, NSNibOwner, topNibObjects, NSNibTopLevelObjects, ((void *)0)];
}

@end

@interface Mixed
- (void)Meth1 __attribute__((availability(macosx,introduced=10.3_0))); // expected-warning {{use same version number separators '_' or '.'}}
- (void)Meth2 __attribute__((availability(macosx,introduced=10_3.1))); // expected-warning {{use same version number separators '_' or '.'}}
@end

@protocol P18804883
- (void)proto_method __attribute__((availability(macosx,introduced=10_1,deprecated=NA))); // means nothing (not deprecated)
@end

@interface A18804883 <P18804883>
- (void)interface_method __attribute__((availability(macosx,introduced=NA))); // expected-note {{'interface_method' has been explicitly marked unavailable here}}
- (void)strange_method __attribute__((availability(macosx,introduced=NA,deprecated=NA)));  // expected-note {{'strange_method' has been explicitly marked unavailable here}}
- (void) always_available __attribute__((availability(macosx,deprecated=NA)));
@end

void foo (A18804883* pa) {
  [pa interface_method]; // expected-error {{'interface_method' is unavailable: not available on macOS}}
  [pa proto_method];
  [pa strange_method]; // expected-error {{'strange_method' is unavailable: not available on macOS}}
  [pa always_available];
}

