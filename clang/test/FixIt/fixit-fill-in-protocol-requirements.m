// RUN: %clang_cc1 -triple=x86_64-apple-macos10.10 -verify -Wno-objc-root-class -serialize-diagnostic-file /dev/null %s
// RUN: %clang_cc1 -triple=x86_64-apple-macos10.10 -fdiagnostics-parseable-fixits -serialize-diagnostic-file /dev/null %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-apple-macos10.12 -fdiagnostics-parseable-fixits -serialize-diagnostic-file /dev/null %s 2>&1 | FileCheck --check-prefix=AVAILABLE %s

@protocol P1

- (void)p2Method;

@end

@protocol P2 <P1>

- (void)p1Method;

@end

@interface I <P2>

@end

@implementation I // expected-warning {{warning: class 'I' does not conform to protocols 'P2' and 'P1'}} expected-note {{add stubs for missing protocol requirements}}

@end
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"- (void)p2Method { \n  <#code#>\n}\n\n- (void)p1Method { \n  <#code#>\n}\n\n"

@protocol P3

+ (void)p3ClassMethod;

@end

@interface I (Category) <P3> // expected-warning {{warning: category 'Category' does not conform to protocols 'P3'}} expected-note {{add stubs for missing protocol requirements}}

@end

@implementation I (Category)

@end
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"+ (void)p3ClassMethod { \n  <#code#>\n}\n\n"

@protocol P4

- (void)anotherMethod;

@end

@interface ThreeProtocols <P2, P1, P3> // expected-warning {{class 'ThreeProtocols' does not conform to protocols 'P2', 'P1' and 'P3'}} expected-note {{add stubs for missing protocol requirements}}
@end
@implementation ThreeProtocols
@end

@interface FourProtocols <P2, P3, P4> // expected-warning {{class 'ThreeProtocols' does not conform to protocols 'P2', 'P1', 'P3', ...}} expected-note {{add stubs for missing protocol requirements}}
@end
@implementation FourProtocols
@end

// Unavailable methods
@protocol TakeAvailabilityIntoAccount

- (void)unavailableMethod __attribute__((availability(macos,unavailable)));
+ (void)notYetAvailableMethod __attribute__((availability(macos,introduced=10.11)));
- (void)availableMethod;
- (void)deprecatedMethod __attribute__((availability(macos,introduced=10.0, deprecated=10.6)));

@end

@interface ImplementsAllAvailable <TakeAvailabilityIntoAccount>
@end

@implementation ImplementsAllAvailable // No warning!

- (void)availableMethod { }
- (void)deprecatedMethod { }

@end

@interface FixitJustAvailable <TakeAvailabilityIntoAccount>
@end

@implementation FixitJustAvailable // expected-warning {{class 'FixitJustAvailable' does not conform to protocol 'TakeAvailabilityIntoAccount'}} expected-note {{add stubs for missing protocol requirements}}

@end
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"- (void)availableMethod { \n  <#code#>\n}\n\n- (void)deprecatedMethod { \n  <#code#>\n}\n\n"
// AVAILABLE: fix-it:{{.*}}:{[[@LINE-2]]:1-[[@LINE-2]]:1}:"- (void)availableMethod { \n  <#code#>\n}\n\n- (void)deprecatedMethod { \n  <#code#>\n}\n\n+ (void)notYetAvailableMethod { \n  <#code#>\n}\n\n"
