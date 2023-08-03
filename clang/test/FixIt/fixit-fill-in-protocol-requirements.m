// RUN: %clang_cc1 -triple=x86_64-apple-macos10.10 -verify -Wno-objc-root-class -fallow-editor-placeholders %s
// RUN: %clang_cc1 -triple=x86_64-apple-macos10.10 -fdiagnostics-parseable-fixits -fallow-editor-placeholders %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-apple-macos10.12 -fdiagnostics-parseable-fixits -fallow-editor-placeholders %s 2>&1 | FileCheck --check-prefix=AVAILABLE %s

@protocol P1

- (void)p2Method;

@end

@protocol P2 <P1>

- (void)p1Method;

@end

@interface I <P2>

@end

@implementation I // expected-warning {{class 'I' does not conform to protocols 'P2' and 'P1'}} expected-note {{add stubs for missing protocol requirements}}

@end
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"- (void)p2Method { \n  <#code#>\n}\n\n- (void)p1Method { \n  <#code#>\n}\n\n"

@protocol P3

+ (void)p3ClassMethod;

@end

@interface I (Category) <P3>

@end

@implementation I (Category) // expected-warning {{category 'Category' does not conform to protocol 'P3'}} expected-note {{add stubs for missing protocol requirements}}

@end
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"+ (void)p3ClassMethod { \n  <#code#>\n}\n\n"

@protocol P4

- (void)anotherMethod;

@end

@interface ThreeProtocols <P2, P1, P3>
@end
@implementation ThreeProtocols // expected-warning {{class 'ThreeProtocols' does not conform to protocols 'P2', 'P1' and 'P3'}} expected-note {{add stubs for missing protocol requirements}}
@end

@interface FourProtocols <P2, P3, P4>
@end
@implementation FourProtocols // expected-warning {{class 'FourProtocols' does not conform to protocols 'P2', 'P1', 'P3', ...}} expected-note {{add stubs for missing protocol requirements}}
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

@implementation ImplementsAllAvailable // expected-warning {{class 'ImplementsAllAvailable' does not conform to protocol 'TakeAvailabilityIntoAccount'}} expected-note {{add stubs for missing protocol requirements}}

- (void)availableMethod { }
- (void)deprecatedMethod { }

@end

@interface FixitJustAvailable <TakeAvailabilityIntoAccount>
@end

@implementation FixitJustAvailable // expected-warning {{class 'FixitJustAvailable' does not conform to protocol 'TakeAvailabilityIntoAccount'}} expected-note {{add stubs for missing protocol requirements}}

@end
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"- (void)availableMethod { \n  <#code#>\n}\n\n- (void)deprecatedMethod { \n  <#code#>\n}\n\n"
// AVAILABLE: fix-it:{{.*}}:{[[@LINE-2]]:1-[[@LINE-2]]:1}:"- (void)availableMethod { \n  <#code#>\n}\n\n- (void)deprecatedMethod { \n  <#code#>\n}\n\n+ (void)notYetAvailableMethod { \n  <#code#>\n}\n\n"

@protocol PReq1
-(void)reqZ1;
@end

@protocol PReq2
-(void)reqZ1;
-(void)reqA2;
-(void)reqA1;
@end

// Ensure optional cannot hide required methods with the same selector.
@protocol POpt
@optional
-(void)reqZ1;
-(void)reqA2;
-(void)reqA1;
@end

@interface MultiReqOpt <PReq1, PReq2, POpt>
@end
@implementation MultiReqOpt // expected-warning {{class 'MultiReqOpt' does not conform to protocols 'PReq1' and 'PReq2'}} expected-note {{add stubs}}
@end
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"
// Z1 is first due to being from the first listed protocol, PReq1
// CHECK-SAME: - (void)reqZ1
// A1 is before A2 because of secondary sort by name.
// CHECK-SAME: - (void)reqA1
// CHECK-SAME: - (void)reqA2
