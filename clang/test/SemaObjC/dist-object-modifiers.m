// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s

@protocol P
- (bycopy id)serverPID; // expected-note {{previous declaration is here}}
- (void)doStuff:(bycopy id)clientId; // expected-note {{previous declaration is here}}
- (bycopy id)Ok;
+ (oneway id) stillMore : (byref id)Arg : (bycopy oneway id)Arg1;  // expected-note 3 {{previous declaration is here}}
@end

@interface I <P>
- (id)Ok;
@end

@implementation I
- (id)serverPID { return 0; } // expected-warning {{conflicting distributed object modifiers on return type in implementation of 'serverPID'}}
- (void)doStuff:(id)clientId { } // expected-warning {{conflicting distributed object modifiers on parameter type in implementation of 'doStuff:'}}
- (bycopy id)Ok { return 0; }
+ (id) stillMore : (id)Arg  : (bycopy id)Arg1 { return Arg; } // expected-warning {{conflicting distributed object modifiers on return type in implementation of 'stillMore::'}} \
                                                              // expected-warning 2{{conflicting distributed object modifiers on parameter type in implementation of 'stillMore::'}}
@end
