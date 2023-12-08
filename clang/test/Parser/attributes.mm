// RUN: %clang_cc1 -verify -fsyntax-only -Wno-objc-root-class %s

// FIXME: Why isn't this supported? Seems useful for availability attributes at
// the very least.
__attribute__((deprecated)) @class B; // expected-error {{prefix attribute must be followed by an interface, protocol, or implementation}}

__attribute__((deprecated)) @interface A @end
__attribute__((deprecated)) @protocol P0;
__attribute__((deprecated)) @protocol P1
@end

#define EXP __attribute__((visibility("default")))
class EXP C {};
EXP class C2 {}; // expected-warning {{attribute 'visibility' is ignored, place it after "class" to apply attribute to type declaration}}

@interface EXP I @end // expected-error {{postfix attributes are not allowed on Objective-C directives, place them in front of '@interface'}}
EXP @interface I2 @end

@implementation EXP I @end // expected-error-re {{postfix attributes are not allowed on Objective-C directives{{$}}}}
EXP @implementation I2 @end

@class EXP OC; // expected-error-re {{postfix attributes are not allowed on Objective-C directives{{$}}}}
EXP @class OC2; // expected-error {{prefix attribute must be followed by an interface, protocol, or implementation}}

@protocol EXP P @end // expected-error {{postfix attributes are not allowed on Objective-C directives, place them in front of '@protocol'}}
EXP @protocol P2 @end
