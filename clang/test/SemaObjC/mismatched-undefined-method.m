// RUN: %clang_cc1  -fsyntax-only -Wno-deprecated-declarations -verify %s

typedef unsigned int CGDirectDisplayID;

@interface NSObject @end

@interface BrightnessAssistant : NSObject {}
- (void)BrightnessAssistantUnregisterForNotifications:(void*) observer; // expected-note {{previous definition is here}}
@end
@implementation BrightnessAssistant // expected-note {{implementation started here}}
- (void)BrightnessAssistantUnregisterForNotifications:(CGDirectDisplayID) displayID, void* observer // expected-warning {{conflicting parameter types in implementation of 'BrightnessAssistantUnregisterForNotifications:': 'void *' vs 'CGDirectDisplayID'}}
@end // expected-error {{expected method body}} // expected-error {{missing '@end'}}
