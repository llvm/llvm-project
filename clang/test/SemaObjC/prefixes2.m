// RUN: %clang_cc1 %s -Wobjc-prefixes=NS,NSCF,NSURL -Wobjc-forbidden-prefixes=XX -fsyntax-only -verify

// Test prefix list rules

@interface Foo // expected-warning {{Objective-C class name prefix not in permitted list}}
@end

@interface NSFoo
@end

@interface NSfoo // expected-warning {{Objective-C class name prefix not in permitted list}}
@end

@interface NSFFoo // expected-warning {{Objective-C class name prefix not in permitted list}}
@end

@interface NSCFFoo
@end

@interface NSURL
@end

@interface NSURLFoo
@end

@interface NSRGBColor // expected-warning {{Objective-C class name prefix not in permitted list}}
@end

@protocol NSRGBColorProtocol // expected-warning {{Objective-C protocol name prefix not in permitted list}}
@end

@interface XXFoo // expected-warning {{Objective-C class name prefix in forbidden list}}
@end

@protocol XXFooProtocol // expected-warning {{Objective-C protocol name prefix in forbidden list}}
@end

@interface Foo (Bar)

@property int flibble; // expected-warning {{Objective-C category method name prefix not in permitted list}}
@property(getter=theFlibble) int NS_flibble; // expected-warning {{Objective-C category method name prefix not in permitted list}}
@property(setter=setTheFlibble:) int NS_flibble2; // expected-warning {{Objective-C category method name prefix not in permitted list}}
@property int NS_flibble3;
@property int XX_flibble; // expected-warning {{Objective-C category method name prefix in forbidden list}}

- (void)bar; // expected-warning {{Objective-C category method name prefix not in permitted list}}
- (void)NSbar;
- (void)NS_bar;
- (void)NSCF_bar;
- (void)ns_bar;
- (void)nsBar;
- (void)nsbar; // expected-warning {{Objective-C category method name prefix not in permitted list}}
- (void)NSXbar; // expected-warning {{Objective-C category method name prefix not in permitted list}}
- (void)nsxBar; // expected-warning {{Objective-C category method name prefix not in permitted list}}
- (void)XXbar; // expected-warning {{Objective-C category method name prefix in forbidden list}}

// "set" prefix is ignored
- (void)setBar:(int)x; // expected-warning {{Objective-C category method name prefix not in permitted list}}
- (void)set_bar:(int)x; // expected-warning {{Objective-C category method name prefix not in permitted list}}
- (void)setNSbar:(int)x;
- (void)set_NSbar:(int)x;

// "is" prefix is ignored
- (int)isNSbar;
- (int)is_NSbar;
- (int)is_ns_bar;
- (int)is_bar; // expected-warning {{Objective-C category method name prefix not in permitted list}}
- (int)isBar; // expected-warning {{Objective-C category method name prefix not in permitted list}}

@end
