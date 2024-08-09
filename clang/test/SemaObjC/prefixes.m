// RUN: %clang_cc1 %s -Wobjc-prefix-length=2 -fsyntax-only -verify

// Test prefix length rules for ObjC interfaces and protocols

// -- Plain interfaces --------------------------------------------------------

@interface _Foo // expected-warning {{un-prefixed Objective-C class name}}
@end

@interface Foo // expected-warning {{un-prefixed Objective-C class name}}
@end

@interface NSFoo
@end

@interface _NSFoo
@end

@interface __NSFoo
@end

// Special case for prefix-length 2
@interface NSCFFoo
@end

@interface _NSCFFoo
@end

@interface NSCFXFoo // expected-warning {{un-prefixed Objective-C class name}}
@end

@interface NSXFoo // expected-warning {{un-prefixed Objective-C class name}}
@end

// -- Categories --------------------------------------------------------------

// Categories don't trigger these warnings, but methods in categories that
// aren't appropriately prefixed are required to be appropriately prefixed

@interface Foo (Bar)

@property int flibble; // expected-warning {{un-prefixed Objective-C method name on category}}
@property(getter=theFlibble) int NS_flibble; // expected-warning {{un-prefixed Objective-C method name on category}}
@property(setter=setTheFlibble:) int NS_flibble2; // expected-warning {{un-prefixed Objective-C method name on category}}
@property int NS_flibble3;

- (void)bar; // expected-warning {{un-prefixed Objective-C method name on category}}
- (void)NSbar;
- (void)NS_bar;
- (void)NSCF_bar;
- (void)NSXbar; // expected-warning {{un-prefixed Objective-C method name on category}}

@end

@interface NSFoo (Bar)

@property int flibble;
@property(getter=theFlibble) int NS_flibble;
@property(setter=setTheFlibble:) int NS_flibble2;
@property int NS_flibble3;

- (void)bar;
- (void)NSbar;
- (void)NS_bar;
- (void)NSCF_bar;
- (void)NSXbar;

@end

@interface NSCFFoo (Bar)

@property int flibble;
@property(getter=theFlibble) int NS_flibble;
@property(setter=setTheFlibble:) int NS_flibble2;
@property int NS_flibble3;

- (void)bar;
- (void)NSbar;
- (void)NS_bar;
- (void)NSCF_bar;
- (void)NSXbar;

@end

@interface NSXFoo (Bar)

@property int flibble; // expected-warning {{un-prefixed Objective-C method name on category}}
@property(getter=theFlibble) int NS_flibble; // expected-warning {{un-prefixed Objective-C method name on category}}
@property(setter=setTheFlibble:) int NS_flibble2; // expected-warning {{un-prefixed Objective-C method name on category}}
@property int NS_flibble3;

- (void)bar; // expected-warning {{un-prefixed Objective-C method name on category}}
- (void)NSbar;
- (void)NS_bar;
- (void)NSCF_bar;
- (void)NSXbar; // expected-warning {{un-prefixed Objective-C method name on category}}

@end

// -- Protocols ---------------------------------------------------------------

@protocol _FooProtocol // expected-warning {{un-prefixed Objective-C protocol name}}
@end

@protocol FooProtocol // expected-warning {{un-prefixed Objective-C protocol name}}
@end

@protocol NSFooProtocol
@end

@protocol _NSFooProtocol
@end

@protocol __NSFooProtocol
@end

// Special case for prefix-length 2
@protocol NSCFFooProtocol
@end

@protocol _NSCFFooProtocol
@end

@protocol NSCFXFooProtocol // expected-warning {{un-prefixed Objective-C protocol name}}
@end

@protocol NSXFooProtocol // expected-warning {{un-prefixed Objective-C protocol name}}
@end

