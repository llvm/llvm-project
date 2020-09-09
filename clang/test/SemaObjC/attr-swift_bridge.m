// RUN: %clang_cc1 -verify -fsyntax-only %s

// expected-error@+1 {{'__swift_bridge__' attribute takes one argument}}
__attribute__((__swift_bridge__))
@interface I
@end

// expected-error@+1 {{'__swift_bridge__' attribute requires a string}}
__attribute__((__swift_bridge__(1)))
@interface J
@end

// expected-error@+1 {{'__swift_bridge__' attribute takes one argument}}
__attribute__((__swift_bridge__("K", 1)))
@interface K
@end

@interface L
// expected-error@+1 {{'__swift_bridge__' attribute only applies to tag types, typedefs, Objective-C interfaces, and Objective-C protocols}}
- (void)method __attribute__((__swift_bridge__("method")));
@end

__attribute__((__swift_bridge__("Array")))
@interface NSArray
@end

__attribute__((__swift_bridge__("ProtocolP")))
@protocol P
@end

typedef NSArray *NSArrayAlias __attribute__((__swift_bridge__("ArrayAlias")));

struct __attribute__((__swift_bridge__("StructT"))) T {};
