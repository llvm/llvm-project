// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface StopAtAtEnd
// This used to eat the @end
int 123 // expected-error{{expected unqualified-id}}
@end

@implementation StopAtAtEnd // no-warning
int 123 // expected-error{{expected unqualified-id}}
@end


@interface StopAtMethodDecls
// This used to eat the method declarations
int 123 // expected-error{{expected unqualified-id}}
- (void)foo; // expected-note{{here}}
int 456 // expected-error{{expected unqualified-id}}
+ (void)bar; // expected-note{{here}}
@end

@implementation StopAtMethodDecls
int 123 // expected-error{{expected unqualified-id}}
- (id)foo {} // expected-warning{{conflicting return type}}
int 456 // expected-error{{expected unqualified-id}}
+ (id)bar {} // expected-warning{{conflicting return type}}
@end


@interface EmbeddedNamespace
// This used to cause an infinite loop.
namespace NS { // expected-error{{expected unqualified-id}}
}
- (id)test; // expected-note{{here}}
@end

@implementation EmbeddedNamespace
int 123 // expected-error{{expected unqualified-id}}
// We should still stop here and parse this namespace.
namespace NS {
  void foo();
}

// Make sure the declaration of -test was recognized.
- (void)test { // expected-warning{{conflicting return type}}
  // Make sure the declaration of NS::foo was recognized.
  NS::foo();
}

@end


@protocol ProtocolWithEmbeddedNamespace
namespace NS { // expected-error{{expected unqualified-id}}

}
- (void)PWEN_foo; // expected-note{{here}}
@end

@interface ImplementPWEN <ProtocolWithEmbeddedNamespace>
@end

@implementation ImplementPWEN
- (id)PWEN_foo {} // expected-warning{{conflicting return type}}
@end
