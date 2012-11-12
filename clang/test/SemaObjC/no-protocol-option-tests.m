// RUN: %clang_cc1 -fsyntax-only -Wno-protocol -verify -Wno-objc-root-class %s
// rdar: // 7056600

@protocol P
- PMeth;
@end

// Test1
@interface I  <P> @end
@implementation I @end //  no warning with -Wno-protocol

// Test2
@interface C -PMeth; @end
@interface C (Category) <P> @end
@implementation C (Category) @end //  no warning with -Wno-protocol

// Test2
@interface super - PMeth; @end
@interface J : super <P>
- PMeth;	// expected-note {{method definition for 'PMeth' not found}}
@end
@implementation J @end	// expected-warning {{incomplete implementation}}

// Test3
@interface K : super <P>
@end
@implementation K @end // no warning with -Wno-protocol

// Test4
@interface Root @end
@interface L : Root<P> @end
@implementation L @end // no warning with -Wno-protocol
