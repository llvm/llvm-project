// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@protocol P0
-bar;
@end

@interface A <P0>
@end

// Interface conforms to inherited protocol

@interface B0 : A <P0>
@end

@implementation B0
@end

// Interface conforms to a protocol which extends another. The other
// protocol is inherited, and extended methods are implemented.

@protocol P1 <P0>
-foo;
@end

@interface B1 : A <P1>
@end

@implementation B1
-foo { return 0; };
@end

// Interface conforms to a protocol whose methods are provided by an
// alternate inherited protocol.

@protocol P2
-bar;
@end

@interface B2 : A <P2>
@end

@implementation B2
@end

// Interface conforms to a protocol whose methods are provided by a base class.

@interface A1 
-bar;
@end

@interface B3 : A1 <P2>
@end

@implementation B3
@end

