// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wno-objc-root-class -Wno-incomplete-implementation -triple x86_64-apple-macosx10.10.0 -verify %s

struct S {
  int throw; // expected-error {{expected member name or ';' after declaration specifiers; 'throw' is a keyword in Objective-C++}}
};

@interface class // expected-error {{expected identifier; 'class' is a keyword in Objective-C++}}
@end

@interface Bar: class // expected-error {{expected identifier; 'class' is a keyword in Objective-C++}}
@end

@protocol P // ok
@end

@protocol new // expected-error {{expected identifier; 'new' is a keyword in Objective-C++}}
@end

@protocol P2;
@protocol delete // expected-error {{expected identifier; 'delete' is a keyword in Objective-C++}}
@end

@class Foo, try; // expected-error {{expected identifier; 'try' is a keyword in Objective-C++}}

@interface Foo

@property (readwrite, nonatomic) int a, b, throw; // expected-error {{expected member name or ';' after declaration specifiers; 'throw' is a keyword in Objective-C++}}

-foo:(int)class; // expected-error {{expected identifier; 'class' is a keyword in Objective-C++}}
+foo:(int)constexpr; // expected-error {{expected identifier; 'constexpr' is a keyword in Objective-C++}}

@end

@interface Foo () <P, new> // expected-error {{expected identifier; 'new' is a keyword in Objective-C++}}
@end

@implementation Foo

@synthesize a = _a; // ok
@synthesize b = virtual; // expected-error {{expected identifier; 'virtual' is a keyword in Objective-C++}}

@dynamic throw; // expected-error {{expected identifier; 'throw' is a keyword in Objective-C++}}

-foo:(int)class { // expected-error {{expected identifier; 'class' is a keyword in Objective-C++}}
}

@end

@implementation class // expected-error {{expected identifier; 'class' is a keyword in Objective-C++}}
@end

@implementation Bar: class // expected-error {{expected identifier; 'class' is a keyword in Objective-C++}}
@end

@compatibility_alias C Foo; // ok
@compatibility_alias const_cast Bar; // expected-error {{expected identifier; 'const_cast' is a keyword in Objective-C++}}
@compatibility_alias C2 class; // expected-error {{expected identifier; 'class' is a keyword in Objective-C++}}

void func() {
  (void)@protocol(P); // ok
  (void)@protocol(delete); // expected-error {{expected identifier; 'delete' is a keyword in Objective-C++}}
}
