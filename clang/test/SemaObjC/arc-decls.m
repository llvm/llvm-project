// RUN: %clang_cc1 -fsyntax-only -fblocks -fobjc-arc -verify -Wno-objc-root-class %s

// rdar://8843524

struct A {
    id x; // expected-error {{ARC forbids Objective-C objects in structs or unions}}
};

union u {
    id u; // expected-error {{ARC forbids Objective-C objects in structs or unions}}
};

@interface I {
   struct A a; 
   struct B {
    id y[10][20]; // expected-error {{ARC forbids Objective-C objects in structs or unions}}
    id z;
   } b;

   union u c; 
};
@end

// rdar://10260525
struct r10260525 {
  id (^block) (); // expected-error {{ARC forbids blocks in structs or unions}}
};

struct S { 
    id __attribute__((objc_ownership(none))) i;
    void * vp;
    int i1;
};

// rdar://9046528

@class NSError;

__autoreleasing id X; // expected-error {{global variables cannot have __autoreleasing ownership}}
__autoreleasing NSError *E; // expected-error {{global variables cannot have __autoreleasing ownership}}


extern id __autoreleasing X1; // expected-error {{global variables cannot have __autoreleasing ownership}}

void func()
{
    id X;
    static id __autoreleasing X1; // expected-error {{global variables cannot have __autoreleasing ownership}}
    extern id __autoreleasing E; // expected-error {{global variables cannot have __autoreleasing ownership}}

}

// rdar://9157348

@interface J
@property (retain) id newFoo; // expected-note {{property declared here}}
@property (strong) id copyBar; // expected-note {{property declared here}}
@property (copy) id allocBaz; // expected-note {{property declared here}}
@property (copy, nonatomic) id new;
@end

@implementation J
@synthesize newFoo;	// expected-error {{property's synthesized getter follows Cocoa naming convention for returning}}
@synthesize copyBar;	// expected-error {{property's synthesized getter follows Cocoa naming convention for returning}}
@synthesize allocBaz;	// expected-error {{property's synthesized getter follows Cocoa naming convention for returning}}
@synthesize new;
- new {return 0; };
@end


// rdar://10187884
@interface Super
- (void)bar:(id)b; // expected-note {{parameter declared here}}
- (void)bar1:(id) __attribute((ns_consumed)) b;
- (void)ok:(id) __attribute((ns_consumed)) b;
- (id)ns_non; // expected-note {{method declared here}}
- (id)not_ret:(id) b __attribute((ns_returns_not_retained)); // expected-note {{method declared here}}
- (id)both__returns_not_retained:(id) b __attribute((ns_returns_not_retained));
@end

@interface Sub : Super
- (void)bar:(id) __attribute((ns_consumed)) b; // expected-error {{overriding method has mismatched ns_consumed attribute on its parameter}}
- (void)bar1:(id)b;
- (void)ok:(id) __attribute((ns_consumed)) b;
- (id)ns_non __attribute((ns_returns_not_retained)); // expected-error {{overriding method has mismatched ns_returns_not_retained attributes}}
- (id)not_ret:(id) b __attribute((ns_returns_retained)); // expected-error {{overriding method has mismatched ns_returns_retained attributes}}
- (id)both__returns_not_retained:(id) b __attribute((ns_returns_not_retained));
// rdar://12173491
@property (copy, nonatomic) __attribute__((ns_returns_retained)) id (^fblock)(void);
@end

// Test that we give a good diagnostic here that mentions the missing
// ownership qualifier.  We don't want this to get suppressed because
// of an invalid conversion.
void test7(void) {
  id x;
  id *px = &x; // expected-error {{pointer to non-const type 'id' with no explicit ownership}}

  I *y;
  J **py = &y; // expected-error {{pointer to non-const type 'J *' with no explicit ownership}} expected-warning {{incompatible pointer types initializing}}
}
