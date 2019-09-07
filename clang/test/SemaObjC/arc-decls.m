// RUN: %clang_cc1 -fsyntax-only -fblocks -fobjc-arc -verify -Wno-objc-root-class %s

// rdar://8843524

struct A {
  id x[4];
  id y;
};

union u {
  id u;
};

// Volatile fields are fine.
struct C {
  volatile int x[4];
  volatile int y;
};

union u_trivial_c {
  volatile int b;
  struct C c;
};

@interface I {
   struct A a; 
   struct B {
    id y[10][20];
    id z;
   } b;

   union u c; 
};
@end

// rdar://10260525
struct r10260525 {
  id (^block) ();
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
// rdar://15757510

@interface J
@property (retain) id newFoo; // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}} expected-note{{explicitly declare getter '-newFoo' with '__attribute__((objc_method_family(none)))' to return an 'unowned' object}}
@property (strong) id copyBar;  // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}} expected-note{{explicitly declare getter '-copyBar' with '__attribute__((objc_method_family(none)))' to return an 'unowned' object}}
@property (copy) id allocBaz; // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}} expected-note{{explicitly declare getter '-allocBaz' with '__attribute__((objc_method_family(none)))' to return an 'unowned' object}}
@property (copy, nonatomic) id new;
@property (retain) id newDFoo; // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}} expected-note{{explicitly declare getter '-newDFoo' with '__attribute__((objc_method_family(none)))' to return an 'unowned' object}}
@property (strong) id copyDBar; // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}} expected-note{{explicitly declare getter '-copyDBar' with '__attribute__((objc_method_family(none)))' to return an 'unowned' object}}
@property (copy) id allocDBaz; // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}} expected-note{{explicitly declare getter '-allocDBaz' with '__attribute__((objc_method_family(none)))' to return an 'unowned' object}}
@end

@implementation J
@synthesize newFoo;
@synthesize copyBar;
@synthesize allocBaz;
@synthesize new;
- new {return 0; };

@dynamic newDFoo;
@dynamic copyDBar; 
@dynamic allocDBaz;
@end


@interface MethodFamilyDiags
@property (retain) id newFoo; // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}}
- (id)newFoo; // expected-note {{explicitly declare getter '-newFoo' with '__attribute__((objc_method_family(none)))' to return an 'unowned' object}}

#define OBJC_METHOD_FAMILY_NONE __attribute__((objc_method_family(none)))
- (id)newBar; // expected-note {{explicitly declare getter '-newBar' with 'OBJC_METHOD_FAMILY_NONE' to return an 'unowned' object}}
@property (retain) id newBar; // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}}

@property (retain) id newBaz; // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}} expected-note {{explicitly declare getter '-newBaz' with 'OBJC_METHOD_FAMILY_NONE' to return an 'unowned' object}}
#undef OBJC_METHOD_FAMILY_NONE

@property (retain, readonly) id newGarply; // expected-error {{property follows Cocoa naming convention for returning 'owned' objects}} expected-note {{explicitly declare getter '-newGarply' with '__attribute__((objc_method_family(none)))' to return an 'unowned' object}}
@end

@interface MethodFamilyDiags (Redeclarations)
- (id)newGarply; // no note here
@end

@implementation MethodFamilyDiags
@synthesize newGarply;
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

void func(void) __attribute__((objc_ownership(none)));  // expected-warning {{'objc_ownership' only applies to Objective-C object or block pointer types; type here is 'void (void)'}}
struct __attribute__((objc_ownership(none))) S2 {}; // expected-error {{'objc_ownership' attribute only applies to variables}}
@interface I2
    @property __attribute__((objc_ownership(frob))) id i; // expected-warning {{'objc_ownership' attribute argument not supported: 'frob'}}
@end

// rdar://15304886
@interface NSObject @end

@interface ControllerClass : NSObject @end

@interface SomeClassOwnedByController
@property (readonly) ControllerClass *controller; // expected-note {{property declared here}}

// rdar://15465916
@property (readonly, weak) ControllerClass *weak_controller;
@end

@interface SomeClassOwnedByController ()
@property (readwrite, weak) ControllerClass *controller; // expected-warning {{primary property declaration is implicitly strong while redeclaration in class extension is weak}}

@property (readwrite, weak) ControllerClass *weak_controller;
@end

@interface I3
@end

@interface D3 : I3
@end

@interface D3 (Cat1)
- (id)method;
@end

@interface I3 (Cat2)
// FIXME: clang should diagnose mismatch between methods in D3(Cat1) and
//        I3(Cat2).
- (id)method __attribute__((ns_returns_retained));
@end

@implementation D3
- (id)method {
  return (id)0;
}
@end
