// RUN: %clang_cc1 -Wmethod-signatures -fsyntax-only -verify -Wno-objc-root-class %s

@interface foo
- (int)meth;
@end

@implementation foo
- (int) meth { return [self meth]; }
@end

// PR2708
@interface MyClass
+- (void)myMethod;   // expected-error {{expected selector for Objective-C method}}
- (vid)myMethod2;    // expected-error {{expected a type}}
@end

@implementation MyClass
- (void)myMethod { }
- (vid)myMethod2 { }	// expected-error {{expected a type}}

@end


@protocol proto;
@protocol NSObject;

//@protocol GrowlPluginHandler <NSObject> @end


@interface SomeClass2
- (int)myMethod1: (id<proto>)
arg; // expected-note {{previous definition is here}}
@end

@implementation SomeClass2
- (int)myMethod1: (id<NSObject>)
  arg { // expected-warning {{conflicting parameter types in implementation of 'myMethod1:': 'id<proto>' vs 'id<NSObject>'}}
  
}
@end
