// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface Foo
- (int)  garf; // expected-note {{previous declaration is here}}
- (int) OK;
+ (int)  cgarf; // expected-note {{previous declaration is here}}
- (int)  InstMeth;
@end

@interface Foo()
- (void)  garf; // expected-error {{duplicate declaration of method 'garf'}}
+ (void)  cgarf; // expected-error {{duplicate declaration of method 'cgarf'}}
+ (int)  InstMeth;
- (int) OK;
@end

@class NSObject;

__attribute__((objc_root_class)) @interface AppDelegate
+ (void)someMethodWithArgument:(NSObject *)argument;
+ (void)someMethodWithArgument:(NSObject *)argument : (NSObject*) argument2; // expected-note {{previous declaration is here}}
@end

@interface AppDelegate ()
- (void)someMethodWithArgument:(float)argument; // OK. No warning to be issued here.
+ (void)someMethodWithArgument:(float)argument : (float)argument2; // expected-error {{duplicate declaration of method 'someMethodWithArgument::'}}
@end

__attribute__((objc_root_class))
@interface Test
-(void)meth; // expected-note {{declared here}}
@end

@interface Test()
-(void)meth;
@end

@implementation Test // expected-warning {{method definition for 'meth' not found}}
@end
