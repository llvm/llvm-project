// RUN: %clang_cc1  -fsyntax-only -Wduplicate-method-arg -verify -Wno-objc-root-class %s

// rdar://8877730

int object;

@class NSString, NSArray;

@interface Test 
- Func:(int)XXXX, id object; // expected-warning {{use of C-style parameters in Objective-C method declarations is deprecated}}

- doSomethingElseWith:(id)object;

- (NSString *)doSomethingWith:(NSString *)object and:(NSArray *)object; // expected-warning {{redeclaration of method parameter 'object'}} \
					  // expected-note {{previous declaration is here}}
@end

@implementation Test

- (NSString *)doSomethingWith:(NSString *)object and:(NSArray *)object // expected-warning {{redefinition of method parameter 'object'}} \
					  // expected-note {{previous declaration is here}}
{
    return object; // expected-warning {{incompatible pointer types returning 'NSArray *' from a function with result type 'NSString *'}}
}

- Func:(int)XXXX, id object { return object; } // expected-warning {{use of C-style parameters in Objective-C method declarations is deprecated}}

- doSomethingElseWith:(id)object { return object; }

@end

struct P;

@interface Test1
- doSomethingWith:(struct S *)object and:(struct P *)obj; // expected-warning {{declaration of 'struct S' will not be visible outside of this function}}
@end

int obj;
