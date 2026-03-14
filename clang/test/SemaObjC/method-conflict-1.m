// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

// This test case tests the default behavior.

@interface NSObject @end

@interface NSArray : NSObject @end

@interface MyClass : NSObject {
}
- (void)myMethod:(NSArray *)object;
- (void)myMethod1:(NSObject *)object; // broken-note {{previous definition is here}}
@end

@implementation MyClass
- (void)myMethod:(NSObject *)object {
}
- (void)myMethod1:(NSArray *)object { // broken-warning {{conflicting parameter types in implementation of 'myMethod1:': 'NSObject *' vs 'NSArray *'}}
}
@end


@protocol MyProtocol @end

@interface MyOtherClass : NSObject <MyProtocol> {
}
- (void)myMethod:(id <MyProtocol>)object; // broken-note {{previous definition is here}}
- (void)myMethod1:(id <MyProtocol>)object; // broken-note {{previous definition is here}}
@end

@implementation MyOtherClass
- (void)myMethod:(MyClass *)object { // broken-warning {{conflicting parameter types in implementation of 'myMethod:': 'id<MyProtocol>' vs 'MyClass *'}}
}
- (void)myMethod1:(MyClass<MyProtocol> *)object { // broken-warning {{conflicting parameter types in implementation of 'myMethod1:': 'id<MyProtocol>' vs 'MyClass<MyProtocol> *'}}
}
@end



@interface A @end
@interface B : A @end

@interface Test1 {}
- (void) test1:(A*) object; // broken-note {{previous definition is here}} 
- (void) test2:(B*) object;
@end

@implementation Test1
- (void) test1:(B*) object {} // broken-warning {{conflicting parameter types in implementation of 'test1:': 'A *' vs 'B *'}}
- (void) test2:(A*) object {}
@end

// wants id -> A* to be an exception
@interface Test2 {}
- (void) test1:(id) object; // broken-note {{previous definition is here}} 
- (void) test2:(A*) object;
@end
@implementation Test2
- (void) test1:(A*) object {} // broken-warning {{conflicting parameter types in implementation of 'test1:': 'id' vs 'A *'}}
- (void) test2:(id) object {}
@end

@interface Test3 {}
- (A*) test1;
- (B*) test2; // broken-note {{previous definition is here}} 
@end

@implementation Test3
- (B*) test1 { return 0; }
- (A*) test2 { return 0; } // broken-warning {{conflicting return type in implementation of 'test2': 'B *' vs 'A *'}}
@end

// The particular case of overriding with an id return is permitted.
@interface Test4 {}
- (id) test1;
- (A*) test2;
@end
@implementation Test4
- (A*) test1 { return 0; }
- (id) test2 { return 0; }
@end
