// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://9091389

@protocol Fooable
- (void)foo;
@end

@protocol SubFooable <Fooable>
@end

@interface AClass
@end

@interface BClass : AClass <SubFooable>
@end

@implementation BClass
- (void)foo {
}
@end

void functionTakingAClassConformingToAProtocol(AClass <Fooable> *instance) { // expected-note {{passing argument to parameter 'instance' here}}
}

int main () {
    AClass *aobject = 0;
    BClass *bobject = 0;
    functionTakingAClassConformingToAProtocol(aobject);  // expected-warning {{incompatible pointer types passing 'AClass *' to parameter of type 'AClass<Fooable> *'}}
    functionTakingAClassConformingToAProtocol(bobject); // Shouldn't warn -  does implement Fooable
    return 0;
}

// rdar://9267196
@interface NSObject @end

@protocol MyProtocol
@end

@interface MyClass : NSObject 
{
}
@end

@implementation MyClass
@end

@interface MySubclass : MyClass <MyProtocol> 
{
}
@end

@interface MyTestClass : NSObject
{
@private
	NSObject <MyProtocol> *someObj;
}

@property (nonatomic, assign) NSObject <MyProtocol> *someObj;

@end

@implementation MyTestClass

@synthesize someObj;

- (void)someMethod
{
	MySubclass *foo;
	[self setSomeObj:foo]; // no warning here!
}

@end
