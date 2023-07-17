// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

/* This test is for categories which don't implement the accessors but some accessors are
   implemented in their base class implementation. In this case,no warning must be issued.
*/

@interface MyClass 
{
    int        _foo;
}
@property(readonly)    int        foo;
@end

@implementation MyClass
- (int) foo        { return _foo; }
@end

@interface MyClass (private)
@property(readwrite)    int        foo;
@end

@implementation MyClass (private)
- (void) setFoo:(int)foo    { _foo = foo; }
@end

@interface MyClass (public)
@property(readwrite)    int        foo;	
@end

@implementation MyClass (public)
@end 

// rdar://12568064
// No warn of unimplemented property of protocols in category,
// when those properties will be implemented in category's primary
// class or one of its super classes.
@interface HBSuperclass
@property (nonatomic) char myProperty;
@property (nonatomic) char myProperty2;
@end

@interface HBClass : HBSuperclass
@end

@protocol HBProtocol
@property (nonatomic) char myProperty;
@property (nonatomic) char myProperty2;
@end

@interface HBSuperclass (HBSCategory)<HBProtocol>
@end

@implementation HBSuperclass (HBSCategory)
@end

@interface HBClass (HBCategory)<HBProtocol>
@end

@implementation HBClass (HBCategory)
@end
