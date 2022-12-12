// RUN: %clang_cc1 -emit-llvm -o %t %s

@interface BaseClass {
    id _delegate;
}
@end

@protocol MyProtocol
@optional
@property(assign) id delegate;
@end

@protocol AnotherProtocol
@optional
@property(assign) id myanother;
@end

@protocol SubProtocol <MyProtocol>
@property(assign) id another;
@end

@interface SubClass : BaseClass <SubProtocol, AnotherProtocol> {
}

@end

@implementation BaseClass @end 

@implementation SubClass
@synthesize delegate = _Subdelegate;
@synthesize another;
@synthesize myanother;
@end
