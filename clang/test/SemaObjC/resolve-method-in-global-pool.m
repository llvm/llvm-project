// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -fblocks -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fsyntax-only -verify -fblocks -Wno-objc-root-class %s
// expected-no-diagnostics

@interface NSObject 
+ (void)clsMethod:(int*)arg;
@end

@class NSDictionary;
@class NSError;

@interface Foo : NSObject
- (void)getDonuts:(void (^)(NSDictionary *, NSError *))replyBlock;
- (void)getCake:(int*)arg, ...;
@end

@protocol Protocol
@required
- (void)getDonuts:(void (^)(NSDictionary *))replyBlock;
- (void)getCake:(float*)arg, ...;
+ (void)clsMethod:(float*)arg;
@end

@implementation Foo
{
  float g;
}

- (void)getDonuts:(void (^)(NSDictionary *, NSError *))replyBlock {
    [(id) 0 getDonuts:^(NSDictionary *replyDict) { }];
}

- (void) getCake:(int*)arg, ... {
    [(id)0 getCake: &g, 1,3.14];
}
@end

void func( Class c, float g ) {
    [c clsMethod: &g];
}

@protocol NSKeyedArchiverDelegate @end

@interface NSKeyedArchiver
@property (assign) id <NSKeyedArchiverDelegate> delegate;
@end

@interface NSConnection
@property (assign) id delegate;
@end

extern id NSApp;

@interface AppDelegate
@end

AppDelegate* GetDelegate(void)
{
    return [NSApp delegate];
}
