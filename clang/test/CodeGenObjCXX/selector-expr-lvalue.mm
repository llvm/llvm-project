// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5  -emit-llvm -o - %s 
// PR7390

@interface NSObject {}
- (void)respondsToSelector:(const SEL&)s : (SEL*)s1;
- (void) setPriority:(int)p;
- (void)Meth;
@end

@implementation  NSObject
- (void)Meth {
    [self respondsToSelector:@selector(setPriority:) : &@selector(setPriority:)];
}
- (void) setPriority:(int)p{}
- (void)respondsToSelector:(const SEL&)s : (SEL*)s1 {}
@end
