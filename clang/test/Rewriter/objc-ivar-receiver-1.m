// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -
// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o - | grep 'newInv->_container'

@interface NSMutableArray 
- (void)addObject:(id)addObject;
@end

@interface NSInvocation {
@private
    id _container;
}
+ (NSInvocation *)invocationWithMethodSignature;

@end

@implementation NSInvocation

+ (NSInvocation *)invocationWithMethodSignature {
    NSInvocation *newInv;
    id obj = newInv->_container;
    [newInv->_container addObject:0];
   return 0;
}
@end
