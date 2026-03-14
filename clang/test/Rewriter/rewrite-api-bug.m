// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

@interface MyDerived
- (void) instanceMethod;
@end

@implementation MyDerived
- (void) instanceMethod {
}
@end

