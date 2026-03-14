// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o -

@protocol P
- (id<P>) Meth: (id<P>) Arg;
@end

@interface INTF<P>
- (id<P>)IMeth;
@end

@implementation INTF
- (id<P>)IMeth { return [(id<P>)self Meth: (id<P>)0]; }
- (id<P>) Meth : (id<P>) Arg { return 0; }
@end
