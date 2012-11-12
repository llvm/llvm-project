// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

@protocol Proto
- (void) ProtoDidget;
@end

@protocol MyProto <Proto>
- (void) widget;
@end

@interface Foo 
- (void)StillMode;
@end

@interface Container
+ (void)MyMeth;
@end

@implementation Container
+ (void)MyMeth
{
  Foo *view;
  [(Foo <MyProto> *)view StillMode];
  [(Foo <MyProto> *)view widget];
  [(Foo <MyProto> *)view ProtoDidget];
}
@end
