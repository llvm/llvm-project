// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o - | grep objc_msgSendSuper | grep MainMethod

typedef struct objc_selector    *SEL;
typedef struct objc_object *id;

@interface SUPER
- (int) MainMethod;
@end

@interface MyDerived : SUPER
- (int) instanceMethod;
@end

@implementation MyDerived 
- (int) instanceMethod {
  return [super MainMethod];
}
@end
