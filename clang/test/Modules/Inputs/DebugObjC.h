@class FwdDecl;

@interface ObjCClass {
  int ivar;
}
+ classMethod;
- instanceMethodWithInt:(int)i;
- (struct OpaqueData*) getSomethingOpaque;
@property int property;
@end

@interface ObjCClass (Category)
- categoryMethod;
@end

@protocol ObjCProtocol

typedef enum {
  e0 = 0
}  InnerEnum;

+ (InnerEnum)protocolMethod;

@end
