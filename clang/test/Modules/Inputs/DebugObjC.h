// -*- ObjC -*-
@class FwdDecl;

@interface ObjCClass {
  int ivar;
}
+ classMethod;
- instanceMethodWithInt:(int)i;
- (struct OpaqueData*) getSomethingOpaque;
@property int property;
@end

@interface ObjCClassWithPrivateIVars {
  int public_ivar;
}
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

struct FwdDeclared;
struct FwdDeclared {
  int i;
};
struct PureForwardDecl;

typedef union { int i; } TypedefUnion;
typedef enum { e1 = 1 } TypedefEnum;
typedef struct { int i; } TypedefStruct;

union { int i; } GlobalUnion;
struct { int i; } GlobalStruct;
enum { e2 = 2 } GlobalEnum;
