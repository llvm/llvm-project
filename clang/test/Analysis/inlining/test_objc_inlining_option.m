// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-ipa=dynamic-bifurcate -analyzer-config objc-inlining=false -verify %s
// expected-no-diagnostics

typedef signed char BOOL;
typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end

// Vanila: ObjC class method is called by name.
@interface MyParent : NSObject
+ (int)getInt;
@end
@interface MyClass : MyParent
+ (int)getInt;
@end
@implementation MyClass
+ (int)testClassMethodByName {
    int y = [MyClass getInt];
    return 5/y; // no-warning
}
+ (int)getInt {
  return 0;
}
@end