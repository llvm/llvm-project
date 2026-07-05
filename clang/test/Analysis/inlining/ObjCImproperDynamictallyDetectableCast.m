// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config ipa=dynamic-bifurcate -verify %s

typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
+(id)new;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end
void clang_analyzer_eval(BOOL);

@interface SomeOtherClass : NSObject
- (int)getZero;
@end
@implementation SomeOtherClass
- (int)getZero { return 0; }
@end

@interface MyClass : NSObject
- (int)getZero;
@end

@implementation MyClass
- (int)getZero { return 1; }

// TODO: Not only we should correctly determine that the type of o at runtime 
// is MyClass, but we should also warn about it. 
+ (void) testCastToParent {
  id a = [[self alloc] init];
  SomeOtherClass *o = a;  
  clang_analyzer_eval([o getZero] == 0); // expected-warning{{FALSE}}
}
@end
