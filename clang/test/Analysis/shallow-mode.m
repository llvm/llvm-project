// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config mode=shallow -verify %s
// expected-no-diagnostics

void clang_analyzer_checkInlined(unsigned);

typedef signed char BOOL;
typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
@end

@interface MyClass : NSObject
+ (void)callee;
+ (void)caller;
@end

@implementation MyClass
+ (void)caller {
    [MyClass callee];
}
+ (void)callee {
  clang_analyzer_checkInlined(0); // The call is not inlined.
}
@end