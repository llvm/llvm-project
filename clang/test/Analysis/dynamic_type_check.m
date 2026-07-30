// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core.DynamicTypeChecker -verify %s


#define nil 0
typedef unsigned long NSUInteger;
typedef int BOOL;

@protocol NSObject
+ (id)alloc;
- (id)init;
@end

@protocol NSCopying
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject>
@end

@interface NSString : NSObject <NSCopying>
@end

@interface NSMutableString : NSString
@end

@interface NSNumber : NSObject <NSCopying>
@end

@class MyType;

void testTypeCheck(NSString* str) {
  id obj = str;
  NSNumber *num = obj; // expected-warning {{}}
  (void)num;
}

void testForwardDeclarations(NSString* str) {
  id obj = str;
  // Do not warn, since no information is available whether MyType is a sub or
  // super class of any other type.
  MyType *num = obj; // no warning
  (void)num;
}
