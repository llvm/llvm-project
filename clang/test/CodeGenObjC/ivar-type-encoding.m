// RUN: %clang_cc1 -S -emit-llvm -fobjc-runtime=gcc -o - %s | FileCheck %s

@protocol NSCopying
@end

@interface NSObject {
  struct objc_object *isa;
}
+ (id) new;
- (id) init;
@end

@interface NSString : NSObject <NSCopying>
+ (NSString *)foo;
@end

@interface TestClass : NSObject {
@public
  NSString    *_stringIvar;
  int         _intIvar;
}
@end
@implementation TestClass

@end

int main() {
  TestClass *c = [TestClass new];
  return 0;
}

// CHECK: @0 = private unnamed_addr constant [12 x i8] c"_stringIvar\00"
// CHECK: @1 = private unnamed_addr constant [12 x i8] c"@\22NSString\22\00"
// CHECK: @2 = private unnamed_addr constant [9 x i8] c"_intIvar\00"
// CHECK: @3 = private unnamed_addr constant [2 x i8] c"i\00"
