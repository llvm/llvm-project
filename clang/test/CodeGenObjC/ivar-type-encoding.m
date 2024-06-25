// RUN: %clang_cc1 -emit-llvm -fobjc-runtime=gcc -o - %s | FileCheck %s

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

int main(void) {
  TestClass *c = [TestClass new];
  return 0;
}

// CHECK: = private unnamed_addr constant [12 x i8] c"_stringIvar\00"
// CHECK: = private unnamed_addr constant [12 x i8] c"@\22NSString\22\00"
// CHECK: = private unnamed_addr constant [9 x i8] c"_intIvar\00"
// CHECK: = private unnamed_addr constant [2 x i8] c"i\00"

@interface Class1 {
  int : 3;
  short : 2;
  long long ll;
  char : 1;
}
@end

@implementation Class1
@end

// CHECK: @{{.*}} = private unnamed_addr constant [5 x i8] c"b0i3\00"
// CHECK: @{{.*}} = private unnamed_addr constant [5 x i8] c"b3s2\00"
// CHECK: @{{.*}} = private unnamed_addr constant [2 x i8] c"q\00"
// CHECK: @{{.*}} = private unnamed_addr constant [{{7|6}} x i8] c"b{{128|96}}c1\00"
